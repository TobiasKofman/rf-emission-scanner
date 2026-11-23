"""Main loop: spectrogram, bursts, features, decisions, events."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classify import ALERT_RANK, BurstWindow, Classifier, Decision, Features
from .config import Config, ReportConfig
from .detect import find_bursts
from .dsp import NoiseFloor, spectrogram, to_db
from .events import DetectionEvent, build_event
from .sources import Capture

log = logging.getLogger(__name__)


@dataclass
class _Track:
    hits: int = 0
    last_sent: float = float("-inf")
    sent_rank: int = 0


class Reporter:
    """Decides which decisions actually get sent so we don't flood the ESP.

    A label has to show up confirm_count times in a row before the first event. After
    that it's resent every report_interval_s, or right away if the alert level goes up.
    """

    def __init__(self, cfg: ReportConfig):
        self.cfg = cfg
        self.tracks: dict[str, _Track] = {}

    def update(self, decisions: list[Decision], t: float) -> list[Decision]:
        min_rank = ALERT_RANK[self.cfg.min_alert]
        active = {d.label: d for d in decisions if ALERT_RANK[d.alert] >= min_rank}
        for label in list(self.tracks):
            if label not in active:
                del self.tracks[label]

        out = []
        for label, d in active.items():
            tr = self.tracks.setdefault(label, _Track())
            tr.hits += 1
            rank = ALERT_RANK[d.alert]
            if tr.hits < self.cfg.confirm_count:
                continue
            if rank > tr.sent_rank or t - tr.last_sent >= self.cfg.report_interval_s:
                out.append(d)
                tr.last_sent, tr.sent_rank = t, rank
        return out


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.noise = NoiseFloor(cfg.detect)
        self.window = BurstWindow(cfg.classify.window_s)
        self.classifier = Classifier(cfg.classify)
        self.reporter = Reporter(cfg.report)
        self.features = Features()
        self.floor_db = float("nan")

    def process(self, cap: Capture) -> list[DetectionEvent]:
        d = self.cfg.dsp
        psd = spectrogram(cap.iq, d.nfft, d.hop, d.window)
        psd_db = to_db(psd)
        self.floor_db = self.noise.update(cap.fc, psd_db)
        bursts = find_bursts(psd, psd_db, cap.fs, cap.fc, cap.t0, d.hop, self.floor_db,
                             self.cfg.detect, cap.iq)

        self.window.add(cap.t0, cap.duration, bursts, cap.fc)
        if not cap.sweep_end:
            return []
        self.features = self.classifier.features(self.window)
        decisions = self.classifier.decide(self.window, self.features)

        t = cap.t0 + cap.duration
        return [build_event(dec, t, cap.gain_db, self.cfg.classify.rssi_cal_db)
                for dec in self.reporter.update(decisions, t)]


def run(cfg: Config, source, sinks, max_captures: int | None = None) -> None:
    pipe = Pipeline(cfg)
    stats_every = cfg.report.stats_interval_s
    n, busy_s, cap_s, last_stats = 0, 0.0, 0.0, time.monotonic()

    for cap in source:
        t_start = time.perf_counter()
        events = pipe.process(cap)
        busy_s += time.perf_counter() - t_start
        cap_s += cap.duration
        n += 1

        for ev in events:
            line = ev.to_json()
            for sink in sinks:
                sink.send(line)
            if cfg.report.save_iq_dir:
                _save_iq(cfg.report.save_iq_dir, cap, ev)

        now = time.monotonic()
        if stats_every and now - last_stats >= stats_every:
            log.info("%d caps, proc %.1f ms/cap (cap %.1f ms), floor %.1f dB | %s",
                     n, 1e3 * busy_s / n, 1e3 * cap_s / n, pipe.floor_db, pipe.features.summary())
            n, busy_s, cap_s, last_stats = 0, 0.0, 0.0, now

        if max_captures is not None:
            max_captures -= 1
            if max_captures <= 0:
                break


def _save_iq(directory: str, cap: Capture, ev: DetectionEvent) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    stem = f"{ev.ts.replace(':', '')}_{ev.label.split('/')[0]}_{cap.fc / 1e6:.0f}MHz"
    np.save(path / f"{stem}.npy", cap.iq)
    (path / f"{stem}.json").write_text(ev.to_json())
