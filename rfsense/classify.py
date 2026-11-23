"""Features over a rolling window of bursts, and some simple rules to label them.

The rules are a first guess, need to tune them on real captures. The stats log prints
the features every few seconds which should help with picking thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import ClassifyConfig
from .detect import Burst

ALERT_RANK = {"none": 0, "info": 1, "warning": 2, "critical": 3}
FRAGMENT_MARGIN_HZ = 1e6

FHSS = "FHSS/control-link-like"
BLUETOOTH = "BT-like"
WIDEBAND = "wideband/video-downlink-like"
WIFI = "WiFi-like"


@dataclass
class Features:
    observed_s: float = 0.0  # time actually captured, not counting gaps
    n_narrow: int = 0
    n_wide: int = 0
    distinct_channels: int = 0
    hop_rate_hz: float = 0.0
    narrow_med_bw_mhz: float = 0.0
    narrow_med_dur_ms: float = 0.0
    wide_duty: float = 0.0  # non Wi-Fi wide signals
    wifi_duty: float = 0.0

    def summary(self) -> str:
        return (f"obs={self.observed_s:.2f}s narrow={self.n_narrow} wide={self.n_wide} "
                f"ch={self.distinct_channels} rate={self.hop_rate_hz:.0f}/s "
                f"bw~{self.narrow_med_bw_mhz:.2f}MHz dur~{self.narrow_med_dur_ms:.2f}ms "
                f"wide_duty={self.wide_duty:.2f} wifi_duty={self.wifi_duty:.2f}")


@dataclass
class Decision:
    label: str
    alert: str
    confidence: float
    bursts: list[Burst] = field(default_factory=list)


class BurstWindow:
    """Bursts and captures from the last window_s seconds"""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self.bursts: deque[Burst] = deque()
        self.captures: deque[tuple[float, float, float]] = deque()  # (t0, duration, fc)

    def add(self, t0: float, duration: float, bursts: list[Burst], fc: float = 0.0) -> None:
        self.captures.append((t0, duration, fc))
        self.bursts.extend(bursts)
        cutoff = t0 + duration - self.window_s
        while self.captures and self.captures[0][0] < cutoff:
            self.captures.popleft()
        while self.bursts and self.bursts[0].t_start < cutoff:
            self.bursts.popleft()

    @property
    def observed_s(self) -> float:
        return sum(d for _, d, _ in self.captures)

    def observed_by_lo(self) -> dict[float, float]:
        out: dict[float, float] = {}
        for _, d, fc in self.captures:
            out[fc] = out.get(fc, 0.0) + d
        return out

    def primary(self) -> list[Burst]:
        """Bursts seen on their nearest LO, so the overlap between LOs isn't counted twice"""
        fcs = np.array(list(self.observed_by_lo()))
        if len(fcs) < 2:
            return list(self.bursts)
        return [b for b in self.bursts if fcs[np.argmin(np.abs(fcs - b.center_hz))] == b.fc_hz]


def _inside_wide(narrow: list[Burst], wide: list[Burst]) -> np.ndarray:
    """Which narrow bursts sit inside a wide one, in both time and frequency"""
    if not narrow or not wide:
        return np.zeros(len(narrow), bool)
    n = np.array([(b.t_start, b.t_start + b.duration_s, b.center_hz, b.fc_hz) for b in narrow])
    w = np.array([(b.t_start, b.t_start + b.duration_s, b.center_hz, b.bw_hz, b.fc_hz) for b in wide])
    hit = np.zeros(len(n), bool)
    for fc in np.unique(n[:, 3]):
        ni, wi = n[:, 3] == fc, w[w[:, 4] == fc]
        if len(wi) == 0:
            continue
        a = n[ni]
        in_time = (a[:, 0, None] < wi[:, 1]) & (a[:, 1, None] > wi[:, 0])
        in_freq = np.abs(a[:, 2, None] - wi[:, 2]) <= wi[:, 3] / 2 + FRAGMENT_MARGIN_HZ
        hit[ni] = (in_time & in_freq).any(axis=1)
    return hit


def _max_duty(bursts: list[Burst], by_lo: dict[float, float]) -> float:
    """Highest on-air fraction of any LO. Per LO, otherwise a constant signal looks like 1/n"""
    best = 0.0
    for fc in {b.fc_hz for b in bursts}:
        spans = sorted((b.t_start, b.t_start + b.duration_s) for b in bursts if b.fc_hz == fc)
        on_s, end = 0.0, float("-inf")
        for t0, t1 in spans:  # overlapping bursts only count once
            on_s += max(0.0, t1 - max(t0, end))
            end = max(end, t1)
        best = max(best, on_s / by_lo[fc])
    return min(1.0, best)


class Classifier:
    def __init__(self, cfg: ClassifyConfig):
        self.cfg = cfg

    def split(self, bursts) -> tuple[list[Burst], list[Burst]]:
        c = self.cfg
        wide = [b for b in bursts if b.bw_hz >= c.wide_min_bw_mhz * 1e6]
        narrow = [b for b in bursts if b.bw_hz <= c.narrow_max_bw_mhz * 1e6]
        # pieces of a wide signal look a lot like hops, so they don't count as narrow bursts
        inside = _inside_wide(narrow, wide)
        return [b for b, x in zip(narrow, inside) if not x], wide

    def features(self, window: BurstWindow) -> Features:
        c = self.cfg
        narrow, wide = self.split(window.primary())
        obs = window.observed_s
        f = Features(observed_s=obs, n_narrow=len(narrow), n_wide=len(wide))
        if obs <= 0:
            return f
        by_lo = window.observed_by_lo()
        if narrow:
            f.distinct_channels = len({round(b.center_hz / (c.channel_res_mhz * 1e6)) for b in narrow})
            # each LO only watches its slice of the band for 1/n of the time. bursts already
            # running when the buffer started would otherwise inflate the rate
            starts = sum(not b.clipped_start for b in narrow)
            f.hop_rate_hz = starts * len(by_lo) / obs
            f.narrow_med_bw_mhz = float(np.median([b.bw_hz for b in narrow])) / 1e6
            whole = [b for b in narrow if not b.truncated] or narrow
            f.narrow_med_dur_ms = float(np.median([b.duration_s for b in whole])) * 1e3
        if wide:
            wifi, other = self._split_wifi(wide)
            f.wide_duty = _max_duty(other, by_lo)
            f.wifi_duty = _max_duty(wifi, by_lo)
        return f

    def _split_wifi(self, wide: list[Burst]) -> tuple[list[Burst], list[Burst]]:
        wifi = [b for b in wide if b.cp_corr >= self.cfg.wifi_min_corr]
        return wifi, [b for b in wide if b.cp_corr < self.cfg.wifi_min_corr]

    def decide(self, window: BurstWindow, f: Features) -> list[Decision]:
        c = self.cfg
        narrow, wide = self.split(window.primary())
        out = []

        # a busy transmitter on a few fixed channels isn't hopping, however fast it keys
        if f.n_narrow >= c.fhss_min_bursts and f.distinct_channels >= c.fhss_min_channels:
            diversity = min(1.0, f.distinct_channels / c.fhss_min_channels)
            rate = min(1.0, f.hop_rate_hz / c.fhss_min_hop_rate_hz)
            lo, hi = c.fhss_bw_range_mhz
            bw_ok = 1.0 if lo <= f.narrow_med_bw_mhz <= hi else 0.0
            conf = 0.45 * diversity + 0.40 * rate + 0.15 * bw_ok
            if f.narrow_med_dur_ms < c.bt_max_burst_ms:
                # BT hops too but packets are sub-ms. not alerting on it for now
                # TODO check this against real drone and BT captures
                out.append(Decision(BLUETOOTH, "info", conf, narrow))
            else:
                out.append(Decision(FHSS, self._alert(conf), conf, narrow))

        if f.wide_duty >= c.wide_min_duty:
            # reaching the duty threshold is worth a warning, more airtime adds confidence
            over = (f.wide_duty - c.wide_min_duty) / max(1e-9, 1 - c.wide_min_duty)
            conf = c.warning_conf + (1 - c.warning_conf) * over
            # max warning for now, only checked against a few drones
            alert = min(self._alert(conf), "warning", key=ALERT_RANK.get)
            out.append(Decision(WIDEBAND, alert, conf, self._split_wifi(wide)[1]))

        if f.wifi_duty >= c.wide_min_duty:
            # could be a Wi-Fi drone (Bebop, AR) but looks the same as a router
            out.append(Decision(WIFI, "info", f.wifi_duty, self._split_wifi(wide)[0]))
        return out

    def _alert(self, conf: float) -> str:
        if conf >= self.cfg.critical_conf:
            return "critical"
        if conf >= self.cfg.warning_conf:
            return "warning"
        return "info"
