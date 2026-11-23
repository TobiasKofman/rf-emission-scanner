"""IQ sources: the Pluto, a recorded file, or a simulator.

Each one yields Capture objects. There are gaps between captures since USB 2.0 can't
keep up with 20 MSPS, so every buffer gets processed on its own.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Capture:
    iq: np.ndarray  # complex64, scaled so full scale is ~1.0
    fs: float
    fc: float
    t0: float  # unix time of first sample
    gain_db: float
    sweep_end: bool = True  # last LO of the sweep, time to run the classifier

    @property
    def duration(self) -> float:
        return len(self.iq) / self.fs


def make_source(cfg: Config) -> Iterator[Capture]:
    kind = cfg.source.kind
    if kind == "pluto":
        return PlutoSource(cfg)
    if kind == "file":
        return FileSource(cfg)
    if kind == "sim":
        return SimSource(cfg)
    raise ValueError(f"unknown source kind: {kind}")


class PlutoSource:
    def __init__(self, cfg: Config):
        import adi  # pyadi-iio, only needed on the pi

        s = cfg.source
        self.fs = s.sample_rate_msps * 1e6
        self.fcs = [f * 1e6 for f in s.center_freqs_mhz]
        self.full_scale = s.full_scale
        self.gain_db = s.gain_db

        log.info("connecting to Pluto at %s", s.uri)
        sdr = adi.Pluto(s.uri)
        sdr.sample_rate = int(self.fs)
        sdr.rx_rf_bandwidth = int(s.rf_bandwidth_mhz * 1e6)
        sdr.rx_buffer_size = s.buffer_size
        sdr.gain_control_mode_chan0 = s.gain_mode
        if s.gain_mode == "manual":
            sdr.rx_hardwaregain_chan0 = s.gain_db
        sdr.rx_lo = int(self.fcs[0])
        try:
            # otherwise rx() right after a retune can hand back old samples
            sdr._rxadc.set_kernel_buffers_count(1)
        except Exception as e: # private api, not a big deal if it fails
            log.warning("could not set kernel buffer count: %s", e)
        self.sdr = sdr

    def __iter__(self) -> Iterator[Capture]:
        i = 0
        while True:
            fc = self.fcs[i % len(self.fcs)]
            if len(self.fcs) > 1:
                self.sdr.rx_lo = int(fc)
                self.sdr.rx()  # throw away, LO was still settling
            iq = self.sdr.rx()
            t_end = time.time()
            iq = (np.asarray(iq) / self.full_scale).astype(np.complex64)
            gain = self.gain_db
            if self.sdr.gain_control_mode_chan0 != "manual":
                gain = float(self.sdr.rx_hardwaregain_chan0)
            i += 1
            yield Capture(iq, self.fs, fc, t_end - len(iq) / self.fs, gain,
                          sweep_end=i % len(self.fcs) == 0)


class FileSource:
    """Replays a .npy (and its .json) from tools/capture.py"""

    def __init__(self, cfg: Config):
        path = Path(cfg.source.file_path or "")
        self.data = np.load(path, mmap_mode="r")
        with open(path.with_suffix(".json")) as f:
            self.meta = json.load(f)

    def __iter__(self) -> Iterator[Capture]:
        m = self.meta
        n_lo = len(set(m["fc"]))
        for k in range(self.data.shape[0]):
            yield Capture(np.asarray(self.data[k], dtype=np.complex64),
                          m["fs"], m["fc"][k], m["t0"][k], m["gain_db"],
                          sweep_end=(k + 1) % n_lo == 0)


class SimSource:
    """Fake signals so we can work without the hardware.

    scenario can be noise, fhss (RC style link, ~1 MHz hops every 5 ms across the band),
    bluetooth (625 us slots, short packets), wideband (constant 10 MHz signal, stand-in
    for a video downlink) or mixed (fhss and wideband together).
    """

    def __init__(self, cfg: Config):
        s, sim = cfg.source, cfg.sim
        self.fs = s.sample_rate_msps * 1e6
        self.fcs = [f * 1e6 for f in s.center_freqs_mhz]
        self.n = s.buffer_size
        self.gain_db = s.gain_db
        self.scenario = sim.scenario
        self.noise_amp = 10 ** (sim.noise_dbfs / 20)
        self.sig_amp = 10 ** (sim.signal_dbfs / 20)
        self.gap = sim.gap_ms / 1e3
        if len(self.fcs) > 1:
            self.gap += self.n / self.fs  # the Pluto throws away a buffer after each retune
        self.wide_fc = self.fcs[len(self.fcs) // 2]  # the video link stays on one channel
        self.seed = sim.seed
        self.rng = np.random.default_rng(sim.seed)
        self.t_start = self.t = time.time()

    def __iter__(self) -> Iterator[Capture]:
        i = 0
        while True:
            fc = self.fcs[i % len(self.fcs)]
            i += 1
            yield Capture(self._synth(fc), self.fs, fc, self.t, self.gain_db,
                          sweep_end=i % len(self.fcs) == 0)
            self.t += self.n / self.fs + self.gap

    def _synth(self, fc: float) -> np.ndarray:
        n, rng = self.n, self.rng
        x = self.noise_amp / np.sqrt(2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        if self.scenario in ("fhss", "mixed"):
            self._add_hopper(x, fc, hop_s=5e-3, burst_s=1.5e-3, n_ch=79,
                             sym_rate=500e3, dev=250e3, p_tx=1.0)
        if self.scenario == "bluetooth":
            self._add_hopper(x, fc, hop_s=625e-6, burst_s=366e-6, n_ch=79,
                             sym_rate=1e6, dev=160e3, p_tx=0.5)
        if self.scenario in ("wideband", "mixed") and fc == self.wide_fc:
            x += self.sig_amp * self._band_noise(n, 10e6)
        return x.astype(np.complex64)

    def _add_hopper(self, x, fc, hop_s, burst_s, n_ch, sym_rate, dev, p_tx):
        t = self.t - self.t_start # relative to start so tests are repeatable
        t_end = t + self.n / self.fs
        for k in range(int(t // hop_s), int(t_end // hop_s) + 1):
            slot_rng = np.random.default_rng((self.seed, k))
            if slot_rng.random() > p_tx:
                continue
            offset = 2402e6 + slot_rng.integers(n_ch) * 1e6 - fc
            if abs(offset) > self.fs / 2 - 1e6:
                continue
            i0 = max(0, int(round((k * hop_s - t) * self.fs)))
            i1 = min(self.n, int(round((k * hop_s + burst_s - t) * self.fs)))
            if i1 <= i0:
                continue
            m = i1 - i0
            sps = max(1, int(self.fs / sym_rate))
            syms = slot_rng.choice([-1.0, 1.0], size=m // sps + 1).repeat(sps)[:m]
            phase = 2 * np.pi * np.cumsum(dev * syms + offset) / self.fs
            x[i0:i1] += self.sig_amp * np.exp(1j * phase)

    def _band_noise(self, n, bw):
        spec = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        spec[np.abs(np.fft.fftfreq(n, 1 / self.fs)) > bw / 2] = 0
        y = np.fft.ifft(spec)
        return y / np.sqrt(np.mean(np.abs(y) ** 2))
