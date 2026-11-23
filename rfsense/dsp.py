"""STFT spectrogram and noise-floor tracking."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from .config import DetectConfig


def spectrogram(iq: np.ndarray, nfft: int, hop: int, window: str = "hann") -> np.ndarray:
    """Power spectrogram, shape (frames, nfft), fftshifted so DC is in the middle.

    Scaled so a frame summed over all bins equals mean |x|^2, which means summing the
    bins of a burst gives its power directly.
    """
    w = get_window(window, nfft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(iq, nfft)[::hop] * w
    X = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    return (np.abs(X) ** 2 / (nfft * np.sum(w ** 2))).astype(np.float32)


def bin_offsets_hz(nfft: int, fs: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))


def to_db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(p + 1e-20)


def lag_corr(iq: np.ndarray, lag: int) -> float:
    """Normalized autocorrelation at one lag. OFDM's cyclic prefix makes a peak here."""
    if len(iq) <= lag:
        return 0.0
    a, b = iq[:-lag], iq[lag:]
    power = 0.5 * float(np.sum(np.abs(a) ** 2 + np.abs(b) ** 2))
    return float(np.abs(np.vdot(a, b))) / power if power > 0 else 0.0


class NoiseFloor:
    """Noise floor in dB per bin, one number per LO freq, smoothed over captures.

    Uses one percentile over the whole capture instead of per bin, otherwise a
    constant signal (like a video downlink) would just become part of the floor.
    TODO: might need a per-bin correction for the passband shape once we see real data.
    """

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.floor: dict[float, float] = {}

    def update(self, fc: float, psd_db: np.ndarray) -> float:
        # noise power per cell is exponentially distributed, so the p-th percentile is
        # -ln(1-p) * mean. correct for that to get the mean noise level
        p = self.cfg.noise_percentile / 100
        est = float(np.percentile(psd_db, 100 * p)) - 10 * np.log10(-np.log(1 - p))
        prev = self.floor.get(fc)
        a = self.cfg.noise_alpha
        self.floor[fc] = est if prev is None else (1 - a) * prev + a * est
        # a wide signal can fill a whole capture and become its own floor, so don't let
        # one LO sit far above the quietest one
        return min(self.floor[fc], min(self.floor.values()) + self.cfg.floor_spread_db)
