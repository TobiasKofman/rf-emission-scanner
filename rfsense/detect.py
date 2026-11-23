"""Find bursts by thresholding the spectrogram and grouping connected cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import DetectConfig
from .dsp import bin_offsets_hz, lag_corr, to_db

WIFI_SYMBOL_S = 3.2e-6  # OFDM symbol without the cyclic prefix
MAX_CORR_SAMPLES = 6000  # plenty for the correlation, keeps it cheap on the Pi


@dataclass
class Burst:
    t_start: float
    duration_s: float
    center_hz: float  # power weighted
    bw_hz: float
    power_dbfs: float
    snr_db: float
    # cut off by the start or end of the buffer, so the real duration is longer
    truncated: bool
    fc_hz: float  # LO it was captured on
    clipped_start: bool = False  # already running when the buffer started
    cp_corr: float = 0.0  # Wi-Fi OFDM fingerprint, only computed for wide bursts


def _occupied_bins(per_bin: np.ndarray, fraction: float) -> int:
    """Bins holding the middle `fraction` of the power, so leakage skirts don't count"""
    cum = np.cumsum(per_bin) / np.sum(per_bin)
    tail = (1 - fraction) / 2
    return int(np.searchsorted(cum, 1 - tail) - np.searchsorted(cum, tail) + 1)


def find_bursts(psd: np.ndarray, psd_db: np.ndarray, fs: float, fc: float, t0: float,
                hop: int, floor_db: float, cfg: DetectConfig,
                iq: np.ndarray | None = None) -> list[Burst]:
    n_frames, nfft = psd.shape
    mask = psd_db > floor_db + cfg.threshold_db
    peak_db = psd_db
    if cfg.dc_notch_bins > 0:
        # ignore the LO spike, but fill it in if both sides are on so a signal
        # sitting on the LO doesn't get split in half
        lo, hi = nfft // 2 - cfg.dc_notch_bins, nfft // 2 + cfg.dc_notch_bins + 1
        mask[:, lo:hi] = mask[:, lo - 1:lo] & mask[:, hi:hi + 1]
        peak_db = np.delete(psd_db, np.arange(lo, hi), axis=1)
    # a strong signal leaks across the whole FFT, anything this far below the frame's
    # peak is more likely leakage than another transmitter
    mask &= psd_db > peak_db.max(axis=1, keepdims=True) - cfg.dynamic_range_db

    # dilate before labeling so broken up pieces of one burst stay together, but
    # only use the real above-threshold cells when measuring it
    joined = ndimage.binary_dilation(mask, structure=np.ones((cfg.merge_frames, cfg.merge_bins), bool))
    labels, _ = ndimage.label(joined)

    offsets = bin_offsets_hz(nfft, fs)
    bin_hz = fs / nfft
    frame_s = hop / fs
    bursts = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        sub = (labels[sl] == k) & mask[sl]
        rows = np.flatnonzero(sub.any(axis=1))
        cols = np.flatnonzero(sub.any(axis=0))
        if len(rows) == 0:
            continue
        n_t = rows[-1] - rows[0] + 1
        n_f = cols[-1] - cols[0] + 1
        if n_t < cfg.min_frames or n_f < cfg.min_bins:
            continue

        p = np.where(sub, psd[sl], 0.0)
        per_bin = p.sum(axis=0)
        center = fc + float(np.sum(per_bin * offsets[sl[1]]) / np.sum(per_bin))
        per_frame = p.sum(axis=1)
        power = float(np.mean(per_frame[per_frame > 0]))
        first = sl[0].start + rows[0]
        last = sl[0].start + rows[-1]
        bw_hz = _occupied_bins(per_bin, cfg.occupied_power) * bin_hz
        cp_corr = 0.0
        if iq is not None and bw_hz >= cfg.ofdm_min_bw_mhz * 1e6:
            a = first * hop
            b = min((last + 1) * hop + nfft - hop, a + MAX_CORR_SAMPLES)
            cp_corr = lag_corr(iq[a:b], int(round(WIFI_SYMBOL_S * fs)))
        bursts.append(Burst(
            t_start=t0 + first * frame_s,
            duration_s=float(n_t * frame_s),
            center_hz=center,
            bw_hz=float(bw_hz),
            power_dbfs=float(to_db(np.float64(power))),
            snr_db=float(np.mean(psd_db[sl][sub]) - floor_db),
            truncated=bool(first == 0 or last == n_frames - 1),
            fc_hz=fc,
            clipped_start=bool(first == 0),
            cp_corr=cp_corr,
        ))
    return bursts
