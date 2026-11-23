"""Runs the whole pipeline on simulated data."""

import json

import numpy as np
import pytest

from rfsense.classify import BLUETOOTH, FHSS, WIDEBAND, WIFI, Classifier
from rfsense.config import ClassifyConfig, Config
from rfsense.detect import Burst
from rfsense.pipeline import Pipeline
from rfsense.sources import Capture, SimSource


def run_sim(scenario: str, n: int = 120):
    cfg = Config()
    cfg.source.kind = "sim"
    cfg.sim.scenario = scenario
    pipe = Pipeline(cfg)
    src = iter(SimSource(cfg))
    events = []
    for _ in range(n):
        events += pipe.process(next(src))
    return pipe, events


def test_noise_only_is_quiet():
    _, events = run_sim("noise")
    assert events == []


def test_fhss_raises_critical_event():
    pipe, events = run_sim("fhss")
    assert events, pipe.features.summary()
    ev = events[-1]
    assert ev.label == FHSS
    assert ev.alert == "critical"
    assert 2430 <= ev.center_mhz <= 2450
    assert 0.5 <= ev.bw_mhz <= 3.0  # sim FSK comes out around 2 MHz


def test_bluetooth_is_not_alerted():
    pipe, events = run_sim("bluetooth")
    assert events == []
    labels = [d.label for d in pipe.classifier.decide(pipe.window, pipe.features)]
    assert BLUETOOTH in labels


def test_wideband_is_labelled_wideband_not_fhss():
    _, events = run_sim("wideband")
    assert {e.label for e in events} == {WIDEBAND}


def test_event_json_matches_esp_schema():
    _, events = run_sim("fhss")
    obj = json.loads(events[0].to_json())
    assert list(obj) == ["type", "ts", "alert", "label", "confidence", "rssi_dbm",
                         "center_mhz", "bw_mhz", "duration_ms"]
    assert obj["type"] == "detection"
    assert obj["ts"].endswith("Z")


def test_sweep_covers_the_band():
    pipe, _ = run_sim("fhss")
    centers = [b.center_hz / 1e6 for b in pipe.window.primary()]
    assert min(centers) < 2420 and max(centers) > 2460
    # sim hops every 5 ms, so the rate should come out near 200/s despite the slicing
    assert 120 <= pipe.features.hop_rate_hz <= 280, pipe.features.summary()


def test_constant_signal_on_one_lo_keeps_full_duty():
    pipe, _ = run_sim("wideband")
    assert pipe.features.wide_duty > 0.9, pipe.features.summary()


def single_lo_pipeline():
    cfg = Config()
    cfg.source.center_freqs_mhz = [2440.0]
    return Pipeline(cfg)


def noise(rng, n, dbfs=-60.0):
    amp = 10 ** (dbfs / 20) / np.sqrt(2)
    return amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def ofdm(rng, n, dbfs=-30.0):
    """802.11 style: 52 of 64 subcarriers at 20 MS/s with a 16 sample cyclic prefix"""
    used = np.r_[1:27, 38:64]
    out = []
    for _ in range(n // 80 + 1):
        spec = np.zeros(64, complex)
        spec[used] = rng.choice([1, -1, 1j, -1j], size=52)
        sym = np.fft.ifft(spec)
        out.append(np.r_[sym[-16:], sym])
    x = np.concatenate(out)[:n]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (dbfs / 20)
    x[int(0.4 * n):] = 0  # packets, not a constant carrier
    return x


def run_iq(pipe, make_iq, n_caps=8):
    n = pipe.cfg.source.buffer_size
    for k in range(n_caps):
        pipe.process(Capture(make_iq(n).astype(np.complex64), 20e6, 2440e6, k * 0.01, 40.0))
    return {d.label for d in pipe.classifier.decide(pipe.window, pipe.features)}


def test_ofdm_is_labelled_wifi_not_video():
    rng = np.random.default_rng(1)
    labels = run_iq(single_lo_pipeline(), lambda n: ofdm(rng, n) + noise(rng, n))
    assert WIFI in labels and WIDEBAND not in labels


def test_continuous_signal_filling_one_lo_is_detected():
    rng = np.random.default_rng(3)
    cfg = Config()
    pipe = Pipeline(cfg)
    n, fcs = cfg.source.buffer_size, [f * 1e6 for f in cfg.source.center_freqs_mhz]
    for k in range(40):
        fc = fcs[k % len(fcs)]
        x = noise(rng, n)
        if fc == 2442e6:
            s = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            s[np.abs(np.fft.fftfreq(n, 1 / 20e6)) > 9e6] = 0
            y = np.fft.ifft(s)
            x = x + y / np.sqrt(np.mean(np.abs(y) ** 2)) * 10 ** (-35 / 20)
        pipe.process(Capture(x.astype(np.complex64), 20e6, fc, k * 0.008, 40.0,
                             sweep_end=(k + 1) % len(fcs) == 0))
    labels = {d.label for d in pipe.classifier.decide(pipe.window, pipe.features)}
    assert WIDEBAND in labels, pipe.features.summary()


def test_strong_tone_is_not_measured_as_wide():
    rng = np.random.default_rng(2)
    n = Config().source.buffer_size
    tone = 0.5 * np.exp(2j * np.pi * 3.3e6 / 20e6 * np.arange(n))
    pipe = single_lo_pipeline()
    run_iq(pipe, lambda m: tone + noise(rng, m, -75.0), n_caps=2)
    assert pipe.window.bursts
    assert max(b.bw_hz for b in pipe.window.bursts) < 1e6


def test_pieces_of_a_wide_burst_are_not_hops():
    wide = Burst(0.0, 1e-3, 2440e6, 16e6, -30.0, 30.0, False, 2440e6)
    inside = [Burst(1e-4, 1e-4, (2434 + k) * 1e6, 5e5, -40.0, 20.0, False, 2440e6) for k in range(12)]
    outside = Burst(1e-4, 1e-4, 2449.5e6, 5e5, -40.0, 20.0, False, 2440e6)
    narrow, _ = Classifier(ClassifyConfig()).split([wide, *inside, outside])
    assert narrow == [outside]


@pytest.mark.parametrize("scenario", ["fhss", "mixed"])
def test_reporter_rate_limits(scenario):
    _, events = run_sim(scenario, n=200)  # about 6 s with the retune time
    per_label = {}
    for e in events:
        per_label[e.label] = per_label.get(e.label, 0) + 1
    assert all(c <= 8 for c in per_label.values()), per_label
