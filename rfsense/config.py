"""Config loading. Defaults here should match config/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    kind: str = "pluto"
    uri: str = "ip:192.168.2.1"
    file_path: str | None = None
    center_freqs_mhz: list[float] = field(
        default_factory=lambda: [2410.0, 2426.0, 2442.0, 2458.0, 2474.0])
    sample_rate_msps: float = 20.0
    rf_bandwidth_mhz: float = 20.0
    buffer_size: int = 65536
    gain_mode: str = "manual"
    gain_db: float = 40.0
    full_scale: float = 2048.0


@dataclass
class SimConfig:
    scenario: str = "fhss"
    noise_dbfs: float = -60.0
    signal_dbfs: float = -45.0
    gap_ms: float = 5.0
    seed: int = 0


@dataclass
class DspConfig:
    nfft: int = 256
    hop: int = 128
    window: str = "hann"


@dataclass
class DetectConfig:
    threshold_db: float = 10.0
    noise_percentile: float = 25.0
    noise_alpha: float = 0.1
    floor_spread_db: float = 3.0
    dc_notch_bins: int = 1
    merge_frames: int = 3
    merge_bins: int = 3
    min_frames: int = 10
    min_bins: int = 2
    dynamic_range_db: float = 35.0
    occupied_power: float = 0.9
    ofdm_min_bw_mhz: float = 8.0


@dataclass
class ClassifyConfig:
    window_s: float = 1.0
    narrow_max_bw_mhz: float = 5.0
    wide_min_bw_mhz: float = 8.0
    channel_res_mhz: float = 1.0
    fhss_min_bursts: int = 6
    fhss_min_channels: int = 8
    fhss_min_hop_rate_hz: float = 20.0
    fhss_bw_range_mhz: list[float] = field(default_factory=lambda: [0.3, 4.0])
    bt_max_burst_ms: float = 0.5
    wide_min_duty: float = 0.2
    wifi_min_corr: float = 0.10
    warning_conf: float = 0.5
    critical_conf: float = 0.75
    rssi_cal_db: float = 0.0


@dataclass
class ReportConfig:
    min_alert: str = "warning"
    confirm_count: int = 3
    report_interval_s: float = 1.0
    stats_interval_s: float = 5.0
    save_iq_dir: str | None = None


@dataclass
class OutputConfig:
    sinks: list[str] = field(default_factory=lambda: ["stdout", "serial"])
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 115200
    udp_host: str = "192.168.4.1"
    udp_port: int = 5005
    file_path: str = "logs/events.jsonl"


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    dsp: DspConfig = field(default_factory=DspConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _apply(obj: Any, data: dict, path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def default_config_path() -> str | None:
    """Use config/local.yaml if there is one, otherwise default.yaml"""
    for p in ("config/local.yaml", "config/default.yaml"):
        if Path(p).exists():
            return p
    return None


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with open(path) as f:
            _apply(cfg, yaml.safe_load(f) or {})
    return cfg
