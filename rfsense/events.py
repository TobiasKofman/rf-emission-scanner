"""Detection events, sent to the ESP as one JSON object per line."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from .classify import Decision


@dataclass
class DetectionEvent:
    type: str
    ts: str # UTC, like 2026-11-13T14:32:05Z
    alert: str # info, warning or critical
    label: str
    confidence: float
    rssi_dbm: float # not calibrated yet
    center_mhz: float # middle of the span the bursts covered
    bw_mhz: float # median burst bandwidth
    duration_ms: float # first burst start to last burst end

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_event(d: Decision, t: float, gain_db: float, rssi_cal_db: float) -> DetectionEvent:
    b = d.bursts
    centers = [x.center_hz for x in b]
    start = min(x.t_start for x in b)
    end = max(x.t_start + x.duration_s for x in b)
    return DetectionEvent(
        type="detection",
        ts=iso_utc(t),
        alert=d.alert,
        label=d.label,
        confidence=round(float(d.confidence), 2),
        rssi_dbm=round(float(np.median([x.power_dbfs for x in b])) - gain_db + rssi_cal_db, 1),
        center_mhz=round((min(centers) + max(centers)) / 2e6, 1),
        bw_mhz=round(float(np.median([x.bw_hz for x in b])) / 1e6, 1),
        duration_ms=round(float(end - start) * 1e3, 1),
    )
