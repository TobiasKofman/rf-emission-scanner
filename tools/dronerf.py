"""Fetch the DroneRF dataset and replay it through the detector.

    python tools/dronerf.py fetch                 # about 4 GB download, 9 GB converted
    python tools/dronerf.py eval                  # tuning segments
    python tools/dronerf.py eval --split holdout  # only look at this once tuning is done

Dataset: Al-Sa'd et al., doi:10.17632/f4c2b4n755.1 (CC BY 4.0). Needs bsdtar to unpack it
(`sudo apt install libarchive-tools` on the Pi).

Each file is 10M real samples at 80 MS/s (125 ms). L covers the lower 40 MHz of the band and
H the upper 40. The rate was measured from the 3.2 us Wi-Fi OFDM symbol in the recordings.
The absolute frequency axis could be off by a few MHz, so don't tune on channel numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np

from rfsense.classify import ALERT_RANK
from rfsense.config import Config, default_config_path, load_config
from rfsense.pipeline import Pipeline
from rfsense.sources import Capture

API = "https://data.mendeley.com/public-api/datasets/f4c2b4n755"
RAW_FS = 80e6
BAND_START = {"L": 2400e6, "H": 2440e6}
GROUPS = {"background": ["00000"], "Bebop": ["10000", "10001", "10010", "10011"],
          "AR": ["10100", "10101", "10110", "10111"], "Phantom": ["11000"]}
MEMBER = re.compile(r"^(\d{5})([LH])_(\d+)\.csv$")
TUNE_FRACTION = 0.6  # first part of each class, the rest is held out
NOISE_LSB = 4.0  # about what the quiet background files measure
USB_GAP_S = 2e-3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _parse_member(name: str):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or len(p.parts) > 2:
        return None
    return MEMBER.match(p.name)


def _convert(job: tuple[Path, str, Path]) -> str:
    archive, member, out_dir = job
    m = _parse_member(member)
    out = out_dir / f"{m[1]}{m[2]}_{int(m[3]):02d}.npy"
    if out.exists():
        return out.name
    raw = subprocess.run(["bsdtar", "-xOf", str(archive), member], check=True,
                         capture_output=True).stdout
    x = np.fromstring(raw.decode("ascii"), sep=",", dtype=np.float64)
    if x.size != 10_000_000 or np.any(x != np.round(x)) or np.abs(x).max() > 32767:
        raise ValueError(f"unexpected data in {member}")
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, x.astype(np.int16))
    tmp.rename(out)
    return out.name


def fetch(dest: Path) -> None:
    raw_dir, npy_dir = dest / "raw", dest / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    meta = subprocess.run(["curl", "-fsSL", "--retry", "5", API], check=True, capture_output=True)
    files = json.loads(meta.stdout)["files"]

    jobs = []
    for item in files:
        d = raw_dir / item["id"]
        d.mkdir(parents=True, exist_ok=True)
        path = d / "archive.rar"
        want = item["content_details"]["sha256_hash"]
        if not path.exists() or _sha256(path) != want:
            print("downloading", item["filename"], flush=True)
            subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(path),
                            item["content_details"]["download_url"]], check=True)
            if _sha256(path) != want:
                raise ValueError(f"hash mismatch: {item['filename']}")
        names = subprocess.run(["bsdtar", "-tf", str(path)], check=True, capture_output=True,
                               text=True).stdout.splitlines()
        jobs += [(path, n, npy_dir) for n in names if _parse_member(n)]

    with ProcessPoolExecutor() as ex:
        for k, _ in enumerate(ex.map(_convert, jobs), start=1):
            if k % 50 == 0:
                print(f"converted {k}/{len(jobs)}", flush=True)
    print(f"{len(jobs)} files in {npy_dir}")


def tune_band(low: np.ndarray, high: np.ndarray, fc: float, n: int, fs: float) -> np.ndarray:
    """Cut what the Pluto would see at fc out of the two real 80 MS/s recordings."""
    decim = int(round(RAW_FS / fs))
    spec = np.r_[np.fft.rfft(low.astype(np.float32))[:len(low) // 2],
                 np.fft.rfft(high.astype(np.float32))[:len(high) // 2]]
    df = RAW_FS / len(low)
    k0 = int(round((fc - BAND_START["L"]) / df)) - n // 2
    band = np.zeros(n, np.complex64)
    a, b = max(k0, 0), min(k0 + n, len(spec))
    if b > a:
        band[a - k0:b - k0] = spec[a:b]
    return np.fft.ifft(np.fft.ifftshift(band)).astype(np.complex64) * (n / len(low)) * decim / 2


def replay(data: Path, code: str, seg: int, cfg: Config, atten_db: float) -> Pipeline:
    low = np.load(data / f"{code}L_{seg:02d}.npy", mmap_mode="r")
    high = np.load(data / f"{code}H_{seg:02d}.npy", mmap_mode="r")
    s = cfg.source
    fs, n = s.sample_rate_msps * 1e6, s.buffer_size
    decim = int(round(RAW_FS / fs))
    fcs = [f * 1e6 for f in s.center_freqs_mhz]
    step = int((2 * n / fs + USB_GAP_S) * RAW_FS)  # buffer + one thrown away after the retune
    rng = np.random.default_rng((int(code), seg))
    pipe = Pipeline(cfg)
    gain = 10 ** (-atten_db / 20)
    k, start = 0, 0
    while start + n * decim <= len(low):
        fc = fcs[k % len(fcs)]
        iq = tune_band(low[start:start + n * decim], high[start:start + n * decim], fc, n, fs)
        noise = NOISE_LSB / np.sqrt(2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        iq = ((iq * gain + noise) / s.full_scale).astype(np.complex64)
        k += 1
        pipe.process(Capture(iq, fs, fc, start / RAW_FS, s.gain_db,
                             sweep_end=k % len(fcs) == 0))
        start += step
    return pipe


def _score(job) -> tuple[str, str, list[str], dict]:
    data, group, code, seg, cfg, atten_db = job
    pipe = replay(data, code, seg, cfg, atten_db)
    decisions = pipe.classifier.decide(pipe.window, pipe.features)
    alerts = [d.label for d in decisions if ALERT_RANK[d.alert] >= ALERT_RANK["warning"]]
    return group, code, alerts, vars(pipe.features)


def segments(data: Path, split: str) -> list[tuple[str, str, int]]:
    out = []
    for group, codes in GROUPS.items():
        for code in codes:
            segs = sorted(int(p.stem[-2:]) for p in data.glob(f"{code}L_*.npy"))
            cut = int(len(segs) * TUNE_FRACTION)
            chosen = {"tune": segs[:cut], "holdout": segs[cut:], "all": segs}[split]
            out += [(group, code, s) for s in chosen]
    return out


def evaluate(data: Path, cfg: Config, split: str, atten_db: float = 0.0, workers: int = 6) -> dict:
    jobs = [(data, g, c, s, cfg, atten_db) for g, c, s in segments(data, split)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_score, jobs, chunksize=2))
    table: dict[str, dict] = {}
    for group in GROUPS:
        rows = [r for r in results if r[0] == group]
        labels = sorted({a for r in rows for a in r[2]})
        table[group] = {"n": len(rows), "alerted": sum(bool(r[2]) for r in rows) / max(1, len(rows)),
                        "labels": {x: sum(x in r[2] for r in rows) / len(rows) for x in labels},
                        "features": [r[3] for r in rows]}
    return table


def print_table(table: dict) -> None:
    for group, row in table.items():
        labels = ", ".join(f"{k} {v:.0%}" for k, v in row["labels"].items()) or "none"
        print(f"  {group:11} n={row['n']:3d}  alerted {row['alerted']:4.0%}  ({labels})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "eval"])
    ap.add_argument("-c", "--config", help="default: config/local.yaml if present, else config/default.yaml")
    ap.add_argument("--dest", default="captures/dronerf", help="where fetch puts the dataset")
    ap.add_argument("--data", help="folder of converted .npy files, default <dest>/npy")
    ap.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    ap.add_argument("--atten", type=float, nargs="+", default=[0.0],
                    help="extra path loss in dB, to mimic a drone further away")
    args = ap.parse_args()

    if args.command == "fetch":
        fetch(Path(args.dest))
        return

    cfg = load_config(args.config or default_config_path())
    cfg = replace(cfg, source=replace(cfg.source, kind="file"))
    data = Path(args.data or Path(args.dest) / "npy")
    for atten in args.atten:
        print(f"{args.split} split, {atten:.0f} dB extra path loss (alert = warning or higher)")
        print_table(evaluate(data, cfg, args.split, atten))


if __name__ == "__main__":
    main()
