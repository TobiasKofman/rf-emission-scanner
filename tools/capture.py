"""Record raw IQ to a file so we can look at it / tune offline.

    python tools/capture.py -n 100 -o captures/quiet_room.npy
    python tools/capture.py -n 100 -o captures/drone_on.npy --gain 30

Saves a .npy with one row per buffer plus a .json with the settings.
Replay it with: python -m rfsense --file captures/drone_on.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rfsense.config import default_config_path, load_config
from rfsense.sources import make_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="default: config/local.yaml if present, else config/default.yaml")
    ap.add_argument("-n", "--num-buffers", type=int, default=50)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--source", choices=["pluto", "sim"], help="override source.kind")
    ap.add_argument("--gain", type=float, help="override source.gain_db (manual gain)")
    ap.add_argument("--fc", type=float, nargs="+", help="override center_freqs_mhz")
    args = ap.parse_args()

    cfg = load_config(args.config or default_config_path())
    if args.source:
        cfg.source.kind = args.source
    if args.gain is not None:
        cfg.source.gain_mode, cfg.source.gain_db = "manual", args.gain
    if args.fc:
        cfg.source.center_freqs_mhz = args.fc

    out = Path(args.out).with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty((args.num_buffers, cfg.source.buffer_size), np.complex64)
    fcs, t0s = [], []
    src = iter(make_source(cfg))
    for k in range(args.num_buffers):
        cap = next(src)
        data[k] = cap.iq
        fcs.append(cap.fc)
        t0s.append(cap.t0)
        peak = 20 * np.log10(np.max(np.abs(cap.iq)) + 1e-12)
        print(f"[{k + 1}/{args.num_buffers}] fc={cap.fc / 1e6:.1f} MHz  peak={peak:.1f} dBFS"
              + ("  << CLIPPING, lower gain" if peak > -1 else ""))

    np.save(out, data)
    meta = {"fs": cap.fs, "fc": fcs, "t0": t0s, "gain_db": cap.gain_db,
            "source": cfg.source.kind, "buffer_size": cfg.source.buffer_size}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({data.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
