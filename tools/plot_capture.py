"""Save a spectrogram PNG of a recorded buffer with the detected bursts boxed in red.

    python tools/plot_capture.py captures/drone_on.npy --index 3
    python tools/plot_capture.py captures/drone_on.npy --all

No display needed so it works over ssh, just open the png in vscode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from rfsense.config import default_config_path, load_config  # noqa: E402
from rfsense.detect import find_bursts  # noqa: E402
from rfsense.dsp import NoiseFloor, bin_offsets_hz, spectrogram, to_db  # noqa: E402


def plot_one(iq, fs, fc, cfg, title, out_png):
    d = cfg.dsp
    psd = spectrogram(iq, d.nfft, d.hop, d.window)
    psd_db = to_db(psd)
    floor = NoiseFloor(cfg.detect).update(fc, psd_db)
    bursts = find_bursts(psd, psd_db, fs, fc, 0.0, d.hop, floor, cfg.detect)

    f_mhz = (fc + bin_offsets_hz(d.nfft, fs)) / 1e6
    t_ms = np.arange(psd.shape[0]) * d.hop / fs * 1e3
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(psd_db.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[t_ms[0], t_ms[-1], f_mhz[0], f_mhz[-1]],
                   vmin=floor - 5, vmax=floor + 40)
    for b in bursts:
        ax.add_patch(Rectangle((b.t_start * 1e3, (b.center_hz - b.bw_hz / 2) / 1e6),
                               b.duration_s * 1e3, b.bw_hz / 1e6,
                               fill=False, edgecolor="red", linewidth=0.8))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(f"{title}  floor={floor:.1f} dB  bursts={len(bursts)}")
    fig.colorbar(im, ax=ax, label="dB / bin")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}  ({len(bursts)} bursts)")
    for b in bursts[:20]:
        print(f"  t={b.t_start * 1e3:7.2f} ms  f={b.center_hz / 1e6:8.2f} MHz  bw={b.bw_hz / 1e6:5.2f} MHz  "
              f"dur={b.duration_s * 1e3:6.2f} ms  snr={b.snr_db:5.1f} dB{'  (trunc)' if b.truncated else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy")
    ap.add_argument("-c", "--config", help="default: config/local.yaml if present, else config/default.yaml")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config or default_config_path())
    path = Path(args.npy)
    data = np.load(path, mmap_mode="r")
    meta = json.loads(path.with_suffix(".json").read_text())
    data = data[None] if data.ndim == 1 else data  # save_iq_dir dumps are a single buffer
    idx = range(data.shape[0]) if args.all else [args.index]
    for k in idx:
        fc = meta["fc"][k] if isinstance(meta.get("fc"), list) else meta.get("fc", 2440e6)
        out_png = path.with_name(f"{path.stem}_{k:03d}.png")
        plot_one(np.asarray(data[k]), meta.get("fs", cfg.source.sample_rate_msps * 1e6), fc, cfg,
                 f"{path.name}[{k}]", out_png)


if __name__ == "__main__":
    main()
