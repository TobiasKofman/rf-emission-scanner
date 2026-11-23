"""Run the detector. Try `python -m rfsense --scenario fhss --sinks stdout` without hardware."""

from __future__ import annotations

import argparse
import logging

from .config import default_config_path, load_config
from .pipeline import run
from .sinks import make_sinks
from .sources import make_source


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rfsense", description=__doc__)
    ap.add_argument("-c", "--config", help="default: config/local.yaml if present, else config/default.yaml")
    ap.add_argument("--source", choices=["pluto", "file", "sim"], help="override source.kind")
    ap.add_argument("--file", help="IQ recording to replay (implies --source file)")
    ap.add_argument("--scenario", help="override sim.scenario (implies --source sim)")
    ap.add_argument("--sinks", help="comma-separated, e.g. stdout,serial,file")
    ap.add_argument("--serial-port", help="override output.serial_port")
    ap.add_argument("-n", "--max-captures", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_path = args.config or default_config_path()
    logging.getLogger("rfsense").info("config: %s", config_path)
    cfg = load_config(config_path)
    if args.file:
        cfg.source.kind, cfg.source.file_path = "file", args.file
    if args.scenario:
        cfg.source.kind, cfg.sim.scenario = "sim", args.scenario
    if args.source:
        cfg.source.kind = args.source
    if args.sinks:
        cfg.output.sinks = [s.strip() for s in args.sinks.split(",") if s.strip()]
    if args.serial_port:
        cfg.output.serial_port = args.serial_port

    sinks = make_sinks(cfg.output)
    try:
        run(cfg, make_source(cfg), sinks, args.max_captures)
    except KeyboardInterrupt:
        pass
    finally:
        for s in sinks:
            s.close()


if __name__ == "__main__":
    main()
