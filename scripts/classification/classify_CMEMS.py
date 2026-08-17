#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga.classify.cmems import CMEMSClassifier


def _comma_split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify native-grid CMEMS sea ice into SI, FI and PI."
    )
    p.add_argument(
        "--root",
        default="/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily",
    )
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--aice-thresh", type=float, default=0.15)
    p.add_argument(
        "--methods",
        default="raw,binary-days,rolling-mean",
    )
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--chunks-latitude", type=int, default=256)
    p.add_argument("--chunks-longitude", type=int, default=540)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-static", action="store_true")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("shuga.classify.cmems")

    runner = CMEMSClassifier(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        hemisphere=args.hemisphere,
        ispd_thresh=args.ispd_thresh,
        aice_thresh=args.aice_thresh,
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
        chunks={
            "time": args.chunks_time,
            "latitude": args.chunks_latitude,
            "longitude": args.chunks_longitude,
        },
        logger=logger,
    )
    outputs = runner.run_methods(
        methods=_comma_split(args.methods),
        overwrite=args.overwrite,
        overwrite_static=args.overwrite_static,
    )
    for key, path in outputs.items():
        logger.info("%s -> %s", key, path)


if __name__ == "__main__":
    main()
