#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging
from pathlib import Path
from shuga.observations.sea_ice_thickness import (
    AWI_ROOT, ESA_ROOT, PROCESSED_ROOT,
    build_source_store, discover_awi_l3cp, discover_esa_l3c,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemisphere", choices=("SH","NH"), required=True)
    ap.add_argument("--esa-root", type=Path, default=ESA_ROOT)
    ap.add_argument("--awi-root", type=Path, default=AWI_ROOT)
    ap.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    ap.add_argument("--sources", default="ESA,AWI")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = {x.strip().upper() for x in args.sources.split(",") if x.strip()}

    if "ESA" in sources:
        files = discover_esa_l3c(args.esa_root, hemisphere=args.hemisphere)
        logging.info("ESA v4.0 L3C files=%d", len(files))
        if files:
            build_source_store(files, output=args.output_root/args.hemisphere/"ESA",
                               source_name="ESA", hemisphere=args.hemisphere,
                               overwrite=args.overwrite)

    if "AWI" in sources:
        files = discover_awi_l3cp(args.awi_root, hemisphere=args.hemisphere)
        logging.info("AWI L3CP files=%d", len(files))
        if files:
            build_source_store(files, output=args.output_root/args.hemisphere/"AWI",
                               source_name="AWI", hemisphere=args.hemisphere,
                               overwrite=args.overwrite)

if __name__ == "__main__":
    main()
