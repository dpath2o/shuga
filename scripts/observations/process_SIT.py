#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shuga.observations.sea_ice_thickness import (
    AWI_ROOT, ESA_ROOT, PROCESSED_ROOT,
    build_continuous_sit, build_source_store,
    discover_awi_l3cp, discover_esa_l3c,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemisphere", choices=("SH", "NH"), required=True)
    ap.add_argument("--esa-root", type=Path, default=ESA_ROOT)
    ap.add_argument("--awi-root", type=Path, default=AWI_ROOT)
    ap.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    ap.add_argument("--prefer", choices=("AWI", "ESA"), default="AWI")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    esa = discover_esa_l3c(args.esa_root, hemisphere=args.hemisphere)
    awi = discover_awi_l3cp(args.awi_root, hemisphere=args.hemisphere)

    logging.info(
        "ESA v4.0 L3C files=%d; AWI L3CP files=%d",
        len(esa), len(awi),
    )

    esa_out = args.output_root / args.hemisphere / "ESA"
    awi_out = args.output_root / args.hemisphere / "AWI"

    esa_ts = None
    awi_ts = None

    if esa:
        build_source_store(
            esa,
            output=esa_out,
            source_name="ESA",
            hemisphere=args.hemisphere,
            overwrite=args.overwrite,
        )
        esa_ts = esa_out / "SIT_timeseries.zarr"
    else:
        logging.warning(
            "No ESA v4.0 L3C files currently found for %s; continuing with AWI.",
            args.hemisphere,
        )

    if awi:
        build_source_store(
            awi,
            output=awi_out,
            source_name="AWI",
            hemisphere=args.hemisphere,
            overwrite=args.overwrite,
        )
        awi_ts = awi_out / "SIT_timeseries.zarr"
    else:
        logging.warning(
            "No AWI L3CP files currently found for %s; continuing with ESA.",
            args.hemisphere,
        )

    if esa_ts is None and awi_ts is None:
        raise FileNotFoundError(
            f"No ESA or AWI gridded SIT files found for {args.hemisphere}."
        )

    combined = (
        args.output_root / args.hemisphere
        / "continuous" / "SIT_timeseries.zarr"
    )

    build_continuous_sit(
        esa_timeseries=esa_ts,
        awi_timeseries=awi_ts,
        output=combined,
        hemisphere=args.hemisphere,
        prefer=args.prefer,
        overwrite=args.overwrite,
    )

    logging.info("Continuous SIT time series -> %s", combined)


if __name__ == "__main__":
    main()
