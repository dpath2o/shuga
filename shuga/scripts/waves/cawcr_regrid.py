#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import (
    CICEGridSpec,
    ClassificationSpec,
    LateralDragSpec,
    ObservationSpec,
    RunSpec,
    WaveForcingSpec,
)
from shuga.waves.cawcr import CAWCRRegridConfig, CAWCRRegridder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare monthly CAWCR spectra regridded to the CICE grid and masked by NSIDC SIC."
    )
    p.add_argument("year", type=int, help="4-digit year, e.g. 1993")
    p.add_argument("month", type=int, help="Month number, 1-12")

    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--sim-name", default="LD-waves-exp01")

    p.add_argument("--hemisphere", default="SH", choices=["SH", "NH"])
    p.add_argument("--sic-threshold", type=float, default=0.15)
    p.add_argument("--k-nearest", type=int, default=5)
    p.add_argument("--idw-power", type=float, default=2.5)
    p.add_argument("--radius-km", type=float, default=1000.0)
    p.add_argument("--time-chunk", type=int, default=4)

    p.add_argument("--overwrite-nc", action="store_true", help="Overwrite output NetCDF")
    p.add_argument("--overwrite-weights", action="store_true", help="Rebuild CAWCR->CICE weights")
    p.add_argument("--overwrite-sic-weights", action="store_true", help="Rebuild NSIDC->CICE weights")

    p.add_argument("--log-file", default=None, help="Optional explicit log file path")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not (1 <= args.month <= 12):
        raise SystemExit("month must be in the range 1..12")

    year = args.year
    month = args.month
    start_date = f"{year:04d}-{month:02d}-01"
    end_date = (Path(start_date).parent if False else None)  # placeholder to keep lint quiet

    # robust month end
    import pandas as pd
    dt0 = pd.Timestamp(start_date)
    dtN = (dt0 + pd.offsets.MonthEnd(0)).normalize()
    end_date = dtN.strftime("%Y-%m-%d")

    run = RunSpec(
        sim_name=args.sim_name,
        start_date=f"{year:04d}-01-01",
        end_date=f"{year:04d}-12-31",
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )
    obs = ObservationSpec()
    grid = CICEGridSpec()
    waves = WaveForcingSpec()
    drag = LateralDragSpec()
    classify = ClassificationSpec()

    paths = ShugaPaths(
        run=run,
        classify=classify,
        observations=obs,
        wave_forcing=waves,
        cice_grid=grid,
        lateral_drag=drag,
    )

    log_file = (
        Path(args.log_file)
        if args.log_file is not None
        else paths.logs_root_path / "waves" / f"cawcr_regrid_{year:04d}{month:02d}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = build_file_logger("shuga.waves.cawcr", log_file)

    cfg = CAWCRRegridConfig(
        hemisphere=args.hemisphere,
        sic_threshold=args.sic_threshold,
        k_nearest=args.k_nearest,
        idw_power=args.idw_power,
        radius_km=args.radius_km,
        output_path=paths.cawcr_regridded_file(year, month),
        weights_path=paths.cawcr2cice_weight_file(year, month),
        sic_weights_path=paths.nsidc2cice_weight_file,
    )

    regridder = CAWCRRegridder(cfg, logger=logger)

    out_path = regridder.prepare_month(
        start_date=start_date,
        end_date=end_date,
        paths=paths,
        overwrite_weights=args.overwrite_weights,
        overwrite_sic_weights=args.overwrite_sic_weights,
        write=True,
        overwrite_output=args.overwrite_nc,
        time_chunk=args.time_chunk,
        return_dataset=False,
    )
    print(out_path)


if __name__ == "__main__":
    main()
