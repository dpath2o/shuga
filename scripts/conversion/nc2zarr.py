#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga import ClassificationSpec, RunSpec, ShugaPaths
from shuga.core.data_conversion import NC2Zarr
from shuga.core.logging import build_file_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert daily CICE iceh NetCDF history files to grouped monthly Zarr.")
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--grid-type", default="Tc")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--afim-output-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--archive-root", default=None)
    p.add_argument("--daily-root", default=None)
    p.add_argument("--cice-store", default=None)
    p.add_argument("--static-store", default=None)
    p.add_argument("--netcdf-engine", default="scipy")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-static", action="store_true")
    p.add_argument("--delete-original", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_parser().parse_args()

    run = RunSpec(
        sim_name=args.sim_name,
        start_date=args.start_date or "1900-01-01",
        end_date=args.end_date or "2100-12-31",
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )
    classify = ClassificationSpec(
        ice_type=args.ice_type,
        grid_type=args.grid_type,
        ispd_thresh=args.ispd_thresh,
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
    )
    paths = ShugaPaths(
        run=run,
        classify=classify,
        afim_output_root=args.afim_output_root,
        logs_root=args.logs_root,
        archive_root=args.archive_root,
        cice_store=args.cice_store,
        static_store=args.static_store,
    )

    log_path = paths.logs_root_path / "classification" / f"nc2zarr_{args.sim_name}.log"
    logger = build_file_logger("shuga.nc2zarr", log_path, level=args.log_level)
    logger.info("Logging to: %s", log_path)

    converter = NC2Zarr(paths=paths, logger=logger, netcdf_engine=args.netcdf_engine)
    result = converter.ensure_iceh_stores(
        dt0_str=args.start_date,
        dtN_str=args.end_date,
        daily_root=args.daily_root,
        overwrite=args.overwrite,
        overwrite_static=args.overwrite_static,
        delete_original=args.delete_original,
    )

    logger.info("CICE grouped store : %s", result.cice_store)
    logger.info("CICE static store  : %s", result.static_store)
    logger.info(
        "Summary months_scanned=%d months_written=%d months_rewritten=%d months_skipped=%d daily_files_seen=%d daily_files_used=%d",
        result.months_scanned,
        result.months_written,
        result.months_rewritten,
        result.months_skipped,
        result.daily_files_seen,
        result.daily_files_used,
    )
    print(result)


if __name__ == "__main__":
    main()
