#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
#####################################################################
# make sure this reflects the correct location of mawsons-chest repo
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
#####################################################################
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga              import (ClassificationSpec,
                                CICEClassifier,
                                RunSpec,
                                ShugaPaths)
from shuga.core.data_conversion import NC2Zarr
from shuga.core.logging import build_file_logger
from shuga.core.naming  import normalize_method

def _comma_split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify CICE fast ice into raw, binary-days, and rolling-mean masks.")
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--grid-type", "--BorC2T-type", dest="grid_type", default="Tc")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--methods", default="raw,binary-days,rolling-mean")
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--aice-thresh", type=float, default=0.15)
    p.add_argument("--cice-store", default=None)
    p.add_argument("--static-store", default=None)
    p.add_argument("--classification-root", default=None)
    p.add_argument("--afim-output-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--archive-root", default=None)
    p.add_argument("--daily-root", default=None)
    p.add_argument("--netcdf-engine", default="scipy")
    p.add_argument("--overwrite-history", action="store_true")
    p.add_argument("--overwrite-static", action="store_true")
    p.add_argument("--delete-original", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p

def main() -> None:
    args         = build_parser().parse_args()
    methods      = [normalize_method(m) for m in _comma_split(args.methods)]
    run          = RunSpec(sim_name   = args.sim_name,
                           start_date = args.start_date,
                           end_date   = args.end_date,
                           hemisphere = args.hemisphere,
                           project    = args.project,
                           user       = args.user)
    classify     = ClassificationSpec(ice_type     = args.ice_type,
                                      grid_type    = args.grid_type,
                                      ispd_thresh  = args.ispd_thresh,
                                      aice_thresh  = args.aice_thresh,
                                      methods      = tuple(methods),
                                      bin_window   = args.bin_window,
                                      bin_min_days = args.bin_min_days,
                                      roll_window  = args.roll_window)
    paths        = ShugaPaths(run                 = run,
                              classify            = classify,
                              afim_output_root    = args.afim_output_root,
                              cice_store          = args.cice_store,
                              static_store        = args.static_store,
                              classification_root = args.classification_root,
                              logs_root           = args.logs_root,
                              archive_root        = args.archive_root)
    logger = build_file_logger("shuga.classify", paths.classification_log_path(), level=args.log_level)
    logger.info("Logging to: %s", paths.classification_log_path())
    converter = NC2Zarr(paths=paths, logger=logger, netcdf_engine=args.netcdf_engine)
    conv      = converter.ensure_iceh_stores(dt0_str          = args.start_date,
                                             dtN_str          = args.end_date,
                                             daily_root       = args.daily_root,
                                             overwrite        = args.overwrite_history,
                                             overwrite_static = args.overwrite_static,
                                             delete_original  = args.delete_original )
    logger.info("Resolved/updated CICE store: %s", conv.cice_store)
    if conv.static_store is not None:
        logger.info("Resolved/updated static store: %s", conv.static_store)
    logger.info("nc2zarr summary: months_scanned=%d months_written=%d months_rewritten=%d months_skipped=%d daily_files_seen=%d daily_files_used=%d",
                conv.months_scanned, conv.months_written, conv.months_rewritten, conv.months_skipped, conv.daily_files_seen, conv.daily_files_used)
    logger.info("Resolved classification root: %s", paths.classification_root_path)
    runner  = CICEClassifier(run=run, classify=classify, paths=paths, logger=logger)
    outputs = runner.run_methods(methods=methods, overwrite=args.overwrite)
    for method, path in outputs.items():
        logger.info("Wrote %s classification: %s", method, path)

if __name__ == "__main__":
    main()
