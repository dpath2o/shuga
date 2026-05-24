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
                                CICEGridSpec,
                                ShugaPaths)
from shuga.core.logging import build_file_logger
from shuga.core.naming  import normalize_method
from shuga.core.data_conversion import NC2Zarr

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
    p.add_argument("--iceh-frequency",
                   choices=["daily", "hourly"],
                   default="daily",
                   help="CICE history frequency. daily -> iceh_daily.zarr/YYYY-MM; hourly -> iceh_hourly.zarr/YYYY_MM_DD.")
    p.add_argument("--hourly-root",
                   default=None,
                   help="Optional root containing hourly CICE NetCDF files, e.g. ~/AFIM_archive/SIM/history/hourly.")
    p.add_argument("--chunks-time",
                   type=int,
                   default=None,
                   help="Time chunk size for conversion/loading. Defaults to 31 for daily and 24 for hourly.")
    p.add_argument("--skip-history-conversion",
                   action="store_true",
                   help="Skip NetCDF-to-Zarr conversion and classify from existing iceh_daily/hourly.zarr stores.")
    p.add_argument("--grid-file", default=None)
    p.add_argument("--kmt-file", default=None)
    p.add_argument("--bathymetry-file", default=None)
    p.add_argument("--f2-file", default=None)
    p.add_argument("--gridcpl-file", default=None)
    p.add_argument("--ice-in-file", default=None)
    p.add_argument("--persist-grid-assets", action="store_true")
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
    if str(args.ice_type).strip().upper() != "FI":
        raise ValueError("The classification workflow must be run with --ice-type FI. "
                         "It writes both FI and PI classification stores from the FI parent mask. "
                         "Use --ice-type PI only in the metrics workflow.")
    methods      = [normalize_method(m) for m in _comma_split(args.methods)]
    run          = RunSpec(sim_name       = args.sim_name,
                           start_date     = args.start_date,
                           end_date       = args.end_date,
                           hemisphere     = args.hemisphere,
                           project        = args.project,
                           user           = args.user,
                           iceh_frequency = args.iceh_frequency)
    classify     = ClassificationSpec(ice_type     = args.ice_type,
                                      grid_type    = args.grid_type,
                                      ispd_thresh  = args.ispd_thresh,
                                      aice_thresh  = args.aice_thresh,
                                      methods      = tuple(methods),
                                      bin_window   = args.bin_window,
                                      bin_min_days = args.bin_min_days,
                                      roll_window  = args.roll_window)
    cice_grid    = CICEGridSpec(grid_file       = args.grid_file,
                                kmt_file        = args.kmt_file,
                                bathymetry_file = args.bathymetry_file,
                                f2_file         = args.f2_file,
                                gridcpl_file    = args.gridcpl_file,
                                ice_in_file     = args.ice_in_file)
    paths        = ShugaPaths(run                 = run,
                              classify            = classify,
                              afim_output_root    = args.afim_output_root,
                              cice_store          = args.cice_store,
                              static_store        = args.static_store,
                              cice_grid           = cice_grid,
                              classification_root = args.classification_root,
                              logs_root           = args.logs_root,
                              archive_root        = args.archive_root)
    logger = build_file_logger("shuga.classify", paths.classification_log_path(), level=args.log_level)
    logger.info("Logging to: %s", paths.classification_log_path())
    has_explicit_grid_assets = any(v is not None for v in (args.grid_file,
                                                           args.kmt_file,
                                                           args.bathymetry_file,
                                                           args.f2_file,
                                                           args.gridcpl_file,
                                                           args.ice_in_file))
    if args.persist_grid_assets and has_explicit_grid_assets:
        cfg = paths.persist_cice_grid_assets(grid_spec=cice_grid, overwrite=True)
        logger.info("Persisted CICE grid assets: %s", cfg)
    elif args.persist_grid_assets:
        logger.warning("--persist-grid-assets requested but no explicit grid assets were provided; "
                       "leaving any existing config unchanged.")
    grid_assets = paths.resolve_cice_grid_assets()
    if grid_assets is None:
        raise RuntimeError("resolve_cice_grid_assets() returned None")
    logger.info("Resolved CICE grid file: %s", grid_assets["grid_file"])
    logger.info("Resolved CICE KMT file : %s", grid_assets["kmt_file"])
    chunks_time = args.chunks_time
    if chunks_time is None:
        chunks_time = 24 if args.iceh_frequency == "hourly" else 31
    chunks = {"time": chunks_time}
    if args.skip_history_conversion:
        logger.info("--skip-history-conversion requested; using existing CICE Zarr/static stores.")
        logger.info("Resolved CICE store target: %s", paths.resolve_cice_store())
        logger.info("Resolved static store    : %s", paths.resolve_static_store())
    else:
        converter = NC2Zarr(paths         = paths,
                            logger        = logger,
                            chunks        = chunks,
                            netcdf_engine = args.netcdf_engine)
        conv = converter.ensure_iceh_stores(dt0_str          = args.start_date,
                                            dtN_str          = args.end_date,
                                            daily_root       = args.daily_root,
                                            hourly_root      = args.hourly_root,
                                            overwrite        = args.overwrite_history,
                                            overwrite_static = args.overwrite_static,
                                            delete_original  = args.delete_original)
        logger.info("Resolved/updated CICE store: %s", conv.cice_store)
        if conv.static_store is not None:
            logger.info("Resolved/updated static store: %s", conv.static_store)
        logger.info("nc2zarr summary: groups_scanned=%d groups_written=%d groups_rewritten=%d "
                    "groups_skipped=%d source_files_seen=%d source_files_used=%d",
                    conv.months_scanned,
                    conv.months_written,
                    conv.months_rewritten,
                    conv.months_skipped,
                    conv.daily_files_seen,
                    conv.daily_files_used)
    logger.info("Resolved classification root: %s", paths.classification_root_path)
    runner = CICEClassifier(run      = run,
                            classify = classify,
                            paths    = paths,
                            chunks   = chunks,
                            logger   = logger)
    # runner  = CICEClassifier(run=run, classify=classify, paths=paths, logger=logger)
    outputs = runner.run_methods(methods=methods, overwrite=args.overwrite)
    for method, path in outputs.items():
        logger.info("Wrote %s classification: %s", method, path)

if __name__ == "__main__":
    main()
