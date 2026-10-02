#!/usr/bin/env python3
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
#####################################################################
# make sure this reflects the correct location of mawsons-chest repo
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
#####################################################################
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.forcing.era5 import ERA5Config, write_month
from shuga.core.logging import build_file_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Build one monthly CICE-ready ERA5 forcing file.")
    parser.add_argument("--year", type = int, required=True)
    parser.add_argument("--month", type = int, required=True)
    parser.add_argument("--regrid-method", type = str, default = None, help = "xESMF regridding method. Default comes from ERA5Config.")
    parser.add_argument("--extrap-method", type = str, default = None, help = "xESMF extrapolation method. Default comes from ERA5Config.")
    parser.add_argument("--weight-file", default = None, help=("Optional explicit xESMF weight filename. "
                                                               "If omitted, filename is derived as "
                                                               "map_ERA5_to_{CICE_GRID_FILENAME_STEM}_{regrid_method}_{extrap_method}.nc"))
    parser.add_argument("--cice-grid-file", type = Path, default = None, help = ("Full path to destination CICE/ACCESS-OM3 grid file. "
                                                                                 "If omitted, uses the default CICE grid from shuga.core.paths.ShugaPaths "
                                                                                 "(normally /g/data/<project>/<user>/grids/ACCESS-OM3-025_Cgrid.nc)."))
    parser.add_argument("--rebuild-weights", action = "store_true", help = "Force xESMF to rebuild the regridding weights.")
    parser.add_argument("--overwrite", action = "store_true")
    parser.add_argument("--log-level", default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type = Path, default = None, help = ("Path to a persistent log file. "
                                                                           "Default: ~/logs/forcing/build_era5_cice_month_YYYY_MM.log"))
    return parser.parse_args()

def main():
    args     = parse_args()
    level    = getattr(logging, args.log_level)
    log_file = args.log_file
    if log_file is None:
        log_file = (Path.home() / "logs" / "forcing" / f"build_era5_cice_month_{args.year:04d}_{args.month:02d}.log")
    logger = build_file_logger("shuga", log_file, level = level)
    logger.info("ERA5 -> CICE monthly forcing build")
    logger.info("year                  = %04d", args.year)
    logger.info("month                 = %02d", args.month)
    logger.info("overwrite             = %s", args.overwrite)
    logger.info("rebuild_weights       = %s", args.rebuild_weights)
    logger.info("regrid_method         = %s", args.regrid_method)
    logger.info("extrap_method         = %s", args.extrap_method)
    logger.info("cice_grid_file        = %s", args.cice_grid_file or "shuga.core.paths default")
    logger.info("weight_file           = %s", args.weight_file or "auto-derived")
    logger.info("log_file              = %s", log_file)
    cfg_kwargs = {"rebuild_weights": args.rebuild_weights}
    if args.cice_grid_file is not None:
        cfg_kwargs["cice_grid_file"] = args.cice_grid_file
    if args.regrid_method is not None:
        cfg_kwargs["regrid_method"] = args.regrid_method
    if args.extrap_method is not None:
        cfg_kwargs["extrap_method"] = args.extrap_method
    if args.weight_file is not None:
        cfg_kwargs["weight_filename"] = args.weight_file
    cfg = ERA5Config(**cfg_kwargs)
    out = write_month(year      = args.year,
                      month     = args.month,
                      cfg       = cfg,
                      overwrite = args.overwrite)
    logger.info("output_file             = %s", out)
    print(out)

if __name__ == "__main__":
    main()
