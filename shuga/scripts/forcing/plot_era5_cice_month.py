#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, logging, sys
from pathlib import Path
import xarray as xr
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.plotting import ERA5Plotter
from shuga.core.paths import ShugaPaths

def parse_region(value: str):
    parts = [float(v) for v in value.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must have four values: lon_min lon_max lat_min lat_max")
    return tuple(parts)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot hourly maps from monthly ERA5 -> CICE forcing files.")
    parser.add_argument("--file",       type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hours",      type=int,  default=48)
    parser.add_argument("--variables",  nargs="+", default=None)
    parser.add_argument("--region",     type=parse_region, default=(-180.0, 180.0, -90.0, -45.0), help="--region=-180,180,-90,-45")
    parser.add_argument("--projection", default="S0/-90/16c")
    parser.add_argument("--stride",     type=int,  default=2)
    parser.add_argument("--show",       action="store_true")
    parser.add_argument("--log-level",  default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return parser.parse_args()

_COORD_VARS = {"LON","LAT","lon","lat","TLON","TLAT","longitude","latitude"}

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    if not args.file.exists():
        raise FileNotFoundError(args.file)

    D_out = args.output_dir
    if D_out is None:
        D_out = ShugaPaths().graphics_root_path / "forcing" / "ERA5"

    # ── probe: discover variables without loading any data ──────────────────
    with xr.open_dataset(args.file, engine="netcdf4") as probe:
        all_data_vars = list(probe.data_vars)
        if args.variables is None:
            variables = [v for v in all_data_vars
                         if v not in _COORD_VARS and "time" in probe[v].dims]
        else:
            variables = args.variables
        ntime_file = int(probe.sizes.get("time", 0))

    ntime = min(args.hours, ntime_file)
    logging.info("File      : %s", args.file)
    logging.info("Variables : %s", ", ".join(variables))
    logging.info("Hours     : %d  (file has %d timesteps)", ntime, ntime_file)
    logging.info("Region    : %s", args.region)
    logging.info("Stride    : %d", args.stride)
    logging.info("Output    : %s", D_out)

    plotter = ERA5Plotter()
    saved   = []

    for variable in variables:
        logging.info("── variable: %s ──", variable)
        # Keep only this variable + coordinate arrays; drop everything else.
        # This caps RAM at  1 var × ntime × ny × nx  (a few hundred MB at most).
        drop = [v for v in all_data_vars if v not in {variable} | _COORD_VARS]
        with xr.open_dataset(args.file, engine="netcdf4", drop_variables=drop) as var_ds:
            for i_t in range(ntime):
                p = plotter.plot_hour(var_ds, variable, i_t,
                                      D_out      = D_out,
                                      region     = args.region,
                                      projection = args.projection,
                                      stride     = args.stride,
                                      show       = args.show)
                saved.append(p)
                logging.info("  [%d/%d] %s", i_t + 1, ntime, p.name)
        gc.collect()   # release the variable's memory before the next one

    logging.info("Saved %d figures", len(saved))
    for path in saved[:10]:
        logging.info("  %s", path)
    if len(saved) > 10:
        logging.info("  ...")

if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse
# import logging
# import sys
# from pathlib import Path
# import xarray as xr
# repo_root = Path(__file__).resolve().parents[3]
# if str(repo_root) not in sys.path:
#     sys.path.insert(0, str(repo_root))
# from shuga.plotting import ERA5Plotter
# from shuga.core.paths import ShugaPaths

# def parse_region(value: str):
#     parts = [float(v) for v in value.replace(",", " ").split()]
#     if len(parts) != 4:
#         raise argparse.ArgumentTypeError("region must have four values: lon_min lon_max lat_min lat_max")
#     return tuple(parts)

# def parse_args():
#     parser = argparse.ArgumentParser(description = "Plot hourly maps from monthly ERA5 -> CICE forcing files.")
#     parser.add_argument("--file", type = Path, required = True, help = "Monthly ERA5 -> CICE NetCDF file, e.g. era5_for_cice6_1994_10.nc.")
#     parser.add_argument("--output-dir", type = Path, default = None, help = "Directory for PNG output. Default: /g/data/gv90/da1339/GRAPHICAL/forcing/ERA5/<variable>.")
#     parser.add_argument("--hours", type = int, default = 48, help = "Number of hourly records to plot from the start of the file.")
#     parser.add_argument("--variables",  nargs = "+",  default = None, help = "Variables to plot. Default: all time-dependent data variables.")
#     parser.add_argument("--region", type = parse_region, default = (-180.0, 180.0, -90.0, -45.0), help = ("Plot region as lon_min,lon_max,lat_min,lat_max. "
#                                                                                                           "If values are negative, prefer --region=-180,180,-90,-45 "
#                                                                                                           "rather than '--region -180,180,-90,-45'. "
#                                                                                                           "Default: -180,180,-90,-45."))
#     parser.add_argument("--projection", default = "S0/-90/16c", help = "PyGMT projection. Default: S0/-90/16c.")
#     parser.add_argument("--stride", type = int, default = 2, help = ("Spatial stride for plotting points. Default 2 for speed. "
#                                                                      "Use 1 for full native regridded resolution."))
#     parser.add_argument("--show", action = "store_true", help = "Show each figure interactively as it is generated.")
#     parser.add_argument("--log-level", default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR"])
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     logging.basicConfig(level = getattr(logging, args.log_level), format = "%(asctime)s | %(levelname)-8s | %(message)s")
#     if not args.file.exists():
#         raise FileNotFoundError(args.file)
#     D_out = args.output_dir
#     if D_out is None:
#         pth_cfg = ShugaPaths()
#         D_out   = pth_cfg.graphics_root_path / "forcing" / "ERA5" 
#     logging.info("Opening %s", args.file)
#     ds        = xr.open_dataset(args.file, chunks={'time': 1})
#     variables = args.variables
#     if variables is None:
#         variables = [v for v in ds.data_vars if v not in {"LON", "LAT", "lon", "lat", "TLON", "TLAT"} and "time" in ds[v].dims]
#     logging.info("Variables: %s", ", ".join(variables))
#     logging.info("Hours    : %s", args.hours)
#     logging.info("Region   : %s", args.region)
#     logging.info("Stride   : %s", args.stride)
#     logging.info("Output   : %s", D_out)
#     plotter = ERA5Plotter()
#     saved   = plotter.plot_hours(ds,
#                                  variables  = variables,
#                                  hours      = args.hours,
#                                  D_out      = D_out,
#                                  region     = args.region,
#                                  projection = args.projection,
#                                  stride     = args.stride,
#                                  show       = args.show)
#     logging.info("Saved %d figures", len(saved))
#     for path in saved[:10]:
#         logging.info("  %s", path)
#     if len(saved) > 10:
#         logging.info("  ...")

# if __name__ == "__main__":
#     main()
