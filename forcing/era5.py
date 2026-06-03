#!/usr/bin/env python3
"""
ERA5 -> CICE6 monthly forcing preprocessing.

This module regrids hourly ERA5 single-level fields to the CICE T grid and
writes monthly CICE-ready NetCDF files. It keeps monthly files as the primary
forcing product rather than concatenating into very large yearly files.
"""
from __future__ import annotations
import calendar, logging, os, sys
from pathlib    import Path
#####################################################################
# make sure this reflects the correct location of mawsons-chest repo
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
#####################################################################
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import numpy          as np
import xarray         as xr
import xesmf          as xe
from dataclasses      import dataclass
from datetime         import datetime
from typing           import Iterable
from dask.diagnostics import ProgressBar
from shuga.grid       import load_cice_tgrid_for_xesmf
from shuga.regridding import (XESMFRegridSpec,
                              build_xesmf_regridder,
                              regrid_dataarray_to_cice_tgrid,
                              format_era5_to_cice_weight_filename)

LOGGER    = logging.getLogger(__name__)
ERA5_VARS_PRI = [("2t", "t2m", "airtmp", "2 m air temperature", "K"),
                 ("2d", "d2m", "dewtmp", "2 m dew-point temperature", "K"),
                 ("sp", "sp", "pair", "surface air pressure", "Pa")]
ERA5_VARS_SEC = [("msdwlwrf", "msdwlwrf", "dlwsfc", "downward longwave radiation flux", "W/m^2"),
                 ("msdwswrf", "msdwswrf", "glbrad", "downward shortwave radiation flux", "W/m^2"),
                 ("mtpr", "mtpr", "ttlpcp", "total precipitation rate", "kg/m^2/s"),
                 ("10u", "u10", "wndewd", "eastward 10 m wind", "m/s"),
                 ("10v", "v10", "wndnwd", "northward 10 m wind", "m/s"),
                 ("blh", "blh", "blh", "boundary layer height", "m"),
                 ("10fg", "fg10", "windgust", "10 m wind gust since previous post-processing", "m/s"),
                 ("100u", "u100", "wnd100ewd", "eastward 100 m wind", "m/s"),
                 ("100v", "v100", "wnd100nwd", "northward 100 m wind", "m/s")]
VAR_NAMES_REQ = ["airtmp", "dlwsfc", "glbrad", "ttlpcp", "wndewd", "wndnwd"]
VAR_NAMES_SUP = ["dewtmp", "pair", "blh", "windgust", "wnd100ewd", "wnd100nwd"]
ALL_VARS_OUT  = VAR_NAMES_REQ + VAR_NAMES_SUP

@dataclass(frozen=True)
class ERA5Config:
    user_name      : str = "da1339"
    project_name   : str = "gv90"
    grid_res       : str = "0p25"
    regrid_method  : str = "bilinear"
    extrap_method  : str = "nearest_s2d"
    era5_root      : Path = Path("/g/data/rt52/era5/single-levels/reanalysis")
    longterm_root  : Path = Path("/g/data/gv90/da1339")
    grid_root      : Path = Path("/g/data/gv90/da1339/grids")
    cice_grid_file : str | Path | None = None
    weight_filename: str | None = None
    rebuild_weights: bool = False
    output_subdir  : str = "monthly_cice6"
    chunk_in_time  : int | None = None
    output_dtype   : str = "float32"

def last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]

def compute_sfc_qsat(d2m: xr.DataArray, sp: xr.DataArray) -> xr.DataArray:
    """
    Compute 2-m specific humidity from dewpoint temperature and surface pressure.

    d2m : 2-m dewpoint temperature [K]
    sp  : surface pressure [Pa]
    """
    rdry = 287.0597
    rvap = 461.5250
    a1 = 611.21
    a3 = 17.502
    a4 = 32.19
    t0 = 273.16
    e = a1 * np.exp(a3 * (d2m - t0) / (d2m - a4))
    q = (rdry / rvap) * e / (sp - ((1.0 - rdry / rvap) * e))
    q.attrs.update(long_name = "specific humidity at 2 m derived from dewpoint and surface pressure",
                   units     = "kg/kg")
    return q

def era5_monthly_filename(var_dir: str, year: int, month: int) -> str:
    day_n = last_day_of_month(year, month)
    return f"{var_dir}_era5_oper_sfc_{year:04d}{month:02d}01-{year:04d}{month:02d}{day_n:02d}.nc"

def era5_path(cfg: ERA5Config, var_dir: str, year: int, month: int) -> Path:
    return cfg.era5_root / var_dir / f"{year:04d}" / era5_monthly_filename(var_dir, year, month)

def open_era5_var(cfg     : ERA5Config,
                  var_dir : str,
                  var_name: str,
                  year    : int,
                  month   : int) -> xr.DataArray:
    path = era5_path(cfg, var_dir, year, month)
    if not path.exists():
        raise FileNotFoundError(f"Missing ERA5 file: {path}")
    LOGGER.info("Opening %s", path)
    if cfg.chunk_in_time is None:
        ds = xr.open_dataset(path)
    else:
        ds = xr.open_dataset(path, chunks={"time": cfg.chunk_in_time})
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not in {path}; available={list(ds.data_vars)}")
    return ds[var_name]

def build_month_dataset(cfg : ERA5Config, year : int, month : int) -> xr.Dataset:
    grid        = load_cice_tgrid_for_xesmf(cfg.cice_grid_file, lon_type = "0-360", logger = LOGGER)
    F_G_CICE    = Path(grid.attrs["source_path"])
    F_wgt       = cfg.weight_filename or format_era5_to_cice_weight_filename(cice_grid_file = F_G_CICE,
                                                                             regrid_method  = cfg.regrid_method,
                                                                             extrap_method  = cfg.extrap_method)
    regrid_spec = XESMFRegridSpec(method          = cfg.regrid_method,
                                  extrap_method   = cfg.extrap_method,
                                  weight_file     = cfg.grid_root / "weights" / F_wgt,
                                  rebuild_weights = cfg.rebuild_weights)
    t2m_src     = open_era5_var(cfg, "2t", "t2m", year, month)
    regridder   = build_xesmf_regridder(t2m_src, grid, regrid_spec, logger=LOGGER)
    fields: dict[str, xr.DataArray] = {}
    fields["airtmp"] = regrid_dataarray_to_cice_tgrid(t2m_src, regridder, "airtmp", long_name = "2 m air temperature", units = "K", dtype = cfg.output_dtype)
    for var_dir, var_name, out_name, long_name, units in ERA5_VARS_SEC:
        src              = open_era5_var(cfg, var_dir, var_name, year, month)
        fields[out_name] = regrid_dataarray_to_cice_tgrid(src, regridder, out_name, long_name = long_name, units = units, dtype = cfg.output_dtype)
    # Surface pressure retained directly for future rhoa / boundary-layer physics.
    sp_src           = open_era5_var(cfg, "sp", "sp", year, month)
    sp_rg            = regrid_dataarray_to_cice_tgrid(sp_src, regridder, "pair", long_name = "surface air pressure", units = "Pa", dtype = cfg.output_dtype)
    # Specific humidity derived from regridded dewpoint and pressure, matching the legacy script.
    d2m_src          = open_era5_var(cfg, "2d", "d2m", year, month)
    d2m_rg           = regrid_dataarray_to_cice_tgrid(d2m_src, regridder, name = "d2m_regridded", long_name = "2 m dewpoint temperature", units = "K" , dtype = cfg.output_dtype)
    qair             = compute_sfc_qsat(d2m_rg, sp_rg).astype(cfg.output_dtype)
    qair.name        = "spchmd"
    qair.attrs.update(long_name="specific humidity", units="kg/kg")
    fields["spchmd"] = qair
    fields["pair"]   = sp_rg.rename("pair")
    # Precipitation phase: prefer ERA5 mean snowfall rate if available.
    # On Gadi this is expected as "msr" / variable "msr", but keep this robust.
    msr_src            = open_era5_var(cfg, "msr", "msr", year, month)
    snow               = regrid_dataarray_to_cice_tgrid(msr_src, regridder, "snow",  long_name = "snowfall rate", units = "kg/m^2/s", dtype = cfg.output_dtype)
    rain               = (fields["ttlpcp"] - snow).clip(min=0.0).astype(cfg.output_dtype)
    rain.name          = "rain"
    rain.attrs.update(long_name="rainfall rate derived as ttlpcp - snowfall", units="kg/m^2/s")
    fields["snow"] = snow
    fields["rain"] = rain
    # coordinates ... likely not be necessary
    # lon = xr.DataArray(grid["lon"].data, dims = ("ny", "nx"), name = "LON", attrs = {"units": "degrees_east"})
    # lat = xr.DataArray(grid["lat"].data, dims = ("ny", "nx"), name = "LAT", attrs = {"units": "degrees_north"})
    # Belt-and-braces: ensure every time-dependent forcing field uses time,ny,nx.
    for key, da in list(fields.items()):
        rename = {}
        if "nj" in da.dims:
            rename["nj"] = "ny"
        if "ni" in da.dims:
            rename["ni"] = "nx"
        if rename:
            da = da.rename(rename)
        if "time" in da.dims:
            da = da.transpose("time", "ny", "nx")
        fields[key] = da
    for key in fields:
        if "time" in fields[key].dims:
            fields[key].attrs["coordinates"] = "LAT LON"
    ds = xr.Dataset(data_vars = fields,
                    coords    = {"time": fields["airtmp"]["time"]}, # "LON" : lon, "LAT" : lat,
                    attrs     = {"creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "Conventions"  : "CCSM data model domain description",
                                 "title"        : "Monthly regridded ERA5 forcing for CICE6 standalone",
                                 "source"       : "ERA5 single-level reanalysis on Gadi /g/data/rt52/era5/single-levels/reanalysis",
                                 "regridding"   : f"xESMF {cfg.regrid_method} with {cfg.extrap_method}",
                                 "weight_file"  : f"{F_wgt}",
                                 "author"       : "Daniel P Atwater",
                                 "email"        : "daniel.atwater@utas.edu.au",
                                 "note"         : "Monthly files are the primary forcing product; no annual ncrcat concatenation."})
    return ds

def output_path(cfg: ERA5Config, year: int, month: int) -> Path:
    out_dir = cfg.longterm_root / "afim_input" / "ERA5" / cfg.grid_res / cfg.regrid_method / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"era5_for_cice6_{year:04d}_{month:02d}.nc"

def write_month(year : int, month : int, cfg : ERA5Config | None = None, overwrite : bool = False) -> Path:
    cfg = cfg or ERA5Config()
    out = output_path(cfg, year, month)
    if out.exists() and not overwrite:
        LOGGER.info("Output exists; skipping %s", out)
        return out
    ds  = build_month_dataset(cfg, year, month)
    for v in VAR_NAMES_OUT:
        if v in ds:
            x = ds[v].isel(time=0)
            print(f"{v:10s}", x.dims, float(x.min()), float(x.max()), float(x.mean()))
    tmp = out.with_suffix(".tmp.nc")
    if tmp.exists():
        tmp.unlink()
    LOGGER.info("Writing uncompressed NetCDF: %s", tmp)
    ds.to_netcdf(tmp, mode="w", engine="netcdf4")
    tmp.replace(out)
    LOGGER.info("Finished writing %s", out)
    return out
