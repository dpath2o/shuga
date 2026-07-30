#!/usr/bin/env python3
"""
Quick comparison of two short CICE kstrength experiments using a static CICE grid
coordinate store.

Example:

python ./kstrength_quick_check.py \
  --exp-a ~/AFIM_archive/kstrength-test01/history/daily \
  --exp-b ~/AFIM_archive/kstrength-test02/history/daily \
  --label-a kstrength-test01_h2match \
  --label-b kstrength-test02_h3match \
  --grid ~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr \
  --outdir ~/AFIM_archive/kstrength_3day_compare
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_VARS = ["aice", "hi", "vice", "strength", "uvel", "vvel", "divu", "shear", "sig1", "sig2"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two short CICE daily-history experiments.")
    p.add_argument("--exp-a", required=True, help="Experiment A daily history directory or glob.")
    p.add_argument("--exp-b", required=True, help="Experiment B daily history directory or glob.")
    p.add_argument("--label-a", default="experiment_A")
    p.add_argument("--label-b", default="experiment_B")
    p.add_argument("--grid", default="~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr",
                   help = "Static CICE coordinate/grid zarr containing TLAT, TLON, tarea.")
    p.add_argument("--outdir", default="./kstrength_3day_compare")
    p.add_argument("--ice-threshold", type=float, default=0.15)
    p.add_argument("--chunks", default="time:1", help="Example: time:1 or none.")
    p.add_argument("--vars", default=",".join(DEFAULT_VARS))
    p.add_argument("--map-vars", default="aice,hi,vice,strength,uvel,vvel")
    # Hemisphere masking for all scalar diagnostics
    p.add_argument("--hemi", choices=["south", "north"], default="south")
    # Map focus region: Mawson Coast to Shackleton Ice Shelf (approx)
    p.add_argument("--map-lon-min", type=float, default=60.0)
    p.add_argument("--map-lon-max", type=float, default=110.0)
    p.add_argument("--map-lat-min", type=float, default=-70.0)
    p.add_argument("--map-lat-max", type=float, default=-60.0)
    # PyGMT side-by-side strength maps
    p.add_argument("--make-strength-pygmt", action = "store_true",
                   help = "Create PyGMT side-by-side strength maps for the two experiments.")
    p.add_argument("--no-final-day-maps", action = "store_true",
                   help = "Skip the existing matplotlib/scatter final-day difference maps.")
    p.add_argument("--strength-map-start", default = "2005-10-01",
                   help = "First date for PyGMT strength side-by-side maps.")
    p.add_argument("--strength-map-days", type = int, default = 10,
                   help="Number of daily PyGMT strength maps to make.")
    p.add_argument("--strength-cmap", default = "cmocean/dense",
                   help = "GMT/PyGMT colormap for strength.")
    p.add_argument("--strength-series", default = "auto",
                   help = "CPT series as min,max,inc or 'auto'. Example: 0,60000,2500")
    p.add_argument("--strength-style", default = "s0.15c",
                   help = "PyGMT point style for native CICE grid cells.")
    p.add_argument("--strength-fig-size", type = float, default = 20.0,
                   help = "Panel projection size in cm.")
    p.add_argument("--no-strength-mean", action = "store_true",
                   help = "Do not create the 10-day mean side-by-side strength map.")
    p.add_argument("--maps-only", action = "store_true",
                   help = "Skip scalar statistics/time-series analysis and proceed directly to requested map plotting.")
    return p.parse_args()

def parse_chunks(text: str) -> dict | None:
    if text.lower() == "none":
        return None
    chunks = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":")
        chunks[key.strip()] = int(value)
    return chunks

def require_pygmt():
    try:
        import pygmt
    except Exception as exc:
        raise ImportError("PyGMT is required for --make-strength-pygmt. Load an environment with pygmt/GMT available.") from exc
    return pygmt

def parse_pygmt_series(text: str | None) -> list[float] | None:
    if text is None:
        return None
    text = str(text).strip().lower()
    if text in {"", "auto", "none"}:
        return None
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("--strength-series must be 'auto' or min,max,inc")
    return vals

def meridian_center_from_region(region: list[float] | tuple[float, float, float, float]) -> float:
    lon_min, lon_max, *_ = region
    a = ((float(lon_min) + 180.0) % 360.0) - 180.0
    b = ((float(lon_max) + 180.0) % 360.0) - 180.0
    if a <= b:
        mc = 0.5 * (a + b)
    else:
        width = (b + 360.0) - a
        mc = a + 0.5 * width
        if mc > 180.0:
            mc -= 360.0
    return mc

def projection_from_region(region: list[float] | tuple[float, float, float, float], fig_size: float = 20.0) -> str:
    _, _, lat_min, lat_max = region
    mc = meridian_center_from_region(region)
    lat_center = 0.5 * (float(lat_min) + float(lat_max))
    pole = -90 if lat_center < 0 else 90
    return f"S{mc}/{pole}/{fig_size}c"

def pygmt_da_prep(da: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray, region: list[float] | tuple[float, float, float, float], *,
                  mask_zero: bool = False) -> pd.DataFrame:
    """
    Prepare a CICE curvilinear field for PyGMT point plotting.

    This mirrors the shuga.plotting.cice pattern:
    regional mask -> flatten lon/lat/z -> retain finite values only.
    """
    lon180 = wrap_lon_180(lon)
    mask = get_sector_mask(xr.Dataset({"TLON": lon, "TLAT": lat}),
                           lon_min = region[0],
                           lon_max = region[1],
                           lat_min = region[2],
                           lat_max = region[3],
                           hemi    = "south" if region[3] <= 0 else "north")
    work = da.where(mask)
    if mask_zero:
        work = work.where(np.abs(work) > 0.0)
    lon_flat = np.asarray(np.ma.filled(lon180.values, np.nan), dtype=float).ravel()
    lat_flat = np.asarray(np.ma.filled(lat.values, np.nan), dtype=float).ravel()
    z_flat   = np.asarray(np.ma.filled(work.values, np.nan), dtype=float).ravel()
    good     = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(z_flat)
    return pd.DataFrame({"lon": lon_flat[good],
                         "lat": lat_flat[good],
                         "z"  : z_flat[good]  })

def select_daily_field(ds: xr.Dataset, var: str, date: pd.Timestamp) -> xr.DataArray:
    if var not in ds:
        raise KeyError(f"{var!r} not found in dataset. Available variables: {list(ds.data_vars)}")
    da = ds[var]
    # Daily CICE history is usually exact, but nearest with a 12-hour tolerance
    # makes this robust to timestamp conventions.
    try:
        return da.sel(time=np.datetime64(date))
    except Exception:
        return da.sel(time=np.datetime64(date), method="nearest", tolerance=np.timedelta64(12, "h"))

def auto_strength_series(ds_a: xr.Dataset, ds_b: xr.Dataset, region: list[float], start_date: str, ndays: int) -> list[float]:
    """
    Build one common CPT range for all daily panels using the combined
    1st--99th percentile over both experiments and the requested date window.
    """
    dates = pd.date_range(start=start_date, periods=ndays, freq="D")
    lon = ds_a["TLON"]
    lat = ds_a["TLAT"]
    vals = []
    for ds in (ds_a, ds_b):
        for date in dates:
            try:
                da = select_daily_field(ds, "strength", date)
            except Exception:
                continue

            data = pygmt_da_prep(da, lon=lon, lat=lat, region=region)
            if not data.empty:
                vals.append(data["z"].to_numpy())
    if not vals:
        raise ValueError("No finite strength values found for automatic CPT scaling.")
    z    = np.concatenate(vals)
    z    = z[np.isfinite(z)]
    vmin = max(0.0, float(np.nanpercentile(z, 1.0)))
    vmax = float(np.nanpercentile(z, 99.0))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(z))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    # Round to useful plotting values.
    if vmax > 1000.0:
        inc = max(500.0, round((vmax - vmin) / 20.0 / 500.0) * 500.0)
        vmin = 0.0
        vmax = np.ceil(vmax / inc) * inc
    else:
        inc = max((vmax - vmin) / 20.0, 1.0)
    return [float(vmin), float(vmax), float(inc)]

def resolve_files(path_or_glob: str) -> list[str]:
    p = Path(path_or_glob).expanduser()
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.nc"))
    else:
        files = sorted(glob.glob(os.path.expanduser(path_or_glob)))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {path_or_glob}")
    return files

def open_daily(path_or_glob: str, chunks: dict | None) -> xr.Dataset:
    files = resolve_files(path_or_glob)
    print(f"  files: {len(files)}")
    return xr.open_mfdataset(files, combine = "by_coords", chunks = chunks, decode_times = True, parallel = False)

def horizontal_dims(ds: xr.Dataset) -> tuple[str, str]:
    """
    Infer CICE horizontal dimensions from common 2-D or 3-D history fields.
    """
    for name in ("aice", "hi", "vice", "strength", "uvel", "vvel"):
        if name not in ds:
            continue
        dims = [d for d in ds[name].dims if d not in ("time", "ncat", "nc", "vertices", "nv")]
        if len(dims) >= 2:
            return tuple(dims[-2:])
    raise ValueError("Could not infer horizontal CICE dimensions from history fields.")

def open_static_grid(path: str, target_ds: xr.Dataset) -> xr.Dataset:
    """
    Open static CICE coordinate store and rename its horizontal dimensions, if required,
    to match the history files.
    """
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Static grid store not found: {grid_path}")
    grid = xr.open_zarr(str(grid_path), consolidated=False)
    target_y, target_x = horizontal_dims(target_ds)
    # Infer grid horizontal dims from tarea/TLAT.
    for name in ("tarea", "TLAT", "TLON", "uarea", "ULAT", "ULON"):
        if name in grid:
            gdims = list(grid[name].dims)
            if len(gdims) >= 2:
                grid_y, grid_x = gdims[-2], gdims[-1]
                break
    else:
        raise ValueError("Could not infer horizontal dimensions from static grid store.")
    rename = {}
    if grid_y != target_y:
        rename[grid_y] = target_y
    if grid_x != target_x:
        rename[grid_x] = target_x
    if rename:
        grid = grid.rename(rename)
    return grid

def attach_grid(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Attach only the static grid variables required for this diagnostic.
    """
    needed = ["tarea", "TLAT", "TLON", "tmask", "uarea", "ULAT", "ULON", "umask"]
    out = ds.copy()
    for name in needed:
        if name in grid and name not in out:
            out[name] = grid[name]
    return out

def wrap_lon_180(lon: xr.DataArray) -> xr.DataArray:
    """
    Convert longitude to [-180, 180).
    """
    return ((lon + 180.0) % 360.0) - 180.0

def lon_between(lon: xr.DataArray, lon_min: float, lon_max: float) -> xr.DataArray:
    """
    Works for standard and dateline-crossing longitude intervals.
    Assumes lon is already in [-180, 180).
    """
    if lon_min <= lon_max:
        return (lon >= lon_min) & (lon <= lon_max)
    else:
        return (lon >= lon_min) | (lon <= lon_max)

def get_hemisphere_mask(ds: xr.Dataset, hemi: str = "south") -> xr.DataArray:
    if "TLAT" not in ds:
        raise ValueError("Dataset has no TLAT. Provide --grid pointing to CICE static coords zarr.")
    if hemi == "south":
        mask = ds["TLAT"] <= 0.0
    else:
        mask = ds["TLAT"] >= 0.0

    if "tmask" in ds:
        mask = mask & (ds["tmask"] > 0)
    return mask

def get_sector_mask(ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float, hemi: str = "south") -> xr.DataArray:
    if "TLAT" not in ds or "TLON" not in ds:
        raise ValueError("Dataset must contain TLAT and TLON.")
    lon = wrap_lon_180(ds["TLON"])
    lat = ds["TLAT"]
    mask = lon_between(lon, lon_min, lon_max) & (lat >= lat_min) & (lat <= lat_max)
    # Also enforce SH if requested
    if hemi == "south":
        mask = mask & (lat <= 0.0)
    elif hemi == "north":
        mask = mask & (lat >= 0.0)
    if "tmask" in ds:
        mask = mask & (ds["tmask"] > 0)
    return mask

def cice_aice_fraction(aice: xr.DataArray) -> xr.DataArray:
    """
    CICE aice is normally 0-1, but this keeps the script safe if percent output appears.
    """
    mx = float(aice.max(skipna=True).compute())
    if mx > 1.5:
        return aice / 100.0
    return aice

def get_area_m2(ds: xr.Dataset) -> xr.DataArray:
    if "tarea" not in ds:
        raise ValueError("Dataset has no tarea. Provide --grid pointing to CICE static coords zarr.")
    area = ds["tarea"]
    units = str(area.attrs.get("units", "")).lower()
    # CICE tarea is often cm^2 in some static files. Convert if metadata or magnitude says so.
    if "cm" in units and "2" in units:
        area = area * 1.0e-4
    else:
        mean_area = float(area.mean(skipna=True).compute())
        if mean_area > 1.0e11:
            area = area * 1.0e-4
    area.name = "tarea_m2"
    return area

def get_region_mask(ds: xr.Dataset, lat_min: float, lat_max: float) -> xr.DataArray:
    if "TLAT" not in ds:
        raise ValueError("Dataset has no TLAT. Provide --grid pointing to CICE static coords zarr.")
    mask = (ds["TLAT"] >= lat_min) & (ds["TLAT"] <= lat_max)
    if "tmask" in ds:
        mask = mask & (ds["tmask"] > 0)
    return mask

def spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    return tuple(d for d in da.dims if d != "time")

def area_sum(da: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    dims = spatial_dims(da)
    return (da.where(mask) * area.where(mask)).sum(dim=dims, skipna=True)

def area_mean(da: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    dims = spatial_dims(da)
    num = (da.where(mask) * area.where(mask)).sum(dim=dims, skipna=True)
    den = area.where(mask & np.isfinite(da)).sum(dim=dims, skipna=True)
    return num / den

def percentile(da: xr.DataArray, q: float, mask: xr.DataArray) -> xr.DataArray:
    dims = spatial_dims(da)
    return da.where(mask).quantile(q, dim=dims, skipna=True)

def compute_summary(ds: xr.Dataset, label: str, var_names: list[str], ice_threshold: float, hemi: str = "south") -> pd.DataFrame:
    area = get_area_m2(ds)
    region_mask = get_hemisphere_mask(ds, hemi=hemi)
    if "aice" not in ds:
        raise ValueError("Need aice in CICE history.")
    aice = cice_aice_fraction(ds["aice"])
    ice_mask = region_mask & (aice >= ice_threshold)
    rows = []
    sia = area_sum(aice, area, region_mask) / 1.0e9
    sie = area.where(region_mask & (aice >= ice_threshold)).sum(dim = spatial_dims(aice), skipna = True) / 1.0e9
    for metric, ts in {"SIA_10^3_km2": sia, "SIE_10^3_km2": sie}.items():
        vals = ts.compute().values
        for t, v in zip(ds["time"].values, vals):
            rows.append({"experiment": label, "time": pd.to_datetime(str(t)), "metric": metric, "value": float(v)})
    derived = {}
    if "uvel" in ds and "vvel" in ds:
        derived["speed"] = np.hypot(ds["uvel"], ds["vvel"])
    if "divu" in ds and "shear" in ds:
        derived["strain_inv"] = np.sqrt(ds["divu"] ** 2 + ds["shear"] ** 2)
    for var in var_names:
        if var in ds:
            da = ds[var]
        elif var in derived:
            da = derived[var]
        else:
            continue
        if "time" not in da.dims:
            continue
        if var == "aice":
            da = cice_aice_fraction(da)
        stats = {f"{var}_mean_ice_mask": area_mean(da, area, ice_mask),
                 f"{var}_p05_ice_mask": percentile(da, 0.05, ice_mask),
                 f"{var}_p50_ice_mask": percentile(da, 0.50, ice_mask),
                 f"{var}_p95_ice_mask": percentile(da, 0.95, ice_mask)}
        for metric, ts in stats.items():
            vals = ts.compute().values
            for t, v in zip(ds["time"].values, vals):
                rows.append({"experiment": label, "time": pd.to_datetime(str(t)), "metric": metric, "value": float(v) if np.isfinite(v) else np.nan})
    return pd.DataFrame(rows)

def make_delta(df: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    wide = df.pivot_table(index=["time", "metric"], columns="experiment", values="value").reset_index()
    wide["delta_B_minus_A"] = wide[label_b] - wide[label_a]
    wide["relative_delta_percent"] = 100.0 * wide["delta_B_minus_A"] / wide[label_a].replace(0.0, np.nan)
    return wide

def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)

def plot_timeseries(df: pd.DataFrame, outdir: Path) -> None:
    preferred = ["SIA_10^3_km2", "SIE_10^3_km2", "aice_mean_ice_mask", "hi_mean_ice_mask", "vice_mean_ice_mask",
                 "strength_mean_ice_mask", "speed_mean_ice_mask", "strain_inv_p95_ice_mask"]
    available = set(df["metric"])
    for metric in preferred:
        if metric not in available:
            continue
        sub = df[df["metric"] == metric].sort_values("time")
        plt.figure(figsize=(8, 4.5))
        for exp, grp in sub.groupby("experiment"):
            plt.plot(grp["time"], grp["value"], marker="o", label=exp)
        plt.title(metric)
        plt.xlabel("time")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"metric_{safe_name(metric)}.png", dpi=160)
        plt.close()

def fig_shift(fig, *, xshift: str = "0c", yshift: str = "0c") -> None:
    """
    Compatibility wrapper.

    PyGMT's public API is normally fig.shift_origin(...), but some older/local
    plotting habits refer to this as fig.shift(...).
    """
    if hasattr(fig, "shift"):
        fig.shift(xshift=xshift, yshift=yshift)
    else:
        fig.shift_origin(xshift=xshift, yshift=yshift)

def plot_strength_side_by_side_pygmt(ds_a: xr.Dataset, ds_b: xr.Dataset, label_a: str, label_b: str, outdir: Path, *,
                                     lon_min     : float,
                                     lon_max     : float,
                                     lat_min     : float,
                                     lat_max     : float,
                                     start_date  : str = "2005-10-01",
                                     ndays       : int = 10,
                                     cmap        : str = "cmocean/dense",
                                     series      : list[float] | None = None,
                                     style       : str = "s0.1c",
                                     fig_size    : float = 13.0,
                                     include_mean: bool = True) -> list[str]:
    """
    Create 2x2 PyGMT maps for two CICE experiments.

    Top row   : strength
    Bottom row: hi

    Outputs one PNG per day plus, optionally, one 10-day mean PNG.
    """
    pygmt = require_pygmt()
    for vname in ("strength", "hi"):
        if vname not in ds_a or vname not in ds_b:
            raise KeyError(f"Both datasets must contain the CICE history variable {vname!r}.")
    region     = [float(lon_min), float(lon_max), float(lat_min), float(lat_max)]
    projection = projection_from_region(region, fig_size = fig_size)
    lon        = ds_a["TLON"]
    lat        = ds_a["TLAT"]
    dates      = pd.date_range(start = start_date, periods = ndays, freq = "D")
    if series is None:
        series = auto_strength_series(ds_a, ds_b, region, start_date, ndays)
    series     = [0, 1e5]
    hi_series  = [0, 5.0]
    hi_cmap    = "cmocean/matter"
    out_strength = outdir / "pygmt_strength"
    out_strength.mkdir(parents = True, exist_ok = True)
    saved = []
    def _plot_panel(fig, data: pd.DataFrame, *, add_title: str | None = None) -> None:
        frame = ["af"]
        if add_title is not None:
            frame.append(f"+t{add_title}")
        fig.basemap(region = region, projection = projection, frame = frame)
        fig.coast(shorelines = "0.5p,black", land = "gray90", water = "white")
        if not data.empty:
            fig.plot(x     = data["lon"],
                     y     = data["lat"],
                     style = style,
                     fill  = data["z"],
                     cmap  = True,
                     pen   = None)
        fig.coast(region = region, projection = projection, shorelines = "0.5p,black")
    def _plot_one_quad(da_str_a     : xr.DataArray,
                       da_str_b     : xr.DataArray,
                       da_hi_a      : xr.DataArray,
                       da_hi_b      : xr.DataArray,
                       date_label   : str,
                       out_name     : str,
                       annotate_date: bool = True) -> str:
        data_str_a = pygmt_da_prep(da_str_a, lon = lon, lat = lat, region = region)
        data_str_b = pygmt_da_prep(da_str_b, lon = lon, lat = lat, region = region)
        data_hi_a  = pygmt_da_prep(da_hi_a,  lon = lon, lat = lat, region = region)
        data_hi_b  = pygmt_da_prep(da_hi_b,  lon = lon, lat = lat, region = region)
        if data_str_a.empty and data_str_b.empty and data_hi_a.empty and data_hi_b.empty:
            print(f"Skipping {out_name}: no finite values in region.")
            return ""
        fig = pygmt.Figure()
        with pygmt.config(MAP_FRAME_TYPE     = "plain",
                          FORMAT_GEO_MAP     = "dddF",
                          FONT_TITLE         = "11p,Helvetica-Bold",
                          FONT_LABEL         = "9p,Helvetica",
                          FONT_ANNOT_PRIMARY = "8p,Helvetica"):
            # ------------------------------------------------------------
            # Top-left panel: strength, experiment A
            # ------------------------------------------------------------
            pygmt.makecpt(cmap = cmap, series = series, background = True)
            _plot_panel(fig, data_str_a, add_title = "kstrength-test01")
            # ------------------------------------------------------------
            # Top-right panel: strength, experiment B
            # ------------------------------------------------------------
            fig_shift(fig, xshift = "21c", yshift = "0c")
            _plot_panel(fig, data_str_b, add_title = "kstrength-test02")
            if annotate_date:
                fig.text(x        = lon_max - 1.0,
                         y        = lat_max - 0.6,
                         text     = date_label,
                         justify  = "TR",
                         font     = "10p,Helvetica-Bold,black",
                         fill     = "white@40",
                         pen      = "0.25p,black")
            # ------------------------------------------------------------
            # Strength colour bar beneath top row
            # ------------------------------------------------------------
            fig_shift(fig, xshift = "-16c", yshift = "-1.0c")
            fig.colorbar(position = "JBC+w20c/0.35c+o7.2c/0c+h",
                         frame    = ["x+lice strength", "y+lN m@+-1@+"])
            # ------------------------------------------------------------
            # Bottom-left panel: hi, experiment A
            # ------------------------------------------------------------
            fig_shift(fig, xshift = "-5c", yshift = "-12c")
            pygmt.makecpt(cmap = hi_cmap, series = hi_series, background = True)
            _plot_panel(fig, data_hi_a, add_title = None)
            # ------------------------------------------------------------
            # Bottom-right panel: hi, experiment B
            # ------------------------------------------------------------
            fig_shift(fig, xshift = "21c", yshift = "0c")
            _plot_panel(fig, data_hi_b, add_title = None)
            # ------------------------------------------------------------
            # hi colour bar beneath bottom row
            # ------------------------------------------------------------
            fig_shift(fig, xshift = "-16c", yshift = "-1.0c")
            fig.colorbar(position = "JBC+w20c/0.35c+o7.2c/0c+h",
                         frame    = ["x+lsea ice thickness", "y+lm"])
        path = out_strength / out_name
        fig.savefig(path)
        print(f"Wrote {path}")
        return str(path)
    for date in dates:
        da_str_a = select_daily_field(ds_a, "strength", date).squeeze(drop = True)
        da_str_b = select_daily_field(ds_b, "strength", date).squeeze(drop = True)
        da_hi_a  = select_daily_field(ds_a, "hi",       date).squeeze(drop = True)
        da_hi_b  = select_daily_field(ds_b, "hi",       date).squeeze(drop = True)
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        path = _plot_one_quad(da_str_a      = da_str_a,
                              da_str_b      = da_str_b,
                              da_hi_a       = da_hi_a,
                              da_hi_b       = da_hi_b,
                              date_label    = date_str,
                              out_name      = f"strength_hi_2x2_{date_str}.png",
                              annotate_date = True)
        if path:
            saved.append(path)
    if include_mean:
        t0 = dates[0]
        t1 = dates[-1]
        da_str_a_mean = ds_a["strength"].sel(time = slice(t0, t1)).mean("time", skipna = True).squeeze(drop = True)
        da_str_b_mean = ds_b["strength"].sel(time = slice(t0, t1)).mean("time", skipna = True).squeeze(drop = True)
        da_hi_a_mean  = ds_a["hi"].sel(time       = slice(t0, t1)).mean("time", skipna = True).squeeze(drop = True)
        da_hi_b_mean  = ds_b["hi"].sel(time       = slice(t0, t1)).mean("time", skipna = True).squeeze(drop = True)
        mean_label = f"{pd.Timestamp(t0).strftime('%Y-%m-%d')} to {pd.Timestamp(t1).strftime('%Y-%m-%d')}"
        path = _plot_one_quad(da_str_a      = da_str_a_mean,
                              da_str_b      = da_str_b_mean,
                              da_hi_a       = da_hi_a_mean,
                              da_hi_b       = da_hi_b_mean,
                              date_label    = mean_label,
                              out_name      = f"strength_hi_2x2_mean_{pd.Timestamp(t0).strftime('%Y%m%d')}_{pd.Timestamp(t1).strftime('%Y%m%d')}.png",
                              annotate_date = True)
        if path:
            saved.append(path)
    return saved

def write_decision(delta: pd.DataFrame, outdir: Path, label_a: str, label_b: str) -> str:
    final_time = delta["time"].max()
    final = delta[delta["time"] == final_time].copy()
    key_metrics = [
        "SIA_10^3_km2",
        "SIE_10^3_km2",
        "aice_mean_ice_mask",
        "hi_mean_ice_mask",
        "vice_mean_ice_mask",
        "strength_mean_ice_mask",
        "speed_mean_ice_mask",
        "strain_inv_p95_ice_mask",
    ]

    lines = []
    lines.append(f"Comparison: {label_b} minus {label_a}")
    lines.append(f"Final comparison date: {final_time}")
    lines.append("")
    lines.append("Final-day differences:")

    rels = {}

    for metric in key_metrics:
        sub = final[final["metric"] == metric]
        if sub.empty:
            continue

        a = float(sub[label_a].iloc[0])
        b = float(sub[label_b].iloc[0])
        d = float(sub["delta_B_minus_A"].iloc[0])
        r = float(sub["relative_delta_percent"].iloc[0])

        rels[metric] = abs(r) if np.isfinite(r) else np.nan

        lines.append(
            f"  {metric:28s}  "
            f"{label_a}={a: .6e}  {label_b}={b: .6e}  "
            f"B-A={d: .6e}  rel={r: .3f}%"
        )

    dyn_rels = [
        rels[m]
        for m in rels
        if any(key in m for key in ("strength", "speed", "strain_inv"))
        and np.isfinite(rels[m])
    ]

    area_rels = [
        rels[m]
        for m in ("SIA_10^3_km2", "SIE_10^3_km2", "aice_mean_ice_mask")
        if m in rels and np.isfinite(rels[m])
    ]

    max_dyn = max(dyn_rels, default=0.0)
    max_area = max(area_rels, default=0.0)

    run_both = (max_dyn >= 5.0) or (max_area >= 0.5)

    lines.append("")
    if run_both:
        lines.append("Recommendation: run both cases longer.")
        lines.append(
            "Rationale: the two normalisations already separate in either the "
            "dynamic response or bulk ice-state metrics over only three days."
        )
    else:
        lines.append("Recommendation: run the 2 m matched case as the primary long case first.")
        lines.append(
            "Rationale: the two short tests are close in the bulk metrics. The 2 m "
            "matched case is the more conservative first test because it remains closer "
            "to the standard strength scale for typical compact Antarctic sea-ice thickness."
        )
        lines.append("Keep the 3 m matched case as a secondary sensitivity or shorter pilot.")

    text = "\n".join(lines)
    (outdir / "decision_summary.txt").write_text(text)
    return text


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    chunks = parse_chunks(args.chunks)

    print(f"Experiment A: {args.label_a}")
    ds_a = open_daily(args.exp_a, chunks)

    print(f"Experiment B: {args.label_b}")
    ds_b = open_daily(args.exp_b, chunks)

    print(f"Static grid: {Path(args.grid).expanduser()}")
    grid = open_static_grid(args.grid, ds_a)

    ds_a = attach_grid(ds_a, grid)
    ds_b = attach_grid(ds_b, grid)

    if args.maps_only:
        if args.make_strength_pygmt:
            plot_strength_side_by_side_pygmt(
                ds_a,
                ds_b,
                args.label_a,
                args.label_b,
                outdir,
                lon_min=args.map_lon_min,
                lon_max=args.map_lon_max,
                lat_min=args.map_lat_min,
                lat_max=args.map_lat_max,
                start_date=args.strength_map_start,
                ndays=args.strength_map_days,
                cmap=args.strength_cmap,
                series=parse_pygmt_series(args.strength_series),
                style=args.strength_style,
                fig_size=args.strength_fig_size,
                include_mean=not args.no_strength_mean,
            )
        else:
            raise ValueError("--maps-only was supplied, but no map task was requested. Add --make-strength-pygmt.")

        print("")
        print(f"Wrote map outputs to: {outdir}")
        return

    var_names = [v.strip() for v in args.vars.split(",") if v.strip()]
    map_vars = [v.strip() for v in args.map_vars.split(",") if v.strip()]

    if "uvel" in ds_a and "vvel" in ds_a and "speed" not in var_names:
        var_names.append("speed")

    if "divu" in ds_a and "shear" in ds_a and "strain_inv" not in var_names:
        var_names.append("strain_inv")

    df_a = compute_summary(
        ds_a,
        args.label_a,
        var_names,
        args.ice_threshold,
        hemi=args.hemi,
    )

    df_b = compute_summary(
        ds_b,
        args.label_b,
        var_names,
        args.ice_threshold,
        hemi=args.hemi,
    )

    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_csv(outdir / "summary_timeseries.csv", index=False)

    delta = make_delta(df, args.label_a, args.label_b)
    delta.to_csv(outdir / "delta_summary.csv", index=False)

    plot_timeseries(df, outdir)

    if not args.no_final_day_maps:
        final_day_maps(
            ds_a,
            ds_b,
            args.label_a,
            args.label_b,
            map_vars,
            outdir,
            lon_min=args.map_lon_min,
            lon_max=args.map_lon_max,
            lat_min=args.map_lat_min,
            lat_max=args.map_lat_max,
            hemi=args.hemi,
        )

    if args.make_strength_pygmt:
        plot_strength_side_by_side_pygmt(
            ds_a,
            ds_b,
            args.label_a,
            args.label_b,
            outdir,
            lon_min=args.map_lon_min,
            lon_max=args.map_lon_max,
            lat_min=args.map_lat_min,
            lat_max=args.map_lat_max,
            start_date=args.strength_map_start,
            ndays=args.strength_map_days,
            cmap=args.strength_cmap,
            series=parse_pygmt_series(args.strength_series),
            style=args.strength_style,
            fig_size=args.strength_fig_size,
            include_mean=not args.no_strength_mean,
        )

    decision = write_decision(delta, outdir, args.label_a, args.label_b)

    print("")
    print(decision)
    print("")
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
