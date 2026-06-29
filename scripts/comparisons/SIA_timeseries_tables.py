#!/usr/bin/env python3
"""Create CICE/NSIDC/OSI-SAF SIA comparison figures and tables.

Outputs
-------
1. Full-period daily SIA time series.
2. Daily climatological seasonal envelope: mean line plus min/max shading by
   day-of-year, plotted with PyGMT.
3. CSV table of daily SIA and optional monthly/annual summaries.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


def _open_zarr(path: str | Path) -> xr.Dataset:
    return xr.open_zarr(str(path), consolidated=False)


def _guess_area(ds: xr.Dataset) -> xr.DataArray:
    for name in ("tarea", "cell_area", "area", "TAREA", "uarea"):
        if name in ds:
            area = ds[name]
            units = str(area.attrs.get("units", "")).lower()
            if "km" in units and "2" in units:
                return area * 1.0e6
            return area
    for xname, yname in (("TLON", "TLAT"), ("lon", "lat"), ("longitude", "latitude")):
        if xname in ds and yname in ds:
            template = ds[xname]
            LOGGER.warning("No area variable found; falling back to nominal 25 km cells for %s", list(ds.data_vars))
            return xr.ones_like(template, dtype="float64") * 25_000.0**2
    raise KeyError("Could not find grid-cell area in dataset.")


def _guess_sia_var(ds: xr.Dataset) -> str | None:
    for name in ("sia", "SIA", "sea_ice_area"):
        if name in ds:
            return name
    return None


def _guess_conc_or_mask_var(ds: xr.Dataset) -> str:
    for name in ("aice", "aice_d", "sic", "siconc", "ice_conc", "sea_ice_area_fraction", "FI", "SI", "mask"):
        if name in ds:
            return name
    # Prefer a variable with time and at least two spatial dimensions.
    for name, da in ds.data_vars.items():
        if "time" in da.dims and da.ndim >= 3:
            return name
    raise KeyError(f"Could not identify concentration/mask variable. Variables: {list(ds.data_vars)}")


def _normalise_fraction(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().strip()
    valid = da.where(np.isfinite(da))
    vmax = float(valid.max(skipna=True).compute())
    if units in {"%", "percent", "percentage"} or vmax > 1.5:
        valid = valid / 100.0
    return valid.clip(min=0.0, max=1.0)


def _to_dataframe(da: xr.DataArray, name: str) -> pd.DataFrame:
    s = da.to_series().rename(name)
    df = s.reset_index()
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[[name]].sort_index()


def load_sia_series_from_store(
    path: str | Path,
    name: str,
    start_date: str,
    end_date: str,
    var_name: str | None = None,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Load an existing SIA variable or compute SIA from concentration/mask."""
    ds = _open_zarr(path).sel(time=slice(start_date, end_date))
    sia_var = var_name or _guess_sia_var(ds)
    if sia_var and sia_var in ds:
        da = ds[sia_var]
        units = str(da.attrs.get("units", "")).lower()
        # Standardise to 10^6 km^2.
        if units in {"m2", "m^2", "square meters", "square metres"}:
            da = da / 1.0e12
        elif units in {"km2", "km^2", "square kilometers", "square kilometres"}:
            da = da / 1.0e6
        return _to_dataframe(da.rename(name), name)

    v = _guess_conc_or_mask_var(ds)
    conc = _normalise_fraction(ds[v])
    area = _guess_area(ds)
    ice_area = conc.where(conc >= threshold, 0.0) * area
    spatial_dims = [d for d in ice_area.dims if d != "time"]
    sia = (ice_area.sum(dim=spatial_dims, skipna=True) / 1.0e12).rename(name)
    return _to_dataframe(sia, name)


def default_cice_metric_path(base_dir: Path, sim_name: str, hemisphere: str, ice_type: str, grid_type: str, method: str, ispd_thresh: str, bin_window: int, bin_min_days: int, roll_window: int) -> Path:
    if method == "binary-days":
        method_dir = f"bin-win-{bin_window:02d}_bin-min-{bin_min_days:02d}"
    elif method == "rolling-mean":
        method_dir = f"roll-days-{roll_window}"
    else:
        method_dir = method
    return base_dir / sim_name / "zarr" / hemisphere / f"ispd_thresh_{ispd_thresh}" / ice_type / grid_type / method_dir / "mets.zarr"


def load_all_series(args: argparse.Namespace) -> pd.DataFrame:
    base = Path(args.afim_base_dir)
    frames = []
    for sim in args.sim_names:
        path = Path(args.cice_store_template.format(sim_name=sim)) if args.cice_store_template else default_cice_metric_path(
            base, sim, args.hemisphere, args.ice_type, args.grid_type, args.method,
            args.ispd_thresh, args.bin_window, args.bin_min_days, args.roll_window,
        )
        LOGGER.info("Loading CICE %s from %s", sim, path)
        frames.append(load_sia_series_from_store(path, sim, args.start_date, args.end_date, var_name=args.cice_sia_var))

    LOGGER.info("Loading NSIDC from %s", args.nsidc_sia_store)
    frames.append(load_sia_series_from_store(args.nsidc_sia_store, "NSIDC", args.start_date, args.end_date, var_name=args.nsidc_sia_var))
    LOGGER.info("Loading OSI-SAF-450 from %s", args.osisaf_sia_store)
    frames.append(load_sia_series_from_store(args.osisaf_sia_store, "OSI-SAF-450", args.start_date, args.end_date, var_name=args.osisaf_sia_var))
    return pd.concat(frames, axis=1).sort_index()


def plot_full_period(df: pd.DataFrame, out_png: Path, title: str) -> None:
    import pygmt
    out_png.parent.mkdir(parents=True, exist_ok=True)
    long = df.reset_index().melt(id_vars="time", var_name="series", value_name="sia").dropna()
    t0 = long.time.min()
    long["days"] = (long.time - t0).dt.total_seconds() / 86400.0
    ymin = max(0.0, np.floor(long.sia.min() * 2) / 2)
    ymax = np.ceil(long.sia.max() * 2) / 2
    region = [0, float(long.days.max()), ymin, ymax]
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="X20c/9c", frame=[f"WSen+t{title}", "xaf+lDays from start", "yaf+lSIA (10@+6@+ km@+2@+)"])
    for series in df.columns:
        sub = long[long.series == series]
        if series == "NSIDC":
            pen = "1.4p,black"
        elif series == "OSI-SAF-450":
            pen = "1.4p,black,-"
        else:
            pen = "0.8p"
        fig.plot(x=sub.days, y=sub.sia, pen=pen)
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
    fig.savefig(out_png)


def _doy_clim(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    tmp = df.copy()
    tmp["doy"] = tmp.index.dayofyear
    # Drop 29 Feb for a stable 365-day climatological year.
    tmp = tmp[~((tmp.index.month == 2) & (tmp.index.day == 29))]
    for col in df.columns:
        g = tmp.groupby("doy")[col]
        out[col] = pd.DataFrame({"doy": g.mean().index, "mean": g.mean().values, "min": g.min().values, "max": g.max().values})
    return out


def plot_seasonal_envelope(df: pd.DataFrame, out_png: Path, title: str) -> None:
    import pygmt
    out_png.parent.mkdir(parents=True, exist_ok=True)
    clim = _doy_clim(df)
    allvals = pd.concat([v[["min", "max"]] for v in clim.values()])
    ymin = max(0.0, np.floor(allvals.min().min() * 2) / 2)
    ymax = np.ceil(allvals.max().max() * 2) / 2
    fig = pygmt.Figure()
    fig.basemap(region=[1, 365, ymin, ymax], projection="X20c/9c", frame=[f"WSen+t{title}", "xa30f15+lDay of year", "yaf+lSIA (10@+6@+ km@+2@+)"])
    for series, c in clim.items():
        # Fill between min and max using a polygon. Leave colour unspecified except greyscale
        # because the observational black-line convention is important here.
        poly_x = np.r_[c.doy.values, c.doy.values[::-1]]
        poly_y = np.r_[c["max"].values, c["min"].values[::-1]]
        if series not in {"NSIDC", "OSI-SAF-450"}:
            fig.plot(x=poly_x, y=poly_y, close=True, fill="gray85", pen="0.1p,gray70")
        pen = "1.8p,black" if series == "NSIDC" else "1.8p,black,-" if series == "OSI-SAF-450" else "1.2p"
        fig.plot(x=c.doy, y=c["mean"], pen=pen, label=series)
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
    fig.savefig(out_png)


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "SIA_daily.csv", index_label="time")
    df.resample("MS").mean().to_csv(out_dir / "SIA_monthly_mean.csv", index_label="time")
    df.resample("YS").mean().to_csv(out_dir / "SIA_annual_mean.csv", index_label="time")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare CICE, NSIDC, and OSI-SAF-450 SIA time series.")
    p.add_argument("--sim-names", nargs="+", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--afim-base-dir", default="/g/data/gv90/da1339/afim_output")
    p.add_argument("--out-dir", default="/g/data/gv90/da1339/GRAPHICAL/AFIM/SIA_comparisons")
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--ice-type", default="SI")
    p.add_argument("--grid-type", default="Tc")
    p.add_argument("--method", default="binary-days", choices=["raw", "binary-days", "rolling-mean"])
    p.add_argument("--ispd-thresh", default="5.0e-4")
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--cice-store-template", default=None, help="Optional Python format string containing {sim_name}.")
    p.add_argument("--cice-sia-var", default=None)
    p.add_argument("--nsidc-sia-store", required=True)
    p.add_argument("--osisaf-sia-store", required=True)
    p.add_argument("--nsidc-sia-var", default=None)
    p.add_argument("--osisaf-sia-var", default="sia")
    p.add_argument("--title", default="Antarctic sea-ice area comparison")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    out_dir = Path(args.out_dir)
    df = load_all_series(args)
    write_tables(df, out_dir)
    years = f"{pd.to_datetime(args.start_date).year}-{pd.to_datetime(args.end_date).year}"
    plot_full_period(df, out_dir / f"SIA_full_period_{years}.png", args.title)
    plot_seasonal_envelope(df, out_dir / f"SIA_daily_climatology_envelope_{years}.png", args.title + " seasonal envelope")
    LOGGER.info("Wrote outputs under %s", out_dir)


if __name__ == "__main__":
    main()
