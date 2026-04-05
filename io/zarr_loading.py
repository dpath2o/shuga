from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import xarray as xr

from shuga.core.paths import ShugaPaths


_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _slice_time(ds: xr.Dataset, dt0: str, dtN: str) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    return ds.sel(time=slice(pd.to_datetime(dt0), pd.to_datetime(dtN)))


def _find_lat_name(ds: xr.Dataset) -> str | None:
    for name in ("TLAT", "ULAT", "lat", "latitude"):
        if name in ds.variables or name in ds.coords:
            return name
    return None


def _apply_hemisphere_mask(ds: xr.Dataset, hemisphere: str) -> xr.Dataset:
    lat_name = _find_lat_name(ds)
    if lat_name is None:
        return ds
    lat = ds[lat_name]
    hemi = hemisphere.upper()
    mask = lat < 0 if hemi == "SH" else lat > 0
    out = ds.copy()
    for name in list(out.data_vars):
        try:
            out[name] = out[name].where(mask)
        except Exception:
            pass
    return out


def _merge_static(ds_all: xr.Dataset, static_store: Path | None, variables: list[str] | None, logger) -> xr.Dataset:
    if static_store is None or not static_store.exists():
        return ds_all
    ds_static_all = xr.open_zarr(static_store, consolidated=False)
    static_name_set = set(ds_static_all.data_vars) | set(ds_static_all.coords)
    static_name_set.discard("time")
    static_name_set.discard("time_bounds")

    if variables is None:
        ds_static_use = ds_static_all
    else:
        static_requested = [v for v in variables if v in static_name_set]
        ds_static_use = xr.Dataset()
        for v in static_requested:
            if v in ds_static_all.data_vars:
                ds_static_use[v] = ds_static_all[v]
            elif v in ds_static_all.coords:
                ds_static_use = ds_static_use.assign_coords({v: ds_static_all.coords[v]})
    if len(ds_static_use.data_vars) > 0 or len(ds_static_use.coords) > 0:
        if logger is not None:
            logger.info("Merging static variables from: %s", static_store)
        ds_all = xr.merge([ds_all, ds_static_use], compat="override", combine_attrs="override")
    return ds_all


def open_cice_history(
    paths: ShugaPaths,
    *,
    variables: list[str] | None = None,
    extend_days: int = 0,
    chunks: dict | None = None,
    logger=None,
) -> xr.Dataset:
    zarr_root = paths.resolve_cice_store()
    static_store = paths.resolve_static_store()
    if not zarr_root.exists():
        raise FileNotFoundError(f"CICE Zarr root does not exist: {zarr_root}")

    dt0 = pd.to_datetime(paths.run.start_date) - pd.Timedelta(days=int(extend_days))
    dtN = pd.to_datetime(paths.run.end_date) + pd.Timedelta(days=int(extend_days))
    dt0_str = dt0.strftime("%Y-%m-%d")
    dtN_str = dtN.strftime("%Y-%m-%d")

    available_groups = sorted(p.name for p in zarr_root.iterdir() if p.is_dir() and _MONTH_RE.match(p.name))
    if available_groups:
        available_dt0 = pd.to_datetime(f"{available_groups[0]}-01")
        available_dtN = pd.to_datetime(f"{available_groups[-1]}-01") + pd.offsets.MonthEnd(1)
        user_dt0 = max(dt0, available_dt0)
        user_dtN = min(dtN, available_dtN)
        if user_dt0 > user_dtN:
            raise ValueError(
                f"Requested window [{dt0_str}, {dtN_str}] does not intersect available "
                f"data [{available_dt0.date()}, {available_dtN.date()}]"
            )
        required_groups = [
            g for g in available_groups
            if (pd.to_datetime(f"{g}-01") <= user_dtN)
            and (pd.to_datetime(f"{g}-01") + pd.offsets.MonthEnd(1) >= user_dt0)
        ]
        if logger is not None:
            logger.info(
                "Opening grouped monthly Zarr between %s and %s (%d groups)",
                user_dt0.date(), user_dtN.date(), len(required_groups)
            )

        ds_list: list[xr.Dataset] = []
        dynamic_requested = None if variables is None else list(dict.fromkeys(variables))
        for g in required_groups:
            ds = xr.open_zarr(zarr_root, group=g, consolidated=False, chunks=chunks)
            if dynamic_requested is not None:
                present = [v for v in dynamic_requested if v in ds.data_vars or v in ds.coords]
                if not present:
                    continue
                ds = ds[present]
            ds = ds.sel(time=slice(user_dt0, user_dtN))
            if ds.sizes.get("time", 0) > 0:
                ds_list.append(ds)
        if not ds_list:
            raise ValueError("No monthly datasets remained after filtering by time/variables.")
        ds_all = xr.concat(
            ds_list,
            dim="time",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
    else:
        if logger is not None:
            logger.info("Opening flat Zarr store: %s", zarr_root)
        ds_all = xr.open_zarr(zarr_root, consolidated=False, chunks=chunks)
        if variables is not None:
            keep = [v for v in variables if v in ds_all.data_vars or v in ds_all.coords]
            ds_all = ds_all[keep]
        ds_all = _slice_time(ds_all, dt0_str, dtN_str)

    ds_all = _merge_static(ds_all, static_store, variables, logger)
    ds_all = _apply_hemisphere_mask(ds_all, paths.hemisphere)
    ds_all = _slice_time(ds_all, dt0_str, dtN_str)
    ds_all = ds_all.sel(time=slice(paths.run.start_date, paths.run.end_date))
    return ds_all
