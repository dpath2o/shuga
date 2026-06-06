from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import resolve_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import RunSpec

_MONTH_GROUP_RE = re.compile(r"^\d{4}-\d{2}$")
_DAY_GROUP_RE   = re.compile(r"^\d{4}_\d{2}_\d{2}$")
LOGGER          = resolve_logger(None, name="shuga.io.iceh_loading")

def normalize_iceh_frequency(value: str | None) -> str:
    token   = str(value or "daily").strip().lower().replace("_", "-")
    aliases = {"d"            : "daily",
               "day"          : "daily",
               "days"         : "daily",
               "daily"        : "daily",
               "h"            : "hourly",
               "hour"         : "hourly",
               "hours"        : "hourly",
               "hourly"       : "hourly",
               "inst"         : "hourly",
               "instantaneous": "hourly"}
    if token not in aliases:
        raise ValueError(f"Unsupported iceh frequency {value!r}; use 'daily' or 'hourly'.")
    return aliases[token]

def _maybe_listify_variables(variables) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [variables]
    return list(dict.fromkeys(list(variables)))

def _is_date_only(value: str | None) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))

def _time_bounds(dt0_str: str | None, dtN_str: str | None, *, frequency: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    dt0 = pd.to_datetime(dt0_str) if dt0_str is not None else None
    dtN = pd.to_datetime(dtN_str) if dtN_str is not None else None
    # For hourly data, end_date="1993-10-31" should mean the whole day,
    # not only exactly 1993-10-31 00:00:00.
    if frequency == "hourly" and _is_date_only(dtN_str):
        dtN = dtN + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return dt0, dtN

def _slice_time(ds: xr.Dataset, dt0_str: str | None, dtN_str: str | None, *, frequency: str) -> xr.Dataset:
    if "time" not in ds.coords or (dt0_str is None and dtN_str is None):
        return ds
    start, end = _time_bounds(dt0_str, dtN_str, frequency=frequency)
    return ds.sel(time=slice(start, end))

def _find_lat_name(ds: xr.Dataset) -> str | None:
    for name in ("TLAT", "ULAT", "NLAT", "ELAT", "lat", "latitude"):
        if name in ds.variables or name in ds.coords:
            return name
    return None

def _apply_hemisphere_mask(ds: xr.Dataset, hemisphere: str | None) -> xr.Dataset:
    if hemisphere is None:
        return ds
    lat_name = _find_lat_name(ds)
    if lat_name is None:
        return ds
    lat = ds[lat_name]
    hemi = str(hemisphere).upper()
    mask = lat < 0 if hemi == "SH" else lat > 0
    return ds.where(mask)

def _split_requested_variables(variables: list[str] | None, paths: ShugaPaths, static_store: Path | None, *,
                               chunks: dict | None = None, logger = LOGGER) -> tuple[list[str] | None, list[str] | None]:
    if variables is None:
        return None, None
    requested = list(dict.fromkeys(variables))
    static_names: set[str] = set()
    ds_static = _open_static_dataset(paths, static_store, variables = None, chunks = chunks, logger = logger)
    if ds_static is not None:
        try:
            static_names = set(ds_static.data_vars) | set(ds_static.coords)
            static_names.discard("time")
            static_names.discard("time_bounds")
        finally:
            ds_static.close()
    static_requested  = [v for v in requested if v in static_names]
    dynamic_requested = [v for v in requested if v not in static_names]
    return dynamic_requested or None, static_requested or None

def _merge_static(ds_all: xr.Dataset, paths: ShugaPaths, static_store: Path | None, variables: list[str] | None, *,
                  chunks: dict | None = None, logger = LOGGER) -> xr.Dataset:
    ds_static_all = _open_static_dataset(paths, static_store, variables = variables, chunks = chunks, logger = logger)
    if ds_static_all is None:
        return ds_all
    try:
        if len(ds_static_all.data_vars) > 0 or len(ds_static_all.coords) > 0:
            ds_all = xr.merge([ds_all, ds_static_all], compat = "override", combine_attrs = "override")
    finally:
        ds_static_all.close()
    return ds_all

def _group_bounds(group: str, *, frequency: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if frequency == "hourly":
        start = pd.to_datetime(group, format="%Y_%m_%d")
        end   = start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return start, end
    start = pd.to_datetime(f"{group}-01")
    end   = start + pd.offsets.MonthEnd(1)
    return start, end

def _valid_group_names(zarr_root: Path, *, frequency: str) -> list[str]:
    pattern = _DAY_GROUP_RE if frequency == "hourly" else _MONTH_GROUP_RE
    return sorted(p.name for p in zarr_root.iterdir() if p.is_dir() and pattern.match(p.name))

def _open_grouped_iceh_store(zarr_root  : Path, *,
                             frequency  : str,
                             dt0_str    : str | None,
                             dtN_str    : str | None,
                             variables  : list[str] | None,
                             chunks     : dict | None,
                             allow_empty: bool = False,
                             logger=LOGGER) -> xr.Dataset:
    if not zarr_root.exists():
        raise FileNotFoundError(f"CICE Zarr root does not exist: {zarr_root}")
    available_groups = _valid_group_names(zarr_root, frequency=frequency)
    if not available_groups:
        logger.info("Opening flat CICE Zarr store: %s", zarr_root)
        ds_all = xr.open_zarr(zarr_root, consolidated=False, chunks=chunks)
        if variables is not None:
            keep = [v for v in variables if v in ds_all.data_vars or v in ds_all.coords]
            ds_all = ds_all[keep]
        return _slice_time(ds_all, dt0_str, dtN_str, frequency=frequency)
    req_dt0, req_dtN = _time_bounds(dt0_str, dtN_str, frequency=frequency)
    avail_dt0, _     = _group_bounds(available_groups[0], frequency=frequency)
    _, avail_dtN     = _group_bounds(available_groups[-1], frequency=frequency)
    user_dt0         = max(req_dt0, avail_dt0) if req_dt0 is not None else avail_dt0
    user_dtN         = min(req_dtN, avail_dtN) if req_dtN is not None else avail_dtN
    if user_dt0 > user_dtN:
        raise ValueError(f"Requested window [{dt0_str}, {dtN_str}] does not intersect available data [{avail_dt0}, {avail_dtN}]")
    required_groups: list[str] = []
    for g in available_groups:
        g0, gN = _group_bounds(g, frequency=frequency)
        if g0 <= user_dtN and gN >= user_dt0:
            required_groups.append(g)
    group_label = "daily" if frequency == "hourly" else "monthly"
    logger.info("Opening grouped %s CICE Zarr between %s and %s (%d groups)", group_label, user_dt0, user_dtN, len(required_groups))
    ds_list: list[xr.Dataset] = []
    dynamic_requested = None if variables is None else list(dict.fromkeys(variables))
    for g in required_groups:
        logger.info("Opening group %s with chunks=%s", g, chunks)
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
        if allow_empty:
            return xr.Dataset()
        raise ValueError("No CICE datasets remained after filtering by time/variables.")
    return xr.concat(ds_list, dim="time", data_vars="minimal",  coords="minimal", compat="override", combine_attrs="override")

def _open_static_dataset(paths: ShugaPaths, static_store: Path | None, *,
                         variables: list[str] | None = None,
                         chunks   : dict | None = None,
                         logger = LOGGER) -> xr.Dataset | None:
    """
    Open the universal CICE static-coordinate store through CICEGridwork.

    This supports both proper xarray-zarr groups and loose-array static stores.
    """
    try:
        from shuga.grid.cice import CICEGridwork
        gridwork = CICEGridwork(paths=paths, logger=logger)
        return gridwork.load_cice_static(P_cice_static_store = static_store,
                                         variables           = variables,
                                         require             = (),
                                         chunks              = chunks,
                                         consolidated        = False,
                                         add_aliases         = True)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Could not open CICE static store %s: %s", static_store, exc)
        return None

class IceHistoryLoader:
    """
    Frequency-aware CICE ice-history loader.

    Supports:
    - daily CICE history stored as iceh_daily.zarr/YYYY-MM
    - hourly instantaneous CICE history stored as iceh_hourly.zarr/YYYY_MM_DD
    """
    def __init__(self, paths: ShugaPaths, *, logger=None) -> None:
        self.paths = paths
        self.run   = paths.run
        self.logger = logger or LOGGER

    @property
    def frequency(self) -> str:
        return normalize_iceh_frequency(getattr(self.run, "iceh_frequency", "daily"))

    def default_chunks(self) -> dict[str, int]:
        if self.frequency == "hourly":
            return {"time": 24}
        return {"time": 31}

    def load(self, *,
             dt0_str     : str | None = None,
             dtN_str     : str | None = None,
             variables                = None,
             hemisphere  : str | None = None,
             cice_store  : str | Path | None = None,
             static_store: str | Path | None = None,
             chunks      : dict | None = None) -> xr.Dataset:
        chunks   = chunks or self.default_chunks()
        var_list = _maybe_listify_variables(variables)
        if var_list is None:
            self.logger.warning("IceHistoryLoader.load() called with variables=None; this may be expensive.")
        zarr_root         = (Path(cice_store).expanduser() if cice_store is not None else self.paths.resolve_cice_store())
        stat_eff          = (Path(static_store).expanduser() if static_store is not None else self.paths.resolve_static_store())
        dyn_req, stat_req = _split_requested_variables(var_list, self.paths, stat_eff, chunks = chunks, logger = self.logger)
        only_stat_req     = var_list is not None and dyn_req is None and stat_req is not None
        if only_stat_req:
            ds_all = xr.Dataset()
        else:
            ds_all = _open_grouped_iceh_store(zarr_root,
                                              frequency   = self.frequency,
                                              dt0_str     = dt0_str or self.run.start_date,
                                              dtN_str     = dtN_str or self.run.end_date,
                                              variables   = dyn_req,
                                              chunks      = chunks,
                                              allow_empty = False,
                                              logger      = self.logger)

        ds_all = _merge_static(ds_all, self.paths, stat_eff, var_list, chunks = chunks, logger = self.logger)
        ds_all = _apply_hemisphere_mask(ds_all, hemisphere or self.run.hemisphere)
        if var_list is not None:
            present = [v for v in var_list if v in ds_all.data_vars or v in ds_all.coords]
            if not present:
                raise ValueError(f"None of the requested variables were found after merging static/dynamic stores: {var_list}")
            ds_all = ds_all[present]
        return ds_all

def load_ice_history(run: RunSpec, paths: ShugaPaths, *,
                     dt0_str     : str | None = None,
                     dtN_str     : str | None = None,
                     variables                = None,
                     hemisphere  : str | None = None,
                     cice_store  : str | Path | None = None,
                     static_store: str | Path | None = None,
                     chunks      : dict | None = None) -> xr.Dataset:
    loader = IceHistoryLoader(paths)
    return loader.load(dt0_str      = dt0_str,
                       dtN_str      = dtN_str,
                       variables    = variables,
                       hemisphere   = hemisphere,
                       cice_store   = cice_store,
                       static_store = static_store,
                       chunks       = chunks)
