from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.context import get_current_context
from shuga.core.logging import resolve_logger
from shuga.core.naming import normalize_method
from shuga.core.paths import ShugaPaths
from shuga.core.store_selection import ResolvedStore, StoreSelection
from shuga.core.types import (ClassificationSpec,
                              MetricsSpec,
                              ObservationSpec,
                              PlottingSpec,
                              RunSpec)
from shuga.io.iceh_loading import IceHistoryLoader
from shuga.io.store_locator import CICEStoreLocator

"""
Public Zarr loading façade.

Implementation ownership:
- CICE history loading belongs to shuga.io.iceh_loading.IceHistoryLoader.
- Classification/metrics store discovery belongs to shuga.io.store_locator.CICEStoreLocator.
- This module resolves user-facing context and applies final masking/slicing.
"""

_CANONICAL_METHODS = ("raw", "binary-days", "rolling-mean")
LOGGER             = resolve_logger(None, name="shuga.io.zarr_loading")

def _with_current_context(run_cfg, cls_cfg, met_cfg, plt_cfg, obs_cfg, pth_cfg, chunks):
    """
    Fill missing loader arguments from the active shuga context.

    Explicit arguments passed to the loader always win over the context.
    """
    ctx = get_current_context()
    return (run_cfg if run_cfg is not None else ctx.run_cfg,
            cls_cfg if cls_cfg is not None else ctx.cls_cfg,
            met_cfg if met_cfg is not None else ctx.met_cfg,
            plt_cfg if plt_cfg is not None else ctx.plt_cfg,
            obs_cfg if obs_cfg is not None else ctx.obs_cfg,
            pth_cfg if pth_cfg is not None else ctx.pth_cfg,
            chunks  if chunks  is not None else ctx.chunks)

def _maybe_listify_variables(variables) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [variables]
    return list(dict.fromkeys(list(variables)))

def _default_value(cls, field_name: str):
    return cls.__dataclass_fields__[field_name].default

def _slice_time(ds: xr.Dataset, dt0: str | None, dtN: str | None) -> xr.Dataset:
    if "time" not in ds.coords or (dt0 is None and dtN is None):
        return ds
    start = pd.to_datetime(dt0) if dt0 is not None else None
    end = pd.to_datetime(dtN) if dtN is not None else None
    return ds.sel(time=slice(start, end))

def _slice_time_overlap(ds: xr.Dataset, dt0_str: str | None, dtN_str: str | None) -> xr.Dataset:
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        return ds
    t0_avail = ds.time.values[0]
    tN_avail = ds.time.values[-1]
    t0_req = np.datetime64(dt0_str) if dt0_str is not None else t0_avail
    tN_req = np.datetime64(dtN_str) if dtN_str is not None else tN_avail
    if t0_req > tN_req:
        raise ValueError(f"Requested time range is reversed: {dt0_str} -> {dtN_str}")
    t0_use = max(t0_req, t0_avail)
    tN_use = min(tN_req, tN_avail)
    if t0_use > tN_use:
        LOGGER.warning("Requested time range %s -> %s has no overlap with metrics store %s -> %s", dt0_str, dtN_str, str(t0_avail), str(tN_avail))
        return ds.isel(time=slice(0, 0))
    return ds.sel(time=slice(t0_use, tN_use))

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
    hemi = str(hemisphere).upper()
    lat  = ds[lat_name]
    mask = lat < 0 if hemi == "SH" else lat > 0
    return ds.where(mask)

def _resolve_run_context(run_cfg: RunSpec | None = None, *,
                         sim_name: str | None = None,
                         dt0_str: str | None = None,
                         dtN_str: str | None = None,
                         hemisphere: str | None = None,
                         iceh_frequency: str | None = None,
                         project: str | None = None,
                         user: str | None = None) -> tuple[RunSpec, str | None, str | None, str | None]:
    LOGGER.info("Resolving run context.")
    if run_cfg is not None:
        sim_name_eff       = sim_name or run_cfg.sim_name
        dt0_eff            = dt0_str or run_cfg.start_date
        dtN_eff            = dtN_str or run_cfg.end_date
        hemisphere_eff     = hemisphere or run_cfg.hemisphere
        project_eff        = project or run_cfg.project
        user_eff           = user or run_cfg.user
        iceh_frequency_eff = iceh_frequency or getattr(run_cfg, "iceh_frequency", "daily")
    else:
        sim_name_eff = sim_name
        if sim_name_eff is None:
            raise ValueError("A simulation context is required. Pass sim_name='...' or run_cfg=RunSpec(...).")
        dt0_eff            = dt0_str
        dtN_eff            = dtN_str
        hemisphere_eff     = hemisphere or _default_value(RunSpec, "hemisphere")
        project_eff        = project or _default_value(RunSpec, "project")
        user_eff           = user or _default_value(RunSpec, "user")
        iceh_frequency_eff = iceh_frequency or _default_value(RunSpec, "iceh_frequency")
    if dt0_eff is not None and dtN_eff is not None:
        if pd.to_datetime(dtN_eff) < pd.to_datetime(dt0_eff):
            raise ValueError(f"dtN_str ({dtN_eff}) must be on or after dt0_str ({dt0_eff}).")
    run_eff = RunSpec(sim_name       = sim_name_eff,
                      start_date     = dt0_eff or "1900-01-01",
                      end_date       = dtN_eff or "2100-12-31",
                      hemisphere     = hemisphere_eff or _default_value(RunSpec, "hemisphere"),
                      project        = project_eff,
                      user           = user_eff,
                      iceh_frequency = iceh_frequency_eff)
    return run_eff, dt0_eff, dtN_eff, hemisphere_eff

def _resolve_classify_context(cls_cfg: ClassificationSpec | None = None, *,
                              classification: str | None = None,
                              grid_type: str | None = None,
                              ice_type: str | None = None,
                              ispd_thresh: float | None = None,
                              bin_window: int | None = None,
                              bin_min_days: int | None = None,
                              roll_window: int | None = None) -> ClassificationSpec:
    LOGGER.info("Resolving classification context.")
    classify_eff = cls_cfg or ClassificationSpec()
    if isinstance(classify_eff.methods, str):
        methods = (classify_eff.methods,)
    else:
        methods = tuple(classify_eff.methods)
    if classification is not None:
        methods = (normalize_method(classification),)
    return replace(classify_eff,
                   ice_type     = ice_type or classify_eff.ice_type,
                   grid_type    = grid_type or classify_eff.grid_type,
                   ispd_thresh  = float(ispd_thresh if ispd_thresh is not None else classify_eff.ispd_thresh),
                   bin_window   = int(bin_window if bin_window is not None else classify_eff.bin_window),
                   bin_min_days = int(bin_min_days if bin_min_days is not None else classify_eff.bin_min_days),
                   roll_window  = int(roll_window if roll_window is not None else classify_eff.roll_window),
                   methods      = methods)

def _build_paths(*, run_cfg: RunSpec,
                 cls_cfg            : ClassificationSpec | None = None,
                 met_cfg            : MetricsSpec | None = None,
                 plt_cfg            : PlottingSpec | None = None,
                 obs_cfg            : ObservationSpec | None = None,
                 pth_cfg            : ShugaPaths | None = None,
                 afim_output_root   : str | Path | None = None,
                 graphics_root      : str | Path | None = None,
                 logs_root          : str | Path | None = None,
                 cice_store         : str | Path | None = None,
                 static_store       : str | Path | None = None,
                 classification_root: str | Path | None = None) -> ShugaPaths:
    LOGGER.info("Building ShugaPaths.")
    base_cls_cfg = cls_cfg or (pth_cfg.cls_cfg if pth_cfg is not None else ClassificationSpec())
    return ShugaPaths(run_cfg             = run_cfg,
                      cls_cfg             = base_cls_cfg,
                      met_cfg             = met_cfg or (pth_cfg.met_cfg if pth_cfg is not None else None),
                      plt_cfg             = plt_cfg or (pth_cfg.plt_cfg if pth_cfg is not None else None),
                      obs_cfg             = obs_cfg or (pth_cfg.obs_cfg if pth_cfg is not None else None),
                      wave_frcg_cfg       = (pth_cfg.wave_frcg_cfg if pth_cfg is not None else None),
                      G_cice_cfg          = (pth_cfg.G_cice_cfg if pth_cfg is not None else None),
                      LD_cfg              = (pth_cfg.LD_cfg if pth_cfg is not None else None),
                      afim_output_root    = afim_output_root if afim_output_root is not None else (pth_cfg.afim_output_root if pth_cfg is not None else None),
                      graphics_root       = graphics_root if graphics_root is not None else (pth_cfg.graphics_root if pth_cfg is not None else None),
                      logs_root           = logs_root if logs_root is not None else (pth_cfg.logs_root if pth_cfg is not None else None),
                      cice_store          = cice_store if cice_store is not None else (pth_cfg.cice_store if pth_cfg is not None else None),
                      static_store        = static_store if static_store is not None else (pth_cfg.static_store if pth_cfg is not None else None),
                      classification_root = (classification_root if classification_root is not None else (pth_cfg.classification_root if pth_cfg is not None else None)),
                      archive_root        = (pth_cfg.archive_root if pth_cfg is not None else None))

def _method_candidates(cls_cfg: ClassificationSpec, explicit_classification: str | None) -> list[str]:
    if explicit_classification is not None:
        return [normalize_method(explicit_classification)]
    if isinstance(cls_cfg.methods, str):
        methods = (cls_cfg.methods,)
    else:
        methods = tuple(cls_cfg.methods or ())
    if len(methods) == 1:
        return [normalize_method(methods[0])]
    if len(methods) > 1:
        return list(dict.fromkeys(normalize_method(m) for m in methods))
    return list(_CANONICAL_METHODS)

def _resolved_store_to_dict(resolved: ResolvedStore) -> dict[str, Any]:
    return {"sim_name"      : resolved.sim_name,
            "method"        : resolved.method,
            "classification": resolved.method,  # backwards-compatible key
            "grid_type"     : resolved.grid_type,
            "store_kind"    : resolved.store_kind,
            "path"          : resolved.path}

def _resolve_store_with_method_candidates(*, locator: CICEStoreLocator, sim_name: str, cls_cfg: ClassificationSpec,
                                          classification: str | None, store_kind: str, selection: StoreSelection,
                                          project: str | None, user: str | None) -> ResolvedStore:
    methods = _method_candidates(cls_cfg, classification)
    matches: list[ResolvedStore] = []
    not_found: list[str] = []
    for method in methods:
        try:
            matches.append(locator.resolve_store(sim_name   = sim_name,
                                                 method     = method,
                                                 store_kind = store_kind,
                                                 selection  = selection,
                                                 project    = project,
                                                 user       = user))
        except FileNotFoundError as exc:
            not_found.append(str(exc))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        method_list = ", ".join(methods)
        details = "\n".join(not_found[-3:])
        raise FileNotFoundError(f"Could not resolve any {store_kind} store for sim={sim_name!r}. "
                                f"Tried methods: {method_list}."
                                + (f"\nRecent lookup failures:\n{details}" if details else ""))
    match_lines = "\n".join(f"- method={m.method}, grid_type={m.grid_type}, path={m.path}" for m in matches)
    raise ValueError(f"Multiple candidate {store_kind} stores found for sim={sim_name!r}. "
                     "Pass classification='raw', 'binary-days', or 'rolling-mean', and/or "
                     "grid_type='Tb'/'Tc'.\n"
                     f"{match_lines}")

def load_cice(run_cfg         : RunSpec | None = None,
              cls_cfg         : ClassificationSpec | None = None,
              met_cfg         : MetricsSpec | None = None,
              plt_cfg         : PlottingSpec | None = None,
              obs_cfg         : ObservationSpec | None = None,
              pth_cfg         : ShugaPaths | None = None, *,
              sim_name        : str | None = None,
              dt0_str         : str | None = None,
              dtN_str         : str | None = None,
              variables = None,
              hemisphere      : str | None = None,
              project         : str | None = None,
              user            : str | None = None,
              iceh_frequency  : str | None = None,
              afim_output_root: str | Path | None = None,
              cice_store      : str | Path | None = None,
              static_store    : str | Path | None = None,
              chunks          : dict | None = None) -> xr.Dataset:
    """
    Load CICE history through IceHistoryLoader.

    This is intentionally a thin public wrapper. Frequency-aware grouped-store
    loading, static/dynamic variable splitting, static-store merging, and final
    CICE-history variable subsetting are owned by IceHistoryLoader.
    """
    run_cfg, cls_cfg, met_cfg, plt_cfg, obs_cfg, pth_cfg, chunks = _with_current_context(run_cfg, cls_cfg, met_cfg, plt_cfg, obs_cfg, pth_cfg, chunks)
    variables_list = _maybe_listify_variables(variables)
    if variables_list is None:
        LOGGER.warning("load_cice() called with variables=None; this can be expensive for long periods.")
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run_cfg,
                                                                     sim_name       = sim_name,
                                                                     dt0_str        = dt0_str,
                                                                     dtN_str        = dtN_str,
                                                                     hemisphere     = hemisphere,
                                                                     iceh_frequency = iceh_frequency,
                                                                     project        = project,
                                                                     user           = user)
    classify_eff = cls_cfg or (pth_cfg.cls_cfg if pth_cfg is not None else ClassificationSpec())
    paths_eff    = _build_paths(run_cfg          = run_eff,
                                cls_cfg          = classify_eff,
                                met_cfg          = met_cfg,
                                plt_cfg          = plt_cfg,
                                obs_cfg          = obs_cfg,
                                pth_cfg          = pth_cfg,
                                afim_output_root = afim_output_root,
                                cice_store       = cice_store,
                                static_store     = static_store)
    loader       = IceHistoryLoader(paths_eff, logger = LOGGER)
    return loader.load(dt0_str      = dt0_eff,
                       dtN_str      = dtN_eff,
                       variables    = variables_list,
                       hemisphere   = hemisphere_eff,
                       cice_store   = cice_store,
                       static_store = static_store,
                       chunks       = chunks)

def load_classified(run_cfg            : RunSpec | None = None,
                    cls_cfg            : ClassificationSpec | None = None,
                    met_cfg            : MetricsSpec | None = None,
                    plt_cfg            : PlottingSpec | None = None,
                    obs_cfg            : ObservationSpec | None = None,
                    pth_cfg            : ShugaPaths | None = None, *,
                    classification     : str | None = None,
                    sim_name           : str | None = None,
                    dt0_str            : str | None = None,
                    dtN_str            : str | None = None,
                    variables = None,
                    hemisphere         : str | None = None,
                    project            : str | None = None,
                    user               : str | None = None,
                    grid_type          : str | None = None,
                    grid_type_map      : dict[str, str] | None = None,
                    ice_type           : str | None = None,
                    ispd_thresh        : float | None = None,
                    bin_window         : int | None = None,
                    bin_min_days       : int | None = None,
                    roll_window        : int | None = None,
                    afim_output_root   : str | Path | None = None,
                    classification_root: str | Path | None = None,
                    chunks             : dict | None = None,
                    return_resolved    : bool = False):
    """
    Load classified CICE fast-ice output from data.zarr.

    Store discovery is delegated to CICES
    """
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run_cfg,
                                                                     sim_name   = sim_name,
                                                                     dt0_str    = dt0_str,
                                                                     dtN_str    = dtN_str,
                                                                     hemisphere = hemisphere,
                                                                     project    = project,
                                                                     user       = user)
    variables_list = _maybe_listify_variables(variables)
    grid_type_eff  = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff   = _resolve_classify_context(cls_cfg or (pth_cfg.cls_cfg if pth_cfg is not None else None),
                                               classification = classification,
                                               grid_type      = grid_type_eff,
                                               ice_type       = ice_type,
                                               ispd_thresh    = ispd_thresh,
                                               bin_window     = bin_window,
                                               bin_min_days   = bin_min_days,
                                               roll_window    = roll_window)
    paths_eff      = _build_paths(run_cfg             = run_eff,
                                  cls_cfg             = classify_eff,
                                  met_cfg             = met_cfg,
                                  plt_cfg             = plt_cfg,
                                  obs_cfg             = obs_cfg,
                                  pth_cfg             = pth_cfg,
                                  afim_output_root    = afim_output_root,
                                  classification_root = classification_root)
    locator        = CICEStoreLocator(run_cfg = run_eff,
                                      cls_cfg = classify_eff,
                                      met_cfg = met_cfg,
                                      plt_cfg = plt_cfg,
                                      obs_cfg = obs_cfg,
                                      pth_cfg = paths_eff,
                                      logger  = LOGGER)
    selection      = StoreSelection(grid_type = grid_type, grid_type_map = grid_type_map)
    resolved       = _resolve_store_with_method_candidates(locator        = locator,
                                                           sim_name       = run_eff.sim_name,
                                                           cls_cfg        = classify_eff,
                                                           classification = classification,
                                                           store_kind     = "classification",
                                                           selection      = selection,
                                                           project        = project,
                                                           user           = user)
    LOGGER.info("Opening classified store for %s [%s/%s]: %s", run_eff.sim_name, resolved.grid_type, resolved.method, resolved.path)
    ds            = xr.open_zarr(resolved.path, consolidated = False, chunks = chunks)
    domain        = str(classify_eff.ice_type).strip().upper()
    expected_mask = f"{domain}_mask"
    # Backward-compatible rename only when there is a single anonymous/legacy var.
    if expected_mask not in ds.data_vars and len(ds.data_vars) == 1:
        only = next(iter(ds.data_vars))
        ds = ds.rename({only: expected_mask})
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        if not keep:
            raise ValueError(f"None of the requested classified variables were found in {resolved.path}: {variables_list}")
        ds = ds[keep]
    else:
        if expected_mask not in ds.data_vars:
            raise KeyError(f"Could not find {expected_mask} in {resolved.path}. Data variables: {list(ds.data_vars)}")
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        if not keep:
            raise ValueError(f"None of the requested classified variables were found in {resolved.path}: {variables_list}")
        ds = ds[keep]
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    ds = _slice_time(ds, dt0_eff, dtN_eff)
    if return_resolved:
        return ds, _resolved_store_to_dict(resolved)
    return ds

def load_metrics(run_cfg            : RunSpec | None = None,
                 cls_cfg            : ClassificationSpec | None = None,
                 met_cfg            : MetricsSpec | None = None,
                 plt_cfg            : PlottingSpec | None = None,
                 obs_cfg            : ObservationSpec | None = None,
                 pth_cfg            : ShugaPaths | None = None, *,
                 classification     : str | None = None,
                 sim_name           : str | None = None,
                 dt0_str            : str | None = None,
                 dtN_str            : str | None = None,
                 variables = None,
                 hemisphere         : str | None = None,
                 project            : str | None = None,
                 user               : str | None = None,
                 grid_type          : str | None = None,
                 grid_type_map      : dict[str, str] | None = None,
                 ice_type           : str | None = None,
                 ispd_thresh        : float | None = None,
                 bin_window         : int | None = None,
                 bin_min_days       : int | None = None,
                 roll_window        : int | None = None,
                 afim_output_root   : str | Path | None = None,
                 classification_root: str | Path | None = None,
                 chunks             : dict | None = None,
                 return_resolved    : bool = False):
    """
    Load method-specific metrics from mets.zarr.

    Store discovery is delegated to CICEStoreLocator. Time slicing remains
    overlap-aware because rolling/windowed metrics can have different available
    bounds from the requested run window.
    """
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run_cfg,
                                                                     sim_name   = sim_name,
                                                                     dt0_str    = dt0_str,
                                                                     dtN_str    = dtN_str,
                                                                     hemisphere = hemisphere,
                                                                     project    = project,
                                                                     user       = user)
    variables_list = _maybe_listify_variables(variables)
    grid_type_eff  = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff   = _resolve_classify_context(cls_cfg or (pth_cfg.cls_cfg if pth_cfg is not None else None),
                                               classification = classification,
                                               grid_type      = grid_type_eff,
                                               ice_type       = ice_type,
                                               ispd_thresh    = ispd_thresh,
                                               bin_window     = bin_window,
                                               bin_min_days   = bin_min_days,
                                               roll_window    = roll_window)

    paths_eff      = _build_paths(run_cfg             = run_eff,
                                  cls_cfg             = classify_eff,
                                  met_cfg             = met_cfg,
                                  plt_cfg             = plt_cfg,
                                  obs_cfg             = obs_cfg,
                                  pth_cfg             = pth_cfg,
                                  afim_output_root    = afim_output_root,
                                  classification_root = classification_root)
    locator        = CICEStoreLocator(run_cfg = run_eff,
                                      cls_cfg = classify_eff,
                                      met_cfg = met_cfg,
                                      plt_cfg = plt_cfg,
                                      obs_cfg = obs_cfg,
                                      pth_cfg = paths_eff,
                                      logger  = LOGGER)
    selection      = StoreSelection(grid_type = grid_type, grid_type_map = grid_type_map)
    resolved       = _resolve_store_with_method_candidates(locator        = locator,
                                                           sim_name       = run_eff.sim_name,
                                                           cls_cfg        = classify_eff,
                                                           classification = classification,
                                                           store_kind     = "metrics",
                                                           selection      = selection,
                                                           project        = project,
                                                           user           = user)
    LOGGER.info("Opening metrics store for %s [%s/%s]: %s", run_eff.sim_name, resolved.grid_type, resolved.method, resolved.path)
    ds = xr.open_zarr(resolved.path, consolidated=False, chunks=chunks)
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        if not keep:
            raise ValueError(f"None of the requested metrics variables were found in {resolved.path}: {variables_list}")
        ds = ds[keep]
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    ds = _slice_time_overlap(ds, dt0_eff, dtN_eff)
    if return_resolved:
        return ds, _resolved_store_to_dict(resolved)
    return ds

def open_cice_history(*args, **kwargs) -> xr.Dataset:
    """
    Backwards-compatible alias for load_cice().
    """
    return load_cice(*args, **kwargs)
