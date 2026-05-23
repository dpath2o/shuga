from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
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
        LOGGER.warning(
            "Requested time range %s -> %s has no overlap with metrics store %s -> %s",
            dt0_str,
            dtN_str,
            str(t0_avail),
            str(tN_avail),
        )
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

def _resolve_run_context(run: RunSpec | None = None, *,
                         sim_name: str | None = None,
                         dt0_str: str | None = None,
                         dtN_str: str | None = None,
                         hemisphere: str | None = None,
                         iceh_frequency: str | None = None,
                         project: str | None = None,
                         user: str | None = None) -> tuple[RunSpec, str | None, str | None, str | None]:
    LOGGER.info("Resolving run context.")
    if run is not None:
        sim_name_eff       = sim_name or run.sim_name
        dt0_eff            = dt0_str or run.start_date
        dtN_eff            = dtN_str or run.end_date
        hemisphere_eff     = hemisphere or run.hemisphere
        project_eff        = project or run.project
        user_eff           = user or run.user
        iceh_frequency_eff = iceh_frequency or getattr(run, "iceh_frequency", "daily")
    else:
        sim_name_eff = sim_name
        if sim_name_eff is None:
            raise ValueError("A simulation context is required. Pass sim_name='...' or run=RunSpec(...).")
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

def _resolve_classify_context(classify: ClassificationSpec | None = None, *,
                              classification: str | None = None,
                              grid_type: str | None = None,
                              ice_type: str | None = None,
                              ispd_thresh: float | None = None,
                              bin_window: int | None = None,
                              bin_min_days: int | None = None,
                              roll_window: int | None = None) -> ClassificationSpec:
    LOGGER.info("Resolving classification context.")
    classify_eff = classify or ClassificationSpec()
    methods      = tuple(classify_eff.methods)
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

def _build_paths(*, run: RunSpec,
                 classify: ClassificationSpec | None = None,
                 metrics: MetricsSpec | None = None,
                 plotting: PlottingSpec | None = None,
                 observations: ObservationSpec | None = None,
                 paths: ShugaPaths | None = None,
                 afim_output_root: str | Path | None = None,
                 graphics_root: str | Path | None = None,
                 logs_root: str | Path | None = None,
                 cice_store: str | Path | None = None,
                 static_store: str | Path | None = None,
                 classification_root: str | Path | None = None) -> ShugaPaths:
    LOGGER.info("Building ShugaPaths.")
    base_classify = classify or (paths.classify if paths is not None else ClassificationSpec())
    return ShugaPaths(run                 = run,
                      classify            = base_classify,
                      metrics             = metrics or (paths.metrics if paths is not None else None),
                      plotting            = plotting or (paths.plotting if paths is not None else None),
                      observations        = observations or (paths.observations if paths is not None else None),
                      wave_forcing        = (paths.wave_forcing if paths is not None else None),
                      cice_grid           = (paths.cice_grid if paths is not None else None),
                      lateral_drag        = (paths.lateral_drag if paths is not None else None),
                      afim_output_root    = afim_output_root if afim_output_root is not None else (paths.afim_output_root if paths is not None else None),
                      graphics_root       = graphics_root if graphics_root is not None else (paths.graphics_root if paths is not None else None),
                      logs_root           = logs_root if logs_root is not None else (paths.logs_root if paths is not None else None),
                      cice_store          = cice_store if cice_store is not None else (paths.cice_store if paths is not None else None),
                      static_store        = static_store if static_store is not None else (paths.static_store if paths is not None else None),
                      classification_root = (classification_root
                                             if classification_root is not None
                                             else (paths.classification_root if paths is not None else None)),
                      archive_root        = (paths.archive_root if paths is not None else None))

def _method_candidates(classify: ClassificationSpec, explicit_classification: str | None) -> list[str]:
    if explicit_classification is not None:
        return [normalize_method(explicit_classification)]
    methods = tuple(classify.methods or ())
    if len(methods) == 1:
        return [normalize_method(methods[0])]
    if len(methods) > 1:
        return list(dict.fromkeys(normalize_method(m) for m in methods))
    return list(_CANONICAL_METHODS)

def _resolved_store_to_dict(resolved: ResolvedStore) -> dict[str, Any]:
    return {"sim_name": resolved.sim_name,
            "method": resolved.method,
            "classification": resolved.method,  # backwards-compatible key
            "grid_type": resolved.grid_type,
            "store_kind": resolved.store_kind,
            "path": resolved.path}

def _resolve_store_with_method_candidates(*, locator: CICEStoreLocator, sim_name: str, classify: ClassificationSpec,
                                          classification: str | None, store_kind: str, selection: StoreSelection,
                                          project: str | None, user: str | None) -> ResolvedStore:
    methods = _method_candidates(classify, classification)
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

def load_cice(run: RunSpec | None = None, classify: ClassificationSpec | None = None, metrics: MetricsSpec | None = None,
              plotting: PlottingSpec | None = None, observations: ObservationSpec | None = None, paths: ShugaPaths | None = None, *,
              sim_name: str | None = None,
              dt0_str: str | None = None,
              dtN_str: str | None = None,
              variables = None,
              hemisphere: str | None = None,
              project: str | None = None,
              user: str | None = None,
              iceh_frequency: str | None = None,
              afim_output_root: str | Path | None = None,
              cice_store: str | Path | None = None,
              static_store: str | Path | None = None,
              chunks: dict | None = None) -> xr.Dataset:
    """
    Load CICE history through IceHistoryLoader.

    This is intentionally a thin public wrapper. Frequency-aware grouped-store
    loading, static/dynamic variable splitting, static-store merging, and final
    CICE-history variable subsetting are owned by IceHistoryLoader.
    """
    variables_list = _maybe_listify_variables(variables)
    if variables_list is None:
        LOGGER.warning("load_cice() called with variables=None; this can be expensive for long periods.")
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run,
                                                                     sim_name=sim_name,
                                                                     dt0_str=dt0_str,
                                                                     dtN_str=dtN_str,
                                                                     hemisphere=hemisphere,
                                                                     iceh_frequency=iceh_frequency,
                                                                     project=project,
                                                                     user=user)
    classify_eff = classify or (paths.classify if paths is not None else ClassificationSpec())
    paths_eff = _build_paths(run=run_eff,
                             classify=classify_eff,
                             metrics=metrics,
                             plotting=plotting,
                             observations=observations,
                             paths=paths,
                             afim_output_root=afim_output_root,
                             cice_store=cice_store,
                             static_store=static_store)
    loader = IceHistoryLoader(paths_eff, logger=LOGGER)
    return loader.load(dt0_str=dt0_eff,
                       dtN_str=dtN_eff,
                       variables=variables_list,
                       hemisphere=hemisphere_eff,
                       cice_store=cice_store,
                       static_store=static_store,
                       chunks=chunks)

def load_classified(run: RunSpec | None = None, classify: ClassificationSpec | None = None, metrics: MetricsSpec | None = None,
                    plotting: PlottingSpec | None = None, observations: ObservationSpec | None = None, paths: ShugaPaths | None = None, *,
                    classification: str | None = None,
                    sim_name: str | None = None,
                    dt0_str: str | None = None,
                    dtN_str: str | None = None,
                    variables=None,
                    hemisphere: str | None = None,
                    project: str | None = None,
                    user: str | None = None,
                    grid_type: str | None = None,
                    grid_type_map: dict[str, str] | None = None,
                    ice_type: str | None = None,
                    ispd_thresh: float | None = None,
                    bin_window: int | None = None,
                    bin_min_days: int | None = None,
                    roll_window: int | None = None,
                    afim_output_root: str | Path | None = None,
                    classification_root: str | Path | None = None,
                    chunks: dict | None = None,
                    return_resolved: bool = False):
    """
    Load classified CICE fast-ice output from data.zarr.

    Store discovery is delegated to CICEStoreLocator.
    """
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run,
                                                                     sim_name=sim_name,
                                                                     dt0_str=dt0_str,
                                                                     dtN_str=dtN_str,
                                                                     hemisphere=hemisphere,
                                                                     project=project,
                                                                     user=user)
    variables_list = _maybe_listify_variables(variables)
    grid_type_eff = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff = _resolve_classify_context(classify or (paths.classify if paths is not None else None),
                                             classification=classification,
                                             grid_type=grid_type_eff,
                                             ice_type=ice_type,
                                             ispd_thresh=ispd_thresh,
                                             bin_window=bin_window,
                                             bin_min_days=bin_min_days,
                                             roll_window=roll_window)
    paths_eff = _build_paths(run=run_eff,
                             classify=classify_eff,
                             metrics=metrics,
                             plotting=plotting,
                             observations=observations,
                             paths=paths,
                             afim_output_root=afim_output_root,
                             classification_root=classification_root)
    locator = CICEStoreLocator(run=run_eff,
                               classify=classify_eff,
                               metrics=metrics,
                               plotting=plotting,
                               observations=observations,
                               paths=paths_eff,
                               logger=LOGGER)
    selection = StoreSelection(grid_type=grid_type, grid_type_map=grid_type_map)
    resolved = _resolve_store_with_method_candidates(locator=locator,
                                                     sim_name=run_eff.sim_name,
                                                     classify=classify_eff,
                                                     classification=classification,
                                                     store_kind="classification",
                                                     selection=selection,
                                                     project=project,
                                                     user=user)
    LOGGER.info("Opening classified store for %s [%s/%s]: %s", run_eff.sim_name, resolved.grid_type, resolved.method, resolved.path)
    ds = xr.open_zarr(resolved.path, consolidated=False, chunks=chunks)
    if "FI_mask" not in ds.data_vars and len(ds.data_vars) == 1:
        only = next(iter(ds.data_vars))
        ds = ds.rename({only: "FI_mask"})
    if "FI_mask" not in ds.data_vars:
        raise KeyError(f"Could not find FI_mask in {resolved.path}. Data variables: {list(ds.data_vars)}")
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

def load_metrics(
    run: RunSpec | None = None,
    classify: ClassificationSpec | None = None,
    metrics: MetricsSpec | None = None,
    plotting: PlottingSpec | None = None,
    observations: ObservationSpec | None = None,
    paths: ShugaPaths | None = None,
    *,
    classification: str | None = None,
    sim_name: str | None = None,
    dt0_str: str | None = None,
    dtN_str: str | None = None,
    variables=None,
    hemisphere: str | None = None,
    project: str | None = None,
    user: str | None = None,
    grid_type: str | None = None,
    grid_type_map: dict[str, str] | None = None,
    ice_type: str | None = None,
    ispd_thresh: float | None = None,
    bin_window: int | None = None,
    bin_min_days: int | None = None,
    roll_window: int | None = None,
    afim_output_root: str | Path | None = None,
    classification_root: str | Path | None = None,
    chunks: dict | None = None,
    return_resolved: bool = False,
):
    """
    Load method-specific metrics from mets.zarr.

    Store discovery is delegated to CICEStoreLocator. Time slicing remains
    overlap-aware because rolling/windowed metrics can have different available
    bounds from the requested run window.
    """
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(
        run,
        sim_name=sim_name,
        dt0_str=dt0_str,
        dtN_str=dtN_str,
        hemisphere=hemisphere,
        project=project,
        user=user,
    )
    variables_list = _maybe_listify_variables(variables)
    grid_type_eff = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff = _resolve_classify_context(
        classify or (paths.classify if paths is not None else None),
        classification=classification,
        grid_type=grid_type_eff,
        ice_type=ice_type,
        ispd_thresh=ispd_thresh,
        bin_window=bin_window,
        bin_min_days=bin_min_days,
        roll_window=roll_window,
    )
    paths_eff = _build_paths(
        run=run_eff,
        classify=classify_eff,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        afim_output_root=afim_output_root,
        classification_root=classification_root,
    )
    locator = CICEStoreLocator(
        run=run_eff,
        classify=classify_eff,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths_eff,
        logger=LOGGER,
    )
    selection = StoreSelection(grid_type=grid_type, grid_type_map=grid_type_map)
    resolved = _resolve_store_with_method_candidates(
        locator=locator,
        sim_name=run_eff.sim_name,
        classify=classify_eff,
        classification=classification,
        store_kind="metrics",
        selection=selection,
        project=project,
        user=user,
    )
    LOGGER.info(
        "Opening metrics store for %s [%s/%s]: %s",
        run_eff.sim_name,
        resolved.grid_type,
        resolved.method,
        resolved.path,
    )
    ds = xr.open_zarr(resolved.path, consolidated=False, chunks=chunks)
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        if not keep:
            raise ValueError(
                f"None of the requested metrics variables were found in "
                f"{resolved.path}: {variables_list}"
            )
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
