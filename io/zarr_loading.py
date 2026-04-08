from __future__ import annotations
import re
import pandas           as pd
import xarray           as xr
from dataclasses        import replace
from pathlib            import Path
from typing             import Any
from shuga.core.logging import resolve_logger
from shuga.core.naming  import method_dirname, normalize_method
from shuga.core.paths   import ShugaPaths
from shuga.core.types   import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec

_MONTH_RE          = re.compile(r"^\d{4}-\d{2}$")
_CANONICAL_METHODS = ("raw", "binary-days", "rolling-mean")
LOGGER             = resolve_logger(None, name="shuga.io.zarr_loading")

def _slice_time(ds: xr.Dataset, dt0: str | None, dtN: str | None) -> xr.Dataset:
    if "time" not in ds.coords or (dt0 is None and dtN is None):
        return ds
    start = pd.to_datetime(dt0) if dt0 is not None else None
    end = pd.to_datetime(dtN) if dtN is not None else None
    return ds.sel(time=slice(start, end))

def _find_lat_name(ds: xr.Dataset) -> str | None:
    for name in ("TLAT", "ULAT", "lat", "latitude"):
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

def _merge_static(ds_all: xr.Dataset, static_store: Path | None, variables: list[str] | None) -> xr.Dataset:
    LOGGER.info(f"merging {static_store} with iceh_daily.zarr dataset")
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
        LOGGER.info("Merging static variables from: %s", static_store)
        ds_all = xr.merge([ds_all, ds_static_use], compat="override", combine_attrs="override")
    return ds_all

def _split_requested_variables(variables: list[str] | None,
                               static_store: Path | None) -> tuple[list[str] | None, list[str] | None]:
    """
    Split requested variable names into dynamic-store and static-store requests.
    """
    if variables is None:
        return None, None
    requested              = list(dict.fromkeys(variables))
    static_names: set[str] = set()
    if static_store is not None and static_store.exists():
        try:
            ds_static    = xr.open_zarr(static_store, consolidated=False)
            static_names = set(ds_static.data_vars) | set(ds_static.coords)
        except Exception:
            static_names = set()
    static_requested  = [v for v in requested if v in static_names]
    dynamic_requested = [v for v in requested if v not in static_names]
    return dynamic_requested or None, static_requested or None

def _maybe_listify_variables(variables) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [variables]
    return list(dict.fromkeys(list(variables)))

def _default_value(cls, field_name: str):
    return cls.__dataclass_fields__[field_name].default

def _resolve_run_context(run        : RunSpec | None = None, *,
                         sim_name   : str | None = None,
                         dt0_str    : str | None = None,
                         dtN_str    : str | None = None,
                         hemisphere : str | None = None,
                         project    : str | None = None,
                         user       : str | None = None) -> tuple[RunSpec, str | None, str | None, str | None]:
    LOGGER.info("resolving run context: sim_name, date start/stop, hemisphere, etc.")
    if run is not None:
        sim_name_eff   = sim_name   or run.sim_name
        dt0_eff        = dt0_str    or run.start_date
        dtN_eff        = dtN_str    or run.end_date
        hemisphere_eff = hemisphere or run.hemisphere
        project_eff    = project    or run.project
        user_eff       = user       or run.user
    else:
        sim_name_eff   = sim_name
        if sim_name_eff is None:
            raise ValueError("A simulation context is required. Pass sim_name='...' or run=RunSpec(...).")
        dt0_eff        = dt0_str
        dtN_eff        = dtN_str
        hemisphere_eff = hemisphere or _default_value(RunSpec, "hemisphere")
        project_eff    = project    or _default_value(RunSpec, "project")
        user_eff       = user       or _default_value(RunSpec, "user")
    if dt0_eff is not None and dtN_eff is not None and pd.to_datetime(dtN_eff) < pd.to_datetime(dt0_eff):
        raise ValueError(f"dtN_str ({dtN_eff}) must be on or after dt0_str ({dt0_eff}).")
    run_eff = RunSpec(sim_name   = sim_name_eff,
                      start_date = dt0_eff or "1900-01-01",
                      end_date   = dtN_eff or "2100-12-31",
                      hemisphere = hemisphere_eff or _default_value(RunSpec, "hemisphere"),
                      project    = project_eff,
                      user       = user_eff)
    return run_eff, dt0_eff, dtN_eff, hemisphere_eff

def _resolve_classify_context(classify       : ClassificationSpec | None = None, *,
                              classification : str | None = None,
                              grid_type      : str | None = None,
                              ice_type       : str | None = None,
                              ispd_thresh    : float | None = None,
                              bin_window     : int | None = None,
                              bin_min_days   : int | None = None,
                              roll_window    : int | None = None) -> ClassificationSpec:
    LOGGER.info("resolving classification context")
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
                 classify            : ClassificationSpec | None = None,
                 metrics             : MetricsSpec | None = None,
                 plotting            : PlottingSpec | None = None,
                 observations        : ObservationSpec | None = None,
                 paths               : ShugaPaths | None = None,
                 afim_output_root    : str | Path | None = None,
                 graphics_root       : str | Path | None = None,
                 logs_root           : str | Path | None = None,
                 cice_store          : str | Path | None = None,
                 static_store        : str | Path | None = None,
                 classification_root : str | Path | None = None) -> ShugaPaths:
    LOGGER.info("building paths")
    base_classify = classify or (paths.classify if paths is not None else ClassificationSpec())
    return ShugaPaths(run                 = run,
                      classify            = base_classify,
                      metrics             = metrics or (paths.metrics if paths is not None else None),
                      plotting            = plotting or (paths.plotting if paths is not None else None),
                      observations        = observations or (paths.observations if paths is not None else None),
                      afim_output_root    = afim_output_root if afim_output_root is not None else (paths.afim_output_root if paths is not None else None),
                      graphics_root       = graphics_root if graphics_root is not None else (paths.graphics_root if paths is not None else None),
                      logs_root           = logs_root if logs_root is not None else (paths.logs_root if paths is not None else None),
                      cice_store          = cice_store if cice_store is not None else (paths.cice_store if paths is not None else None),
                      static_store        = static_store if static_store is not None else (paths.static_store if paths is not None else None),
                      classification_root = classification_root if classification_root is not None else (paths.classification_root if paths is not None else None))

def _open_grouped_cice_store(zarr_root : Path, *,
                             dt0_str   : str | None,
                             dtN_str   : str | None,
                             variables : list[str] | None,
                             chunks    : dict | None,
                             allow_empty : bool = False) -> xr.Dataset:
    if not zarr_root.exists():
        raise FileNotFoundError(f"CICE Zarr root does not exist: {zarr_root}")
    available_groups = sorted(p.name for p in zarr_root.iterdir() if p.is_dir() and _MONTH_RE.match(p.name))
    if not available_groups:
        LOGGER.info("Opening flat Zarr store: %s", zarr_root)
        ds_all = xr.open_zarr(zarr_root, consolidated=False, chunks=chunks)
        if variables is not None:
            keep = [v for v in variables if v in ds_all.data_vars or v in ds_all.coords]
            ds_all = ds_all[keep]
        return _slice_time(ds_all, dt0_str, dtN_str)
    dt0 = pd.to_datetime(dt0_str) if dt0_str is not None else None
    dtN = pd.to_datetime(dtN_str) if dtN_str is not None else None
    available_dt0 = pd.to_datetime(f"{available_groups[0]}-01")
    available_dtN = pd.to_datetime(f"{available_groups[-1]}-01") + pd.offsets.MonthEnd(1)
    user_dt0 = max(dt0, available_dt0) if dt0 is not None else available_dt0
    user_dtN = min(dtN, available_dtN) if dtN is not None else available_dtN
    if user_dt0 > user_dtN:
        raise ValueError(f"Requested window [{dt0_str}, {dtN_str}] does not intersect available "
                         f"data [{available_dt0.date()}, {available_dtN.date()}]")
    required_groups = [g for g in available_groups
                       if (pd.to_datetime(f"{g}-01") <= user_dtN)
                       and (pd.to_datetime(f"{g}-01") + pd.offsets.MonthEnd(1) >= user_dt0)]
    LOGGER.info("Opening grouped monthly Zarr between %s and %s (%d groups)",user_dt0.date(), user_dtN.date(), len(required_groups))
    ds_list: list[xr.Dataset] = []
    dynamic_requested = None if variables is None else list(dict.fromkeys(variables))
    LOGGER.info(f"{zarr_root} on ...")
    for g in required_groups:
        LOGGER.info(f"\tgroup {g} with chunks {chunks}")
        ds = xr.open_zarr(zarr_root, group=g, consolidated=False, chunks=chunks)
        if dynamic_requested is not None:
            LOGGER.debug(f"\tsubset of variables requested--checking existence in dataset")
            present = [v for v in dynamic_requested if v in ds.data_vars or v in ds.coords]
            if not present:
                continue
            ds = ds[present]
        LOGGER.debug(f"\tslicing the time")
        ds = ds.sel(time=slice(user_dt0, user_dtN))
        if ds.sizes.get("time", 0) > 0:
            ds_list.append(ds)
            LOGGER.debug("\tappended to dataset list")
    if not ds_list:
        if allow_empty:
            return xr.Dataset()
        raise ValueError("No monthly datasets remained after filtering by time/variables.")
    LOGGER.info(f"concatenating along time dim and returning")
    return xr.concat(ds_list,
                     dim           = "time",
                     data_vars     = "minimal",
                     coords        = "minimal",
                     compat        = "override",
                     combine_attrs = "override")

def _grid_type_candidates(classify: ClassificationSpec, explicit_grid_type: str | None, parent: Path) -> list[str]:
    requested: list[str] = []
    if explicit_grid_type is not None:
        requested.append(str(explicit_grid_type))
    else:
        gt = classify.grid_type
        if isinstance(gt, str):
            requested.append(gt)
        else:
            requested.extend(str(x) for x in gt)
    if parent.exists():
        for p in sorted(parent.iterdir()):
            if p.is_dir() and p.name not in requested:
                requested.append(p.name)
    # de-dup while preserving order
    return list(dict.fromkeys(requested))

def _method_candidates(classify: ClassificationSpec, explicit_classification: str | None) -> list[str]:
    if explicit_classification is not None:
        return [normalize_method(explicit_classification)]
    methods = tuple(classify.methods or ())
    if len(methods) == 1:
        return [normalize_method(methods[0])]
    if len(methods) > 1:
        return list(dict.fromkeys(normalize_method(m) for m in methods))
    return list(_CANONICAL_METHODS)

def _resolve_class_store_path(paths: ShugaPaths, classify: ClassificationSpec, *,
                              classification : str | None,
                              grid_type      : str | None,
                              store_name     : str) -> dict[str, Any]:
    parent            = paths.classification_root_path.parent
    grid_candidates   = _grid_type_candidates(classify, grid_type, parent)
    method_candidates = _method_candidates(classify, classification)
    matches: list[dict[str, Any]] = []
    for gt in grid_candidates:
        for method in method_candidates:
            candidate = parent / gt / method_dirname(method,
                                                     bin_window   = classify.bin_window,
                                                     bin_min_days = classify.bin_min_days,
                                                     roll_window  = classify.roll_window) / store_name
            if candidate.exists():
                matches.append({"path": candidate, "grid_type": gt, "classification": method})
    if len(matches) == 1:
        return matches[0]
    if not matches:
        look = f"{parent}/*/<method>/{store_name}"
        raise FileNotFoundError(f"Could not find any store for sim={paths.run.sim_name!r}. Looked under: {look}")
    if classification is None:
        method_set = sorted({m["classification"] for m in matches})
        if len(method_set) > 1:
            raise ValueError(f"Multiple classification methods are available for sim={paths.run.sim_name!r}: {method_set}. "
                             "Pass classification='raw', 'binary-days', or 'rolling-mean'.")
    if grid_type is None:
        grid_set = sorted({m["grid_type"] for m in matches})
        if len(grid_set) > 1:
            raise ValueError(f"Multiple grid_type branches are available for sim={paths.run.sim_name!r}: {grid_set}. "
                             "Pass grid_type='Tb' or 'Tc' (etc.).")
    match_lines = "\n".join(str(m["path"]) for m in matches)
    raise ValueError(f"Ambiguous store resolution for sim={paths.run.sim_name!r}. Matching stores:\n{match_lines}")

def load_cice(run: RunSpec | None = None,
              classify: ClassificationSpec | None = None,
              metrics: MetricsSpec | None = None,
              plotting: PlottingSpec | None = None,
              observations: ObservationSpec | None = None,
              paths: ShugaPaths | None = None,
              *,
              sim_name: str | None = None,
              dt0_str: str | None = None,
              dtN_str: str | None = None,
              variables=None,
              hemisphere: str | None = None,
              project: str | None = None,
              user             : str | None = None,
              afim_output_root : str | Path | None = None,
              cice_store       : str | Path | None = None,
              static_store     : str | Path | None = None,
              chunks           : dict | None = None) -> xr.Dataset:
    if chunks is None:
        chunks = {"time": 31}
    variables_list = _maybe_listify_variables(variables)
    if variables_list is None:
        LOGGER.warning("load_cice() called with variables=None; loading all variables from iceh_daily.zarr "
                       "can be very memory-intensive for long date ranges.")
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(
        run,
        sim_name=sim_name,
        dt0_str=dt0_str,
        dtN_str=dtN_str,
        hemisphere=hemisphere,
        project=project,
        user=user,
    )
    classify_eff = classify or (paths.classify if paths is not None else ClassificationSpec())
    paths_eff = _build_paths(
        run=run_eff,
        classify=classify_eff,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        afim_output_root=afim_output_root,
        cice_store=cice_store,
        static_store=static_store,
    )
    zarr_root = paths_eff.resolve_cice_store()
    static_eff = Path(static_store).expanduser() if static_store is not None else paths_eff.resolve_static_store()
    dynamic_requested, static_requested = _split_requested_variables(
        variables_list,
        static_eff,
    )
    # If user requested only static variables, allow grouped monthly open to return empty
    allow_empty_dynamic = variables_list is not None and dynamic_requested is None
    ds_all = _open_grouped_cice_store(
        zarr_root,
        dt0_str=dt0_eff,
        dtN_str=dtN_eff,
        variables=dynamic_requested,
        chunks=chunks,
        allow_empty=allow_empty_dynamic,
    )
    ds_all = _merge_static(ds_all, static_eff, variables_list)
    ds_all = _apply_hemisphere_mask(ds_all, hemisphere_eff)
    # Final subset on merged dataset
    if variables_list is not None:
        present = [v for v in variables_list if v in ds_all.data_vars or v in ds_all.coords]
        if not present:
            raise ValueError(
                f"None of the requested variables were found after merging static/dynamic stores: {variables_list}"
            )
        ds_all = ds_all[present]
    return ds_all
    # zarr_root = paths_eff.resolve_cice_store()
    # static_eff = Path(static_store).expanduser() if static_store is not None else paths_eff.resolve_static_store()
    # ds_all = _open_grouped_cice_store(
    #     zarr_root,
    #     dt0_str=dt0_eff,
    #     dtN_str=dtN_eff,
    #     variables=variables_list,
    #     chunks=chunks
    # )
    # ds_all = _merge_static(ds_all, static_eff, variables_list)
    # ds_all = _apply_hemisphere_mask(ds_all, hemisphere_eff)
    # ds_all = _slice_time(ds_all, dt0_eff, dtN_eff)
    # return ds_all

def load_classified(
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
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(
        run,
        sim_name=sim_name,
        dt0_str=dt0_str,
        dtN_str=dtN_str,
        hemisphere=hemisphere,
        project=project,
        user=user,
    )
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
    resolved = _resolve_class_store_path(
        paths_eff,
        classify_eff,
        classification=classification,
        grid_type=grid_type_eff,
        store_name="data.zarr",
    )
    LOGGER.info(
        "Opening classified store for %s [%s/%s]: %s",
        run_eff.sim_name,
        resolved["grid_type"],
        resolved["classification"],
        resolved["path"],
    )
    ds = xr.open_zarr(resolved["path"], consolidated=False, chunks=chunks)
    if "FI_mask" in ds.data_vars:
        ds = ds[["FI_mask"]]
    elif len(ds.data_vars) == 1:
        only = next(iter(ds.data_vars))
        ds = ds[[only]].rename({only: "FI_mask"})
    else:
        raise KeyError(f"Could not find FI_mask in {resolved['path']}. Data variables: {list(ds.data_vars)}")
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    ds = _slice_time(ds, dt0_eff, dtN_eff)
    if return_resolved:
        return ds, resolved
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
    variables_list = _maybe_listify_variables(variables)
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(
        run,
        sim_name=sim_name,
        dt0_str=dt0_str,
        dtN_str=dtN_str,
        hemisphere=hemisphere,
        project=project,
        user=user,
    )
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
    resolved = _resolve_class_store_path(
        paths_eff,
        classify_eff,
        classification=classification,
        grid_type=grid_type_eff,
        store_name="mets.zarr",
    )
    LOGGER.info(
            "Opening metrics store for %s [%s/%s]: %s",
            run_eff.sim_name,
            resolved["grid_type"],
            resolved["classification"],
            resolved["path"],
    )
    ds = xr.open_zarr(resolved["path"], consolidated=False, chunks=chunks)
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        ds = ds[keep]
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    ds = _slice_time(ds, dt0_eff, dtN_eff)
    if return_resolved:
        return ds, resolved
    return ds


def open_cice_history(
    paths: ShugaPaths,
    *,
    variables: list[str] | None = None,
    extend_days: int = 0,
    chunks: dict | None = None,
) -> xr.Dataset:
    dt0 = pd.to_datetime(paths.run.start_date) - pd.Timedelta(days=int(extend_days))
    dtN = pd.to_datetime(paths.run.end_date) + pd.Timedelta(days=int(extend_days))
    return load_cice(
        run=paths.run,
        classify=paths.classify,
        metrics=paths.metrics,
        plotting=paths.plotting,
        observations=paths.observations,
        paths=paths,
        dt0_str=dt0.strftime("%Y-%m-%d"),
        dtN_str=dtN.strftime("%Y-%m-%d"),
        variables=variables,
        hemisphere=paths.run.hemisphere,
        chunks=chunks,
    )
