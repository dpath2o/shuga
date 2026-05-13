from __future__ import annotations
import re
import pandas           as pd
import xarray           as xr
import numpy            as np
from dataclasses        import replace
from pathlib            import Path
from typing             import Any
from shuga.core.logging import resolve_logger
from shuga.core.naming  import method_dirname, normalize_method
from shuga.core.paths   import ShugaPaths
from shuga.core.types   import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec
from shuga.io.iceh_loading import IceHistoryLoader

"""
Zarr loading utilities for CICE history, classified fast-ice masks, and
derived metrics.

Design notes
------------
- ``load_cice()`` reads dynamic fields from the grouped history store and
  merges static fields from the static store when needed.
- ``load_classified()`` normalises classified outputs to a single variable
  named ``FI_mask``.
- ``load_metrics()`` reads ``mets.zarr`` products and uses overlap-aware time
  slicing, which is important for rolling/windowed metrics.
- Most loaders can resolve context from either spec objects or explicit
  keyword overrides.
"""

_MONTH_RE          = re.compile(r"^\d{4}-\d{2}$")
_CANONICAL_METHODS = ("raw", "binary-days", "rolling-mean")
LOGGER             = resolve_logger(None, name="shuga.io.zarr_loading")

def _slice_time(ds: xr.Dataset, dt0: str | None, dtN: str | None) -> xr.Dataset:
    if "time" not in ds.coords or (dt0 is None and dtN is None):
        return ds
    start = pd.to_datetime(dt0) if dt0 is not None else None
    end   = pd.to_datetime(dtN) if dtN is not None else None
    return ds.sel(time=slice(start, end))

def _slice_time_overlap(ds: xr.Dataset, dt0_str: str | None, dtN_str: str | None) -> xr.Dataset:
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        return ds
    t0_avail = ds.time.values[0]
    tN_avail = ds.time.values[-1]
    t0_req   = np.datetime64(dt0_str) if dt0_str is not None else t0_avail
    tN_req   = np.datetime64(dtN_str) if dtN_str is not None else tN_avail
    if t0_req > tN_req:
        raise ValueError(f"Requested time range is reversed: {dt0_str} -> {dtN_str}")
    t0_use = max(t0_req, t0_avail)
    tN_use = min(tN_req, tN_avail)
    if t0_use > tN_use:
        LOGGER.warning("Requested time range %s -> %s has no overlap with metrics store %s -> %s", dt0_str, dtN_str, str(t0_avail), str(tN_avail))
        return ds.isel(time=slice(0, 0))
    return ds.sel(time=slice(t0_use, tN_use))

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
                         iceh_frequency : str | None = None, 
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
        iceh_frequency_eff = iceh_frequency or getattr(run, "iceh_frequency", "daily")
    else:
        sim_name_eff   = sim_name
        if sim_name_eff is None:
            raise ValueError("A simulation context is required. Pass sim_name='...' or run=RunSpec(...).")
        dt0_eff        = dt0_str
        dtN_eff        = dtN_str
        hemisphere_eff = hemisphere or _default_value(RunSpec, "hemisphere")
        project_eff    = project    or _default_value(RunSpec, "project")
        user_eff       = user       or _default_value(RunSpec, "user")
        iceh_frequency_eff = iceh_frequency or _default_value(RunSpec, "iceh_frequency")
    if dt0_eff is not None and dtN_eff is not None and pd.to_datetime(dtN_eff) < pd.to_datetime(dt0_eff):
        raise ValueError(f"dtN_str ({dtN_eff}) must be on or after dt0_str ({dt0_eff}).")
    run_eff = RunSpec(sim_name   = sim_name_eff,
                      start_date = dt0_eff or "1900-01-01",
                      end_date   = dtN_eff or "2100-12-31",
                      hemisphere = hemisphere_eff or _default_value(RunSpec, "hemisphere"),
                      project    = project_eff,
                      user       = user_eff,
                      iceh_frequency = iceh_frequency_eff)
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

def load_cice(run              : RunSpec | None = None,
              classify         : ClassificationSpec | None = None,
              metrics          : MetricsSpec | None = None,
              plotting         : PlottingSpec | None = None,
              observations     : ObservationSpec | None = None,
              paths            : ShugaPaths | None = None, *,
              sim_name         : str | None = None,
              dt0_str          : str | None = None,
              dtN_str          : str | None = None,
              variables                     = None,
              hemisphere       : str | None = None,
              project          : str | None = None,
              user             : str | None = None,
              iceh_frequency   : str | None = None,
              afim_output_root : str | Path | None = None,
              cice_store       : str | Path | None = None,
              static_store     : str | Path | None = None,
              chunks           : dict | None = None) -> xr.Dataset:
    """
    Load CICE history output from the grouped daily Zarr store, optionally
    merging in static variables from a separate static store.

    This is the primary loader for model history data. It resolves the run
    context from either a supplied ``RunSpec``/``ShugaPaths`` or from explicit
    keyword overrides such as ``sim_name``, ``dt0_str``, ``dtN_str``,
    ``hemisphere``, ``project``, and ``user``. Dynamic variables are opened
    from the grouped CICE history store, while static variables are merged from
    the static store when requested.

    Parameters
    ----------
    run : RunSpec | None, optional
        Run configuration describing the simulation, date range, hemisphere,
        and user/project context.
    classify, metrics, plotting, observations : spec objects, optional
        Optional companion spec objects used when resolving paths via
        ``_build_paths``.
    paths : ShugaPaths | None, optional
        Pre-built path bundle. When provided, its attributes may be used as
        defaults for missing run/classification context.
    sim_name : str | None, optional
        Simulation name override.
    dt0_str, dtN_str : str | None, optional
        Start and end date overrides in ``YYYY-MM-DD`` format.
    variables : str | list[str] | None, optional
        Variable name or list of variable names to load. If ``None``, all
        variables are requested from the dynamic store, which may be expensive
        for long periods.
    hemisphere : str | None, optional
        Hemisphere selection passed through to the hemisphere masking helper.
    project, user : str | None, optional
        Project and username overrides used during run/path resolution.
    afim_output_root : str | Path | None, optional
        Root AFIM output directory override.
    cice_store : str | Path | None, optional
        Explicit path to the grouped CICE history Zarr store.
    static_store : str | Path | None, optional
        Explicit path to the static Zarr store containing non-time-varying
        fields.
    chunks : dict | None, optional
        Dask chunk mapping passed to the grouped store opener. Defaults to
        ``{"time": 31}``.

    Returns
    -------
    xr.Dataset
        Dataset containing the requested CICE variables, with static variables
        merged in when relevant and the final result masked to the requested
        hemisphere.

    Raises
    ------
    ValueError
        If ``variables`` was provided but none of the requested variables are
        present after dynamic/static merging.

    Notes
    -----
    - Requested variables are split into dynamic and static groups before
      opening stores.
    - If only static variables are requested, the grouped monthly dynamic open
      is allowed to return empty.
    - A final subset is applied after static merging so the returned dataset
      contains only the variables actually requested.
    """
    if chunks is None:
        chunks = {"time": 31}
    variables_list = _maybe_listify_variables(variables)
    if variables_list is None:
        LOGGER.warning("load_cice() called with variables=None; loading all variables from iceh_daily.zarr "
                       "can be very memory-intensive for long date ranges.")
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run,
                                                                     sim_name   = sim_name,
                                                                     dt0_str    = dt0_str,
                                                                     dtN_str    = dtN_str,
                                                                     hemisphere = hemisphere,
                                                                     project    = project,
                                                                     user       = user)
    classify_eff = classify or (paths.classify if paths is not None else ClassificationSpec())
    paths_eff    = _build_paths(run              = run_eff,
                                classify         = classify_eff,
                                metrics          = metrics,
                                plotting         = plotting,
                                observations     = observations,
                                paths            = paths,
                                afim_output_root = afim_output_root,
                                cice_store       = cice_store,
                                static_store     = static_store)
    loader = IceHistoryLoader(paths_eff, logger=LOGGER)
    return loader.load(dt0_str      = dt0_eff,
                       dtN_str      = dtN_eff,
                       variables    = variables,
                       hemisphere   = hemisphere_eff,
                       cice_store   = cice_store,
                       static_store = static_store,
                       chunks       = chunks)

def load_classified(run                 : RunSpec | None = None,
                    classify            : ClassificationSpec | None = None,
                    metrics             : MetricsSpec | None = None,
                    plotting            : PlottingSpec | None = None,
                    observations        : ObservationSpec | None = None,
                    paths               : ShugaPaths | None = None, *,
                    classification      : str | None = None,
                    sim_name            : str | None = None,
                    dt0_str             : str | None = None,
                    dtN_str             : str | None = None,
                    variables                        = None,
                    hemisphere          : str | None = None,
                    project             : str | None = None,
                    user                : str | None = None,
                    grid_type           : str | None = None,
                    grid_type_map       : dict[str, str] | None = None,
                    ice_type            : str | None = None,
                    ispd_thresh         : float | None = None,
                    bin_window          : int | None = None,
                    bin_min_days        : int | None = None,
                    roll_window         : int | None = None,
                    afim_output_root    : str | Path | None = None,
                    classification_root : str | Path | None = None,
                    chunks              : dict | None = None,
                    return_resolved     : bool = False):
    """
    Load a classified fast-ice store and return a dataset containing a single
    ``FI_mask`` variable.

    This loader resolves the run and classification context, constructs the
    classification-store path, opens ``data.zarr``, and normalises the output
    so that the returned dataset always contains a variable named
    ``FI_mask``. If the source dataset contains exactly one data variable with
    a different name, that variable is renamed to ``FI_mask``.

    Parameters
    ----------
    run : RunSpec | None, optional
        Run configuration used as the primary source of simulation/date
        context.
    classify : ClassificationSpec | None, optional
        Classification configuration describing classification method, grid,
        thresholds, and window lengths.
    metrics, plotting, observations : spec objects, optional
        Optional companion specs used for path construction.
    paths : ShugaPaths | None, optional
        Pre-built path bundle whose ``classify`` section may provide defaults.
    classification : str | None, optional
        Classification method name override, for example a raw, binary-days,
        or rolling-mean classification label.
    sim_name : str | None, optional
        Simulation name override.
    dt0_str, dtN_str : str | None, optional
        Start and end date overrides in ``YYYY-MM-DD`` format.
    hemisphere : str | None, optional
        Hemisphere selection used during masking.
    project, user : str | None, optional
        Project and username overrides used during run/path resolution.
    grid_type : str | None, optional
        Grid-type override used when resolving the classification store path.
    grid_type_map : dict[str, str] | None, optional
        Optional mapping from simulation name to grid type. If provided, this
        takes precedence over ``grid_type`` for matching simulations.
    ice_type : str | None, optional
        Ice-type override used in classification resolution.
    ispd_thresh : float | None, optional
        Speed-threshold override for the classification context.
    bin_window, bin_min_days, roll_window : int | None, optional
        Classification window parameters used when resolving binary-days or
        rolling-mean stores.
    afim_output_root : str | Path | None, optional
        Root AFIM output directory override.
    classification_root : str | Path | None, optional
        Explicit classification-root override.
    chunks : dict | None, optional
        Dask chunk mapping passed to ``xr.open_zarr``.
    return_resolved : bool, optional
        If ``True``, also return the resolved path/context dictionary produced
        by ``_resolve_class_store_path``.

    Returns
    -------
    xr.Dataset
        Dataset containing a single data variable named ``FI_mask``, masked to
        the requested hemisphere and sliced to the requested time range.
    tuple[xr.Dataset, dict]
        Returned when ``return_resolved=True``. The second element contains the
        resolved classification metadata/path dictionary.

    Raises
    ------
    KeyError
        If the opened classified store does not contain ``FI_mask`` and does
        not contain exactly one alternative data variable that can be renamed.

    Notes
    -----
    - The function opens ``data.zarr`` under the resolved classification path.
    - Time slicing uses ``_slice_time`` rather than overlap-based slicing.
    - This loader is intentionally strict about returning a single canonical
      mask variable.
    """
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run,
                                                                     sim_name   = sim_name,
                                                                     dt0_str    = dt0_str,
                                                                     dtN_str    = dtN_str,
                                                                     hemisphere = hemisphere,
                                                                     project    = project,
                                                                     user       = user)
    variables_list = _maybe_listify_variables(variables)
    grid_type_eff  = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff   = _resolve_classify_context(classify or (paths.classify if paths is not None else None),
                                               classification = classification,
                                               grid_type      = grid_type_eff,
                                               ice_type       = ice_type,
                                               ispd_thresh    = ispd_thresh,
                                               bin_window     = bin_window,
                                               bin_min_days   = bin_min_days,
                                               roll_window    = roll_window)
    paths_eff      = _build_paths(run                 = run_eff,
                                  classify            = classify_eff,
                                  metrics             = metrics,
                                  plotting            = plotting,
                                  observations        = observations,
                                  paths               = paths,
                                  afim_output_root    = afim_output_root,
                                  classification_root = classification_root)
    resolved       = _resolve_class_store_path(paths_eff, classify_eff,
                                               classification = classification,
                                               grid_type      = grid_type_eff,
                                               store_name     = "data.zarr")
    LOGGER.info("Opening classified store for %s [%s/%s]: %s", run_eff.sim_name, resolved["grid_type"], resolved["classification"], resolved["path"])
    ds = xr.open_zarr(resolved["path"], consolidated=False, chunks=chunks)
    if "FI_mask" not in ds.data_vars and len(ds.data_vars) == 1:
        only = next(iter(ds.data_vars))
        ds = ds.rename({only: "FI_mask"})
    if "FI_mask" not in ds.data_vars:
        raise KeyError(f"Could not find FI_mask in {resolved['path']}. Data variables: {list(ds.data_vars)}")
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        if not keep:
            raise ValueError(f"None of the requested classified variables were found in {resolved['path']}: {variables_list}")
        ds = ds[keep]
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    ds = _slice_time(ds, dt0_eff, dtN_eff)
    if return_resolved:
        return ds, resolved
    return ds

def load_metrics(run                 : RunSpec | None         = None,
                 classify            : ClassificationSpec | None = None,
                 metrics             : MetricsSpec | None     = None,
                 plotting            : PlottingSpec | None    = None,
                 observations        : ObservationSpec | None = None,
                 paths               : ShugaPaths | None      = None, *,
                 classification      : str | None             = None,
                 sim_name            : str | None             = None,
                 dt0_str             : str | None             = None,
                 dtN_str             : str | None             = None,
                 variables                                    = None,
                 hemisphere          : str | None             = None,
                 project             : str | None             = None,
                 user                : str | None             = None,
                 grid_type           : str | None             = None,
                 grid_type_map       : dict[str, str] | None  = None,
                 ice_type            : str | None             = None,
                 ispd_thresh         : float | None           = None,
                 bin_window          : int | None             = None,
                 bin_min_days        : int | None             = None,
                 roll_window         : int | None             = None,
                 afim_output_root    : str | Path | None      = None,
                 classification_root : str | Path | None      = None,
                 chunks              : dict | None            = None,
                 return_resolved     : bool                   = False):
    """
    Load a classification metrics store from ``mets.zarr``.

    This function resolves run and classification context, opens the resolved
    metrics Zarr store, optionally subsets variables, applies a hemisphere
    mask, and slices the dataset to the requested time window using overlap-
    aware time slicing.

    Parameters
    ----------
    run : RunSpec | None, optional
        Run configuration used as the primary source of simulation/date
        context.
    classify : ClassificationSpec | None, optional
        Classification configuration describing classification method, grid,
        thresholds, and window lengths.
    metrics, plotting, observations : spec objects, optional
        Optional companion specs used during path construction.
    paths : ShugaPaths | None, optional
        Pre-built path bundle whose ``classify`` section may supply defaults.
    classification : str | None, optional
        Classification method override used to resolve the metrics store path.
    sim_name : str | None, optional
        Simulation name override.
    dt0_str, dtN_str : str | None, optional
        Start and end date overrides in ``YYYY-MM-DD`` format.
    variables : str | list[str] | None, optional
        Metric variable or variables to keep after opening the store. If
        omitted, all variables are returned.
    hemisphere : str | None, optional
        Hemisphere selection used during masking.
    project, user : str | None, optional
        Project and username overrides used during run/path resolution.
    grid_type : str | None, optional
        Grid-type override used to resolve the metrics store path.
    grid_type_map : dict[str, str] | None, optional
        Optional mapping from simulation name to grid type. If provided, it is
        checked before the direct ``grid_type`` argument.
    ice_type : str | None, optional
        Ice-type override used in classification resolution.
    ispd_thresh : float | None, optional
        Speed-threshold override used in classification resolution.
    bin_window, bin_min_days, roll_window : int | None, optional
        Classification window parameters used to resolve method-specific metric
        stores.
    afim_output_root : str | Path | None, optional
        Root AFIM output directory override.
    classification_root : str | Path | None, optional
        Explicit classification-root override.
    chunks : dict | None, optional
        Dask chunk mapping passed to ``xr.open_zarr``.
    return_resolved : bool, optional
        If ``True``, also return the resolved path/context dictionary produced
        by ``_resolve_class_store_path``.

    Returns
    -------
    xr.Dataset
        Metrics dataset, optionally variable-subset, hemisphere-masked, and
        overlap-sliced in time.
    tuple[xr.Dataset, dict]
        Returned when ``return_resolved=True``. The second element contains the
        resolved classification metadata/path dictionary.

    Notes
    -----
    - The function opens ``mets.zarr`` under the resolved classification path.
    - Variable subsetting is applied before hemisphere masking and time slicing.
    - Time slicing uses ``_slice_time_overlap()``, which is useful for metrics
      generated from rolling or windowed methods whose effective support may
      extend beyond the nominal target range.
    - Extensive logging is emitted describing the resolved context, opened
      store, retained variables, and dataset sizes before and after slicing.
    """
    variables_list = _maybe_listify_variables(variables)
    run_eff, dt0_eff, dtN_eff, hemisphere_eff = _resolve_run_context(run,
                                                                     sim_name   = sim_name,
                                                                     dt0_str    = dt0_str,
                                                                     dtN_str    = dtN_str,
                                                                     hemisphere = hemisphere,
                                                                     project    = project,
                                                                     user       = user)
    LOGGER.info("load_metrics resolved run context: sim=%s dt0=%s dtN=%s hemisphere=%s",run_eff.sim_name, dt0_eff, dtN_eff, hemisphere_eff)
    grid_type_eff = (grid_type_map or {}).get(run_eff.sim_name, grid_type)
    classify_eff  = _resolve_classify_context(classify or (paths.classify if paths is not None else None),
                                              classification = classification,
                                              grid_type      = grid_type_eff,
                                              ice_type       = ice_type,
                                              ispd_thresh    = ispd_thresh,
                                              bin_window     = bin_window,
                                              bin_min_days   = bin_min_days,
                                              roll_window    = roll_window)

    paths_eff     = _build_paths(run                 = run_eff,
                                 classify            = classify_eff,
                                 metrics             = metrics,
                                 plotting            = plotting,
                                 observations        = observations,
                                 paths               = paths,
                                 afim_output_root    = afim_output_root,
                                 classification_root = classification_root)

    resolved = _resolve_class_store_path(paths_eff, classify_eff,
                                         classification = classification,
                                         grid_type      = grid_type_eff,
                                         store_name     = "mets.zarr")

    LOGGER.info("Opening metrics store for %s [%s/%s]: %s", run_eff.sim_name, resolved["grid_type"], resolved["classification"], resolved["path"])
    ds = xr.open_zarr(resolved["path"], consolidated=False, chunks=chunks)
    LOGGER.info("raw metrics ds: sizes=%s time=%s -> %s", dict(ds.sizes), ds.time.values[0] if ds.sizes.get("time", 0) else None, ds.time.values[-1] if ds.sizes.get("time", 0) else None)
    if variables_list is not None:
        keep = [v for v in variables_list if v in ds.data_vars or v in ds.coords]
        LOGGER.info("keeping variables: %s", keep)
        ds = ds[keep]
        LOGGER.info("after variable subset: sizes=%s", dict(ds.sizes))
    ds = _apply_hemisphere_mask(ds, hemisphere_eff)
    LOGGER.info("after hemisphere mask: sizes=%s", dict(ds.sizes))
    ds = _slice_time_overlap(ds, dt0_eff, dtN_eff)
    LOGGER.info("after time slice: sizes=%s time=%s -> %s", dict(ds.sizes), ds.time.values[0] if ds.sizes.get("time", 0) else None, ds.time.values[-1] if ds.sizes.get("time", 0) else None)
    if return_resolved:
        return ds, resolved
    return ds

def open_cice_history(paths: ShugaPaths, *,
                      variables   : list[str] | None = None,
                      extend_days : int              = 0,
                      chunks      : dict | None      = None) -> xr.Dataset:
    """
    Convenience wrapper around :func:`load_cice` using dates and configuration
    from a ``ShugaPaths`` object.

    The loader expands the run date range by ``extend_days`` on both ends,
    then delegates to ``load_cice`` using the run, classification, metrics,
    plotting, and observation specs stored on ``paths``.

    Parameters
    ----------
    paths : ShugaPaths
        Path/configuration bundle containing the run start/end dates and all
        associated spec objects.
    variables : list[str] | None, optional
        Variables to load from the CICE history store. Passed directly to
        ``load_cice``.
    extend_days : int, optional
        Number of days to extend the requested time range both before the run
        start date and after the run end date. Defaults to ``0``.
    chunks : dict | None, optional
        Dask chunk mapping passed through to ``load_cice``.

    Returns
    -------
    xr.Dataset
        CICE history dataset loaded over the expanded date range.

    Notes
    -----
    - ``dt0`` is computed as ``paths.run.start_date - extend_days``.
    - ``dtN`` is computed as ``paths.run.end_date + extend_days``.
    - The hemisphere is taken directly from ``paths.run.hemisphere``.
    - This helper is useful when classification or diagnostic workflows need a
      small temporal halo around the nominal analysis interval.
    """
    dt0 = pd.to_datetime(paths.run.start_date) - pd.Timedelta(days=int(extend_days))
    dtN = pd.to_datetime(paths.run.end_date) + pd.Timedelta(days=int(extend_days))
    return load_cice(run          = paths.run,
                     classify     = paths.classify,
                     metrics      = paths.metrics,
                     plotting     = paths.plotting,
                     observations = paths.observations,
                     paths        = paths,
                     dt0_str      = dt0.strftime("%Y-%m-%d"),
                     dtN_str      = dtN.strftime("%Y-%m-%d"),
                     variables    = variables,
                     hemisphere   = paths.run.hemisphere,
                     chunks       = chunks)
