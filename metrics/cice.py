from __future__            import annotations
import shutil
from pathlib               import Path
from datetime              import datetime
import numpy               as np
import pandas              as pd
import xarray              as xr
from shuga.core.logging    import build_file_logger
from shuga.core.naming     import normalize_method
from shuga.core.paths      import ShugaPaths
from shuga.core.regions    import ANTARCTIC_8_REGIONS
from shuga.core.types      import ClassificationSpec, MetricsSpec, RunSpec
from shuga.io.zarr_loading import load_cice, load_classified
from shuga.io.zarr_writing import sanitise_for_zarr_write
from shuga.metrics.registry import (CORE_FI,
                                    CORE_SI,
                                    REGIONAL,
                                    SPATIAL,
                                    SUMMARY,
                                    STRESS,
                                    METRIC_GROUPS,
                                    FIPSI_NAMES,
                                    FIA_SKILL_NAMES,
                                    FIT_SKILL_NAMES,
                                    FIA_SEASONAL_NAMES,
                                    FIT_SEASONAL_NAMES,
                                    SIA_SEASONAL_NAMES,
                                    SIT_SEASONAL_NAMES,
                                    expand_metric_names)
from shuga.metrics.skill import skill_stats
from shuga.metrics.temporal import (month_window_bounds,
                                    linear_rate_per_day,
                                    seasonal_rate_record,
                                    compute_extrema_table)
from shuga.metrics.regional import (detect_lonlat,
                                    ensure_2d_static,
                                    lon_to_180,
                                    region_mask,
                                    spatial_dims)
from shuga.metrics.calculations import (compute_area_series,
                                        compute_volume_series,
                                        compute_thickness_series,
                                        compute_strength_series,
                                        compute_persistence_mask,
                                        compute_temporal_mean,
                                        convert_thickness_tendency_to_m_per_day,
                                        compute_volume_rate,
                                        compute_area_rate,
                                        compute_spatial_rate_year,
                                        compute_region_series,
                                        compute_area_weighted_stress)
from shuga.metrics.io import (output_chunk_map,
                              open_existing_metrics,
                              backup_legacy_store)
from shuga.metrics.dispatch import (MetricDispatchContext,
                                    MetricDispatcher,
                                    PRIMARY_METRIC_NAMES,
                                    PRIMARY_METRIC_SET,
                                    needs_fast_ice_mask)
from shuga.metrics.stress import (compute_stress_dataset,
                                  stress_requested)
from shuga.metrics.secondary import (attach_common_metrics_attrs,
                                     compute_fipsi_dataset,
                                     compute_obs_skill_dataset,
                                     compute_seasonal_summary_dataset)

"""
Incremental CICE metrics builder for fast-ice and sea-ice diagnostics.

The class computes time-series, spatial, regional, seasonal-summary,
persistence, skill, and stress metrics from classified masks and CICE history
fields, and writes them to method-specific metrics Zarr stores.
"""

def _publish_matching(out: xr.Dataset, dsi: xr.Dataset, requested: set[str]) -> xr.Dataset:
    for name in dsi.data_vars:
        if name in requested:
            out[name] = dsi[name]
    return out

def publish_da(name: str, dispatcher) -> xr.DataArray | None:
    da = dispatcher.get(name)
    if da is not None:
        out[name] = da
    return da

class CICEMetrics:
    """
    Incremental metrics builder for CICE fast-ice and sea-ice diagnostics.

    This class loads classified masks, CICE history fields, and existing metric
    stores as needed, then computes and writes derived diagnostics to a
    method-specific metrics Zarr store. It supports both time-series and
    spatial diagnostics for fast ice (FI) and sea ice (SI), including regional
    products, seasonal summary statistics, skill metrics, persistence metrics,
    and stress diagnostics.

    Metrics are organised through registry-style name groups such as
    ``fi_core``, ``si_core``, ``regional``, ``spatial``, ``summary``, and
    ``stress``. These groups expand to concrete metric names via
    ``METRIC_GROUPS`` and can be requested in bulk when building or updating a
    metrics store.

    Parameters
    ----------
    run : RunSpec
        Run-level configuration describing the simulation, analysis period, and
        hemisphere.
    classify : ClassificationSpec
        Classification configuration describing the mask source, ice type,
        grid type, and classification method settings used by downstream
        metrics.
    metrics : MetricsSpec | None, optional
        Metrics configuration containing scaling factors, requested metric
        groups, and related output settings. If omitted, a default
        ``MetricsSpec()`` is created.
    paths : ShugaPaths | None, optional
        Path bundle used to locate input history, classification, and metrics
        stores. If omitted, a new ``ShugaPaths`` instance is built from
        ``run`` and ``classify``.
    chunks : dict | None, optional
        Dask chunking used when opening model and metrics datasets. Defaults to
        ``{"time": 31}``.
    logger : logging.Logger | None, optional
        Logger used for status, write, and cache messages. If omitted, a file
        logger is created using the metrics log path.

    Attributes
    ----------
    run : RunSpec
        Active run configuration.
    classify : ClassificationSpec
        Active classification configuration.
    metrics : MetricsSpec
        Active metrics configuration.
    paths : ShugaPaths
        Resolved path bundle for input and output stores.
    chunks : dict
        Chunking policy used when opening datasets.
    logger : logging.Logger
        Logger for metrics progress and diagnostics.
    region_defs : object
        Regional definitions used for Antarctic regional metrics.
    _cice_cache : xr.Dataset | None
        Cache for loaded CICE history data.
    _classified_cache : dict[str, xr.Dataset]
        Per-method cache for classified mask datasets.
    _metrics_cache : dict[str, xr.Dataset]
        Per-method cache for opened or recently written metrics datasets.

    Notes
    -----
    - The class is incremental in the sense that it can compute only missing
      metrics and append them to an existing store.
    - Metric groups provide a stable public interface while allowing the
      underlying dataset to contain many individual variables.
    - Output stores are method-specific and typically written as
      ``mets.zarr`` products.
    """
    # Backwards-compatible registry aliases. The canonical definitions live in
    # shuga.metrics.registry.
    CORE_FI            = CORE_FI
    CORE_SI            = CORE_SI
    REGIONAL           = REGIONAL
    SPATIAL            = SPATIAL
    SUMMARY            = SUMMARY
    STRESS             = STRESS
    METRIC_GROUPS      = METRIC_GROUPS
    FIPSI_NAMES        = FIPSI_NAMES
    FIA_SKILL_NAMES    = FIA_SKILL_NAMES
    FIT_SKILL_NAMES    = FIT_SKILL_NAMES
    FIA_SEASONAL_NAMES = FIA_SEASONAL_NAMES
    FIT_SEASONAL_NAMES = FIT_SEASONAL_NAMES
    SIA_SEASONAL_NAMES = SIA_SEASONAL_NAMES
    SIT_SEASONAL_NAMES = SIT_SEASONAL_NAMES

    #----------------------------------------------------------------------------------
    # class initialisation
    #----------------------------------------------------------------------------------
    def __init__(self, run: RunSpec, classify: ClassificationSpec,
                 metrics : MetricsSpec | None = None,
                 paths   : ShugaPaths | None  = None, *,
                 chunks  : dict | None        = None,
                 logger                       = None) -> None:
        """
        Initialise a metrics builder for a single simulation and classification
        context.

        Parameters
        ----------
        run : RunSpec
            Run-level configuration describing the simulation, requested date
            range, and hemisphere.
        classify : ClassificationSpec
            Classification settings used to locate classified-mask inputs and to
            annotate output metrics with ice type, grid type, and method context.
        metrics : MetricsSpec | None, optional
            Metrics configuration controlling scale factors, requested metric
            collections, and related processing defaults. If omitted, a default
            ``MetricsSpec()`` instance is used.
        paths : ShugaPaths | None, optional
            Path bundle for locating CICE history, classification stores, metrics
            stores, and log files. If omitted, one is constructed from ``run`` and
            ``classify``.
        chunks : dict | None, optional
            Chunking to use when opening datasets. Defaults to ``{"time": 31}``.
        logger : logging.Logger | None, optional
            Logger for status and diagnostic messages. If not supplied, a file
            logger is created automatically.

        Notes
        -----
        - Antarctic regional definitions are initialised from
          ``ANTARCTIC_8_REGIONS``.
        - Three internal caches are created:
          ``_cice_cache`` for model history data,
          ``_classified_cache`` for per-method classification datasets, and
          ``_metrics_cache`` for per-method metrics datasets.
        - If no explicit metrics specification is provided, the class uses a
          default ``MetricsSpec()`` so scaling and output settings remain
          available.
        """
        self.run                                      = run
        self.classify                                 = classify
        self.metrics                                  = metrics or MetricsSpec()
        self.paths                                    = paths or ShugaPaths(run=run, classify=classify)
        self.chunks                                   = chunks or {"time": 31}
        self.logger                                   = logger or build_file_logger("shuga.metrics", self.paths.metrics_log_path())
        self.region_defs                              = ANTARCTIC_8_REGIONS
        self._cice_cache: xr.Dataset | None           = None
        self._classified_cache: dict[str, xr.Dataset] = {}
        self._metrics_cache: dict[str, xr.Dataset]    = {}

    #----------------------------------------------------------------------------------
    # property and static methods
    #----------------------------------------------------------------------------------
    @property
    def mask_var_name(self) -> str:
        return f"{self.classify.ice_type}_mask"

    # Backwards-compatible pure-helper aliases. The canonical implementations
    # live in shuga.metrics.skill and shuga.metrics.temporal.
    _skill_stats          = staticmethod(skill_stats)
    _month_window_bounds  = staticmethod(month_window_bounds)
    _linear_rate_per_day  = staticmethod(linear_rate_per_day)
    _seasonal_rate_record = staticmethod(seasonal_rate_record)
    compute_extrema_table = staticmethod(compute_extrema_table)

    # Backwards-compatible regional/spatial helper aliases.
    _ensure_2d_static = staticmethod(ensure_2d_static)
    _lon_to_180       = staticmethod(lon_to_180)
    _detect_lonlat    = staticmethod(detect_lonlat)
    _spatial_dims     = staticmethod(spatial_dims)
    def _region_mask(self, template: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        return region_mask(template    = template,
                           lon         = lon,
                           lat         = lat,
                           region_defs = self.region_defs)

    # Backwards-compatible metric-calculation aliases. The canonical
    # implementations live in shuga.metrics.calculations.
    compute_area_series                      = staticmethod(compute_area_series)
    compute_volume_series                    = staticmethod(compute_volume_series)
    compute_thickness_series                 = staticmethod(compute_thickness_series)
    compute_strength_series                  = staticmethod(compute_strength_series)
    compute_persistence_mask                 = staticmethod(compute_persistence_mask)
    compute_temporal_mean                    = staticmethod(compute_temporal_mean)
    _convert_thickness_tendency_to_m_per_day = staticmethod(convert_thickness_tendency_to_m_per_day)
    compute_volume_rate                      = staticmethod(compute_volume_rate)
    compute_area_rate                        = staticmethod(compute_area_rate)
    compute_spatial_rate_year                = staticmethod(compute_spatial_rate_year)
    compute_region_series                    = staticmethod(compute_region_series)
    compute_area_weighted_stress             = staticmethod(compute_area_weighted_stress)

    #----------------------------------------------------------------------------------
    # helpers
    #----------------------------------------------------------------------------------
    def _require_pygmt(self):
        try:
            import pygmt
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyGMT is required for plotting methods.") from exc
        return pygmt

    def _plotter(self):
        from shuga.plotting import CICEPlotter
        return CICEPlotter(run      = self.run,
                           classify = self.classify,
                           metrics  = self.metrics,
                           paths    = self.paths,
                           chunks   = self.chunks,
                           logger   = self.logger)

    def plot_fip(self, *args, **kwargs):
        return self._plotter().plot_fip(*args, **kwargs)

    def plot_timeseries(self, *args, **kwargs):
        return self._plotter().plot_timeseries(*args, **kwargs)

    def plot_var_split_hemisphere(self, *args, **kwargs):
        return self._plotter().plot_var_split_hemisphere(*args, **kwargs)

    def plot_var_by_region(self, *args, **kwargs):
        return self._plotter().plot_var_by_region(*args, **kwargs)

    def plot_triptych(self, *args, **kwargs):
        return self._plotter().plot_triptych(*args, **kwargs)

    def _output_chunk_map(self, ds: xr.Dataset) -> dict[str, int]:
        return output_chunk_map(ds)

    def _si_mask(self, aice: xr.DataArray) -> xr.DataArray:
        thresh = float(getattr(self.classify, "aice_thresh", 0.15))
        return xr.where(aice >= thresh, True, False)

    def _expand_metric_names(self, metric_names=None, metric_groups=None) -> list[str]:
        return expand_metric_names(metric_names=metric_names, metric_groups=metric_groups)

    def _open_existing_metrics(self, method: str) -> xr.Dataset | None:
        return open_existing_metrics(paths  = self.paths,
                                     cache  = self._metrics_cache,
                                     method = method)

    def _backup_legacy_store(self, store: Path) -> Path:
        return backup_legacy_store(store, logger=self.logger)

    def _get_cice(self) -> xr.Dataset:
        if self._cice_cache is None:
            requested = ["aice", "hi", "strength", "dvidtt", "dvidtd", "daidtt", "daidtd",
                         "KuxE", "KuxN", "KuyE", "KuyN", "earea", "narea", "uarea",
                         "tarea", "TLON", "TLAT", "ULON", "ULAT"]
            self.logger.info("Resolved CICE store: %s", self.paths.resolve_cice_store())
            static_store = self.paths.resolve_static_store()
            if static_store is not None:
                self.logger.info("Resolved static store: %s", static_store)
            self._cice_cache = load_cice(run        = self.run,
                                         classify   = self.classify,
                                         metrics    = self.metrics,
                                         paths      = self.paths,
                                         variables  = requested,
                                         hemisphere = self.run.hemisphere,
                                         chunks     = self.chunks)
        return self._cice_cache

    def _get_classified(self, method: str) -> xr.Dataset:
        norm = normalize_method(method)
        if norm not in self._classified_cache:
            self._classified_cache[norm] = load_classified(run            = self.run,
                                                           classify       = self.classify,
                                                           metrics        = self.metrics,
                                                           paths          = self.paths,
                                                           classification = norm,
                                                           dt0_str        = self.run.start_date,
                                                           dtN_str        = self.run.end_date,
                                                           variables      = "FI_mask",
                                                           hemisphere     = self.run.hemisphere,
                                                           chunks         = self.chunks)
        return self._classified_cache[norm]

    def _obs_skill_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.metrics.obs_metrics_store:
            return xr.Dataset()
        store = Path(self.metrics.obs_metrics_store).expanduser()
        if not store.exists():
            self.logger.warning("Observation metrics store does not exist: %s", store)
            return xr.Dataset()
        obs = xr.open_zarr(store, consolidated=False)
        out = {}
        if self.metrics.obs_fia_var in obs and "FIA" in ds:
            mod, ref = xr.align(ds["FIA"], obs[self.metrics.obs_fia_var], join="inner")
            stats    = self._skill_stats(mod.values, ref.values)
            for k, v in stats.items():
                out[f"FIA_{k}"] = xr.DataArray(v)
        if self.metrics.obs_fit_var in obs and "FIT" in ds:
            mod, ref = xr.align(ds["FIT"], obs[self.metrics.obs_fit_var], join="inner")
            stats    = self._skill_stats(mod.values, ref.values)
            for k, v in stats.items():
                out[f"FIT_{k}"] = xr.DataArray(v)
        return xr.Dataset(out)

    def _match_mask_to_field(self, mask: xr.DataArray | None, field: xr.DataArray) -> xr.DataArray | None:
        if mask is None:
            return None
        try:
            return mask.broadcast_like(field)
        except Exception:
            return None

    #----------------------------------------------------------------------------------
    # the following functions are the backbone of this module
    # essentially, _compute_requested_metrics does all the heavy lifting for batch
    # processing of metric computations
    #---------------------------------------------------------------------------------
    def _compute_requested_metrics(self, method: str, requested: set[str]) -> xr.Dataset:
        """
        Compute requested metrics for a classification method.

        Primary metric dispatch is delegated to MetricDispatcher. This method
        remains responsible for:
        - loading CICE/classification context;
        - constructing region masks;
        - running seasonal summaries;
        - running FIPSI diagnostics;
        - running obs skill diagnostics;
        - running stress diagnostics;
        - attaching common metadata.
        """
        ds            = self._get_cice()
        aice          = ds["aice"]
        hi            = ds["hi"]
        area          = self._ensure_2d_static(ds["tarea"])
        lon, lat      = self._detect_lonlat(ds)
        regional_mask = self._region_mask(area, lon, lat)
        need_fi       = needs_fast_ice_mask(requested, self.FIPSI_NAMES)
        ds_mask       = self._get_classified(method) if need_fi else None
        fi_mask       = ds_mask["FI_mask"].astype(bool) if ds_mask is not None else None
        if fi_mask is not None:
            aice, hi, fi_mask = xr.align(aice, hi, fi_mask, join="inner")
        si_mask = self._si_mask(aice)
        ctx     = MetricDispatchContext(ds           = ds,
                                        aice         = aice,
                                        hi           = hi,
                                        area         = area,
                                        region_mask  = regional_mask,
                                        fi_mask      = fi_mask,
                                        si_mask      = si_mask,
                                        area_scale   = self.metrics.area_scale,
                                        volume_scale = self.metrics.volume_scale)
        dispatcher = MetricDispatcher(context=ctx, calculator=self)
        out        = xr.Dataset()
        # def publish(name: str) -> xr.DataArray | None:
        #     da = dispatcher.get(name)
        #     if da is not None:
        #         out[name] = da
        #     return da
        # ------------------------------------------------------------------
        # Primary metrics
        # ------------------------------------------------------------------
        for name in PRIMARY_METRIC_NAMES:
            if name in requested:
                publish_da(name)
                # publish(name)
        # ------------------------------------------------------------------
        # Seasonal scalar summaries derived from primary 1-D series.
        # ------------------------------------------------------------------
        seasonal_requests = {"FIA": self.FIA_SEASONAL_NAMES,
                             "FIT": self.FIT_SEASONAL_NAMES,
                             "SIA": self.SIA_SEASONAL_NAMES,
                             "SIT": self.SIT_SEASONAL_NAMES}
        seasonal_ds = compute_seasonal_summary_dataset(requested                = requested,
                                                       dispatcher               = dispatcher,
                                                       output                   = out,
                                                       seasonal_requests        = seasonal_requests,
                                                       compute_seasonal_summary = self.compute_seasonal_summary)
        out = xr.merge([out, seasonal_ds], compat="override")
        # ------------------------------------------------------------------
        # Persistence-stability diagnostics.
        # ------------------------------------------------------------------
        fipsi_ds = compute_fipsi_dataset(requested=requested,
                                         fipsi_names=self.FIPSI_NAMES,
                                         fi_mask=fi_mask,
                                         area=area,
                                         persistence_stability_index=self.persistence_stability_index)
        out = xr.merge([out, fipsi_ds], compat="override")
        # ------------------------------------------------------------------
        # Observation skill diagnostics.
        # ------------------------------------------------------------------
        skill_ds = compute_obs_skill_dataset(requested         = requested,
                                             dispatcher        = dispatcher,
                                             output            = out,
                                             fia_skill_names   = self.FIA_SKILL_NAMES,
                                             fit_skill_names   = self.FIT_SKILL_NAMES,
                                             obs_skill_dataset = self._obs_skill_dataset)
        out = xr.merge([out, skill_ds], compat="override")
        # ------------------------------------------------------------------
        # Stress diagnostics.
        # ------------------------------------------------------------------
        if stress_requested(requested, "FI") and fi_mask is not None:
            stress_ds = compute_stress_dataset(ds         = ds,
                                               area       = area,
                                               requested  = requested,
                                               prefix     = "FI",
                                               mask       = fi_mask,
                                               calculator = self.compute_area_weighted_stress)
            out = xr.merge([out, stress_ds], compat="override")
        if stress_requested(requested, "SI"):
            stress_ds = compute_stress_dataset(ds         =ds,
                                               area       = area,
                                               requested  = requested,
                                               prefix     = "SI",
                                               mask       = si_mask,
                                               calculator = self.compute_area_weighted_stress)
            out = xr.merge([out, stress_ds], compat="override")
        # ------------------------------------------------------------------
        # Common output metadata.
        # ------------------------------------------------------------------
        out = attach_common_metrics_attrs(out,
                                          sim_name   = self.run.sim_name,
                                          start_date = self.run.start_date,
                                          end_date   = self.run.end_date,
                                          hemisphere = self.run.hemisphere,
                                          ice_type   = self.classify.ice_type,
                                          grid_type  = self.classify.grid_type,
                                          method     = method)
        missing = sorted(name for name in requested if name not in out.data_vars)
        if missing:
            self.logger.info("Requested metrics not produced for %s/%s, likely because required inputs are absent: %s",
                             self.run.sim_name, method, missing)
        return out

    def _strip_aux_coords(self, da: xr.DataArray) -> xr.DataArray:
        keep_non_dim = {"time", "region"}
        drop = [c for c in da.coords if (c not in da.dims and c not in keep_non_dim)]
        if drop:
            da = da.reset_coords(drop, drop=True)
        return da

    def _assert_same_indexes(self, existing: xr.Dataset, ds_new: xr.Dataset, dims: tuple[str, ...] = ("time", "region")) -> None:
        for dim in dims:
            if dim in existing.coords and dim in ds_new.coords:
                idx_existing = existing.indexes.get(dim, None)
                idx_new      = ds_new.indexes.get(dim, None)
                if idx_existing is None or idx_new is None:
                    continue
                if not idx_existing.equals(idx_new):
                    extra = ""
                    if dim == "time":
                        try:
                            extra = (f" existing[{existing.sizes.get(dim)}]"
                                     f"={str(idx_existing[0])}..{str(idx_existing[-1])};"
                                     f" new[{ds_new.sizes.get(dim)}]"
                                     f"={str(idx_new[0])}..{str(idx_new[-1])}")
                        except Exception:
                            extra = (f" existing={existing.sizes.get(dim)}"
                                     f" new={ds_new.sizes.get(dim)}")
                    raise ValueError(f"Cannot combine metrics because coordinate '{dim}' differs "
                                     f"between existing and new datasets.{extra} "
                                     f"This usually means the existing mets.zarr was produced by an "
                                     f"older merge path and should be rebuilt.")

    def _encoding_from_dataset(self, ds: xr.Dataset) -> dict[str, dict]:
        encoding: dict[str, dict] = {}
        for name, var in ds.variables.items():
            chunks = getattr(var.data, "chunks", None)
            if chunks is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in chunks)}
        return encoding

    def _prepare_output_dataset(self, ds_out: xr.Dataset) -> xr.Dataset:
        cleaned = {name: self._strip_aux_coords(ds_out[name]) for name in ds_out.data_vars}
        coords  = {}
        if "time" in ds_out.coords:
            coords["time"] = ds_out["time"]
        if "region" in ds_out.coords:
            coords["region"] = ds_out["region"]
        ds_out    = xr.Dataset(cleaned, coords=coords, attrs=ds_out.attrs)
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking metrics output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)
        ds_out = sanitise_for_zarr_write(ds_out)
        #ds_out = _sanitize_for_zarr_write(ds_out)
        return ds_out

    def compute_seasonal_summary(self, da: xr.DataArray, prefix: str) -> dict[str, xr.DataArray]:
        """
        Compute simple annual extrema summary statistics for a time series.

        For each calendar year present in the input series, this method extracts
        the annual maximum, annual minimum, day-of-year of the maximum, and
        day-of-year of the minimum. It then returns the across-year mean and
        standard deviation for each of those quantities.

        Parameters
        ----------
        da : xr.DataArray
            Input time series with a ``time`` dimension.
        prefix : str
            Prefix used to construct the returned metric names.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary containing up to eight scalar metrics:
            ``{prefix}_max_mean``, ``{prefix}_max_std``,
            ``{prefix}_min_mean``, ``{prefix}_min_std``,
            ``{prefix}_doy_max_mean``, ``{prefix}_doy_max_std``,
            ``{prefix}_doy_min_mean``, ``{prefix}_doy_min_std``.

        Notes
        -----
        - If the input has no ``time`` dimension, an empty dictionary is returned.
        - Missing values are dropped before processing.
        - The standard deviation uses ``ddof=0``.
        """
        if "time" not in da.dims:
            return {}
        series = da.to_series().dropna()
        if series.empty:
            return {}
        rows = []
        for year, grp in series.groupby(series.index.year):
            if grp.empty:
                continue
            rows.append({"year"    : year,
                         "max"     : float(grp.max()),
                         "min"     : float(grp.min()),
                         "doy_max" : float(pd.Timestamp(grp.idxmax()).dayofyear),
                         "doy_min" : float(pd.Timestamp(grp.idxmin()).dayofyear)})
        if not rows:
            return {}
        df  = pd.DataFrame(rows)
        out = {}
        for col in ("max", "min", "doy_max", "doy_min"):
            vals                        = df[col].to_numpy(dtype=float)
            out[f"{prefix}_{col}_mean"] = xr.DataArray(np.nanmean(vals))
            out[f"{prefix}_{col}_std"]  = xr.DataArray(np.nanstd(vals, ddof=0))
        return out

    def persistence_stability_index(self, mask: xr.DataArray, area: xr.DataArray,
                                    persistence_threshold : float = 0.8,
                                    winter_months         : tuple[int, ...] = (5, 6, 7, 8, 9, 10)) -> dict[str, xr.DataArray]:
        """
        Compute the winter persistence stability index and related area diagnostics.

        This diagnostic evaluates how much of the winter fast-ice domain is
        persistent. Winter persistence is defined as the fraction of selected
        winter time steps for which a cell is classified as ice-covered.

        Parameters
        ----------
        mask : xr.DataArray
            Boolean classification mask with a ``time`` dimension.
        area : xr.DataArray
            Grid-cell area field in square metres.
        persistence_threshold : float, optional
            Threshold used to define persistent winter ice. Cells with persistence
            greater than or equal to this value contribute to the persistent area.
        winter_months : tuple[int, ...], optional
            Calendar months used to define winter. Defaults to May through October.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary containing:
            - ``FIPSI`` : persistent winter area divided by ever-winter area
            - ``persistent_winter_area`` : area with persistence above threshold,
              in ``10^3 km^2``
            - ``ever_winter_area`` : area with any winter occurrence, in
              ``10^3 km^2``

        Notes
        -----
        - If no winter time steps are present, all three outputs are returned as
          ``NaN`` scalars.
        - Persistence is computed as the mean of the boolean mask over winter time.
        - Area outputs are scaled using ``self.metrics.area_scale``.
        """
        winter = mask.sel(time=mask.time.dt.month.isin(winter_months))
        if winter.sizes.get("time", 0) == 0:
            return {"FIPSI"                  : xr.DataArray(np.nan),
                    "persistent_winter_area" : xr.DataArray(np.nan),
                    "ever_winter_area"       : xr.DataArray(np.nan)}
        persistence     = winter.astype("float32").mean("time")
        persistent_area = xr.where(persistence >= persistence_threshold, area, 0.0).sum()
        ever_area       = xr.where(winter.any("time"), area, 0.0).sum()
        ratio           = xr.where(ever_area > 0, persistent_area / ever_area, np.nan)
        return {"FIPSI"                  : ratio,
                "persistent_winter_area" : persistent_area / self.metrics.area_scale,
                "ever_winter_area"       : ever_area / self.metrics.area_scale}

    def compute_metrics(self, method: str, *,
                        overwrite                 : bool                       = False,
                        metric_names              : str | Iterable[str] | None = None,
                        metric_groups             : str | Iterable[str] | None = None,
                        update_missing_only       : bool                       = True,
                        rebuild_on_index_mismatch : bool                       = False) -> str:
        """
        Compute, merge, and write requested metrics for a classification method.

        This is the main metrics orchestration method. It resolves the requested
        metric names, determines which metrics still need to be computed, builds
        the new metrics dataset, and writes the result to the method-specific
        metrics store.

        Depending on the state of an existing store and the chosen update policy,
        the method may:
        - create a new store,
        - append only missing variables,
        - merge new and existing variables into a rebuilt store,
        - back up and rebuild a legacy store if index incompatibilities are found.

        Parameters
        ----------
        method : str
            Classification method whose metrics store should be updated. This is
            normalised via ``normalize_method()`` before use.
        overwrite : bool, optional
            If ``True``, discard any existing metrics store and rebuild it from the
            requested metrics.
        metric_names : str | Iterable[str] | None, optional
            Explicit metric names to compute.
        metric_groups : str | Iterable[str] | None, optional
            Metric-group aliases to expand into concrete metric names.
        update_missing_only : bool, optional
            If ``True`` and a store already exists, compute only metrics that are
            not already present.
        rebuild_on_index_mismatch : bool, optional
            If ``True``, back up and rebuild the metrics store when existing and
            newly computed metrics do not share compatible ``time`` or ``region``
            indexes.

        Returns
        -------
        str
            Path to the final metrics store.

        Raises
        ------
        ValueError
            If no metrics were requested.
        ValueError
            If appending would introduce overlapping variable names into an
            existing store.
        ValueError
            If index compatibility checks fail and
            ``rebuild_on_index_mismatch=False``.

        Notes
        -----
        - New metrics are computed via ``_compute_requested_metrics()``.
        - Output metadata include simulation name, dates, hemisphere, ice type,
          grid type, and classification method.
        - Writes are staged through a temporary ``.tmp`` store before replacement.
        - Zarr output is written with ``consolidated=False`` and
          ``zarr_format=2``.
        - Successful writes refresh the in-memory metrics cache for the requested
          method.
        """
        norm      = normalize_method(method)
        requested = set(self._expand_metric_names(metric_names=metric_names, metric_groups=metric_groups))
        self.logger.info("Resolved class store for %s: %s", norm, self.paths.classification_store(norm))
        self.logger.info("Resolved metrics store for %s: %s", norm, self.paths.metrics_store(norm))
        if not requested:
            raise ValueError("No metrics requested.")
        store    = self.paths.metrics_store(norm)
        existing = None if overwrite else self._open_existing_metrics(norm)
        if existing is None:
            to_compute = requested
        elif update_missing_only:
            to_compute = {name for name in requested if name not in existing.data_vars}
        else:
            to_compute = requested
        if not to_compute and existing is not None:
            self.logger.info("All requested metrics already present for %s; nothing to do.", norm)
            return str(store)
        self.logger.info("Requested metrics (%d): %s", len(requested), ", ".join(sorted(requested)))
        self.logger.info("Computing metrics (%d): %s", len(to_compute), ", ".join(sorted(to_compute)))
        ds_new = self._compute_requested_metrics(norm, to_compute)
        ds_new.attrs.update({"sim_name"   : self.run.sim_name,
                             "start_date" : self.run.start_date,
                             "end_date"   : self.run.end_date,
                             "hemisphere" : self.run.hemisphere,
                             "ice_type"   : self.classify.ice_type,
                             "grid_type"  : self.classify.grid_type,
                             "method"     : norm})
        store.parent.mkdir(parents=True, exist_ok=True)
        if existing is not None and update_missing_only and not overwrite:
            overlap = set(ds_new.data_vars) & set(existing.data_vars)
            if overlap:
                raise ValueError(f"Refusing to append overlapping metric names to existing store: {sorted(overlap)}")
            try:
                self._assert_same_indexes(existing, ds_new, dims=("time", "region"))
            except ValueError:
                if not rebuild_on_index_mismatch:
                    raise
                self.logger.warning("Existing metrics store has incompatible indexes; backing up and rebuilding.")
                self._metrics_cache.pop(norm, None)
                self._backup_legacy_store(store)
                ds_out = self._compute_requested_metrics(norm, requested)
                ds_out.attrs.update({"sim_name"   : self.run.sim_name,
                                     "start_date" : self.run.start_date,
                                     "end_date"   : self.run.end_date,
                                     "hemisphere" : self.run.hemisphere,
                                     "ice_type"   : self.classify.ice_type,
                                     "grid_type"  : self.classify.grid_type,
                                     "method"     : norm})
                ds_out    = self._prepare_output_dataset(ds_out)
                tmp_store = store.with_name(store.name + ".tmp")
                if tmp_store.exists():
                    shutil.rmtree(tmp_store)
                self.logger.info("Writing rebuilt metrics to %s", tmp_store)
                ds_out.to_zarr(tmp_store,
                               mode         = "w",
                               consolidated = False,
                               encoding     = self._encoding_from_dataset(ds_out),
                               zarr_format  = 2)
                if store.exists():
                    shutil.rmtree(store)
                tmp_store.rename(store)
                self._metrics_cache[norm] = ds_out
                return str(store)
            ds_new = self._prepare_output_dataset(ds_new)
            self.logger.info("Appending %d new metrics to existing store: %s", len(ds_new.data_vars), store)
            ds_new.to_zarr(store,
                           mode         = "a",
                           consolidated = False,
                           encoding     = self._encoding_from_dataset(ds_new),
                           zarr_format  = 2)
            self._metrics_cache.pop(norm, None)
            self._metrics_cache[norm] = xr.open_zarr(store, consolidated=False)
            return str(store)
        if existing is not None and not overwrite:
            try:
                self._assert_same_indexes(existing, ds_new, dims=("time", "region"))
            except ValueError:
                if not rebuild_on_index_mismatch:
                    raise
                self.logger.warning("Existing metrics store has incompatible indexes; backing up and rebuilding.")
                self._metrics_cache.pop(norm, None)
                self._backup_legacy_store(store)
                existing = None
            if existing is not None:
                ds_out = xr.merge([existing, ds_new], compat="override", combine_attrs="override", join="exact")
            else:
                ds_out = ds_new
        else:
            ds_out = ds_new
        ds_out.attrs.update({"sim_name"   : self.run.sim_name,
                             "start_date" : self.run.start_date,
                             "end_date"   : self.run.end_date,
                             "hemisphere" : self.run.hemisphere,
                             "ice_type"   : self.classify.ice_type,
                             "grid_type"  : self.classify.grid_type,
                             "method"     : norm})
        ds_out    = self._prepare_output_dataset(ds_out)
        tmp_store = store.with_name(store.name + ".tmp")
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        if overwrite and store.exists():
            shutil.rmtree(store)
        self.logger.info("Writing metrics to %s", tmp_store)
        ds_out.to_zarr(tmp_store,
                       mode         = "w",
                       consolidated = False,
                       encoding     = self._encoding_from_dataset(ds_out),
                       zarr_format  = 2)
        if store.exists():
            shutil.rmtree(store)
        tmp_store.rename(store)
        self._metrics_cache.pop(norm, None)
        self._metrics_cache[norm] = ds_out
        return str(store)

    def report_metric_extrema(self, method: str, *,
                              variable                : str = "FIA",
                              year_mode               : str = "calendar",
                              compute_missing         : bool = False,
                              include_mean            : bool = True,
                              include_overall         : bool = True,
                              growth_window           : tuple[int, int] | None = (4, 7),
                              retreat_window          : tuple[int, int] | None = (12, 3),
                              require_full_rate_window: bool = True,
                              rate_min_points         : int = 20,
                              drop_partial_periods    : bool = False) -> pd.DataFrame:
        """
        Load a metric from the method-specific metrics store and return
        per-year extrema diagnostics as a table.
        """
        norm = normalize_method(method)
        ds = self._open_existing_metrics(norm)
        if ds is None or variable not in ds.data_vars:
            if not compute_missing:
                store = self.paths.metrics_store(norm)
                raise KeyError(f"Metric {variable!r} is not available in {store}. Run metrics first, or call with compute_missing=True.")
            self.logger.info("Metric %s missing for %s; computing it now.", variable, self.run.sim_name)
            self.compute_metrics(norm,
                                 metric_names        = [variable],
                                 metric_groups       = [],
                                 update_missing_only = True)
            ds = self._open_existing_metrics(norm)
        if ds is None or variable not in ds.data_vars:
            raise KeyError(f"Metric {variable!r} could not be loaded after metrics computation.")
        da = ds[variable]
        if "time" in da.coords:
            da = da.sel(time = slice(pd.to_datetime(self.run.start_date), pd.to_datetime(self.run.end_date)))
        out = self.compute_extrema_table(da,
                                         variable                 = variable,
                                         sim_name                 = self.run.sim_name,
                                         year_mode                = year_mode,
                                         include_mean             = include_mean,
                                         include_overall          = include_overall,
                                         growth_window            = growth_window,
                                         retreat_window           = retreat_window,
                                         require_full_rate_window = require_full_rate_window,
                                         rate_min_points          = rate_min_points,
                                         drop_partial_periods     = drop_partial_periods)
        out.insert(1, "method", norm)
        out.insert(2, "grid_type", self.classify.grid_type)
        out.insert(3, "ice_type", self.classify.ice_type)
        out.insert(4, "hemisphere", self.run.hemisphere)
        return out

