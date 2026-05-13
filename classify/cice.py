
from __future__            import annotations
import shutil
from collections.abc       import Callable
import numpy               as np
import pandas              as pd
import xarray              as xr
from shuga.core.logging    import build_file_logger
from shuga.core.naming     import normalize_method
from shuga.core.paths      import ShugaPaths
from shuga.core.types      import ClassificationSpec, RunSpec
from shuga.io.zarr_loading import load_cice
from shuga.regridding.cice import compute_tgrid_speed, parse_grid_selection


# POSSIBLE MOVE THIS TO shuga/IO/zarr_writing.py **NEW FILE**
def _sanitize_for_zarr_write(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    # Drop inherited backend encoding that can poison writes.
    for name in ds.variables:
        ds[name].encoding = {}
    # Dataset-level encoding can also carry backend state.
    ds.encoding = {}
    return ds

# POSSIBLE MOVE THIS TO shuga/core/netcdf_helper.py **NEW FILE**
def _strip_to_classification_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Keep only the minimal coordinates needed for classification output.
    Retain time coordinate if present; drop all spatial/static coords.
    """
    time_coord = da["time"] if "time" in da.coords else None
    clean      = xr.DataArray(da.data,
                              dims   = da.dims,
                              coords = {"time": time_coord} if time_coord is not None else None,
                              name   = da.name,
                              attrs  = da.attrs)
    return clean

class CICEClassifier:
    """
    Standalone fast-ice classifier for CICE Zarr history output.

    This class loads CICE history fields, reconstructs sea-ice speed on the
    target T grid, applies one or more fast-ice classification methods, and
    writes the resulting masks to method-specific Zarr stores.

    The implementation is designed to follow the AFIM classification workflow
    closely by reconstructing a T-grid speed field before thresholding. The
    classifier supports several grid-handling modes via
    ``self.classify.grid_type``:

    - ``"Tc"``
        Reconstruct T-grid speed from C-grid edge components
        ``uvelE``, ``uvelN``, ``vvelE``, and ``vvelN``. This mode is exclusive.
    - ``"Ta"``
        Reconstruct T-grid speed from a B-grid 2x2 corner mean, with missing
        values propagating through the average.
    - ``"Tb"``
        Reconstruct T-grid speed from a B-grid 2x2 corner mean, with missing
        values filled as ``0.0`` before averaging. This behaves more like a
        no-slip treatment near coasts.
    - ``"Tx"``
        Use an explicit B-grid to T-grid regridding path. This mode requires a
        ``regridder`` callable to be supplied at initialisation.

    Parameters
    ----------
    run : RunSpec
        Run-level configuration describing the simulation, analysis period, and
        hemisphere.
    classify : ClassificationSpec
        Classification configuration defining variables, thresholds, grid type,
        methods, and rolling/window parameters.
    paths : ShugaPaths | None, optional
        Path bundle used to locate input stores and classification outputs. If
        omitted, a new ``ShugaPaths`` instance is built from ``run`` and
        ``classify``.
    chunks : dict | None, optional
        Dask chunking used when loading CICE history data. Defaults to
        ``{"time": 31}``.
    regridder : Callable[[xr.DataArray], xr.DataArray] | None, optional
        Optional callable used for explicit B-grid to T-grid remapping when the
        selected grid mode requires it.
    logger : logging.Logger | None, optional
        Logger used for progress and write messages. If omitted, a file logger
        is created using the classification log path.

    Attributes
    ----------
    run : RunSpec
        Active run configuration.
    classify : ClassificationSpec
        Active classification configuration.
    paths : ShugaPaths
        Resolved path bundle for inputs and outputs.
    chunks : dict
        Chunking policy used when opening history data.
    regridder : callable | None
        Optional explicit regridder for ``"Tx"``-style workflows.
    logger : logging.Logger
        Logger for classification progress and diagnostics.

    Notes
    -----
    - All classification methods ultimately derive from a reconstructed T-grid
      speed magnitude.
    - Raw, binary-days, and rolling-mean classification products are supported.
    - Output stores are written as Zarr version 2 with a canonical variable
      name of ``FI_mask``.
    """

    def __init__(self, run: RunSpec, classify: ClassificationSpec,
                 paths    : ShugaPaths | None                             = None, *,
                 chunks   : dict | None                                   = None,
                 regridder: Callable[[xr.DataArray], xr.DataArray] | None = None,
                 logger                                                   = None) -> None:
        """
        Initialise a fast-ice classifier for a single simulation and
        classification configuration.

        Parameters
        ----------
        run : RunSpec
            Run-level configuration describing the simulation and requested time
            window.
        classify : ClassificationSpec
            Classification settings, including thresholds, variables, methods, and
            grid reconstruction mode.
        paths : ShugaPaths | None, optional
            Path bundle for locating model history input and classification output
            stores. If omitted, one is constructed from ``run`` and ``classify``.
        chunks : dict | None, optional
            Chunking to use when reading CICE history data. Defaults to
            ``{"time": 31}``.
        regridder : Callable[[xr.DataArray], xr.DataArray] | None, optional
            Optional callable for explicit B-grid to T-grid remapping.
        logger : logging.Logger | None, optional
            Logger for status and diagnostic messages. If not supplied, a file
            logger is created automatically.

        Notes
        -----
        - A dataset cache attribute ``_ds_cache`` is initialised to ``None`` for
          reuse of loaded history data.
        - The logger defaults to a file-backed classifier logger when not
          explicitly provided.
        """
        self.run       = run
        self.classify  = classify
        self.paths     = paths or ShugaPaths(run=run, classify=classify)
        self.chunks    = chunks or {"time": 31}
        self.regridder = regridder
        self.logger    = logger or build_file_logger("shuga.classify", self.paths.classification_log_path())
        self._ds_cache: xr.Dataset | None = None

    @property
    def mask_var_name(self) -> str:
        return f"{self.classify.ice_type}_mask"

    @property
    def grid_selection(self) -> tuple[str, ...]:
        return parse_grid_selection(self.classify.grid_type)

    #----------------------------------------------------------------------------
    # helpers
    #---------------------------------------------------------------------------
    def _required_padding_days(self, methods: list[str] | tuple[str, ...]) -> int:
        pads = [0]
        methods = [normalize_method(m) for m in methods]
        if "binary-days" in methods:
            pads.append(self.classify.bin_window // 2)
        if "rolling-mean" in methods:
            pads.append(self.classify.roll_window // 2)
        return max(pads)

    def _target_da(self, ds: xr.Dataset) -> xr.DataArray:
        target = ds[self.classify.aice_var]
        if target.ndim < 3:
            raise ValueError(f"Expected {self.classify.aice_var!r} to have time,y,x dims; got {target.dims!r}")
        return target

    def _required_velocity_vars(self) -> list[str]:
        sel = set(self.grid_selection)
        if "Tc" in sel:
            return [self.classify.uvelE_var,
                    self.classify.uvelN_var,
                    self.classify.vvelE_var,
                    self.classify.vvelN_var]
        vars_keep = [self.classify.speed_var_u, self.classify.speed_var_v]
        return vars_keep

    def _crop_requested_window(self, da: xr.DataArray) -> xr.DataArray:
        return da.sel(time=slice(self.run.start_date, self.run.end_date))

    def load_cice(self, methods: list[str] | tuple[str, ...] | None = None) -> xr.Dataset:
        methods     = list(methods or self.classify.methods)
        extend_days = self._required_padding_days(methods)
        if self._ds_cache is None:
            vars_keep    = [self.classify.aice_var, *self._required_velocity_vars(), "TLON", "TLAT"]
            self.logger.info("Resolved CICE store: %s", self.paths.resolve_cice_store())
            static_store = self.paths.resolve_static_store()
            if static_store is not None:
                self.logger.info("Resolved static store: %s", static_store)
            dt0            = (pd.to_datetime(self.run.start_date) - pd.Timedelta(days=int(extend_days))).strftime("%Y-%m-%d")
            dtN            = (pd.to_datetime(self.run.end_date)   + pd.Timedelta(days=int(extend_days))).strftime("%Y-%m-%d")
            self._ds_cache = load_cice(run        = self.run,
                                       classify   = self.classify,
                                       paths      = self.paths,
                                       dt0_str    = dt0,
                                       dtN_str    = dtN,
                                       variables  = vars_keep,
                                       hemisphere = self.run.hemisphere,
                                       chunks     = self.chunks)
        return self._ds_cache

    def _output_chunk_map(self, ds_out: xr.Dataset) -> dict[str, int]:
        chunk_map: dict[str, int] = {}
        if "time" in ds_out.dims:
            chunk_map["time"] = int(self.chunks.get("time", 31))
        for dim in ds_out.dims:
            if dim != "time":
                chunk_map[dim] = -1
        return chunk_map

    #----------------------------------------------------------------------------
    # APIs
    #---------------------------------------------------------------------------
    def compute_speed(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Reconstruct sea-ice speed magnitude on the target T grid.

        This method delegates the grid-specific reconstruction to
        ``compute_tgrid_speed()``, using the variable names and grid settings from
        the active ``ClassificationSpec``. The returned speed field is renamed to
        ``"ice_speed"``, annotated with standard metadata, and cast to
        ``float32``.

        Parameters
        ----------
        ds : xr.Dataset
            Input CICE history dataset containing the velocity fields required by
            the configured grid-reconstruction mode.

        Returns
        -------
        xr.DataArray
            T-grid sea-ice speed magnitude with name ``"ice_speed"``.

        Notes
        -----
        - The exact velocity inputs used depend on ``self.classify.grid_type``.
        - ``grid_type`` metadata is written from ``self.grid_selection``.
        - The output units are metres per second.
        """
        target = self._target_da(ds)
        speed = compute_tgrid_speed(ds, target,
                                    grid_type     = self.classify.grid_type,
                                    u_var         = self.classify.speed_var_u,
                                    v_var         = self.classify.speed_var_v,
                                    uvelE_var     = self.classify.uvelE_var,
                                    uvelN_var     = self.classify.uvelN_var,
                                    vvelE_var     = self.classify.vvelE_var,
                                    vvelN_var     = self.classify.vvelN_var,
                                    wrap_x        = bool(self.classify.wrap_x),
                                    cgrid_combine = self.classify.cgrid_combine,
                                    regridder     = self.regridder,
                                    logger        = self.logger)
        speed.name = "ice_speed"
        speed.attrs.update({"long_name": "Sea-ice speed magnitude on T-grid",
                            "units"    : "m s-1",
                            "grid_type": " ".join(self.grid_selection)})
        return speed.astype(np.float32)

    def compute_raw_mask(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Compute the raw daily fast-ice mask from instantaneous speed and ice
        concentration.

        A cell is classified as fast ice when all of the following hold:

        - concentration exceeds ``aice_thresh``,
        - reconstructed speed is finite,
        - speed is greater than zero,
        - speed is less than or equal to ``ispd_thresh``.

        Parameters
        ----------
        ds : xr.Dataset
            Input CICE history dataset containing the fields required for speed
            reconstruction and concentration thresholding.

        Returns
        -------
        xr.DataArray
            Boolean raw classification mask with the classifier's configured mask
            variable name.

        Notes
        -----
        - The mask is based on the reconstructed T-grid speed from
          :meth:`compute_speed`.
        - Metadata records the ice-speed threshold, concentration threshold,
          classification method, and grid selection.
        """
        speed     = self.compute_speed(ds)
        aice      = ds[self.classify.aice_var]
        mask      = ((aice > float(self.classify.aice_thresh))
                     & np.isfinite(speed)
                     & (speed > 0)
                     & (speed <= float(self.classify.ispd_thresh)))
        mask.name = self.mask_var_name
        mask.attrs.update({"long_name"            : f"{self.classify.ice_type} raw daily mask",
                           "ispd_thresh_m_s"      : float(self.classify.ispd_thresh),
                           "aice_thresh"          : float(self.classify.aice_thresh),
                           "classification_method": "raw",
                           "grid_type"            : " ".join(self.grid_selection)})
        return mask.astype("bool")

    def classify_raw(self, ds: xr.Dataset | None = None) -> xr.DataArray:
        """
        Generate the raw fast-ice classification mask for the requested analysis
        window.

        Parameters
        ----------
        ds : xr.Dataset | None, optional
            Preloaded CICE history dataset. If omitted, history data are loaded
            internally for the ``"raw"`` method.

        Returns
        -------
        xr.DataArray
            Boolean raw fast-ice mask cropped to the requested output time window.

        Notes
        -----
        - This is the direct threshold-based classification with no temporal
          persistence or smoothing.
        - If ``ds`` is not supplied, history data are loaded via ``load_cice()``.
        """
        ds = ds if ds is not None else self.load_cice(methods=("raw",))
        return self._crop_requested_window(self.compute_raw_mask(ds))

    def classify_binary_days(self, ds: xr.Dataset | None = None) -> xr.DataArray:
        """
        Generate a binary-days fast-ice mask using a centred rolling persistence
        test.

        The raw daily mask is first converted to integer form and then summed over
        a centred rolling time window. A day is classified as fast ice when the
        number of raw fast-ice days within the window is greater than or equal to
        ``bin_min_days``.

        Parameters
        ----------
        ds : xr.Dataset | None, optional
            Preloaded CICE history dataset. If omitted, history data are loaded
            internally for the ``"binary-days"`` method.

        Returns
        -------
        xr.DataArray
            Boolean binary-days fast-ice mask cropped to the requested output time
            window.

        Notes
        -----
        - The rolling window uses ``center=True``.
        - ``min_periods`` is set to ``bin_min_days``.
        - Output metadata records the classification method and the binary-days
          window settings.
        """
        ds        = ds if ds is not None else self.load_cice(methods=("binary-days",))
        raw       = self.compute_raw_mask(ds).astype("int16")
        mask      = raw.rolling(time=self.classify.bin_window, center=True, min_periods=self.classify.bin_min_days).sum() >= self.classify.bin_min_days
        mask      = self._crop_requested_window(mask.astype("bool"))
        mask.name = self.mask_var_name
        mask.attrs.update({"long_name"            : f"{self.classify.ice_type} binary-days mask",
                           "classification_method": "binary-days",
                           "bin_window"           : int(self.classify.bin_window),
                           "bin_min_days"         : int(self.classify.bin_min_days),
                           "grid_type"            : " ".join(self.grid_selection)})
        return mask

    def classify_rolling_mean(self, ds: xr.Dataset | None = None) -> xr.DataArray:
        """
        Generate a rolling-mean fast-ice mask using temporally averaged ice speed.

        Speed is first reconstructed on the T grid and then smoothed with a
        centred rolling mean over ``roll_window`` days. The classification is then
        applied using the smoothed speed field together with the concentration
        threshold.

        Parameters
        ----------
        ds : xr.Dataset | None, optional
            Preloaded CICE history dataset. If omitted, history data are loaded
            internally for the ``"rolling-mean"`` method.

        Returns
        -------
        xr.DataArray
            Boolean rolling-mean fast-ice mask cropped to the requested output time
            window.

        Notes
        -----
        - The rolling average uses ``center=True``.
        - ``min_periods`` is set equal to ``roll_window``, so full temporal support
          is required.
        - Output metadata records the classification method and rolling window
          length.
        """
        ds         = ds if ds is not None else self.load_cice(methods=("rolling-mean",))
        speed      = self.compute_speed(ds)
        aice       = ds[self.classify.aice_var]
        roll_speed = speed.rolling(time=self.classify.roll_window, center=True, min_periods=self.classify.roll_window).mean()
        mask       = ((aice > float(self.classify.aice_thresh))
                      & np.isfinite(roll_speed)
                      & (roll_speed > 0)
                      & (roll_speed <= float(self.classify.ispd_thresh)))
        mask       = self._crop_requested_window(mask.astype("bool"))
        mask.name  = self.mask_var_name
        mask.attrs.update({"long_name"             : f"{self.classify.ice_type} rolling-mean mask",
                           "classification_method" : "rolling-mean",
                           "roll_window"           : int(self.classify.roll_window),
                           "grid_type"             : " ".join(self.grid_selection)})
        return mask

    def classify_method(self, method: str, ds: xr.Dataset | None = None) -> xr.DataArray:
        """
        Dispatch to the requested classification method.

        Parameters
        ----------
        method : str
            Classification method name. This is normalised via
            ``normalize_method()`` before dispatch.
        ds : xr.Dataset | None, optional
            Preloaded CICE history dataset to reuse across method calls. If
            omitted, the selected method will load data as needed.

        Returns
        -------
        xr.DataArray
            Boolean fast-ice mask for the requested method.

        Notes
        -----
        - Supported normalised methods are ``"raw"``, ``"binary-days"``, and
          ``"rolling-mean"``.
        - Any normalised method other than ``"raw"`` and ``"binary-days"``
          currently falls through to ``classify_rolling_mean()``.
        """
        norm = normalize_method(method)
        if norm == "raw":
            return self.classify_raw(ds)
        if norm == "binary-days":
            return self.classify_binary_days(ds)
        return self.classify_rolling_mean(ds)

    def write_classification(self, method: str, data: xr.Dataset | xr.DataArray, *,
                             overwrite: bool = False) -> str:
        """
        Write classified output to its method-specific Zarr store.

        Accepts either:
        - a legacy mask DataArray, or
        - a Dataset containing FI_mask and optional classified diagnostics.
        """
        store = self.paths.classification_store(method)
        store.parent.mkdir(parents=True, exist_ok=True)
        if store.exists():
            if not overwrite:
                self.logger.info("Classification store exists and overwrite=False, skipping: %s", store)
                return str(store)
            shutil.rmtree(store)
        if isinstance(data, xr.DataArray):
            mask  = _strip_to_classification_coords(data)
            t_org = mask["time"] if "time" in mask.coords else None
            mask  = xr.DataArray(mask.data,
                                 dims   = mask.dims,
                                 coords = {"time": t_org} if t_org is not None else None,
                                 name   = "FI_mask",
                                 attrs  = mask.attrs)
            ds_out = xr.Dataset({"FI_mask": mask})
        else:
            if "FI_mask" not in data.data_vars:
                if len(data.data_vars) == 1:
                    only = next(iter(data.data_vars))
                    data = data.rename({only: "FI_mask"})
                else:
                    raise KeyError(f"Classification dataset must contain FI_mask; got {list(data.data_vars)}")
            cleaned = {}
            for name in ("FI_mask", "FI_ispd", "FI_aice"):
                if name in data.data_vars:
                    da    = _strip_to_classification_coords(data[name])
                    t_org = da["time"] if "time" in da.coords else None
                    cleaned[name] = xr.DataArray(da.data,
                                                 dims   = da.dims,
                                                 coords = {"time": t_org} if t_org is not None else None,
                                                 name   = name,
                                                 attrs  = da.attrs)
            ds_out = xr.Dataset(cleaned)
        ds_out.attrs.update({"sim_name"  : self.run.sim_name,
                             "start_date": self.run.start_date,
                             "end_date"  : self.run.end_date,
                             "hemisphere": self.run.hemisphere,
                             "ice_type"  : self.classify.ice_type,
                             "grid_type" : self.classify.grid_type,
                             "method"    : normalize_method(method)})
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking classification output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)
        ds_out = _sanitize_for_zarr_write(ds_out)
        encoding = {}
        for name, var in ds_out.data_vars.items():
            if getattr(var.data, "chunks", None) is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in var.chunks)}
        for name, var in ds_out.coords.items():
            if getattr(var.data, "chunks", None) is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in var.chunks)}
        self.logger.info("Writing %s classification to %s", normalize_method(method), store)
        ds_out.to_zarr(store, mode="w", consolidated=False, encoding=encoding, zarr_format=2)
        return str(store)

    def build_classification_output(self, method: str, ds: xr.Dataset | None = None) -> xr.Dataset:
        """
        Build the method-specific classified output dataset written to data.zarr.

        Returns a dataset containing:
        - FI_mask : final classified fast-ice mask
        - FI_ispd : speed field actually used by the classifier, masked by FI_mask
        - FI_aice : sea-ice concentration masked by FI_mask
        """
        norm = normalize_method(method)
        ds   = ds if ds is not None else self.load_cice(methods=(norm,))
        aice = ds[self.classify.aice_var]
        if norm == "raw":
            speed_used = self.compute_speed(ds)
            mask       = ((aice > float(self.classify.aice_thresh))
                          & np.isfinite(speed_used)
                          & (speed_used > 0)
                          & (speed_used <= float(self.classify.ispd_thresh)))
            mask       = self._crop_requested_window(mask.astype("bool"))
            mask.attrs.update({"long_name"             : f"{self.classify.ice_type} raw daily mask",
                               "ispd_thresh_m_s"       : float(self.classify.ispd_thresh),
                               "aice_thresh"           : float(self.classify.aice_thresh),
                               "classification_method" : "raw",
                               "grid_type"             : " ".join(self.grid_selection)})
            speed_source = "instantaneous_tgrid_speed"
        elif norm == "binary-days":
            speed_used = self.compute_speed(ds)
            raw        = ((aice > float(self.classify.aice_thresh))
                          & np.isfinite(speed_used)
                          & (speed_used > 0)
                          & (speed_used <= float(self.classify.ispd_thresh)))
            mask       = (raw.astype("int16").rolling(time=self.classify.bin_window,
                                                      center=True,
                                                      min_periods=self.classify.bin_min_days).sum() >= self.classify.bin_min_days)
            mask       = self._crop_requested_window(mask.astype("bool"))
            mask.attrs.update({"long_name"              : f"{self.classify.ice_type} binary-days mask",
                               "classification_method"  : "binary-days",
                               "bin_window"             : int(self.classify.bin_window),
                               "bin_min_days"           : int(self.classify.bin_min_days),
                               "grid_type"              : " ".join(self.grid_selection)})
            speed_source = "instantaneous_tgrid_speed"
        else:
            inst_speed = self.compute_speed(ds)
            speed_used = inst_speed.rolling(time=self.classify.roll_window, center=True, min_periods=self.classify.roll_window).mean()
            mask       = ((aice > float(self.classify.aice_thresh))
                          & np.isfinite(speed_used)
                          & (speed_used > 0)
                          & (speed_used <= float(self.classify.ispd_thresh)))
            mask       = self._crop_requested_window(mask.astype("bool"))
            mask.attrs.update({"long_name"             : f"{self.classify.ice_type} rolling-mean mask",
                               "classification_method" : "rolling-mean",
                               "roll_window"           : int(self.classify.roll_window),
                               "grid_type"             : " ".join(self.grid_selection)})
            speed_source = "rolling_mean_tgrid_speed"
        mask.name      = "FI_mask"
        speed_out      = self._crop_requested_window(speed_used).where(mask).astype(np.float32)
        speed_out.name = "FI_ispd"
        speed_out.attrs.update({"long_name"             : f"{self.classify.ice_type} speed used for classification",
                                "units"                 : "m s-1",
                                "classification_method" : norm,
                                "speed_source"          : speed_source,
                                "grid_type"             : " ".join(self.grid_selection)})
        if norm == "rolling-mean":
            speed_out.attrs["roll_window"] = int(self.classify.roll_window)
        aice_out      = self._crop_requested_window(aice).where(mask).astype(np.float32)
        aice_out.name = "FI_aice"
        aice_out.attrs.update({"long_name"             : f"{self.classify.ice_type} sea-ice concentration",
                               "units"                 : aice.attrs.get("units", "1"),
                               "classification_method" : norm,
                               "grid_type"             : " ".join(self.grid_selection)})
        return xr.Dataset({"FI_mask": mask,
                           "FI_ispd": speed_out,
                           "FI_aice": aice_out})

    def run_methods(self, methods: list[str] | tuple[str, ...] | None = None, *,
                    overwrite: bool = False) -> dict[str, str]:
        """
        Run one or more classification methods and write their outputs.

        This is the main orchestration entry point for the classifier. It resolves
        the requested methods, loads the necessary CICE history data once, applies
        each classification method in turn, writes each result to its method-
        specific store, and returns the output paths.

        Parameters
        ----------
        methods : list[str] | tuple[str, ...] | None, optional
            Methods to run. If omitted, the methods configured in
            ``self.classify.methods`` are used.
        overwrite : bool, optional
            If ``True``, existing classification stores are replaced. If ``False``,
            existing stores are left in place and may be skipped during writing.

        Returns
        -------
        dict[str, str]
            Mapping from normalised method name to output store path.

        Notes
        -----
        - Method names are normalised before execution.
        - CICE history is loaded once and reused across all requested methods.
        - The resolved classification root and speed-reconstruction mode are logged
          before processing begins.
        """
        methods = list(methods or self.classify.methods)
        methods = [normalize_method(m) for m in methods]
        self.logger.info("Resolved classification root: %s", self.paths.classification_root_path)
        self.logger.info("Classification speed reconstruction mode(s): %s", ", ".join(self.grid_selection))
        ds                  = self.load_cice(methods=methods)
        out: dict[str, str] = {}
        for method in methods:
            self.logger.info("Classifying method: %s", method)
            ds_method   = self.build_classification_output(method, ds)
            out[method] = self.write_classification(method, ds_method, overwrite=overwrite)
        return out

