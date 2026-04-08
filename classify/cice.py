
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

def _sanitize_for_zarr_write(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    # Drop inherited backend encoding that can poison writes.
    for name in ds.variables:
        ds[name].encoding = {}
    # Dataset-level encoding can also carry backend state.
    ds.encoding = {}
    return ds

def _strip_to_classification_coords(mask: xr.DataArray) -> xr.DataArray:
    """
    Keep only the minimal coordinates needed for classification output.
    Retain time coordinate if present; drop all spatial/static coords.
    """
    time_coord = mask["time"] if "time" in mask.coords else None
    clean = xr.DataArray(mask.data,
                         dims   = mask.dims,
                         coords = {"time": time_coord} if time_coord is not None else None,
                         name   = mask.name or "FI_mask",
                         attrs  = mask.attrs)
    return clean

class CICEClassifier:
    """Standalone fast-ice classification for CICE Zarr history output.

    This implementation mirrors the AFIM classification logic more closely than the
    earlier shuga draft by reconstructing a T-grid ice-speed field before thresholding.
    Supported `grid_type` tokens are:

    - ``Tc``: C-grid edge east/north components ``uvelE,uvelN,vvelE,vvelN`` -> T-grid speed.
      This mode is exclusive.
    - ``Ta``: B-grid 2x2 corner mean with NaNs propagating.
    - ``Tb``: B-grid 2x2 corner mean with NaNs->0.0 (no-slip-like near coast).
    - ``Tx``: explicit B-grid->T-grid regridding, only when a regridder callable is supplied.
    """

    def __init__(self, run: RunSpec, classify: ClassificationSpec, paths: ShugaPaths | None = None, *,
                 chunks   : dict | None                                   = None,
                 regridder: Callable[[xr.DataArray], xr.DataArray] | None = None,
                 logger                                                   = None) -> None:
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

    def compute_speed(self, ds: xr.Dataset) -> xr.DataArray:
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

    def _crop_requested_window(self, da: xr.DataArray) -> xr.DataArray:
        return da.sel(time=slice(self.run.start_date, self.run.end_date))

    def classify_raw(self, ds: xr.Dataset | None = None) -> xr.DataArray:
        ds = ds if ds is not None else self.load_cice(methods=("raw",))
        return self._crop_requested_window(self.compute_raw_mask(ds))

    def classify_binary_days(self, ds: xr.Dataset | None = None) -> xr.DataArray:
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
        norm = normalize_method(method)
        if norm == "raw":
            return self.classify_raw(ds)
        if norm == "binary-days":
            return self.classify_binary_days(ds)
        return self.classify_rolling_mean(ds)

    def _output_chunk_map(self, ds_out: xr.Dataset) -> dict[str, int]:
        chunk_map: dict[str, int] = {}
        if "time" in ds_out.dims:
            chunk_map["time"] = int(self.chunks.get("time", 31))
        for dim in ds_out.dims:
            if dim != "time":
                chunk_map[dim] = -1
        return chunk_map

    def write_classification(self, method: str, mask: xr.DataArray, *, overwrite: bool = False) -> str:
        store = self.paths.classification_store(method)
        store.parent.mkdir(parents=True, exist_ok=True)
        if store.exists():
            if not overwrite:
                self.logger.info("Classification store exists and overwrite=False, skipping: %s", store)
                return str(store)
            shutil.rmtree(store)
        mask  = _strip_to_classification_coords(mask)
        t_org = mask["time"] if "time" in mask.coords else None
        mask  = xr.DataArray(mask.data,
                             dims   = mask.dims,
                             coords = {"time": t_org} if t_org is not None else None,
                             name   =  "FI_mask",
                             attrs  = mask.attrs)
        ds_out = xr.Dataset({"FI_mask": mask})
        ds_out.attrs.update({"sim_name": self.run.sim_name,
                             "start_date": self.run.start_date,
                             "end_date": self.run.end_date,
                             "hemisphere": self.run.hemisphere,
                             "ice_type": self.classify.ice_type,
                             "grid_type": self.classify.grid_type,
                             "method": normalize_method(method)})
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking classification output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)
        # IMPORTANT: strip inherited encodings before setting fresh ones
        ds_out = _sanitize_for_zarr_write(ds_out)
        encoding = {}
        for name, var in ds_out.data_vars.items():
            if getattr(var.data, "chunks", None) is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in var.chunks)}
        # coordinates should not carry stale backend encoding either
        for name, var in ds_out.coords.items():
            if getattr(var.data, "chunks", None) is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in var.chunks)}
        self.logger.info("Writing %s classification to %s", normalize_method(method), store)
        ds_out.to_zarr(store, mode="w", consolidated=False, encoding=encoding, zarr_format=2) #enforce 2
        return str(store)

    def run_methods(self, methods: list[str] | tuple[str, ...] | None = None, *,
                    overwrite: bool = False) -> dict[str, str]:
        methods = list(methods or self.classify.methods)
        methods = [normalize_method(m) for m in methods]
        self.logger.info("Resolved classification root: %s", self.paths.classification_root_path)
        self.logger.info("Classification speed reconstruction mode(s): %s", ", ".join(self.grid_selection))
        ds = self.load_cice(methods=methods)
        out: dict[str, str] = {}
        for method in methods:
            self.logger.info("Classifying method: %s", method)
            mask = self.classify_method(method, ds)
            out[method] = self.write_classification(method, mask, overwrite=overwrite)
        return out

