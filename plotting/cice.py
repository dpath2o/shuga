from __future__           import annotations
from importlib.resources  import files as resource_files
from dataclasses          import replace, dataclass
from pathlib              import Path
from typing               import Mapping, Sequence
import numpy              as np
import pandas             as pd
import xarray             as xr
from shuga.classify.cice  import CICEClassifier
from shuga.io             import load_cice, load_classified, load_metrics
from shuga.core.logging   import build_file_logger
from shuga.core.naming    import normalize_method
from shuga.core.paths     import ShugaPaths
from shuga.core.regions   import ANTARCTIC_8_REGIONS
from shuga.core.types     import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec
from shuga.metrics.cice   import CICEMetrics
from shuga.observations   import SeaIceObservations
from shuga.plotting.cawcr import plot_regridded_hs_sic_panel

@dataclass(frozen=True)
class SIAStyle:
    """Line and envelope styling for one SIA series."""
    pen: str
    fill: str
DEFAULT_SIA_STYLES: dict[str, SIAStyle] = {"NSIDC"      : SIAStyle("2.4p,black"     , "gray40@82"),
                                           "OSI-SAF-450": SIAStyle("2.4p,gray35,5_2", "gray65@82"),
                                           "ORAS"       : SIAStyle("2.4p,#61D97B"   , "#61D97B@80")}

@dataclass(frozen=True)
class SITStyle:
    """Line and envelope styling for one SIT series."""
    pen: str
    fill: str
DEFAULT_SIT_STYLES: dict[str, SITStyle] = {"ESA-CCI": SITStyle("2.6p,black", "gray45@84"),
                                           "AWI"    : SITStyle("2.6p,gray35,5_2", "gray65@84"),
                                           "CMEMS"  : SITStyle("2.4p,#61D97B", "#61D97B@82")}

def _any_not_none(*vals) -> bool:
    return any(v is not None for v in vals)

def _method_tuple(method: str | Sequence[str] | None = None,
                  methods: str | Sequence[str] | None = None) -> tuple[str, ...] | None:
    """
    Normalize method/methods input.

    Accepts:
        method="binary-days"
        methods="binary-days"
        methods=("binary-days", "rolling-mean")
        methods=["binary-days", "rolling-mean"]

    Returns
    -------
    tuple[str, ...] | None
    """
    value = methods if methods is not None else method
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)

class CICEPlotter:
    """PyGMT plotting helpers for shuga classification, metrics, and observations."""

    def __init__(self,
                 run_cfg       : RunSpec            | None = None,
                 cls_cfg       : ClassificationSpec | None = None,
                 met_cfg       : MetricsSpec        | None = None,
                 plt_cfg       : PlottingSpec       | None = None,
                 obs_cfg       : ObservationSpec    | None = None,
                 pth_cfg       : ShugaPaths         | None = None, *,
                 sim_name      : str | None = None,
                 start_date    : str | None = None,
                 end_date      : str | None = None,
                 method        : str | Sequence[str] | None = None,
                 methods       : str | Sequence[str] | None = None,
                 ice_type      : str = "FI",
                 hemisphere    : str = "SH",
                 project       : str = "gv90",
                 user          : str = "da1339",
                 iceh_frequency: str = "daily",
                 chunks        : dict | None = None,
                 logger=None) -> None:
        """
        Construct a CICEPlotter from either full config objects or lightweight
        scalar arguments.

        Supported patterns
        ------------------
        Full configuration mode:

            CICEPlotter(run_cfg          = RunSpec(...),
                        cls_cfg     = ClassificationSpec(...),
                        met_cfg      = MetricsSpec(...),
                        plt_cfg     = PlottingSpec(...),
                        obs_cfg = ObservationSpec(...))

        Lightweight mode:

            CICEPlotter(sim_name   = "LD-blend-base",
                        start_date = "2000-01-01",
                        end_date   = "2003-12-31",
                        ice_type   = "FI",
                        method     = "binary-days")
        """
        method_tuple = _method_tuple(method=method, methods=methods)
        if run_cfg is None:
            missing = [name for name, value in {"sim_name": sim_name,
                                                "start_date": start_date,
                                                "end_date": end_date}.items() if value is None]
            if missing:
                raise ValueError("CICEPlotter requires either run_cfg=RunSpec(...) or the scalar "
                                 f"arguments sim_name=..., start_date=..., end_date=.... "
                                 f"Missing: {', '.join(missing)}")
            run_cfg = RunSpec(sim_name       = sim_name,
                              start_date     = start_date,
                              end_date       = end_date,
                              hemisphere     = hemisphere,
                              project        = project,
                              user           = user,
                              iceh_frequency = iceh_frequency)
        else:
            # Optional scalar overrides when a Run_CfgSpec is supplied.
            updates = {}
            if sim_name is not None:
                updates["sim_name"] = sim_name
            if start_date is not None:
                updates["start_date"] = start_date
            if end_date is not None:
                updates["end_date"] = end_date
            if hemisphere is not None:
                updates["hemisphere"] = hemisphere
            if project is not None:
                updates["project"] = project
            if user is not None:
                updates["user"] = user
            if iceh_frequency is not None:
                updates["iceh_frequency"] = iceh_frequency
            if updates:
                run_cfg = replace(run_cfg, **updates)
        if cls_cfg is None:
            classify_kwargs = {"ice_type": ice_type}
            if method_tuple is not None:
                classify_kwargs["methods"] = method_tuple
            cls_cfg = ClassificationSpec(**classify_kwargs)
        else:
            updates = {}
            if ice_type is not None:
                updates["ice_type"] = ice_type
            if method_tuple is not None:
                updates["methods"] = method_tuple
            if updates:
                cls_cfg = replace(cls_cfg, **updates)
        if met_cfg is None:
            if method_tuple is not None:
                met_cfg = MetricsSpec(methods=method_tuple)
            else:
                met_cfg = MetricsSpec()
        elif method_tuple is not None:
            met_cfg = replace(met_cfg, methods=method_tuple)
        self.run_cfg        = run_cfg
        self.cls_cfg        = cls_cfg
        self.met_cfg        = met_cfg
        self.plt_cfg        = plt_cfg or PlottingSpec()
        self.obs_cfg        = obs_cfg or ObservationSpec()
        self.pth_cfg        = pth_cfg or ShugaPaths(run_cfg = run_cfg, cls_cfg = cls_cfg, obs_cfg = self.obs_cfg)
        self.chunks         = chunks or {"time": 31}
        self.logger         = logger or build_file_logger("shuga.plt_cfg", self.pth_cfg.logs_root_path / "plotting" / f"plotting_{run_cfg.sim_name}.log")
        self.metrics_runner = CICEMetrics(run_cfg  = run_cfg,
                                          cls_cfg  = cls_cfg,
                                          met_cfg  = self.met_cfg,
                                          pth_cfg  = self.pth_cfg,
                                          chunks   = self.chunks,
                                          logger   = self.logger)
        self.classifier     = CICEClassifier(run_cfg = run_cfg,
                                             cls_cfg = cls_cfg,
                                             pth_cfg = self.pth_cfg,
                                             chunks  = self.chunks,
                                             logger  = self.logger)
        self.obs            = SeaIceObservations(run_cfg = run_cfg,
                                                 obs_cfg = self.obs_cfg,
                                                 pth_cfg = self.pth_cfg,
                                                 chunks  = self.chunks,
                                                 logger  = self.logger)

    # ------------------------------------------------------------
    def _require_pygmt(self):
        try:
            import pygmt
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyGMT is required for plotting methods.") from exc
        return pygmt

    # ------------------------------------------------------------
    # static functions
    # ------------------------------------------------------------
    @staticmethod
    def _detect_lonlat(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        lon_name = next((n for n in ("TLON", "ULON", "lon", "longitude", "ELON", "NLON") if n in ds.variables or n in ds.coords), None)
        lat_name = next((n for n in ("TLAT", "ULAT", "lat", "latitude", "ELAT", "NLAT") if n in ds.variables or n in ds.coords), None)
        if lon_name is None or lat_name is None:
            raise KeyError("Could not find longitude/latitude fields in dataset.")
        return ds[lon_name], ds[lat_name]

    @staticmethod
    def _lon_to_180(lon: xr.DataArray) -> xr.DataArray:
        return ((lon + 180.0) % 360.0) - 180.0

    @staticmethod
    def meridian_center_from_region(region: Sequence[float]) -> float:
        lon_min, lon_max, *_ = region
        a = ((float(lon_min) + 180.0) % 360.0) - 180.0
        b = ((float(lon_max) + 180.0) % 360.0) - 180.0
        if a <= b:
            mc = 0.5 * (a + b)
        else:
            width = (b + 360.0) - a
            mc = a + 0.5 * width
            if mc > 180.0:
                mc -= 360.0
        return mc

    @classmethod
    def projection_from_region(cls, region: Sequence[float], fig_size: float = 20.0) -> str:
        _, _, lat_min, lat_max = region
        mc = cls.meridian_center_from_region(region)
        lat_center = 0.5 * (float(lat_min) + float(lat_max))
        pole = -90 if lat_center < 0 else 90
        return f"S{mc}/{pole}/{fig_size}c"

    @staticmethod
    def _region_mask(lon: xr.DataArray, lat: xr.DataArray, region: Sequence[float]) -> xr.DataArray:
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        lon_min, lon_max, lat_min, lat_max = [float(v) for v in region]
        if lon_min <= lon_max:
            lon_mask = (lon180 >= lon_min) & (lon180 <= lon_max)
        else:
            lon_mask = (lon180 >= lon_min) | (lon180 <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        return lon_mask & lat_mask

    @staticmethod
    def _resolve_plot_window(dt0_str: str | None = None, dtN_str: str | None = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        dt0 = pd.Timestamp(dt0_str) if dt0_str is not None else None
        dtN = pd.Timestamp(dtN_str) if dtN_str is not None else None
        if dt0 is not None and dtN is not None and dtN < dt0:
            raise ValueError(f"dtN_str ({dtN.date()}) must be on or after dt0_str ({dt0.date()}).")
        return dt0, dtN

    @staticmethod
    def _validate_f2020_window(dt0: pd.Timestamp | None, dtN : pd.Timestamp | None, *,
                               add_f2020  : bool,
                               f2020_mode : str) -> None:
        if not add_f2020:
            return
        if str(f2020_mode).strip().lower() != "climatology":
            return
        if dt0 is None or dtN is None:
            return
        ndays = (dtN - dt0).days + 1
        if ndays < 15:
            raise ValueError(f"When add_f2020=True and f2020_mode='climatology', the requested window must be at least 15 days. "
                             f"Got {ndays} days ({dt0.strftime('%Y-%m-%d')} to {dtN.strftime('%Y-%m-%d')}).")

    @staticmethod
    def sia_to_million_km2(da: xr.DataArray) -> xr.DataArray:
        units = str(da.attrs.get("units", "")).strip().lower().replace("²", "^2").replace(" ", "")
        if units in {"m2", "m^2", "m**2"}:
            out = da / 1.0e12
        elif units in {"km2", "km^2", "km**2"}:
            out = da / 1.0e6
        elif any(token in units for token in ("10^3km^2", "10^3km2", "10^3*km^2", "10^3*km2")):
            out = da / 1.0e3
        elif any(token in units for token in ("10^6km^2", "10^6km2", "10^6*km^2", "10^6*km2")):
            out = da
        else:
            raise ValueError(f"Unsupported/missing SIA units {da.attrs.get('units')!r} for {da.name!r}.")
        out = out.rename(da.name)
        out.attrs.update(da.attrs)
        out.attrs["units"] = "10^6 km^2"
        return out

    @staticmethod
    def dataarray_to_series(da: xr.DataArray, name: str) -> pd.Series:
        if "time" not in da.dims or [d for d in da.dims if d != "time"]:
            raise ValueError(f"{name}: expected a one-dimensional time series; got {da.dims}")
        values = da.compute().to_series()
        values.index = pd.DatetimeIndex(values.index)
        values = values[~values.index.duplicated(keep="first")].sort_index()
        values.name = name
        return values.astype(float)

    @staticmethod
    def _open_sia_dataset(path: str | Path, chunks: dict | None = None) -> xr.Dataset:
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir() or path.suffix == ".zarr":
            try:
                return xr.open_zarr(path, consolidated=True, chunks=chunks)
            except (KeyError, ValueError, FileNotFoundError):
                return xr.open_zarr(path, consolidated=False, chunks=chunks)
        return xr.open_dataset(path, chunks=chunks)

    @classmethod
    def load_sia_store(cls, path: str | Path, *,
                       label         : str,
                       start_date    : str | None = None,
                       end_date      : str | None = None,
                       variable      : str | None = None,
                       candidates    : Sequence[str] = ("SIA", "sia", "sea_ice_area"),
                       units_override: str | None = None) -> pd.Series:
        """
        Load a one-dimensional SIA time series from NetCDF or Zarr.

        Parameters
        ----------
        path
            NetCDF or Zarr store.
        label
            Output pandas Series name.
        start_date, end_date
            Optional inclusive time subset.
        variable
            Explicit SIA variable name. If omitted, ``candidates`` are searched.
        candidates
            Candidate SIA variable names.
        units_override
            Explicit source units when the stored variable does not carry reliable
            units metadata. The override is applied before conversion to
            ``10^6 km^2``.

            This should only be used when the source units are independently known.
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir() or path.suffix == ".zarr":
            try:
                ds = xr.open_zarr(path, consolidated = True)
            except (KeyError, ValueError, FileNotFoundError):
                ds = xr.open_zarr(path, consolidated = False)
        else:
            ds = xr.open_dataset(path)
        if variable is not None:
            if variable not in ds:
                raise KeyError(f"{label}: variable {variable!r} not found in {path}. Available variables: {list(ds.data_vars)}")
            var = variable
        else:
            var = next( (name for name in candidates if name in ds), None)
            if var is None:
                raise KeyError(f"{label}: could not identify an SIA variable in {path}. Available variables: {list(ds.data_vars)}")
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError(f"{label}: expected {var!r} to contain a time dimension; got {da.dims}")
        if start_date is not None or end_date is not None:
            da = da.sel(time = slice(start_date, end_date))
        if da.sizes.get("time", 0) == 0:
            raise ValueError(f"{label}: no SIA data between {start_date or 'start'} and {end_date or 'end'}")
        if units_override is not None:
            da = da.copy()
            da.attrs["units"] = units_override
        return cls.dataarray_to_series(cls.sia_to_million_km2(da), label)

    @staticmethod
    def sit_monthly_climatology(series: pd.Series, *, envelope: str = "p10-p90") -> pd.DataFrame:
        s = series.dropna().sort_index()
        if s.empty:
            raise ValueError(f"{series.name}: no finite SIT values")
        s.index = pd.DatetimeIndex(s.index)
        monthly = s.resample("MS").mean().dropna()
        grp     = monthly.groupby(monthly.index.month)
        mean    = grp.mean().reindex(range(1,13))
        key     = envelope.lower()
        if key == "minmax":
            lower, upper = grp.min().reindex(range(1,13)), grp.max().reindex(range(1,13))
        elif key == "std":
            std = grp.std(ddof=1).reindex(range(1,13))
            lower, upper = mean-std, mean+std
        elif key in {"p10-p90","p10p90"}:
            lower = grp.quantile(.10).reindex(range(1,13))
            upper = grp.quantile(.90).reindex(range(1,13))
        else:
            raise ValueError("envelope must be minmax, std, or p10-p90")
        n = grp.count().reindex(range(1,13)).fillna(0).astype(int)
        return pd.DataFrame({"month" : range(1,13),
                             "mean"  : mean.to_numpy(float),
                             "lower" : lower.to_numpy(float),
                             "upper" : upper.to_numpy(float),
                             "n"     : n.to_numpy(int)})

    @classmethod
    def load_sit_store(cls, path: str | Path, *,
                       label     : str,
                       start_date: str | None = None,
                       end_date  : str | None = None,
                       variable  : str | None = None,
                       candidates: Sequence[str] = ("SIT", "sit", "sea_ice_thickness", "sithick")) -> pd.Series:
        path = Path(path).expanduser()
        if path.is_dir() or path.suffix == ".zarr":
            try:
                ds = xr.open_zarr(path, consolidated=True)
            except Exception:
                ds = xr.open_zarr(path, consolidated=False)
        else:
            ds = xr.open_dataset(path)
        var = variable or next((n for n in candidates if n in ds), None)
        if var is None or var not in ds:
            raise KeyError(f"{label}: SIT variable not found in {path}; available={list(ds.data_vars)}")
        da = ds[var]
        if "time" not in da.dims or [d for d in da.dims if d != "time"]:
            raise ValueError(f"{label}: expected 1-D SIT time series; got {da.dims}")
        if start_date is not None or end_date is not None:
            da = da.sel(time=slice(start_date, end_date))
        return cls.dataarray_to_series(da, label)

    @staticmethod
    def _noleap_doy(index: pd.DatetimeIndex) -> np.ndarray:
        ref = pd.DatetimeIndex(pd.to_datetime({"year": np.full(len(index), 2001, dtype=int), "month": index.month, "day": index.day}))
        return ref.dayofyear.to_numpy()

    @staticmethod
    def _circular_rolling(values: pd.Series, window: int) -> pd.Series:
        window = int(window)
        if window <= 1: return values.copy()
        if window % 2 == 0: raise ValueError("smooth_days must be odd")
        pad = window // 2; arr = values.to_numpy(dtype=float)
        ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
        sm = pd.Series(ext).rolling(window=window, center=True, min_periods=1).mean().iloc[pad:pad+len(arr)].to_numpy()
        return pd.Series(sm, index=values.index, name=values.name)

    @classmethod
    def daily_climatology_envelopes(cls, df: pd.DataFrame, *, start_date: str, end_date: str,
                                    envelope: str = "minmax", smooth_days: int = 1) -> dict[str, pd.DataFrame]:
        subset = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].copy()
        subset = subset[~((subset.index.month == 2) & (subset.index.day == 29))]
        if subset.empty: raise ValueError("No SIA data overlap requested period")
        subset["__doy__"] = cls._noleap_doy(subset.index)
        full = pd.Index(np.arange(1,366), name="doy"); key = envelope.lower()
        out = {}
        for name in df.columns:
            grp = subset.groupby("__doy__")[name]; mean = grp.mean().reindex(full)
            if key == "minmax": lower, upper = grp.min().reindex(full), grp.max().reindex(full)
            elif key == "std":
                std = grp.std(ddof=1).reindex(full); lower, upper = mean-std, mean+std
            elif key in {"p10-p90","p10p90"}: lower, upper = grp.quantile(.1).reindex(full), grp.quantile(.9).reindex(full)
            else: raise ValueError("envelope must be minmax, std, or p10-p90")
            if smooth_days > 1:
                mean, lower, upper = (cls._circular_rolling(v, smooth_days) for v in (mean, lower, upper))
            out[name] = pd.DataFrame({"doy": full.to_numpy(int), "mean": mean.to_numpy(float), "lower": lower.to_numpy(float), "upper": upper.to_numpy(float)})
        return out

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _resolve_regions(self,
                         region_name : str | None                           = None,
                         region      : Sequence[float] | None               = None,
                         regions     : Mapping[str, Sequence[float]] | None = None) -> dict[str, Sequence[float]]:
        if region is not None:
            key = region_name or "custom"
            return {key: region}
        if region_name is not None:
            if region_name not in ANTARCTIC_8_REGIONS:
                raise KeyError(f"Unknown default region {region_name!r}")
            return {region_name: ANTARCTIC_8_REGIONS[region_name].get("plot_region", ANTARCTIC_8_REGIONS[region_name]["geo_region"])}
        if regions is not None:
            return dict(regions)
        return {k: v.get("plot_region", v["geo_region"]) for k, v in ANTARCTIC_8_REGIONS.items()}

    def _nsidc_contours(self, date_str: str, hemisphere: str, threshold: float | None = None) -> list[np.ndarray]:
        ds  = self.obs.load_nsidc_daily(start_date=date_str, end_date=date_str, hemisphere=hemisphere)
        sic = ds[self.obs_cfg.nsidc_sic_var].isel(time=0).astype(float)
        x   = ds["x"].values
        y   = ds["y"].values
        import matplotlib.pyplot as plt  # local import to avoid hard dependency outside plotting use
        fig, ax = plt.subplots()
        cs      = ax.contour(x, y, sic.values, levels=[float(threshold if threshold is not None else self.obs_cfg.nsidc_threshold)])
        plt.close(fig)
        lines: list[np.ndarray] = []
        lonlat = xr.open_dataset(self.obs.nsidc_latlon_file(self.obs.canonical_hemisphere(hemisphere)))[["longitude", "latitude"]]
        lon    = lonlat["longitude"].values
        lat    = lonlat["latitude"].values
        x0, y0 = x[0], y[0]
        dx     = x[1] - x[0]
        dy     = y[1] - y[0]
        ny, nx = lon.shape
        for coll in cs.collections:
            for path in coll.get_paths():
                verts = path.vertices
                ix    = np.clip(np.rint((verts[:, 0] - x0) / dx).astype(int), 0, nx - 1)
                iy    = np.clip(np.rint((verts[:, 1] - y0) / dy).astype(int), 0, ny - 1)
                line  = np.column_stack([(((lon[iy, ix] + 180.0) % 360.0) - 180.0), lat[iy, ix]])
                lines.append(line)
        return lines

    def _load_field(self, variable: str, date_str: str | None = None, method: str | None = None) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        var = variable
        if var == self.classifier.mask_var_name or var.lower() == "fi_mask":
            if method is None:
                raise ValueError("method is required when plotting FI_mask")
            ds_mask  = load_classified(run_cfg        = self.run_cfg,
                                       cls_cfg        = self.cls_cfg,
                                       met_cfg        = self.met_cfg,
                                       plt_cfg        = self.plt_cfg,
                                       obs_cfg        = self.obs_cfg,
                                       pth_cfg        = self.pth_cfg,
                                       classification = method,
                                       hemisphere     = self.run_cfg.hemisphere,
                                       chunks         = self.chunks,
                                       logger         = self.logger)
            da       = ds_mask["FI_mask"]
            ds       = load_cice(run_cfg    = self.run_cfg,
                                 cls_cfg    = self.cls_cfg,
                                 met_cfg    = self.met_cfg,
                                 plt_cfg    = self.plt_cfg,
                                 obs_cfg    = self.obs_cfg,
                                 pth_cfg    = self.pth_cfg,
                                 variables  = ["TLON", "TLAT"],
                                 hemisphere = self.run_cfg.hemisphere,
                                 chunks     = self.chunks,
                                 logger     = self.logger)
            lon, lat = self._detect_lonlat(ds)
        elif var.lower() in {"ispd", "ice_speed"}:
            ds = self.classifier.load_cice(methods=(method or "raw",))
            da = self.classifier.compute_speed(ds)
            lon, lat = self._detect_lonlat(ds)
        else:
            ds = load_cice(run_cfg    = self.run_cfg,
                           cls_cfg    = self.cls_cfg,
                           met_cfg    = self.met_cfg,
                           plt_cfg    = self.plt_cfg,
                           obs_cfg    = self.obs_cfg,
                           pth_cfg    = self.pth_cfg,
                           variables  = [var, "TLON", "TLAT"],
                           hemisphere = self.run_cfg.hemisphere,
                           chunks     = self.chunks,
                           logger     = self.logger)
            if var not in ds:
                raise KeyError(f"Variable {var!r} not found in CICE history")
            da = ds[var]
            lon, lat = self._detect_lonlat(ds)
        if date_str is not None and "time" in da.dims:
            da = da.sel(time=date_str)
        return da, lon, lat

    def _build_obs_timeseries(self, variable: str, *,
                              plot_dt0: pd.Timestamp,
                              plot_dtN: pd.Timestamp,
                              hemisphere: str,
                              add_obs: bool,
                              f2020_mode: str = "climatology") -> pd.DataFrame | None:
        """
        Return an observational time series DataFrame with columns ['time', 'value'],
        or None if no observation overlay is available/applicable.
        """
        if not add_obs:
            return None
        var  = str(variable).upper()
        hemi = str(hemisphere).strip().upper()
        # ------------------------------------------------------------
        # FIA -> AF2020 (south only)
        # ------------------------------------------------------------
        if var == "FIA":
            if hemi != "SH":
                self.logger.info("No AF2020 FIA observational overlay available for hemisphere=%s; plotting model only.", hemi)
                return None
            self._validate_f2020_window(plot_dt0, plot_dtN, add_f2020=True, f2020_mode=f2020_mode)
            mode = str(f2020_mode).strip().lower()
            if mode not in {"climatology", "overlap"}:
                raise ValueError(f"f2020_mode must be 'climatology' or 'overlap', got {f2020_mode!r}")
            if mode == "climatology":
                obs_da = self.obs.repeat_af2020_fia_daily_climatology(plot_dt0.strftime("%Y-%m-%d"), plot_dtN.strftime("%Y-%m-%d"))
            else:
                obs_da = self.obs.subset_af2020_fia_daily(plot_dt0.strftime("%Y-%m-%d"), plot_dtN.strftime("%Y-%m-%d"))
            obs_df = pd.DataFrame({"time"  : pd.to_datetime(obs_da["time"].values),
                                   "value" : np.asarray(obs_da.values, dtype=float)}).dropna()
            return None if obs_df.empty else obs_df
        # ------------------------------------------------------------
        # SIA -> NSIDC (north or south)
        # ------------------------------------------------------------
        if var == "SIA":
            nsidc = self.obs.compute_nsidc_sia_sie(start_date = plot_dt0.strftime("%Y-%m-%d"),
                                                   end_date   = plot_dtN.strftime("%Y-%m-%d"),
                                                   hemisphere = hemi)
            if "SIA" not in nsidc:
                raise KeyError("NSIDC observational dataset did not return 'SIA'.")
            obs_da = nsidc["SIA"]
            obs_df = pd.DataFrame({"time" : pd.to_datetime(obs_da["time"].values),
                                   "value": np.asarray(obs_da.values, dtype=float)}).dropna()
            return None if obs_df.empty else obs_df
        # ------------------------------------------------------------
        # Other variables -> no observational overlay implemented
        # ------------------------------------------------------------
        self.logger.info("No observational overlay implemented for variable=%s; plotting model only.", var)
        return None

    def _load_static_lonlat(self, sim_name: str | None = None) -> xr.Dataset:
        """
        Load lon/lat fields from the universal CICE static-coordinate store.

        This avoids relying on simulation-local zarr/iceh_static.zarr. The
        universal default is:

            ~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
        """
        from shuga.grid.cice import CICEGridwork
        gridwork = CICEGridwork(pth_cfg = self.pth_cfg, logger = self.logger)
        ds = gridwork.load_cice_static(variables   = ["TLON", "TLAT", "ULON", "ULAT", "ELON", "ELAT", "NLON", "NLAT"],
                                       require     = ("TLON", "TLAT"),
                                       chunks      = self.chunks,
                                       add_aliases = True)
        wanted = [v for v in ("TLON", "TLAT", "ULON", "ULAT", "ELON", "ELAT", "NLON", "NLAT") if v in ds]
        if not wanted:
            raise KeyError("No recognised lon/lat variables found in universal CICE static store. "
                           "Expected one or more of TLON, TLAT, ULON, ULAT, ELON, ELAT, NLON, NLAT.")
        return ds[wanted]

    # ------------------------------------------------------------
    # primary APIs
    # ------------------------------------------------------------
    def plot_sia_daily_climatology_envelope(self, df: pd.DataFrame, out_file: str | Path, *,
                                            start_date: str,
                                            end_date: str,
                                            envelope: str="minmax",
                                            smooth_days: int=1,
                                            order: Sequence[str] | None=None,
                                            styles: Mapping[str, SIAStyle] | None=None,
                                            colors: Mapping[str,str] | None=None,
                                            y_min: float=0.0,
                                            y_max: float=20.0,
                                            title: str | None=None,
                                            projection: str="X20c/14c",
                                            legend_position: str="JTL+jTL+o0.2c",
                                            write_csv: bool=True):
        pygmt        = self._require_pygmt();
        out_file     = Path(out_file); out_file.parent.mkdir(parents=True, exist_ok=True)
        clim         = self.daily_climatology_envelopes(df, start_date=start_date, end_date=end_date, envelope=envelope, smooth_days=smooth_days)
        series_order = [n for n in (list(order) if order is not None else list(df.columns)) if n in clim]
        style_map    = dict(DEFAULT_SIA_STYLES)
        if styles: style_map.update(styles)
        if colors:
            for name, color in colors.items():
                if name in {"NSIDC", "OSI-SAF-450", "ORAS"}: continue
                style_map[name] = SIAStyle(f"2.4p,{color}", f"{color}@80")
        month_starts = np.array([1,32,60,91,121,152,182,213,244,274,305,335,366]);
        month_mids   = .5*(month_starts[:-1]+month_starts[1:]-1)
        frame        = ["WSen","x0","ya2f1+lSea Ice Area (10@+6@+ km@+2@+)"]
        if title: frame[0] += f"+t{title}"
        with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica", FONT_LABEL="14p,Helvetica", MAP_FRAME_PEN="1p,black"):
            fig = pygmt.Figure();
            fig.basemap(region = [1,365,y_min,y_max], projection = projection, frame = frame)
            for x in month_starts[1:-1]:
                fig.plot(x = [x,x], y = [y_min,y_max], pen = "0.35p,gray55")
            for y in np.arange(5.,y_max+.001,5.):
                fig.plot(x = [1,365], y = [y,y], pen = "0.35p,gray55")
            for name in series_order:
                t     = clim[name];
                st    = style_map.get(name,SIAStyle("1.6p,black","gray80@80"));
                valid = np.isfinite(t.lower)&np.isfinite(t.upper)
                x     = t.loc[valid,"doy"].to_numpy();
                lo    = t.loc[valid,"lower"].to_numpy();
                hi    = t.loc[valid,"upper"].to_numpy()
                if len(x):
                    fig.plot(x = np.r_[x,x[::-1]], y = np.r_[hi,lo[::-1]], close = True, fill = st.fill, pen = "0p")
            for name in series_order:
                self.logger.info(f"adding {name} to figure")
                t      = clim[name];
                st     = style_map.get(name,SIAStyle("1.6p,black","gray80@80"));
                valid  = np.isfinite(t["mean"])
                x_mean = np.ascontiguousarray(t.loc[valid, "doy"].to_numpy(dtype = np.float64))
                y_mean = np.ascontiguousarray(t.loc[valid, "mean"].to_numpy(dtype = np.float64))
                fig.plot(x = x_mean, y = y_mean, pen = st.pen, label = name)
                #fig.plot(x = t.loc[valid,"doy"], y = t.loc[valid,"mean"], pen = st.pen, label = name)
            fig.text(x = month_mids, y = np.full(12, y_min-.55), text = list("JFMAMJJASOND"), font = "12p,Helvetica", justify = "TC", no_clip = True)
            fig.legend(position = legend_position, box = "+gwhite+p0.7p,black");
            fig.savefig(out_file, dpi = 600)
        if write_csv:
            rows = []
            for name in series_order:
                t = clim[name].copy();
                t.insert(0,"series",name);
                rows.append(t)
            pd.concat(rows, ignore_index = True).to_csv(out_file.with_suffix(".csv"), index = False)
        return clim

    def plot_sit_monthly_climatology_envelope(self, series: Mapping[str,pd.Series], output: str | Path, *,
                                              envelope: str = "p10-p90",
                                              order: Sequence[str] | None = None,
                                              colors: Mapping[str,str] | None = None,
                                              y_min: float = 0.0,
                                              y_max: float = 4.0,
                                              title: str | None = None,
                                              write_csv: bool = True) -> dict[str,pd.DataFrame]:
        pygmt      = self._require_pygmt()
        stats      = {name:self.sit_monthly_climatology(s,envelope=envelope) for name,s in series.items()}
        plot_order = [n for n in (list(order) if order else list(stats)) if n in stats]
        style_map  = dict(DEFAULT_SIT_STYLES)
        for name,color in dict(colors or {}).items():
            if name not in style_map:
                style_map[name] = SITStyle(f"2.2p,{color}", f"{color}@82")
        x_month = np.arange(1,13,dtype=float)
        frame   = ["WSen","x0","ya0.5f0.25+lSea Ice Thickness (m)"]
        if title:
            frame[0] += f"+t{title}"
        with pygmt.config(FONT_ANNOT_PRIMARY = "12p,Helvetica",
                          FONT_LABEL         = "14p,Helvetica",
                          MAP_FRAME_PEN      = "1p,black"):
            fig = pygmt.Figure()
            fig.basemap(region=[0.5,12.5,y_min,y_max],
                        projection="X24c/16c", frame=frame)
            for x in np.arange(1.5,12.0,1.0):
                fig.plot(x=[x,x], y=[y_min,y_max], pen="0.35p,gray55")
            for y in np.arange(0.5,y_max+.001,0.5):
                fig.plot(x=[0.5,12.5], y=[y,y], pen="0.35p,gray55")
            for name in plot_order:
                t     = stats[name]
                st    = style_map.get(name,SITStyle("1.8p,black","gray80@82"))
                valid = np.isfinite(t["lower"]) & np.isfinite(t["upper"])
                x     = x_month[valid]
                lo    = t.loc[valid,"lower"].to_numpy(float)
                hi    = t.loc[valid,"upper"].to_numpy(float)
                if len(x):
                    fig.plot(x=np.r_[x,x[::-1]], y=np.r_[hi,lo[::-1]],
                             close=True, fill=st.fill, pen="0p")
            for name in plot_order:
                t     = stats[name]
                st    = style_map.get(name,SITStyle("1.8p,black","gray80@82"))
                valid = np.isfinite(t["mean"])
                x     = np.ascontiguousarray(x_month[valid], dtype=np.float64)
                y     = np.ascontiguousarray(t.loc[valid,"mean"].to_numpy(float), dtype=np.float64)
                if len(x):
                    fig.plot(x=x,y=y,pen=st.pen,label=name)
            fig.text(x = x_month, y = np.full(12,y_min-.12), text = list("JFMAMJJASOND"), font = "12p,Helvetica", justify = "TC", no_clip = True)
            fig.legend(position = "jTR+jTR+o0.2c", box = "+gwhite+p0.8p")
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output,dpi=600)
        if write_csv:
            rows = []
            for name in plot_order:
                t = stats[name].copy()
                t.insert(0,"series",name)
                rows.append(t)
            pd.concat(rows,ignore_index=True).to_csv(Path(output).with_suffix(".csv"),index=False)
        return stats

    def plot_sit_daily_climatology_envelope(self, df: pd.DataFrame, output: str | Path, *,
                                            start_date: str,
                                            end_date: str,
                                            envelope: str = "minmax",
                                            smooth_days: int = 7,
                                            order: Sequence[str] | None = None,
                                            colors: Mapping[str, str] | None = None,
                                            y_min: float = 0.0,
                                            y_max: float = 4.0,
                                            title: str | None = None,
                                            write_csv: bool = True) -> Path:
        pygmt = self._require_pygmt()
        frame = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].copy()
        if frame.empty:
            raise ValueError("No SIT data in requested period.")
        stats = {}
        for name in frame.columns:
            s = frame[name].dropna()
            if s.empty:
                continue
            tmp = pd.DataFrame({"value": s})
            tmp = tmp.loc[~((tmp.index.month == 2) & (tmp.index.day == 29))]
            tmp["doy"] = tmp.index.dayofyear
            grp = tmp.groupby("doy")["value"]
            mean = grp.mean()
            if envelope == "minmax":
                lower, upper = grp.min(), grp.max()
            elif envelope == "std":
                std = grp.std()
                lower, upper = mean - std, mean + std
            else:
                lower, upper = grp.quantile(0.10), grp.quantile(0.90)
            t = pd.DataFrame({"mean":mean, "lower":lower, "upper":upper})
            if smooth_days > 1:
                t = t.rolling(smooth_days, center=True, min_periods=1).mean()
            t["doy"] = t.index.astype(float)
            stats[name] = t
        fig = pygmt.Figure()
        fig.basemap(region = [1,365,y_min,y_max], projection = "X24c/16c", frame = ["WSen","xa30g30","ya0.5g0.5","y+lSea Ice Thickness (m)"])
        colors     = dict(colors or {})
        plot_order = list(order) if order else list(stats)
        for name in plot_order:
            if name not in stats:
                continue
            t     = stats[name]
            valid = np.isfinite(t["mean"].values)
            x     = np.ascontiguousarray(t.loc[valid,"doy"].to_numpy(dtype=np.float64))
            y     = np.ascontiguousarray(t.loc[valid,"mean"].to_numpy(dtype=np.float64))
            lo    = np.ascontiguousarray(t.loc[valid,"lower"].to_numpy(dtype=np.float64))
            hi    = np.ascontiguousarray(t.loc[valid,"upper"].to_numpy(dtype=np.float64))
            if len(x) == 0:
                continue
            if name in DEFAULT_SIT_STYLES:
                st = DEFAULT_SIT_STYLES[name]
            else:
                colour = colors.get(name, "black")
                st = SITStyle(f"2.2p,{colour}", f"{colour}@82")
            fig.plot(x = np.ascontiguousarray(np.concatenate([x, x[::-1]])),
                     y = np.ascontiguousarray(np.concatenate([lo, hi[::-1]])),
                     close=True, fill=st.fill, pen="0.2p,gray60")
            fig.plot(x = x, y = y, pen = st.pen, label = name)
        fig.legend(position = "jTR+jTR+o0.2c", box = "+gwhite+p0.8p")
        output = Path(output)
        output.parent.mkdir(parents = True, exist_ok = True)
        fig.savefig(output)
        if write_csv:
            rows = []
            for name, t in stats.items():
                tt = t.copy()
                tt.insert(0, "series", name)
                rows.append(tt.reset_index(drop=True))
            pd.concat(rows, ignore_index=True).to_csv(output.with_suffix(".csv"), index=False)
        return output

    def pygmt_da_prep(self, da: xr.DataArray,
                      lon       : xr.DataArray | None    = None,
                      lat       : xr.DataArray | None    = None, *,
                      mask_zero : bool                   = False,
                      region    : Sequence[float] | None = None) -> pd.DataFrame:
        if lon is None or lat is None:
            if "lon" in da.coords and "lat" in da.coords:
                lon, lat = da["lon"], da["lat"]
            else:
                raise ValueError("lon/lat must be supplied when not present on the DataArray.")
        lon_da = self._lon_to_180(lon)
        work   = da
        if region is not None:
            mask = self._region_mask(lon_da, lat, region)
            work = work.where(mask)
        if mask_zero:
            work = work.where(np.abs(work) > 0)
        lon_flat = lon_da.values.ravel()
        lat_flat = lat.values.ravel()
        z_flat   = work.values.ravel()
        good     = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(z_flat)
        return pd.DataFrame({"lon": lon_flat[good], "lat": lat_flat[good], "z": z_flat[good]})

    def pygmt_base_layer(self, fig, region: Sequence[float], projection: str, *,
                         title      : str | None = None,
                         shorelines : str | None = None,
                         land       : str | None = None,
                         water      : str | None = None):
        frame = ["af"]
        if title:
            frame.append(f'+t{title}')
        fig.basemap(region=list(region), projection=projection, frame=frame)
        fig.coast(shorelines = shorelines or self.plt_cfg.shorelines,
                  land       = land       or self.plt_cfg.land,
                  water      = water      or self.plt_cfg.water)

    def plot_fip(self,
                 method            : str = "binary-days", *,
                 source            : str = "sim",
                 field             : str = "FIP",
                 sim_name          : str | None = None,
                 grid_type         : str | None = None,
                 af2020_store      : str | Path | None = None,
                 dataset           : xr.Dataset | xr.DataArray | str | Path | None = None,
                 af2020_start      : str | None = None,
                 af2020_end        : str | None = None,
                 FIP_plot_thresh   : float | None = 0.05,
                 output_path       : str | Path | None = None,
                 output_root       : str | Path | None = None,
                 region_name       : str | None = None,
                 region            : Sequence[float] | None = None,
                 regions           : Mapping[str, Sequence[float]] | None = None,
                 fig_size          : float | None = None,
                 cmap              : str | Path | None = None,
                 series            : Sequence[float] | None = None,
                 title             : str | None = None,
                 shorelines        : str | None = None,
                 land              : str | None = None,
                 water             : str | None = None,
                 grid_style        : str | None = None,
                 colorbar_position : str | None = "JMB+w8c/0.4c+o0.8c/0c",
                 colorbar_frame    : Sequence[str] | None = None,
                 categorical_labels: Sequence[str] = ("agreement", "model-dominant", "obs-dominant"),
                 categorical_colors: Sequence[str] = ("#FDAE61", "#2CA25F", "#2171B5"),
                 show              : bool = False) -> str | dict[str, str]:
        """
        Plot precomputed fast-ice persistence products.

        This method deliberately does *not* recompute simulation FIP. Simulation
        FIP is loaded from the shuga metrics store:

            [SIM_NAME]/zarr/SH/ispd_thresh_<VAL>/FI/[GRID]/[BIN-DAYS]/mets.zarr/FIP

        Supported sources
        -----------------
        source="sim"
            Plot simulation FIP from the metrics store. Dates are not accepted,
            because the stored metric already represents the analysis window used
            when metrics were generated.

        source="af2020"
            Plot AF2020 FIP from a persistent AF2020 common-grid zarr store.
            If af2020_start/af2020_end are supplied, FIP is computed from the
            stored native-15-day AF2020 FIC variable over that period. Otherwise
            the stored FIP variable is used.

        source="dataset"
            Plot an explicitly supplied dataset/dataarray/path. Use this for
            FIP['diff'] or FIP['diff_cat'] generated by FIP_differencing.py.

        field
        -----
        "FIP", "diff", or "diff_cat" are the intended values, but any variable
        present in the supplied dataset can be plotted.
        """
        pygmt       = self._require_pygmt()
        norm        = normalize_method(method)
        source_l    = str(source).strip().lower()
        field_l     = str(field).strip()
        target_sim  = sim_name or self.run_cfg.sim_name
        target_grid = grid_type or self.cls_cfg.grid_type
        lon         = lat = None
        label       = target_sim
        if source_l == "sim":
            if af2020_start is not None or af2020_end is not None:
                raise ValueError("Do not pass af2020_start/af2020_end when source='sim'. Simulation FIP is loaded from the precomputed metrics store.")
            ds = load_metrics(run_cfg        = self.run_cfg,
                              cls_cfg        = self.cls_cfg,
                              met_cfg        = self.met_cfg,
                              plt_cfg        = self.plt_cfg,
                              obs_cfg        = self.obs_cfg,
                              pth_cfg        = self.pth_cfg,
                              classification = norm,
                              sim_name       = target_sim,
                              variables      = ["FIP"],
                              hemisphere     = self.run_cfg.hemisphere,
                              grid_type      = target_grid,
                              chunks         = self.chunks)
            if "FIP" not in ds:
                raise KeyError(f"Could not find FIP in metrics store for {target_sim}/{norm}/{target_grid}.")
            da        = ds["FIP"].squeeze(drop = True) / 100
            static_ds = self._load_static_lonlat(sim_name = target_sim)
            lon, lat  = self._detect_lonlat(static_ds)
            label     = target_sim
        elif source_l == "af2020":
            if af2020_store is None:
                raise ValueError("source='af2020' requires af2020_store=...")
            ods = xr.open_zarr(Path(af2020_store).expanduser(), consolidated = False, chunks=self.chunks)
            if af2020_start is not None or af2020_end is not None:
                if "FIC" not in ods:
                    raise KeyError("AF2020 store does not contain FIC; cannot compute period-specific AF2020 FIP.")
                da = ods["FIC"].sel(time = slice(af2020_start, af2020_end)).mean("time", skipna = True).rename("FIP")
                da.attrs.update(long_name  = "AF2020 fast ice persistence",
                                units      = "1",
                                time_start = str(pd.Timestamp(af2020_start).date()) if af2020_start else str(pd.Timestamp(ods.time.values[0]).date()),
                                time_end   = str(pd.Timestamp(af2020_end).date()) if af2020_end else str(pd.Timestamp(ods.time.values[-1]).date()))
            else:
                if "FIP" not in ods:
                    raise KeyError("AF2020 store does not contain FIP.")
                da = ods["FIP"].squeeze(drop=True)
            lon   = ods["lon"] if "lon" in ods else da.coords.get("lon")
            lat   = ods["lat"] if "lat" in ods else da.coords.get("lat")
            label = "AF2020"
        elif source_l == "dataset":
            if dataset is None:
                raise ValueError("source='dataset' requires dataset=...")
            if isinstance(dataset, xr.DataArray):
                da = dataset
            else:
                if isinstance(dataset, xr.Dataset):
                    ds = dataset
                else:
                    p  = Path(dataset).expanduser()
                    ds = xr.open_zarr(p, consolidated = False, chunks = self.chunks) if p.suffix == ".zarr" or p.is_dir() else xr.open_dataset(p, chunks = self.chunks)
                if field_l not in ds:
                    raise KeyError(f"{field_l!r} not found in supplied dataset. Available variables: {list(ds.data_vars)}")
                da  = ds[field_l]
                lon = ds["lon"] if "lon" in ds else da.coords.get("lon")
                lat = ds["lat"] if "lat" in ds else da.coords.get("lat")
            label = field_l
        else:
            raise ValueError("source must be one of: 'sim', 'af2020', 'dataset'.")
        if "time" in da.dims:
            if da.sizes.get("time", 0) != 1:
                raise ValueError("plot_fip expects a 2-D field. Select a single time or compute persistence before plotting.")
            da = da.isel(time=0, drop=True)
        if lon is None or lat is None:
            if "lon" in da.coords and "lat" in da.coords:
                lon, lat = da["lon"], da["lat"]
            else:
                raise ValueError("Could not determine lon/lat for plot_fip.")
        is_diff = field_l.lower() in {"diff", "fip_diff"}
        is_cat  = field_l.lower() in {"diff_cat", "fip_diff_cat"}
        if FIP_plot_thresh is not None and not is_diff and not is_cat:
            da_plot = da.where(da > float(FIP_plot_thresh))
        else:
            da_plot = da
        if series is None:
            if is_diff:
                series = [-1.0, 1.0, 0.05]
            elif is_cat:
                series = [0, 2, 1]
            else:
                series = [0.0, 1.0, 0.01]
        if cmap is None and self.plt_cfg is not None:
            if is_diff:
                cmap = str(Path(self.pth_cfg.fip_diff_con_cmap).expanduser())
            elif is_cat:
                cmap = str(Path(self.pth_cfg.fip_diff_cat_cmap).expanduser())
            else:
                cmap = str(Path(self.pth_cfg.fip_cmap).expanduser())
        region_map = self._resolve_regions(region_name = region_name, region = region, regions = regions)
        saved: dict[str, str] = {}
        for name, reg in region_map.items():
            data = self.pygmt_da_prep(da_plot, lon = lon, lat = lat, mask_zero = False, region = reg)
            if output_path and len(region_map) == 1:
                path = Path(output_path).expanduser()
            else:
                if output_root is not None:
                    root = Path(output_root).expanduser()
                else:
                    root = self.pth_cfg.figure_root() / "FIP"
                safe_label = str(label).replace("/", "_")
                safe_field = field_l.replace("/", "_")
                path       = root / safe_label / name / f"{safe_label}_{safe_field}_{norm}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig  = pygmt.Figure()
            proj = self.projection_from_region(reg, fig_size=fig_size or self.plt_cfg.fip_fig_size)
            if is_cat:
                # CPT has already been written above.
                pass
            else:
                pygmt.makecpt(cmap = str(cmap), series = list(series), continuous = True)
            plot_title = title or f"{label} {name} {field_l}"
            self.pygmt_base_layer(fig, reg, proj, title=plot_title, shorelines=shorelines, land=land, water=water)
            fig.plot(x = data["lon"], y = data["lat"], style = grid_style or self.plt_cfg.grid_style, fill = data["z"], cmap = True if not is_cat else str(cmap))
            fig.coast(region = reg, projection = proj, shorelines = shorelines or self.plt_cfg.shorelines)
            if colorbar_position:
                if colorbar_frame is not None:
                    frame = list(colorbar_frame)
                elif is_diff:
                    frame = ['xaf+lFIP difference (model - AF2020)']
                elif is_cat:
                    frame = ['+lFIP difference category']
                else:
                    frame = ['xaf+lpersistence']
                fig.colorbar(position = colorbar_position, frame = frame, cmap = str(cmap) if is_cat else None)
            fig.savefig(path)
            if show:
                fig.show()
            saved[name] = str(path)
        return next(iter(saved.values())) if len(saved) == 1 else saved

    def plot_timeseries(self, variable: str, method: str,
                        region      : str        = "total",
                        output_path : str | None = None,
                        add_f2020   : bool       = True,
                        f2020_mode  : str        = "climatology",
                        title       : str | None = None,
                        obs_pen     : str        = "1p,black",
                        model_pen   : str        = "1p,blue",
                        show        : bool       = False,
                        sim_name    : str | None = None,
                        dt0_str     : str | None = None,
                        dtN_str     : str | None = None) -> str:
        pygmt = self._require_pygmt()
        norm  = normalize_method(method)
        var   = variable.upper()
        sim   = sim_name or self.run_cfg.sim_name
        dt0   = dt0_str  or self.run_cfg.start_date
        dtN   = dtN_str  or self.run_cfg.end_date
        hemi  = self.run_cfg.hemisphere
        # Use overridden run context when sim_name changes
        run_cfg = replace(self.run_cfg, sim_name=sim)
        self.logger.info(f"loading metrics for {sim} over period {dt0} -- {dtN} for hemisphere {hemi} ...")
        ds  = load_metrics(run_cfg        = run_cfg,
                           cls_cfg        = self.cls_cfg,
                           met_cfg        = self.met_cfg,
                           plt_cfg        = self.plt_cfg,
                           obs_cfg        = self.obs_cfg,
                           pth_cfg        = self.pth_cfg,
                           classification = norm,
                           dt0_str        = dt0,
                           dtN_str        = dtN,
                           hemisphere     = hemi,
                           chunks         = self.chunks)
        # ------------------------------------------------------------
        # Select requested series
        # ------------------------------------------------------------
        if region.lower() == "total":
            if var not in ds:
                raise KeyError(f"Variable {var!r} not found in metrics dataset.")
            series     = ds[var]
            region_key = "total"
        else:
            key = f"{var}_by_region"
            if key not in ds:
                raise KeyError(f"Regional metric {key!r} not found in metrics dataset.")
            series = ds[key].sel(region=region)
            region_key = region
        model = pd.DataFrame({"time" : pd.to_datetime(series["time"].values),
                              "value": np.asarray(series.values, dtype=float)}).dropna()
        if model.empty:
            raise ValueError("No model time series available for plotting.")
        # Respect explicit override dates if supplied; otherwise use the actual model range
        plot_dt0 = pd.Timestamp(dt0) if dt0_str is not None or sim_name is not None else model["time"].min()
        plot_dtN = pd.Timestamp(dtN) if dtN_str is not None or sim_name is not None else model["time"].max()
        # Crop model frame again defensively in case ds carried a wider range
        model = model[(model["time"] >= plot_dt0) & (model["time"] <= plot_dtN)].copy()
        if model.empty:
            raise ValueError("Model time series is empty after applying requested date window.")
        # ------------------------------------------------------------
        # Observational overlay
        # Keep add_f2020 name for backward compatibility, but use it as
        # 'add observation where available'
        # ------------------------------------------------------------
        self.logger.info(f"constructing observational time-series")
        obs_df = self._build_obs_timeseries(var,
                                            plot_dt0   = plot_dt0,
                                            plot_dtN   = plot_dtN,
                                            hemisphere = hemi,
                                            add_obs    = add_f2020,
                                            f2020_mode = f2020_mode)
        # ------------------------------------------------------------
        # Y range
        # ------------------------------------------------------------
        yvals = [model["value"].to_numpy()]
        if obs_df is not None and not obs_df.empty:
            yvals.append(obs_df["value"].to_numpy())
        ymin = float(min(np.nanmin(v) for v in yvals))
        ymax = float(max(np.nanmax(v) for v in yvals))
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        else:
            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad
        # ------------------------------------------------------------
        # Output path
        # Prefer existing helper where possible; fall back to explicit path
        # if sim/date overrides are used and the helper is fixed to self.run.
        # ------------------------------------------------------------
        if not output_path:
            P_png = self.pth_cfg.timeseries_plot_path(var, norm, region_key)
        else:
            P_png = output_path
        # ------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------
        self.logger.info(f"creating figure")
        fig       = pygmt.Figure()
        title_str = title or (f"{sim} {var}" + (f" {region_key}" if region_key != "total" else "") + f" ({norm})")
        y_label   = var
        # GMT/PyGMT is much safer with ISO strings than pandas.Timestamp
        x0 = pd.Timestamp(plot_dt0).strftime("%Y-%m-%dT%H:%M:%S")
        x1 = pd.Timestamp(plot_dtN).strftime("%Y-%m-%dT%H:%M:%S")
        model_plot          = model.copy()
        model_plot["time"]  = pd.to_datetime(model_plot["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        model_plot["value"] = model_plot["value"].astype("float64")
        if obs_df is not None and not obs_df.empty:
            obs_plot          = obs_df.copy()
            obs_plot["time"]  = pd.to_datetime(obs_plot["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
            obs_plot["value"] = obs_plot["value"].astype("float64")
        else:
            obs_plot = None
        self.logger.info(f"     basemap")
        fig.basemap(region=[x0, x1, ymin, ymax], projection="X16c/6c", frame=["WSen+t" + title_str, "xaf", f'yaf+l"{y_label}"'])
        if obs_plot is not None and not obs_plot.empty:
            self.logger.info(f"     observations")
            self.logger.info(obs_plot.head())
            fig.plot(x=obs_plot["time"].to_numpy(), y=obs_plot["value"].to_numpy(), pen=obs_pen)
        self.logger.info(f"     model results")
        fig.plot(x=model_plot["time"].to_numpy(), y=model_plot["value"].to_numpy(), pen=model_pen)
        self.logger.info(f"     saving")
        fig.savefig(P_png)
        if show:
            fig.show()
        return str(P_png)

    def plot_timeseries_multi(self, variable: str, method: str, simulations,
                              region         : str                    = "total",
                              output_path    : str | None             = None,
                              add_f2020      : bool                   = True,
                              f2020_mode     : str                    = "climatology",
                              title          : str | None             = None,
                              obs_pen        : str                    = "1.2p,black",
                              default_pens   : tuple[str, ...]        = ("1.2p,blue",
                                                                         "1.2p,red",
                                                                         "1.2p,green3",
                                                                         "1.2p,orange",
                                                                         "1.2p,purple",
                                                                         "1.2p,brown",
                                                                         "1.2p,cyan4",
                                                                         "1.2p,magenta"),
                              legend_position : str                   = "JTR+jTR+o0.2c",
                              legend_placement: str                   = "BR", # TL, BL, TC, etc.
                              legend_width    : str                   = "5cm", # if none then one column
                              show            : bool                  = False, *,
                              dt0_str         : str | None            = None,
                              dtN_str         : str | None            = None,
                              grid_type       : str | None            = None,
                              grid_type_map   : dict[str, str] | None = None) -> str:
        pygmt            = self._require_pygmt()
        var              = variable.upper()
        norm             = normalize_method(method)
        req_dt0, req_dtN = self._resolve_plot_window(dt0_str=dt0_str, dtN_str=dtN_str)
        series_list      = []
        yvals            = []
        for i, spec in enumerate(simulations):
            if isinstance(spec, str):
                spec = {"sim_name": spec}
            sim_name      = spec["sim_name"]
            label         = spec.get("label", sim_name)
            pen           = spec.get("pen", default_pens[i % len(default_pens)])
            sim_grid_type = spec.get("grid_type") or (grid_type_map or {}).get(sim_name, grid_type)
            ds, resolved  = load_metrics(run_cfg             = self.run_cfg,
                                         cls_cfg        = self.cls_cfg,
                                         met_cfg         = self.met_cfg,
                                         plt_cfg        = self.plt_cfg,
                                         obs_cfg    = self.obs_cfg,
                                         pth_cfg           = self.pth_cfg,
                                         classification  = norm,
                                         sim_name        = sim_name,
                                         dt0_str         = dt0_str,
                                         dtN_str         = dtN_str,
                                         hemisphere      = self.run_cfg.hemisphere,
                                         grid_type       = sim_grid_type,
                                         chunks          = self.chunks,
                                         return_resolved = True)

            if region.lower() == "total":
                da = ds[var]
            else:
                da = ds[f"{var}_by_region"].sel(region=region)
            df = pd.DataFrame({"time": pd.to_datetime(da["time"].values), "value": np.asarray(da.values, dtype=float)}).dropna()
            if req_dt0 is not None:
                df = df[df["time"] >= req_dt0]
            if req_dtN is not None:
                df = df[df["time"] <= req_dtN]
            if df.empty:
                continue
            if self.logger is not None:
                self.logger.info("Loaded %s for %s using grid_type=%s", var, sim_name, resolved["grid_type"])
            series_list.append({"sim_name": sim_name, "label": label, "pen": pen, "df": df, "grid_type": resolved["grid_type"]})
            yvals.append(df["value"].to_numpy())
        if not series_list:
            raise ValueError("No model time series available for plotting.")
        model_xmin = min(s["df"]["time"].min() for s in series_list)
        model_xmax = max(s["df"]["time"].max() for s in series_list)
        plot_dt0   = req_dt0 if req_dt0 is not None else model_xmin
        plot_dtN   = req_dtN if req_dtN is not None else model_xmax
        self._validate_f2020_window(plot_dt0, plot_dtN, add_f2020=add_f2020, f2020_mode=f2020_mode)
        obs_df = None
        if add_f2020:
            if var != "FIA":
                raise ValueError("AF2020 overlay is currently implemented for FIA only.")
            mode = str(f2020_mode).strip().lower()
            if mode not in {"climatology", "overlap"}:
                raise ValueError(f"f2020_mode must be 'climatology' or 'overlap', got {f2020_mode!r}")

            if mode == "climatology":
                obs_da = self.obs.repeat_af2020_fia_daily_climatology(plot_dt0.strftime("%Y-%m-%d"), plot_dtN.strftime("%Y-%m-%d"))
            else:
                obs_da = self.obs.subset_af2020_fia_daily(plot_dt0.strftime("%Y-%m-%d"), plot_dtN.strftime("%Y-%m-%d"))

            obs_df = pd.DataFrame({"time": pd.to_datetime(obs_da["time"].values), "value": np.asarray(obs_da.values, dtype=float)}).dropna()
            if not obs_df.empty:
                yvals.append(obs_df["value"].to_numpy())
            else:
                obs_df = None
        xmin = plot_dt0
        xmax = plot_dtN
        ymin = float(min(np.nanmin(v) for v in yvals))
        ymax = float(max(np.nanmax(v) for v in yvals))
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        else:
            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad
        region_key = region if region.lower() != "total" else "total"
        if output_path is None:
            path = self.pth_cfg.multi_timeseries_plot_path(variable    = var,
                                                         method      = norm,
                                                         simulations = series_list,
                                                         region      = region_key,
                                                         dt0_str     = plot_dt0.strftime("%Y-%m-%d"),
                                                         dtN_str     = plot_dtN.strftime("%Y-%m-%d"))
        else:
            path = Path(output_path)
        self.logger.info(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig       = pygmt.Figure()
        title_str = title or (f"{var} {region_key} ({norm})" if region_key != "total" else f"{var} ({norm})")
        fig.basemap(region=[xmin, xmax, ymin, ymax], projection="X16c/6c", frame=["WSen", "xaf", f'yaf+l{var}'])#, f'+t{title_str}'],
        if obs_df is not None:
            fig.plot(x=obs_df["time"], y=obs_df["value"], pen=obs_pen, label="F2020")
        for srec in series_list:
            fig.plot(x=srec["df"]["time"], y=srec["df"]["value"], pen=srec["pen"], label=srec["label"])
        fig.legend(position=legend_position, box="+gwhite+p0.5p") #legend_position old: "JTR+jTR+o0.2c"
        fig.savefig(path)
        if show:
            fig.show()
        return str(path)

    def plot_var_split_hemisphere(self, date_str: str, variable: str, *,
                                  method         : str | None = None,
                                  add_nsidc_south: bool = True,
                                  add_nsidc_north: bool = False,
                                  output_path    : str | None = None,
                                  fig_size       : float | None = None,
                                  cmap           : str = "viridis",
                                  series         : Sequence[float] | None = None,
                                  title          : str | None = None,
                                  grid_style     : str | None = None) -> str:
        pygmt = self._require_pygmt()
        da, lon, lat = self._load_field(variable, date_str=date_str, method=method)
        path = Path(output_path).expanduser() if output_path else self.pth_cfg.split_hemisphere_plot_path(variable, date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        south = self.pygmt_da_prep(da.where(lat < 0), lon=lon, lat=lat, mask_zero=False)
        north = self.pygmt_da_prep(da.where(lat > 0), lon=lon, lat=lat, mask_zero=False)
        fig = pygmt.Figure()
        with pygmt.config(MAP_FRAME_TYPE="plain"):
            fig.subplot(nrows=1, ncols=2, figsize=(f"{2*(fig_size or self.plt_cfg.split_fig_size)}c", f"{fig_size or self.plt_cfg.split_fig_size}c"), margins=["0.3c", "0.3c"])
            with fig.set_panel(panel=0):
                reg = [-180, 180, -90, -45]
                proj = "S0/-90/12c"
                self.pygmt_base_layer(fig, reg, proj, title=(title or f"{variable} {date_str}") + " (SH)")
                pygmt.makecpt(cmap=cmap, series=series, continuous=True)
                fig.plot(x=south["lon"], y=south["lat"], style=grid_style or self.plt_cfg.grid_style, fill=south["z"], cmap=True)
                if add_nsidc_south:
                    for line in self._nsidc_contours(date_str, "south"):
                        fig.plot(x=line[:, 0], y=line[:, 1], pen=self.plt_cfg.nsidc_pen)
            with fig.set_panel(panel=1):
                reg = [-180, 180, 45, 90]
                proj = "S0/90/12c"
                self.pygmt_base_layer(fig, reg, proj, title=(title or f"{variable} {date_str}") + " (NH)")
                pygmt.makecpt(cmap=cmap, series=series, continuous=True)
                fig.plot(x=north["lon"], y=north["lat"], style=grid_style or self.plt_cfg.grid_style, fill=north["z"], cmap=True)
                if add_nsidc_north:
                    for line in self._nsidc_contours(date_str, "north"):
                        fig.plot(x=line[:, 0], y=line[:, 1], pen=self.plt_cfg.nsidc_pen)
        fig.savefig(path)
        return str(path)

    def plot_var_by_region(self, date_str: str, variable: str, *,
                           method     : str | None = None,
                           region_name: str | None = None,
                           region     : Sequence[float] | None = None,
                           regions    : Mapping[str, Sequence[float]] | None = None,
                           output_path: str | None = None,
                           output_root: str | Path | None = None,
                           fig_size   : float | None = None,
                           cmap       : str = "viridis",
                           series     : Sequence[float] | None = None,
                           title      : str | None = None,
                           grid_style : str | None = None,
                           add_nsidc  : bool | None = None) -> str | dict[str, str]:
        pygmt = self._require_pygmt()
        da, lon, lat = self._load_field(variable, date_str=date_str, method=method)
        region_map = self._resolve_regions(region_name=region_name, region=region, regions=regions)
        out: dict[str, str] = {}
        for name, reg in region_map.items():
            data = self.pygmt_da_prep(da, lon=lon, lat=lat, region=reg)
            path = Path(output_path).expanduser() if output_path and len(region_map) == 1 else (Path(output_root).expanduser() / self.run_cfg.sim_name / name / variable / f"{date_str}.png"
                                                                                                if output_root is not None else self.pth_cfg.regional_var_plot_path(variable, date_str, name))
            path.parent.mkdir(parents=True, exist_ok=True)
            fig  = pygmt.Figure()
            proj = self.projection_from_region(reg, fig_size=fig_size or self.plt_cfg.region_fig_size)
            pygmt.makecpt(cmap=cmap, series=series, continuous=True)
            self.pygmt_base_layer(fig, reg, proj, title=title or f"{self.run_cfg.sim_name} {name} {variable} {date_str}")
            fig.plot(x=data["lon"], y=data["lat"], style=grid_style or self.plt_cfg.grid_style, fill=data["z"], cmap=True)
            if add_nsidc or (add_nsidc is None and float(np.mean(reg[2:])) < 0):
                for line in self._nsidc_contours(date_str, "south"):
                    lonline = line[:, 0]
                    latline = line[:, 1]
                    keep = self._region_mask(xr.DataArray(lonline, dims="p"), xr.DataArray(latline, dims="p"), reg).values
                    if np.any(keep):
                        fig.plot(x=lonline[keep], y=latline[keep], pen=self.plt_cfg.nsidc_pen)
            fig.colorbar(position=self.plt_cfg.colorbar_position)
            fig.savefig(path)
            out[name] = str(path)
        return next(iter(out.values())) if len(out) == 1 else out

    def plot_triptych(self, region: Sequence[float], panels: Sequence[Mapping], *,
                      fig_size   : float = 20.0,
                      output_path: str | Path | None = None,
                      panel_gap  : str = "1.5c") -> str:
        """Generic 3-panel regional plotter with per-panel layer stacks.

        Each panel dict may contain:
        - title: str
        - layers: list of layer dicts with keys:
            data: pandas.DataFrame or xarray.DataArray
            lon/lat: optional DataArray when data is xarray
            mask_zero: bool
            style: str
            fill: color string or 'z'
            cmap: cmap path/name
            series: list/tuple for makecpt
            colorbar: dict(position=..., frame=[...])
            pen: optional outline pen
        """
        pygmt = self._require_pygmt()
        if len(panels) != 3:
            raise ValueError("plot_triptych currently expects exactly three panels.")
        path = Path(output_path).expanduser() if output_path is not None else self.pth_cfg.figure_root() / "comparison" / f"{self.run_cfg.start_date}_{self.run_cfg.end_date}_triptych.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        proj = self.projection_from_region(region, fig_size=fig_size)
        fig = pygmt.Figure()
        for i, panel in enumerate(panels):
            if i == 0:
                self.pygmt_base_layer(fig, region, proj, title=panel.get("title"))
            else:
                fig.shift_origin(xshift=f"1w+{panel_gap}")
                self.pygmt_base_layer(fig, region, proj, title=panel.get("title"))
            for layer in panel.get("layers", []):
                layer_data = layer.get("data")
                if isinstance(layer_data, pd.DataFrame):
                    data = layer_data
                elif isinstance(layer_data, xr.DataArray):
                    data = self.pygmt_da_prep(layer_data, lon=layer.get("lon"), lat=layer.get("lat"), mask_zero=layer.get("mask_zero", False), region=region)
                else:
                    raise TypeError("Layer data must be a pandas.DataFrame or xarray.DataArray")
                if layer.get("cmap") is not None:
                    pygmt.makecpt(cmap=layer["cmap"], series=layer.get("series"), continuous=True)
                fill = data["z"] if layer.get("fill", "z") == "z" else layer.get("fill")
                fig.plot(x=data["lon"], y=data["lat"], style=layer.get("style", self.plt_cfg.grid_style), fill=fill, cmap=bool(layer.get("cmap")), pen=layer.get("pen"))
                cbar = layer.get("colorbar")
                if cbar:
                    fig.colorbar(**cbar)
        fig.savefig(path)
        return str(path)

