from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec

@dataclass(slots=True)
class AF2020Spec:
    """Variable and coordinate names used by the raw AF2020 gridded product."""
    variable : str = "Fast_Ice_Time_series"
    lon      : str = "longitude"
    lat      : str = "latitude"
    area     : str = "area"
    time     : str = "time"
    D_org_nc : str | Path | None = "/g/data/jk72/af1544/fraser2020_data"
    D_reG    : str | Path | None = Path("/g/data/gv90/da1339/SeaIce/FI_obs")
    F_reG    : str = "AF-FI-2020db_common-5km_pyresample.zarr"
    threshold: float = 4.0

class AF2020Observations:
    """
    Loader and transformation utilities for the Fraser et al. Antarctic fast-ice dataset.

    This module intentionally owns AF2020-specific logic that was previously mixed into
    ``shuga.observations.cice.SeaIceObservations``. It supports two distinct products:

    1. ``FastIce_70_YYYY.nc`` native 15-day gridded rasters used for FIP maps and
       FIP differences.
    2. ``FIA_AF2020_daily.nc`` daily circum-Antarctic/regional time-series products.

    Raw AF2020 rasters are converted to a binary mask using ``Fast_Ice_Time_series >= 4``
    by default, matching the historical AFIM workflow. Temporal interpolation to daily
    fields is provided for FIC side-by-side plots; FIP comparisons default to native
    15-day sampling unless the caller explicitly requests daily sampling.
    """
    def __init__(self,
                 run_cfg : RunSpec | None = None,
                 obs_cfg : ObservationSpec | None = None,
                 pth_cfg : ShugaPaths | None = None, *,
                 D_org_nc: str | Path | None = None,
                 D_reG   : str | Path | None = None,
                 chunks  : dict | None = None,
                 af20_cfg: AF2020Spec | None = None,
                 logger  = None) -> None:
        self.af20_cfg = af20_cfg or AF2020Spec()
        self.run_cfg  = run_cfg or RunSpec(sim_name = "AF2020", start_date = "2000-01-01", end_date = "2018-12-31")
        self.obs_cfg  = obs_cfg or ObservationSpec()
        self.pth_cfg  = pth_cfg or ShugaPaths(run_cfg = self.run_cfg, classify = None, obs_cfg = self.obs_cfg)  # type: ignore[arg-type]
        org_root      = D_org_nc if D_org_nc is not None else self.af20_cfg.D_org_nc
        reg_root      = D_reG    if D_reG    is not None else self.af20_cfg.D_reG
        if org_root is None:
            org_root = self.pth_cfg.fi_obs_root_path / "org"
        if reg_root is None:
            reg_root = self.pth_cfg.fi_obs_root_path
        self.D_org_nc = Path(org_root).expanduser()
        self.D_reG    = Path(reg_root).expanduser()
        self.chunks   = chunks or {"time": "auto"}
        self.logger   = logger or build_file_logger("shuga.obs_cfg.AF2020", Path.home() / "logs" / "observations" / "shuga_AF2020.log")
        self._org_cache: xr.Dataset | None = None
        self._fia_daily_cache: xr.Dataset | None = None

    # ---------------------------------------------------------------------
    # file discovery and loading
    # ---------------------------------------------------------------------
    def org_files(self, start_date: str | None = None, end_date: str | None = None) -> list[Path]:
        """Return origin AF2020 ``FastIce_70_YYYY.nc`` files covering a date range."""
        start = pd.Timestamp(start_date or self.run_cfg.start_date)
        end   = pd.Timestamp(end_date or self.run_cfg.end_date)
        years = range(int(start.year), int(end.year) + 1)
        files = [self.D_org_nc / f"FastIce_70_{year:04d}.nc" for year in years]
        files = [p for p in files if p.exists()]
        if not files:
            raise FileNotFoundError(f"No AF2020 origin FastIce_70_YYYY.nc files found in {self.D_org_nc} for {start:%Y-%m-%d}..{end:%Y-%m-%d}.")
        return files

    def open_org(self, start_date: str | None = None, end_date: str | None = None) -> xr.Dataset:
        """Open origin AF2020 native 15-day rasters and subset them to the requested dates."""
        files = self.org_files(start_date, end_date)
        self.logger.info("Opening %s AF2020 origin files from %s", len(files), self.D_org_nc)
        ds = xr.open_mfdataset(files, engine = "netcdf4", combine = "by_coords", chunks = self.chunks, data_vars = "minimal", coords = "minimal", compat = "override")
        ds = self._normalise_org(ds)
        if start_date is not None or end_date is not None:
            ds = ds.sel(time=slice(start_date, end_date))
        self._org_cache = ds
        return ds

    def open_raw(self, start_date: str | None = None, end_date: str | None = None) -> xr.Dataset:
        """Backward-compatible alias for open_org()."""
        return self.open_org(start_date=start_date, end_date=end_date)

    def raw_files(self, start_date: str | None = None, end_date: str | None = None) -> list[Path]:
        """Backward-compatible alias for org_files()."""
        return self.org_files(start_date=start_date, end_date=end_date)

    def _normalise_org(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardise origin AF2020 names while preserving the native grid."""
        n = self.af20_cfg
        rename: dict[str, str] = {}
        if n.time in ds and n.time != "time":
            rename[n.time] = "time"
        if rename:
            ds = ds.rename(rename)
        if "time" in ds:
            ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values)).sortby("time")
        for required in (n.variable, n.lon, n.lat):
            if required not in ds:
                raise KeyError(f"Required AF2020 variable/coordinate {required!r} not found. Available: {list(ds.variables)}")
        return ds

    # ---------------------------------------------------------------------
    # native masks, FIC, FIP
    # ---------------------------------------------------------------------
    @staticmethod
    def _with_standard_spatial_dims(da: xr.DataArray, *, ydim: str = "nj", xdim: str = "ni") -> xr.DataArray:
        """Rename common AF2020 spatial dimensions to ``nj``/``ni`` when possible."""
        spatial_dims = [d for d in da.dims if d != "time"]
        if len(spatial_dims) != 2:
            return da
        rename = {}
        if spatial_dims[0] != ydim:
            rename[spatial_dims[0]] = ydim
        if spatial_dims[1] != xdim:
            rename[spatial_dims[1]] = xdim
        return da.rename(rename) if rename else da

    def native_mask(self, ds: xr.Dataset | None = None, *, threshold: float | None = None, name: str = "AF_FI_mask") -> xr.DataArray:
        """
        Return the native 15-day AF2020 binary fast-ice mask.

        The default threshold is ``Fast_Ice_Time_series >= 4``. Values outside the
        threshold are zero rather than NaN so persistence is a true occupancy fraction.
        """
        ds     = ds if ds is not None else (self._org_cache or self.open_org())
        n      = self.af20_cfg
        thresh = float(self.af20_cfg.threshold if threshold is None else threshold)
        mask   = xr.where(ds[n.variable] >= thresh, 1.0, 0.0).astype("float32")
        mask   = self._with_standard_spatial_dims(mask).rename(name)
        # Attach AF2020 lon/lat using the same spatial dims as mask.
        spatial_dims = tuple(d for d in mask.dims if d != "time")
        lon          = ds[n.lon]
        lat          = ds[n.lat]
        if lon.dims != spatial_dims:
            lon = xr.DataArray(lon.values, dims = spatial_dims)
        if lat.dims != spatial_dims:
            lat = xr.DataArray(lat.values, dims = spatial_dims)
        mask = mask.assign_coords(lon = (spatial_dims, lon.values), lat = (spatial_dims, lat.values))
        mask.attrs.update(long_name       = "AF2020 native fast-ice mask",
                          units           = "1",
                          threshold       = thresh,
                          source_variable = n.variable,
                          sampling        = "native 15-day AF2020")
        return mask

    def native_area(self, ds: xr.Dataset | None = None) -> xr.DataArray | None:
        """Return native AF2020 cell area in m2 when present."""
        ds = ds if ds is not None else (self._org_cache or self.open_org())
        n  = self.af20_cfg
        if n.area not in ds:
            return None
        area = ds[n.area]
        if "time" in area.dims:
            area = area.isel(time=0, drop=True)
        area = self._with_standard_spatial_dims(area).astype("float64").rename("AF_cell_area")
        area.attrs.update(long_name="AF2020 native cell area", units="m2")
        return area

    def daily_fic(self, mask: xr.DataArray | None = None, *,
                  start_date  : str | None = None,
                  end_date    : str | None = None,
                  method      : str = "linear",
                  max_gap_days: int | None = 35,
                  name        : str = "AF_FIC") -> xr.DataArray:
        """
        Interpolate native 15-day AF2020 masks to daily fractional occupancy.

        ``method='linear'`` gives a fractional daily FIC-like occupancy during transition
        intervals. ``method='nearest'`` gives a stepwise daily mask. Long temporal gaps are
        masked if ``max_gap_days`` is not ``None``.
        """
        if method not in {"linear", "nearest"}:
            raise ValueError("method must be 'linear' or 'nearest'.")
        if mask is None:
            ds = self.open_org(start_date, end_date)
            mask = self.native_mask(ds)
        start       = pd.Timestamp(start_date or str(pd.to_datetime(mask.time.values[0]).date()))
        end         = pd.Timestamp(end_date or str(pd.to_datetime(mask.time.values[-1]).date()))
        target_time = pd.date_range(start, end, freq="D")
        out         = mask.sortby("time").interp(time = target_time, method = method).clip(0.0, 1.0).astype("float32")
        if max_gap_days is not None:
            src_t = pd.to_datetime(mask.time.values)
            dist  = np.array([np.min(np.abs((src_t - t).days)) for t in target_time], dtype = "float32")
            valid = xr.DataArray(dist <= float(max_gap_days), dims = ("time"), coords = {"time": target_time})
            out   = out.where(valid)
        out = out.rename(name)
        out.attrs.update(long_name              = "AF2020 daily fast-ice concentration/fractional occupancy",
                         units                  = "1",
                         temporal_interpolation = method,
                         max_gap_days           = -1 if max_gap_days is None else int(max_gap_days))
        return out

    @staticmethod
    def persistence(mask: xr.DataArray, *, start_date: str | None = None, end_date: str | None = None, name: str = "FIP") -> xr.DataArray:
        """Return fast-ice persistence as the mean occupancy over time."""
        da = mask
        if start_date is not None or end_date is not None:
            da = da.sel(time=slice(start_date, end_date))
        if da.sizes.get("time", 0) == 0:
            raise ValueError("Cannot compute persistence from an empty time slice.")
        out = da.mean("time", skipna=True).clip(0.0, 1.0).astype("float32").rename(name)
        out.attrs.update(long_name="fast-ice persistence", units="1")
        return out

    def native_fip(self, ds: xr.Dataset | None = None, *,
                   start_date: str | None = None,
                   end_date  : str | None = None,
                   threshold : float | None = None,
                   name      : str = "AF_FIP_native") -> xr.DataArray:
        """Compute AF2020 FIP on the native 15-day AF2020 grid."""
        ds = ds if ds is not None else self.open_org(start_date, end_date)
        mask = self.native_mask(ds, threshold=threshold)
        return self.persistence(mask, start_date=start_date, end_date=end_date, name=name)

    # ---------------------------------------------------------------------
    # FIA time-series products formerly in observations/cice.py
    # ---------------------------------------------------------------------
    def load_fia_daily(self) -> xr.Dataset:
        """Load the existing daily AF2020 FIA NetCDF product."""
        if self._fia_daily_cache is None:
            path = self.pth_cfg.fi_obs_root_path / self.obs_cfg.af2020_fia_daily_file
            if not path.exists():
                raise FileNotFoundError(f"AF2020 FIA daily file does not exist: {path}")
            ds = xr.open_dataset(path, chunks={"time": 366})
            if self.obs_cfg.af2020_fia_daily_var not in ds:
                raise KeyError(f"Variable {self.obs_cfg.af2020_fia_daily_var!r} not found in {path}")
            self._fia_daily_cache = ds
        return self._fia_daily_cache

    def get_fia_daily(self) -> xr.DataArray:
        """Return daily AF2020 FIA as ``FIA`` in 10^3 km2."""
        da = self.load_fia_daily()[self.obs_cfg.af2020_fia_daily_var].astype("float32").rename("FIA")
        da.attrs.update(long_name="Observed Fast Ice Area", units="10^3 km^2")
        return da

    def subset_fia_daily(self, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        start_date = start_date or self.run_cfg.start_date
        end_date = end_date or self.run_cfg.end_date
        return self.get_fia_daily().sel(time=slice(start_date, end_date))

    def fia_daily_climatology(self) -> xr.DataArray:
        """Return daily AF2020 FIA climatology over day-of-year."""
        da = self.get_fia_daily().dropna("time", how="all")
        clim = da.groupby("time.dayofyear").mean("time").rename({"dayofyear": "doy"}).rename("FIA_clim")
        clim.attrs.update(long_name="Observed Fast Ice Area Climatology", units="10^3 km^2")
        return clim

    def repeat_fia_daily_climatology(self, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        """Repeat AF2020 daily FIA climatology over an arbitrary daily period."""
        start_date = start_date or self.run_cfg.start_date
        end_date = end_date or self.run_cfg.end_date
        clim = self.fia_daily_climatology()
        t = pd.date_range(start_date, end_date, freq="D")
        doy_vals = np.asarray(clim["doy"].values).astype(int)
        clim_vals = np.asarray(clim.values, dtype=float)
        lut = {int(d): float(v) for d, v in zip(doy_vals, clim_vals)}
        values = np.array([lut.get(365 if d == 366 else int(d), np.nan) for d in t.dayofyear], dtype=float)
        out = xr.DataArray(values, dims=("time",), coords={"time": t}, name="FIA_clim_repeat")
        out.attrs.update(clim.attrs)
        return out

    # ---------------------------------------------------------------------
    # generic utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_fia_from_mask(mask: xr.DataArray, area: xr.DataArray, *, scale: float = 1e9, name: str = "FIA") -> xr.DataArray:
        """Compute FIA from a mask/fractional occupancy field and m2 cell area."""
        spatial_dims = [d for d in mask.dims if d != "time"]
        area2d = area.isel(time=0, drop=True) if "time" in area.dims else area
        out = (mask.astype("float64") * area2d.astype("float64")).sum(dim=spatial_dims, skipna=True) / float(scale)
        out = out.rename(name)
        out.attrs.update(long_name="Fast Ice Area", units="10^3 km^2" if scale == 1e9 else f"m2/{scale:g}")
        return out

# Backwards-compatible alias for scripts that prefer a generic class name.
SeaIceAF2020 = AF2020Observations
