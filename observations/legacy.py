from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec
from shuga.observations.AF2020 import AF2020Observations
from shuga.observations.NSIDC import NSIDCObservations

class SeaIceObservations:
    """
    Backwards-compatible observation facade for older shuga callers.

    The old ``shuga.observations.cice.SeaIceObservations`` mixed NSIDC and
    AF2020 methods in one class. The observation code has now been split into:

        AF2020Observations  -> Antarctic fast-ice observations
        NSIDCObservations   -> sea-ice concentration observations

    This adapter preserves the old public method names used by
    ``shuga.plotting.cice.CICEPlotter``, ``shuga.waves.cawcr``, older scripts,
    and notebooks while delegating implementation to the new classes.

    New code should prefer using ``AF2020Observations`` or
    ``NSIDCObservations`` directly.
    """

    def __init__(self,
                 run_cfg: RunSpec,
                 obs_cfg: ObservationSpec | None = None,
                 pth_cfg: ShugaPaths | None = None, *,
                 chunks : dict | None = None,
                 logger = None) -> None:
        self.run_cfg = run_cfg
        self.obs_cfg = obs_cfg or ObservationSpec()
        self.pth_cfg = pth_cfg or ShugaPaths(run_cfg = run_cfg, classify = None, obs_cfg = self.obs_cfg)  # type: ignore[arg-type]
        self.chunks  = chunks or {"time": 31}
        self.logger  = logger or build_file_logger("shuga.observations.legacy", Path.home() / "logs" / "observations" / "shuga_observations_legacy.log")
        self.nsidc   = NSIDCObservations(run_cfg = self.run_cfg,
                                         obs_cfg = self.obs_cfg,
                                         pth_cfg = self.pth_cfg,
                                         chunks  = self.chunks,
                                         logger  = self.logger)
        self.af2020  = AF2020Observations(run_cfg = self.run_cfg,
                                          obs_cfg = self.obs_cfg,
                                          pth_cfg = self.pth_cfg,
                                          chunks  = self.chunks,
                                          logger  = self.logger)
        # Preserve the old cache attribute names for any external callers that
        # inspect them. The delegated classes hold the actual NSIDC cache.
        self._nsidc_cache = self.nsidc._cache
        self._af2020_cache: dict[str, xr.Dataset] = {}

    # ------------------------------------------------------------------
    # NSIDC compatibility methods
    # ------------------------------------------------------------------
    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        return NSIDCObservations.canonical_hemisphere(value)

    def _nsidc_aux_suffix(self, hemisphere: str) -> str:
        hemi = self.canonical_hemisphere(hemisphere)
        return self.nsidc._aux_suffix(hemi)

    def nsidc_area_file(self, hemisphere: str) -> Path:
        return self.nsidc.area_file(self.canonical_hemisphere(hemisphere))

    def nsidc_latlon_file(self, hemisphere: str) -> Path:
        return self.nsidc.latlon_file(self.canonical_hemisphere(hemisphere))

    def _nsidc_daily_files(self, start_date: str, end_date: str, hemisphere: str) -> list[Path]:
        return self.nsidc.daily_files(start_date, end_date, self.canonical_hemisphere(hemisphere))

    def load_nsidc_daily(self,
                         start_date: str | None = None,
                         end_date  : str | None = None,
                         hemisphere: str | None = None) -> xr.Dataset:
        return self.nsidc.load_daily(start_date=start_date, end_date=end_date, hemisphere=hemisphere)

    def compute_nsidc_sia_sie(self,
                              start_date: str | None = None,
                              end_date  : str | None = None,
                              hemisphere: str | None = None,
                              threshold : float | None = None) -> xr.Dataset:
        return self.nsidc.compute_sia_sie(start_date = start_date,
                                          end_date   = end_date,
                                          hemisphere = hemisphere,
                                          threshold  = threshold)

    # ------------------------------------------------------------------
    # AF2020 daily FIA compatibility methods
    # ------------------------------------------------------------------
    def load_af2020_fia_daily(self) -> xr.Dataset:
        return self.af2020.load_fia_daily()

    def get_af2020_fia_daily(self) -> xr.DataArray:
        return self.af2020.get_fia_daily()

    def subset_af2020_fia_daily(self, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        return self.af2020.subset_fia_daily(start_date=start_date, end_date=end_date)

    def compute_af2020_fia_daily_climatology(self) -> xr.DataArray:
        return self.af2020.fia_daily_climatology()

    def repeat_af2020_fia_daily_climatology(self, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        return self.af2020.repeat_fia_daily_climatology(start_date=start_date, end_date=end_date)

    # ------------------------------------------------------------------
    # AF2020 gridded-store compatibility methods
    # ------------------------------------------------------------------
    def load_af2020_regridded(self) -> xr.Dataset:
        key = "af2020_regridded"
        if key not in self._af2020_cache:
            path = self.pth_cfg.fi_obs_root_path / self.obs_cfg.af2020_regridded_store
            if not path.exists():
                raise FileNotFoundError(f"AF2020 regridded store does not exist: {path}")
            self._af2020_cache[key] = xr.open_zarr(path, consolidated = False, chunks = self.chunks)
        return self._af2020_cache[key]

    def load_af2020_climatology(self) -> xr.Dataset:
        key = "af2020_clim"
        if key not in self._af2020_cache:
            path = self.pth_cfg.fi_obs_root_path / self.obs_cfg.af2020_climatology_store
            if not path.exists():
                raise FileNotFoundError(f"AF2020 climatology store does not exist: {path}")
            self._af2020_cache[key] = xr.open_zarr(path, consolidated=True if (path / ".zmetadata").exists() else False, chunks = self.chunks)
        return self._af2020_cache[key]

    @staticmethod
    def _ensure_2d_area(area: xr.DataArray) -> xr.DataArray:
        return area.isel(time=0, drop=True) if "time" in area.dims else area

    def compute_af2020_fia_from_regridded(self, area: xr.DataArray, var_name: str | None = None) -> xr.Dataset:
        ds       = self.load_af2020_regridded()
        area2d   = self._ensure_2d_area(area)
        var_name = var_name or self.obs_cfg.af2020_regridded_var
        if var_name not in ds:
            raise KeyError(f"Variable {var_name!r} not found in AF2020 regridded store.")
        da = ds[var_name]
        if self.obs_cfg.af2020_time_var in da.dims:
            da = da.rename({self.obs_cfg.af2020_time_var: "time"})
        mask         = da > 0
        spatial_dims = [d for d in mask.dims if d != "time"]
        fia          = (mask.astype("float32") * area2d).sum(dim=spatial_dims, skipna=True) / 1e9
        out          = xr.Dataset({"FIA": fia})
        out["FIA"].attrs.update(long_name="Observed Fast Ice Area", units="10^3 km^2")
        return out

    def compute_af2020_fia_climatology(self, area: xr.DataArray | None = None, var_name: str | None = None) -> xr.Dataset:
        """
        Return AF2020 FIA climatology derived from the daily observed FIA NetCDF.

        The gridded AF2020 climatology store is intentionally not used here,
        matching the previous SeaIceObservations behaviour.
        """
        return xr.Dataset({"FIA_clim": self.compute_af2020_fia_daily_climatology()})

    def repeat_daily_climatology(self, clim: xr.DataArray, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        """
        Repeat/interpolate a day-of-year climatology over an arbitrary daily window.

        This is copied from the previous mixed SeaIceObservations API because
        some plotting paths used it generically for observation climatologies.
        """
        start_date = start_date or self.run_cfg.start_date
        end_date   = end_date or self.run_cfg.end_date
        if clim.name == "FIA_clim" or clim.attrs.get("long_name", "").startswith("Observed Fast Ice Area"):
            t = pd.date_range(start_date, end_date, freq="D")
            if "doy" not in clim.dims and "doy" not in clim.coords:
                raise ValueError(f"repeat_daily_climatology expected a 'doy' axis; got dims={clim.dims}")
            if "doy" not in clim.dims:
                src_dim = clim.dims[0]
                clim    = clim.swap_dims({src_dim: "doy"}).drop_vars(src_dim, errors="ignore")
            doy_vals  = np.asarray(clim["doy"].values).astype(int)
            clim_vals = np.asarray(clim.values, dtype=float)
            lut       = {int(d): float(v) for d, v in zip(doy_vals, clim_vals)}
            values    = np.array([lut.get(365 if d == 366 else int(d), np.nan) for d in t.dayofyear], dtype=float)
            out       = xr.DataArray(values, dims=("time",), coords={"time": t}, name=f"{clim.name}_repeat")
            out.attrs.update(clim.attrs)
            return out
        t = pd.date_range(start_date, end_date, freq="D")
        if "doy" not in clim.dims and "doy" not in clim.coords:
            raise ValueError(f"repeat_daily_climatology expected a climatology with 'doy'; got dims={clim.dims}")
        if "doy" not in clim.dims:
            src_dim = clim.dims[0]
            clim    = clim.swap_dims({src_dim: "doy"}).drop_vars(src_dim, errors="ignore")
        clim       = clim.sortby("doy")
        src_doy    = np.asarray(clim["doy"].values).astype(int)
        src_val    = np.asarray(clim.values, dtype=float)
        xp         = np.r_[src_doy[0] - 365, src_doy, src_doy[-1] + 365]
        fp         = np.r_[src_val[-1], src_val, src_val[0]]
        target_doy = t.dayofyear.to_numpy().astype(int)
        target_doy = np.where(target_doy == 366, 365, target_doy)
        values     = np.interp(target_doy, xp, fp)
        out        = xr.DataArray(values, dims=("time",), coords={"time": t}, name=f"{clim.name}_repeat")
        out.attrs.update(clim.attrs)
        return out
