from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec


class SeaIceObservations:
    """Load NSIDC concentration and AF2020 fast-ice products for shuga workflows."""

    def __init__(
        self,
        run: RunSpec,
        observations: ObservationSpec | None = None,
        paths: ShugaPaths | None = None,
        *,
        chunks: dict | None = None,
        logger=None,
    ) -> None:
        self.run = run
        self.observations = observations or ObservationSpec()
        self.paths = paths or ShugaPaths(run=run, classify=None, observations=self.observations)  # type: ignore[arg-type]
        self.chunks = chunks or {"time": 31}
        self.logger = logger or build_file_logger("shuga.observations", Path.home() / "logs" / "observations" / "shuga_observations.log")
        self._nsidc_cache: dict[tuple[str, str, str], xr.Dataset] = {}
        self._af2020_cache: dict[str, xr.Dataset] = {}

    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        token = str(value).strip().lower()
        return "south" if token in {"s", "sh", "south", "southern"} else "north"

    def _nsidc_aux_suffix(self, hemisphere: str) -> str:
        hemi = "S" if hemisphere == "south" else "N"
        return f"{hemi}25km_v1.1.nc"

    def nsidc_area_file(self, hemisphere: str) -> Path:
        return self.paths.nsidc_aux_root_path / f"NSIDC0771_CellArea_PS_{self._nsidc_aux_suffix(hemisphere)}"

    def nsidc_latlon_file(self, hemisphere: str) -> Path:
        return self.paths.nsidc_aux_root_path / f"NSIDC0771_LatLon_PS_{self._nsidc_aux_suffix(hemisphere)}"

    def _nsidc_daily_files(self, start_date: str, end_date: str, hemisphere: str) -> list[Path]:
        hemi = self.canonical_hemisphere(hemisphere)
        root = self.paths.nsidc_root_path / hemi / "daily"
        if not root.exists():
            raise FileNotFoundError(f"NSIDC daily directory does not exist: {root}")
        dates = pd.date_range(start_date, end_date, freq="D")
        files: list[Path] = []
        for dt in dates:
            patt = f"sic_ps{'s' if hemi == 'south' else 'n'}25_{dt:%Y%m%d}_*_v06r00.nc"
            matches = sorted(root.glob(patt))
            if matches:
                files.append(matches[0])
        if not files:
            raise FileNotFoundError(f"No NSIDC daily files found in {root} between {start_date} and {end_date}")
        return files

    def load_nsidc_daily(self, start_date: str | None = None, end_date: str | None = None, hemisphere: str | None = None) -> xr.Dataset:
        start_date = start_date or self.run.start_date
        end_date = end_date or self.run.end_date
        hemi = self.canonical_hemisphere(hemisphere or self.run.hemisphere)
        key = (start_date, end_date, hemi)
        if key in self._nsidc_cache:
            return self._nsidc_cache[key]
        files = self._nsidc_daily_files(start_date, end_date, hemi)
        self.logger.info("Opening %s NSIDC daily files for %s hemisphere", len(files), hemi)

        def _prep(ds: xr.Dataset) -> xr.Dataset:
            keep = [v for v in (self.observations.nsidc_sic_var,) if v in ds]
            if keep:
                ds = ds[keep]
            return ds

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=True, preprocess=_prep, chunks=self.chunks)
        latlon = xr.open_dataset(self.nsidc_latlon_file(hemi))[["latitude", "longitude"]]
        area = xr.open_dataset(self.nsidc_area_file(hemi))[["cell_area"]]
        ds = xr.merge([ds, latlon, area], compat="override", combine_attrs="drop_conflicts")
        self._nsidc_cache[key] = ds
        return ds

    def compute_nsidc_sia_sie(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        hemisphere: str | None = None,
        threshold: float | None = None,
    ) -> xr.Dataset:
        ds = self.load_nsidc_daily(start_date=start_date, end_date=end_date, hemisphere=hemisphere)
        sic = ds[self.observations.nsidc_sic_var].astype("float32")
        mask = sic >= float(threshold if threshold is not None else self.observations.nsidc_threshold)
        area = ds["cell_area"].astype("float64")
        sia = (sic.where(mask, 0.0) * area).sum(dim=("y", "x")) / 1e12
        sie = (mask.astype("float32") * area).sum(dim=("y", "x")) / 1e12
        out = xr.Dataset({"SIA": sia, "SIE": sie})
        out["SIA"].attrs.update({"long_name": "Sea Ice Area", "units": "10^6 km^2"})
        out["SIE"].attrs.update({"long_name": "Sea Ice Extent", "units": "10^6 km^2"})
        return out

    def load_af2020_fia_daily(self) -> xr.Dataset:
        key = "af2020_fia_daily"
        if key not in self._af2020_cache:
            path = self.paths.fi_obs_root_path / self.observations.af2020_fia_daily_file
            if not path.exists():
                raise FileNotFoundError(f"AF2020 FIA daily file does not exist: {path}")
            ds = xr.open_dataset(path, chunks=self.chunks)
            if self.observations.af2020_fia_daily_var not in ds:
                raise KeyError(
                    f"Variable {self.observations.af2020_fia_daily_var!r} not found in AF2020 FIA daily file: {path}"
                )
            self._af2020_cache[key] = ds
        return self._af2020_cache[key]

    def get_af2020_fia_daily(self) -> xr.DataArray:
        ds = self.load_af2020_fia_daily()
        da = ds[self.observations.af2020_fia_daily_var].astype("float32")
        da = da.rename("FIA")
        da.attrs.update({"long_name": "Observed Fast Ice Area", "units": "10^3 km^2"})
        return da

    def subset_af2020_fia_daily(self, start_date: str | None = None, end_date: str | None = None) -> xr.DataArray:
        da = self.get_af2020_fia_daily()
        start_date = start_date or self.run.start_date
        end_date = end_date or self.run.end_date
        return da.sel(time=slice(start_date, end_date))

    def compute_af2020_fia_daily_climatology(self) -> xr.DataArray:
        da = self.get_af2020_fia_daily().dropna("time", how="all")
        clim = da.groupby("time.dayofyear").mean("time")
        clim = clim.rename({"dayofyear": "doy"}).rename("FIA_clim")
        clim.attrs.update({"long_name": "Observed Fast Ice Area Climatology", "units": "10^3 km^2"})
        return clim

    def repeat_af2020_fia_daily_climatology(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> xr.DataArray:
        start_date = start_date or self.run.start_date
        end_date = end_date or self.run.end_date
        clim = self.compute_af2020_fia_daily_climatology()
        t = pd.date_range(start_date, end_date, freq="D")
        doy_vals = np.asarray(clim["doy"].values).astype(int)
        clim_vals = np.asarray(clim.values, dtype=float)
        lut = {int(d): float(v) for d, v in zip(doy_vals, clim_vals)}
        values = np.array([lut.get(365 if d == 366 else int(d), np.nan) for d in t.dayofyear], dtype=float)
        out = xr.DataArray(values, dims=("time",), coords={"time": t}, name="FIA_clim_repeat")
        out.attrs.update(clim.attrs)
        return out

    def load_af2020_regridded(self) -> xr.Dataset:
        key = "af2020_regridded"
        if key not in self._af2020_cache:
            path = self.paths.fi_obs_root_path / self.observations.af2020_regridded_store
            if not path.exists():
                raise FileNotFoundError(f"AF2020 regridded store does not exist: {path}")
            self._af2020_cache[key] = xr.open_zarr(path, consolidated=False, chunks=self.chunks)
        return self._af2020_cache[key]

    def load_af2020_climatology(self) -> xr.Dataset:
        key = "af2020_clim"
        if key not in self._af2020_cache:
            path = self.paths.fi_obs_root_path / self.observations.af2020_climatology_store
            if not path.exists():
                raise FileNotFoundError(f"AF2020 climatology store does not exist: {path}")
            self._af2020_cache[key] = xr.open_zarr(path, consolidated=True if (path / ".zmetadata").exists() else False, chunks=self.chunks)
        return self._af2020_cache[key]

    @staticmethod
    def _ensure_2d_area(area: xr.DataArray) -> xr.DataArray:
        return area.isel(time=0, drop=True) if "time" in area.dims else area

    def compute_af2020_fia_from_regridded(self, area: xr.DataArray, var_name: str | None = None) -> xr.Dataset:
        ds = self.load_af2020_regridded()
        area2d = self._ensure_2d_area(area)
        var_name = var_name or self.observations.af2020_regridded_var
        if var_name not in ds:
            raise KeyError(f"Variable {var_name!r} not found in AF2020 regridded store.")
        da = ds[var_name]
        if self.observations.af2020_time_var in ds:
            da = da.rename({self.observations.af2020_time_var: "time"}) if self.observations.af2020_time_var in da.dims else da
        mask = da > 0
        fia = (mask.astype("float32") * area2d).sum(dim=[d for d in mask.dims if d != "time"]) / 1e9
        out = xr.Dataset({"FIA": fia})
        out["FIA"].attrs.update({"long_name": "Observed Fast Ice Area", "units": "10^3 km^2"})
        return out

    def compute_af2020_fia_climatology(self, area: xr.DataArray | None = None, var_name: str | None = None) -> xr.Dataset:
        """Return AF2020 FIA climatology derived from the daily observed FIA NetCDF.

        The gridded AF2020 climatology store is not used here because it overestimates circum-Antarctic FIA.
        """
        return xr.Dataset({"FIA_clim": self.compute_af2020_fia_daily_climatology()})

    def repeat_daily_climatology(
        self,
        clim: xr.DataArray,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> xr.DataArray:
        start_date = start_date or self.run.start_date
        end_date = end_date or self.run.end_date
        if clim.name == "FIA_clim" or clim.attrs.get("long_name", "").startswith("Observed Fast Ice Area"):
            # treat as a day-of-year climatology on a doy axis
            t = pd.date_range(start_date, end_date, freq="D")
            if "doy" not in clim.dims and "doy" not in clim.coords:
                raise ValueError(f"repeat_daily_climatology expected a 'doy' axis; got dims={clim.dims}")
            if "doy" not in clim.dims:
                src_dim = clim.dims[0]
                clim = clim.swap_dims({src_dim: "doy"}).drop_vars(src_dim, errors="ignore")
            doy_vals = np.asarray(clim["doy"].values).astype(int)
            clim_vals = np.asarray(clim.values, dtype=float)
            lut = {int(d): float(v) for d, v in zip(doy_vals, clim_vals)}
            values = np.array([lut.get(365 if d == 366 else int(d), np.nan) for d in t.dayofyear], dtype=float)
            out = xr.DataArray(values, dims=("time",), coords={"time": t}, name=f"{clim.name}_repeat")
            out.attrs.update(clim.attrs)
            return out

        # sparse climatologies on doy-like coordinates: interpolate to daily
        t = pd.date_range(start_date, end_date, freq="D")
        if "doy" not in clim.dims and "doy" not in clim.coords:
            raise ValueError(f"repeat_daily_climatology expected a climatology with 'doy'; got dims={clim.dims}")
        if "doy" not in clim.dims:
            src_dim = clim.dims[0]
            clim = clim.swap_dims({src_dim: "doy"}).drop_vars(src_dim, errors="ignore")
        clim = clim.sortby("doy")
        src_doy = np.asarray(clim["doy"].values).astype(int)
        src_val = np.asarray(clim.values, dtype=float)
        xp = np.r_[src_doy[0] - 365, src_doy, src_doy[-1] + 365]
        fp = np.r_[src_val[-1], src_val, src_val[0]]
        target_doy = t.dayofyear.to_numpy().astype(int)
        target_doy = np.where(target_doy == 366, 365, target_doy)
        values = np.interp(target_doy, xp, fp)
        out = xr.DataArray(values, dims=("time",), coords={"time": t}, name=f"{clim.name}_repeat")
        out.attrs.update(clim.attrs)
        return out
