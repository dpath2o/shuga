from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shuga.core.logging import build_file_logger
from shuga.core.naming import normalize_method
from shuga.core.paths import ShugaPaths
from shuga.core.regions import ANTARCTIC_8_REGIONS
from shuga.core.types import ClassificationSpec, MetricsSpec, RunSpec
from shuga.io.zarr_loading import load_cice, load_classified

def _sanitize_for_zarr_write(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    # Clear inherited backend-specific encodings on all variables and coords
    for name in ds.variables:
        ds[name].encoding = {}
    ds.encoding = {}
    return ds

class CICEMetrics:
    """Compute and load fast-ice metrics for raw, binary-days, and rolling-mean masks."""

    def __init__(
        self,
        run: RunSpec,
        classify: ClassificationSpec,
        metrics: MetricsSpec | None = None,
        paths: ShugaPaths | None = None,
        *,
        chunks: dict | None = None,
        logger=None,
    ) -> None:
        self.run = run
        self.classify = classify
        self.metrics = metrics or MetricsSpec()
        self.paths = paths or ShugaPaths(run=run, classify=classify)
        self.chunks = chunks or {"time": 31}
        self.logger = logger or build_file_logger(
            "shuga.metrics",
            self.paths.metrics_log_path(),
        )
        self.region_defs = ANTARCTIC_8_REGIONS
        self._cice_cache: xr.Dataset | None = None
        self._metrics_cache: dict[str, xr.Dataset] = {}

    @property
    def mask_var_name(self) -> str:
        return f"{self.classify.ice_type}_mask"

    def _get_cice(self, variables: list[str] | None = None) -> xr.Dataset:
        if self._cice_cache is None:
            vars_keep = variables or ["aice", "hi", "tarea", "TLON", "TLAT"]
            self.logger.info("Resolved CICE store: %s", self.paths.resolve_cice_store())
            static_store = self.paths.resolve_static_store()
            if static_store is not None:
                self.logger.info("Resolved static store: %s", static_store)
            self._cice_cache = load_cice(
                run=self.run,
                classify=self.classify,
                metrics=self.metrics,
                paths=self.paths,
                variables=vars_keep,
                hemisphere=self.run.hemisphere,
                chunks=self.chunks)
        ds = self._cice_cache
        if variables is not None:
            keep = [v for v in variables if v in ds.variables or v in ds.coords]
            ds = ds[keep]
        return ds

    @staticmethod
    def _ensure_2d_static(da: xr.DataArray) -> xr.DataArray:
        return da.isel(time=0, drop=True) if "time" in da.dims else da

    @staticmethod
    def _lon_to_180(lon: xr.DataArray) -> xr.DataArray:
        return ((lon + 180.0) % 360.0) - 180.0

    def _detect_lonlat(self, ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        lon_name = next((n for n in ("TLON", "ULON", "lon", "longitude") if n in ds.variables or n in ds.coords), None)
        lat_name = next((n for n in ("TLAT", "ULAT", "lat", "latitude") if n in ds.variables or n in ds.coords), None)
        if lon_name is None or lat_name is None:
            raise KeyError("Could not find longitude/latitude fields in the CICE dataset.")
        return ds[lon_name], ds[lat_name]

    def _spatial_dims(self, da: xr.DataArray) -> list[str]:
        return [dim for dim in da.dims if dim != "time" and dim != "region"]

    def _output_chunk_map(self, ds: xr.Dataset) -> dict[str, int]:
        chunk_map: dict[str, int] = {}
        if "time" in ds.dims:
            chunk_map["time"] = int(self.chunks.get("time", 31))
        for dim in ds.dims:
            if dim != "time":
                chunk_map[dim] = -1
        return chunk_map

    def _region_mask(self, template: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        lon180 = self._lon_to_180(lon)
        region_masks = []
        names = []
        for region_name, spec in self.region_defs.items():
            lon_min, lon_max, lat_min, lat_max = spec["geo_region"]
            if lon_min <= lon_max:
                lon_mask = (lon180 >= lon_min) & (lon180 <= lon_max)
            else:
                lon_mask = (lon180 >= lon_min) | (lon180 <= lon_max)
            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            region_masks.append((lon_mask & lat_mask).astype(bool))
            names.append(region_name)
        out = xr.concat(region_masks, dim=pd.Index(names, name="region"))
        return out.transpose("region", *template.dims)

    def compute_fia(self, aice: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        da = (aice.where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice)) / self.metrics.area_scale
        da.name = "FIA"
        da.attrs.update({"long_name": "Fast Ice Area", "units": "10^3 km^2"})
        return da

    def compute_fiv(self, aice: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        da = (aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice)) / self.metrics.volume_scale
        da.name = "FIV"
        da.attrs.update({"long_name": "Fast Ice Volume", "units": "10^3 km^3"})
        return da

    def compute_fit(self, aice: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        volume = (aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice))
        area_num = (aice.where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice))
        da = xr.where(area_num > 0, volume / area_num, np.nan)
        da.name = "FIT"
        da.attrs.update({"long_name": "Fast Ice Thickness", "units": "m"})
        return da

    def compute_fip(self, mask: xr.DataArray) -> xr.DataArray:
        da = mask.astype("float32").mean(dim="time")
        da.name = "FIP"
        da.attrs.update({"long_name": "Fast Ice Persistence", "units": "1"})
        return da

    def compute_regional_fia_fit(
        self,
        aice: xr.DataArray,
        hi: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray,
        region_mask: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        weighted_area = aice.where(mask, 0.0) * area
        weighted_vol = aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area
        spatial_dims = self._spatial_dims(aice)
        region_area = weighted_area.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        region_vol = weighted_vol.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        fia_reg = (region_area.sum(dim=spatial_dims) / self.metrics.area_scale).transpose("time", "region")
        fit_num = region_vol.sum(dim=spatial_dims).transpose("time", "region")
        fit_den = region_area.sum(dim=spatial_dims).transpose("time", "region")
        fit_reg = xr.where(fit_den > 0, fit_num / fit_den, np.nan)
        fia_reg.name = "FIA_by_region"
        fit_reg.name = "FIT_by_region"
        fia_reg.attrs.update({"long_name": "Fast Ice Area by Antarctic sector", "units": "10^3 km^2"})
        fit_reg.attrs.update({"long_name": "Fast Ice Thickness by Antarctic sector", "units": "m"})
        return fia_reg, fit_reg

    def compute_seasonal_summary(self, da: xr.DataArray, prefix: str) -> dict[str, xr.DataArray]:
        if "time" not in da.dims:
            return {}
        series = da.to_series().dropna()
        if series.empty:
            return {}
        rows = []
        for year, grp in series.groupby(series.index.year):
            if grp.empty:
                continue
            rows.append(
                {
                    "year": year,
                    "max": float(grp.max()),
                    "min": float(grp.min()),
                    "doy_max": float(pd.Timestamp(grp.idxmax()).dayofyear),
                    "doy_min": float(pd.Timestamp(grp.idxmin()).dayofyear),
                }
            )
        if not rows:
            return {}
        df = pd.DataFrame(rows)
        out = {}
        for col in ("max", "min", "doy_max", "doy_min"):
            vals = df[col].to_numpy(dtype=float)
            out[f"{prefix}_{col}_mean"] = xr.DataArray(np.nanmean(vals))
            out[f"{prefix}_{col}_std"] = xr.DataArray(np.nanstd(vals, ddof=0))
        return out

    def persistence_stability_index(
        self,
        mask: xr.DataArray,
        area: xr.DataArray,
        persistence_threshold: float = 0.8,
        winter_months: tuple[int, ...] = (5, 6, 7, 8, 9, 10),
    ) -> dict[str, xr.DataArray]:
        winter = mask.sel(time=mask.time.dt.month.isin(winter_months))
        if winter.sizes.get("time", 0) == 0:
            return {
                "FIPSI": xr.DataArray(np.nan),
                "persistent_winter_area": xr.DataArray(np.nan),
                "ever_winter_area": xr.DataArray(np.nan),
            }
        persistence = winter.astype("float32").mean("time")
        persistent_area = xr.where(persistence >= persistence_threshold, area, 0.0).sum()
        ever_area = xr.where(winter.any("time"), area, 0.0).sum()
        ratio = xr.where(ever_area > 0, persistent_area / ever_area, np.nan)
        return {
            "FIPSI": ratio,
            "persistent_winter_area": persistent_area / self.metrics.area_scale,
            "ever_winter_area": ever_area / self.metrics.area_scale,
        }

    @staticmethod
    def _skill_stats(mod_data: np.ndarray, obs_data: np.ndarray) -> dict[str, float]:
        good = np.isfinite(mod_data) & np.isfinite(obs_data)
        if good.sum() == 0:
            return {"Bias": np.nan, "RMSE": np.nan, "MAE": np.nan, "Corr": np.nan}
        m = mod_data[good]
        o = obs_data[good]
        corr = np.corrcoef(m, o)[0, 1] if good.sum() > 1 else np.nan
        return {
            "Bias": float(np.mean(m - o)),
            "RMSE": float(np.sqrt(np.mean((m - o) ** 2))),
            "MAE": float(np.mean(np.abs(m - o))),
            "Corr": float(corr),
        }

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
            stats = self._skill_stats(mod.values, ref.values)
            for k, v in stats.items():
                out[f"FIA_{k}"] = xr.DataArray(v)
        if self.metrics.obs_fit_var in obs and "FIT" in ds:
            mod, ref = xr.align(ds["FIT"], obs[self.metrics.obs_fit_var], join="inner")
            stats = self._skill_stats(mod.values, ref.values)
            for k, v in stats.items():
                out[f"FIT_{k}"] = xr.DataArray(v)
        return xr.Dataset(out)

    def compute_metrics(self, method: str, *, overwrite: bool = False) -> str:
        norm = normalize_method(method)
        self.logger.info("Resolved class store for %s: %s", norm, self.paths.classification_store(norm))
        self.logger.info("Resolved metrics store for %s: %s", norm, self.paths.metrics_store(norm))
        ds = self._get_cice(["aice", "hi", "tarea", "TLON", "TLAT"])
        ds_mask = load_classified(
            run=self.run,
            classify=self.classify,
            metrics=self.metrics,
            paths=self.paths,
            classification=norm,
            dt0_str=self.run.start_date,
            dtN_str=self.run.end_date,
            hemisphere=self.run.hemisphere,
            chunks=self.chunks)
        mask = ds_mask["FI_mask"].astype(bool)
        aice, hi, mask = xr.align(ds["aice"], ds["hi"], mask, join="inner")
        area = self._ensure_2d_static(ds["tarea"])
        lon, lat = self._detect_lonlat(ds)
        region_mask = self._region_mask(area, lon, lat)

        fia = self.compute_fia(aice, area, mask)
        fiv = self.compute_fiv(aice, hi, area, mask)
        fit = self.compute_fit(aice, hi, area, mask)
        fip = self.compute_fip(mask)
        fia_reg, fit_reg = self.compute_regional_fia_fit(aice, hi, area, mask, region_mask)
        seasonal = {}
        seasonal.update(self.compute_seasonal_summary(fia, "FIA"))
        seasonal.update(self.compute_seasonal_summary(fit, "FIT"))
        fipsi = self.persistence_stability_index(mask, area)
        skill = self._obs_skill_dataset(xr.Dataset({"FIA": fia, "FIT": fit}))

        ds_out = xr.Dataset(
            {
                "FIA": fia,
                "FIV": fiv,
                "FIT": fit,
                "FIP": fip,
                "FIA_by_region": fia_reg,
                "FIT_by_region": fit_reg,
                **seasonal,
                **fipsi,
            }
        )
        if len(skill.data_vars) > 0:
            ds_out = xr.merge([ds_out, skill], compat="override", combine_attrs="override")
        ds_out.attrs.update(
            {
                "sim_name": self.run.sim_name,
                "start_date": self.run.start_date,
                "end_date": self.run.end_date,
                "hemisphere": self.run.hemisphere,
                "ice_type": self.classify.ice_type,
                "grid_type": self.classify.grid_type,
                "method": norm,
            }
        )
        store = self.paths.metrics_store(norm)
        store.parent.mkdir(parents=True, exist_ok=True)
        if store.exists():
            if not overwrite:
                self.logger.info("Metrics store exists and overwrite=False, skipping: %s", store)
                return str(store)
            shutil.rmtree(store)
        # Rechunk first, as you already do
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking metrics output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)

        # Optional but recommended: strip non-dimension coords from metrics outputs too
        keep_coord_names = [c for c in ds_out.coords if c in {"time", "region"}]
        ds_out = ds_out.reset_coords(drop=True)
        for cname in keep_coord_names:
            if cname in ds_out:
                pass
            elif cname in ds_out.variables:
                pass

        # Better: rebuild with only desired coords
        coords = {}
        if "time" in ds_out.coords:
            coords["time"] = ds_out["time"]
        if "region" in ds_out.coords:
            coords["region"] = ds_out["region"]
        ds_out = xr.Dataset(
            data_vars={name: ds_out[name] for name in ds_out.data_vars},
            coords=coords,
            attrs=ds_out.attrs,
        )

        # Strip inherited encodings
        ds_out = _sanitize_for_zarr_write(ds_out)

        # Rebuild clean chunk encoding only
        encoding = {}
        for name, var in ds_out.variables.items():
            chunks = getattr(var.data, "chunks", None)
            if chunks is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in chunks)}

        self.logger.info("Writing metrics to %s", store)
        ds_out.to_zarr(
            store,
            mode="w",
            consolidated=False,
            encoding=encoding,
            zarr_format=2,
        )
        self._metrics_cache[norm] = ds_out
        return str(store)

    def _require_pygmt(self):
        try:
            import pygmt
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyGMT is required for plotting methods.") from exc
        return pygmt

    def _plotter(self):
        from shuga.plotting import CICEPlotter
        return CICEPlotter(
            run=self.run,
            classify=self.classify,
            metrics=self.metrics,
            paths=self.paths,
            chunks=self.chunks,
            logger=self.logger,
        )

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
