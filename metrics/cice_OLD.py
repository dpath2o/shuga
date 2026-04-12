
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

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
    for name in ds.variables:
        ds[name].encoding = {}
    ds.encoding = {}
    return ds


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


class CICEMetrics:
    """Incremental CICE metrics builder for FI and SI diagnostics."""

    # registry-style metric group aliases
    CORE_FI = [
        "FIA", "FIV", "FIT", "FIP",
        "FIS", "FITVR", "FIMVR", "FITAR", "FIMAR",
    ]
    CORE_SI = [
        "SIA", "SIV", "SIT", "SIP",
        "SIS", "SITVR", "SIMVR", "SITAR", "SIMAR",
    ]
    REGIONAL = ["FIA_by_region", "FIT_by_region", "SIA_by_region", "SIT_by_region"]
    SPATIAL = [
        "FIHI", "FIST", "FITVR_YR", "FIMVR_YR", "FITAR_YR", "FIMAR_YR",
        "SIHI", "SIST", "SITVR_YR", "SIMVR_YR", "SITAR_YR", "SIMAR_YR",
    ]
    SUMMARY = [
        "FIA_max_mean", "FIA_max_std", "FIA_min_mean", "FIA_min_std",
        "FIA_doy_max_mean", "FIA_doy_max_std", "FIA_doy_min_mean", "FIA_doy_min_std",
        "FIT_max_mean", "FIT_max_std", "FIT_min_mean", "FIT_min_std",
        "FIT_doy_max_mean", "FIT_doy_max_std", "FIT_doy_min_mean", "FIT_doy_min_std",
        "SIA_max_mean", "SIA_max_std", "SIA_min_mean", "SIA_min_std",
        "SIA_doy_max_mean", "SIA_doy_max_std", "SIA_doy_min_mean", "SIA_doy_min_std",
        "SIT_max_mean", "SIT_max_std", "SIT_min_mean", "SIT_min_std",
        "SIT_doy_max_mean", "SIT_doy_max_std", "SIT_doy_min_mean", "SIT_doy_min_std",
        "FIPSI", "persistent_winter_area", "ever_winter_area",
        "FIA_Bias", "FIA_RMSE", "FIA_MAE", "FIA_Corr",
        "FIT_Bias", "FIT_RMSE", "FIT_MAE", "FIT_Corr",
    ]
    STRESS = [
        "FIKuxE_mean", "FIKuxE_abs_mean", "FIKuxE_valid_area_m2",
        "FIKuyE_mean", "FIKuyE_abs_mean", "FIKuyE_valid_area_m2",
        "FIKuxN_mean", "FIKuxN_abs_mean", "FIKuxN_valid_area_m2",
        "FIKuyN_mean", "FIKuyN_abs_mean", "FIKuyN_valid_area_m2",
        "FIKuE_mag_mean", "FIKuE_mag_abs_mean", "FIKuE_mag_valid_area_m2",
        "FIKuN_mag_mean", "FIKuN_mag_abs_mean", "FIKuN_mag_valid_area_m2",
        "SIKuxE_mean", "SIKuxE_abs_mean", "SIKuxE_valid_area_m2",
        "SIKuyE_mean", "SIKuyE_abs_mean", "SIKuyE_valid_area_m2",
        "SIKuxN_mean", "SIKuxN_abs_mean", "SIKuxN_valid_area_m2",
        "SIKuyN_mean", "SIKuyN_abs_mean", "SIKuyN_valid_area_m2",
        "SIKuE_mag_mean", "SIKuE_mag_abs_mean", "SIKuE_mag_valid_area_m2",
        "SIKuN_mag_mean", "SIKuN_mag_abs_mean", "SIKuN_mag_valid_area_m2",
    ]

    METRIC_GROUPS = {
        "fi_core": CORE_FI,
        "si_core": CORE_SI,
        "regional": REGIONAL,
        "spatial": SPATIAL,
        "summary": SUMMARY,
        "stress": STRESS,
        "default": CORE_FI + CORE_SI + REGIONAL + SPATIAL + SUMMARY + STRESS,
        "all": CORE_FI + CORE_SI + REGIONAL + SPATIAL + SUMMARY + STRESS,
    }

    FIPSI_NAMES = {"FIPSI", "persistent_winter_area", "ever_winter_area"}
    FIA_SKILL_NAMES = {"FIA_Bias", "FIA_RMSE", "FIA_MAE", "FIA_Corr"}
    FIT_SKILL_NAMES = {"FIT_Bias", "FIT_RMSE", "FIT_MAE", "FIT_Corr"}
    FIA_SEASONAL_NAMES = {
        "FIA_max_mean", "FIA_max_std", "FIA_min_mean", "FIA_min_std",
        "FIA_doy_max_mean", "FIA_doy_max_std", "FIA_doy_min_mean", "FIA_doy_min_std",
    }
    FIT_SEASONAL_NAMES = {
        "FIT_max_mean", "FIT_max_std", "FIT_min_mean", "FIT_min_std",
        "FIT_doy_max_mean", "FIT_doy_max_std", "FIT_doy_min_mean", "FIT_doy_min_std",
    }
    SIA_SEASONAL_NAMES = {
        "SIA_max_mean", "SIA_max_std", "SIA_min_mean", "SIA_min_std",
        "SIA_doy_max_mean", "SIA_doy_max_std", "SIA_doy_min_mean", "SIA_doy_min_std",
    }
    SIT_SEASONAL_NAMES = {
        "SIT_max_mean", "SIT_max_std", "SIT_min_mean", "SIT_min_std",
        "SIT_doy_max_mean", "SIT_doy_max_std", "SIT_doy_min_mean", "SIT_doy_min_std",
    }

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
        self.logger = logger or build_file_logger("shuga.metrics", self.paths.metrics_log_path())
        self.region_defs = ANTARCTIC_8_REGIONS
        self._cice_cache: xr.Dataset | None = None
        self._classified_cache: dict[str, xr.Dataset] = {}
        self._metrics_cache: dict[str, xr.Dataset] = {}

    @property
    def mask_var_name(self) -> str:
        return f"{self.classify.ice_type}_mask"

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
        return [dim for dim in da.dims if dim not in {"time", "region"}]

    def _output_chunk_map(self, ds: xr.Dataset) -> dict[str, int]:
        chunk_map: dict[str, int] = {}
        if "time" in ds.dims:
            chunk_map["time"] = int(self.chunks.get("time", 31))
        for dim in ds.dims:
            if dim != "time":
                chunk_map[dim] = -1
        return chunk_map

    def _assert_same_indexes(self, existing: xr.Dataset, ds_new: xr.Dataset, dims=("time", "region")) -> None:
        for dim in dims:
            if dim in existing.coords and dim in ds_new.coords:
                a = existing.indexes.get(dim, None)
                b = ds_new.indexes.get(dim, None)
                if a is not None and b is not None and not a.equals(b):
                    raise ValueError(f"Cannot append metrics: coordinate mismatch on '{dim}'. "
                                     f"existing={existing.sizes.get(dim)} new={ds_new.sizes.get(dim)}")

    def _encoding_from_dataset(self, ds: xr.Dataset) -> dict[str, dict]:
        encoding = {}
        for name, var in ds.variables.items():
            chunks = getattr(var.data, "chunks", None)
            if chunks is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in chunks)}
        return encoding

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

    def _si_mask(self, aice: xr.DataArray) -> xr.DataArray:
        thresh = float(getattr(self.classify, "aice_thresh", 0.15))
        return xr.where(aice >= thresh, True, False)

    def _expand_metric_names(self, metric_names=None, metric_groups=None) -> list[str]:
        explicit = _as_list(metric_names)
        groups = _as_list(metric_groups) or ["default"]
        out = set(explicit)
        for group in groups:
            key = group.strip().lower()
            if key not in self.METRIC_GROUPS:
                raise ValueError(
                    f"Unknown metric group '{group}'. "
                    f"Valid groups: {sorted(self.METRIC_GROUPS)}"
                )
            out.update(self.METRIC_GROUPS[key])
        return sorted(out)

    def _open_existing_metrics(self, method: str) -> xr.Dataset | None:
        norm = normalize_method(method)
        if norm in self._metrics_cache:
            return self._metrics_cache[norm]
        store = self.paths.metrics_store(norm)
        if not store.exists():
            return None
        ds = xr.open_zarr(store, consolidated=False)
        self._metrics_cache[norm] = ds
        return ds

    def _get_cice(self) -> xr.Dataset:
        if self._cice_cache is None:
            requested = [
                "aice", "hi", "strength",
                "dvidtt", "dvidtd", "daidtt", "daidtd",
                "KuxE", "KuxN", "KuyE", "KuyN",
                "earea", "narea", "uarea",
                "tarea", "TLON", "TLAT", "ULON", "ULAT",
            ]
            self.logger.info("Resolved CICE store: %s", self.paths.resolve_cice_store())
            static_store = self.paths.resolve_static_store()
            if static_store is not None:
                self.logger.info("Resolved static store: %s", static_store)
            self._cice_cache = load_cice(
                run=self.run,
                classify=self.classify,
                metrics=self.metrics,
                paths=self.paths,
                variables=requested,
                hemisphere=self.run.hemisphere,
                chunks=self.chunks,
            )
        return self._cice_cache

    def _get_classified(self, method: str) -> xr.Dataset:
        norm = normalize_method(method)
        if norm not in self._classified_cache:
            self._classified_cache[norm] = load_classified(
                run=self.run,
                classify=self.classify,
                metrics=self.metrics,
                paths=self.paths,
                classification=norm,
                dt0_str=self.run.start_date,
                dtN_str=self.run.end_date,
                hemisphere=self.run.hemisphere,
                chunks=self.chunks,
            )
        return self._classified_cache[norm]

    def compute_area_series(
        self,
        sic: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
        scale: float | None = None,
    ) -> xr.DataArray:
        weighted = sic.where(mask, 0.0) if mask is not None else sic
        da = (weighted * area).sum(dim=self._spatial_dims(sic))
        if scale is not None:
            da = da / scale
            units = "10^3 km^2"
        else:
            units = "m^2"
        da.name = name
        da.attrs.update({"long_name": long_name, "units": units})
        return da

    def compute_volume_series(
        self,
        sic: xr.DataArray,
        hi: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
        scale: float | None = None,
    ) -> xr.DataArray:
        c = sic.where(mask, 0.0) if mask is not None else sic
        h = hi.fillna(0.0).where(mask, 0.0) if mask is not None else hi.fillna(0.0)
        da = (c * h * area).sum(dim=self._spatial_dims(sic))
        if scale is not None:
            da = da / scale
            units = "10^3 km^3"
        else:
            units = "m^3"
        da.name = name
        da.attrs.update({"long_name": long_name, "units": units})
        return da

    def compute_thickness_series(
        self,
        sic: xr.DataArray,
        hi: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
    ) -> xr.DataArray:
        c = sic.where(mask, 0.0) if mask is not None else sic
        h = hi.fillna(0.0).where(mask, 0.0) if mask is not None else hi.fillna(0.0)
        vol = (c * h * area).sum(dim=self._spatial_dims(sic))
        are = (c * area).sum(dim=self._spatial_dims(sic))
        da = xr.where(are > 0, vol / are, np.nan)
        da.name = name
        da.attrs.update({"long_name": long_name, "units": "m"})
        return da

    def compute_persistence_mask(self, mask: xr.DataArray, *, name: str, long_name: str) -> xr.DataArray:
        da = mask.astype("float32").mean(dim="time")
        da.name = name
        da.attrs.update({"long_name": long_name, "units": "1"})
        return da

    def compute_temporal_mean(self, da: xr.DataArray, *, name: str, long_name: str) -> xr.DataArray:
        out = da.mean(dim="time")
        out.name = name
        out.attrs.update({"long_name": long_name, "units": da.attrs.get("units", "")})
        return out

    def compute_strength_series(
        self,
        sic: xr.DataArray,
        hi: xr.DataArray,
        strength: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
    ) -> xr.DataArray:
        valid = hi > 0
        if mask is not None:
            valid = valid & mask
        pressure_pa = xr.where(valid, strength / hi.where(hi > 0), np.nan)
        weights = xr.where(valid, sic * area, 0.0)
        num = (pressure_pa * weights).sum(dim=self._spatial_dims(sic), skipna=True)
        den = weights.sum(dim=self._spatial_dims(sic))
        da = xr.where(den > 0, num / den / 100.0, np.nan)
        da.name = name
        da.attrs.update({"long_name": long_name, "units": "hPa"})
        return da

    def _convert_thickness_tendency_to_m_per_day(self, da: xr.DataArray) -> xr.DataArray:
        units = str(da.attrs.get("units", "")).lower().replace(" ", "")
        if units in {"cm/day", "cmday-1", "cmd-1"}:
            return da / 100.0
        if units in {"m/day", "mday-1", "md-1"}:
            return da
        if units in {"m/s", "ms-1"}:
            return da * 86400.0
        return da / 100.0

    def compute_volume_rate(
        self,
        dvt: xr.DataArray,
        sic: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
    ) -> xr.DataArray:
        thick_rate = self._convert_thickness_tendency_to_m_per_day(dvt)
        c = sic.where(mask, 0.0) if mask is not None else sic
        dV_day = (thick_rate.where(mask, 0.0) if mask is not None else thick_rate) * c * area
        da = dV_day.sum(dim=self._spatial_dims(sic)) / self.metrics.volume_scale
        da.name = name
        da.attrs.update({"long_name": long_name, "units": "10^3 km^3/day"})
        return da

    def compute_area_rate(
        self,
        dat: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
    ) -> xr.DataArray:
        field = dat.where(mask, 0.0) if mask is not None else dat
        da = (field * area).sum(dim=self._spatial_dims(field))
        da = da / 1e6 * 86400.0 / 1e3
        da.name = name
        da.attrs.update({"long_name": long_name, "units": "10^3 km^2/day"})
        return da

    def compute_spatial_rate_year(
        self,
        da: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        name: str,
        long_name: str,
        area: xr.DataArray | None = None,
    ) -> xr.DataArray:
        field = da.where(mask, np.nan) if mask is not None else da
        units = str(da.attrs.get("units", "")).lower().replace(" ", "")
        if area is not None:
            out = ((field * area) / 31536000.0).mean(dim="time")
            out.attrs["units"] = "m^2 yr^-1"
        elif units in {"cm/day", "cmday-1", "cmd-1"}:
            out = (field / 100.0) * 365.0
            out = out.mean(dim="time")
            out.attrs["units"] = "m/yr"
        elif units in {"m/day", "mday-1", "md-1"}:
            out = (field * 365.0).mean(dim="time")
            out.attrs["units"] = "m/yr"
        elif units in {"m/s", "ms-1"}:
            out = (field * 31536000.0).mean(dim="time")
            out.attrs["units"] = "m/yr"
        else:
            out = field.mean(dim="time")
            out.attrs["units"] = da.attrs.get("units", "")
        out.name = name
        out.attrs["long_name"] = long_name
        return out

    def compute_region_series(
        self,
        sic: xr.DataArray,
        hi: xr.DataArray,
        area: xr.DataArray,
        region_mask: xr.DataArray,
        mask: xr.DataArray | None = None,
        *,
        area_name: str,
        thickness_name: str,
        area_long_name: str,
        thickness_long_name: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        c = sic.where(mask, 0.0) if mask is not None else sic
        h = hi.fillna(0.0).where(mask, 0.0) if mask is not None else hi.fillna(0.0)
        weighted_area = c * area
        weighted_vol = c * h * area
        spatial_dims = self._spatial_dims(sic)
        region_area = weighted_area.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        region_vol = weighted_vol.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        area_reg = (region_area.sum(dim=spatial_dims) / self.metrics.area_scale).transpose("time", "region")
        thick_num = region_vol.sum(dim=spatial_dims).transpose("time", "region")
        thick_den = region_area.sum(dim=spatial_dims).transpose("time", "region")
        thick_reg = xr.where(thick_den > 0, thick_num / thick_den, np.nan)
        area_reg.name = area_name
        thick_reg.name = thickness_name
        area_reg.attrs.update({"long_name": area_long_name, "units": "10^3 km^2"})
        thick_reg.attrs.update({"long_name": thickness_long_name, "units": "m"})
        return area_reg, thick_reg

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

    def _match_mask_to_field(self, mask: xr.DataArray | None, field: xr.DataArray) -> xr.DataArray | None:
        if mask is None:
            return None
        try:
            return mask.broadcast_like(field)
        except Exception:
            return None

    def compute_area_weighted_stress(
        self,
        tau: xr.DataArray,
        area: xr.DataArray,
        mask: xr.DataArray | None,
        *,
        base_name: str,
    ) -> xr.Dataset:
        spatial_dims = self._spatial_dims(tau)
        valid = np.isfinite(tau)
        mask_use = self._match_mask_to_field(mask, tau)
        if mask_use is not None:
            valid = valid & mask_use
        weights = xr.where(valid, area, 0.0)
        num_mean = (tau.where(valid, 0.0) * weights).sum(dim=spatial_dims)
        num_abs = (np.abs(tau.where(valid, 0.0)) * weights).sum(dim=spatial_dims)
        den = weights.sum(dim=spatial_dims)
        mean = xr.where(den > 0, num_mean / den, np.nan)
        abs_mean = xr.where(den > 0, num_abs / den, np.nan)
        mean.name = f"{base_name}_mean"
        abs_mean.name = f"{base_name}_abs_mean"
        den.name = f"{base_name}_valid_area_m2"
        mean.attrs.update({"long_name": f"{base_name} area-weighted mean stress", "units": tau.attrs.get("units", "Pa")})
        abs_mean.attrs.update({"long_name": f"{base_name} area-weighted mean absolute stress", "units": tau.attrs.get("units", "Pa")})
        den.attrs.update({"long_name": f"{base_name} valid area", "units": "m^2"})
        return xr.Dataset({mean.name: mean, abs_mean.name: abs_mean, den.name: den})

    def _compute_requested_metrics(self, method: str, requested: set[str]) -> xr.Dataset:
        ds = self._get_cice()
        aice = ds["aice"]
        hi = ds["hi"]
        area = self._ensure_2d_static(ds["tarea"])
        lon, lat = self._detect_lonlat(ds)
        region_mask = self._region_mask(area, lon, lat)

        need_fi = any(name.startswith("FI") or name.startswith("FIA_") or name.startswith("FIT_") or name in self.FIPSI_NAMES for name in requested)
        ds_mask = self._get_classified(method) if need_fi else None
        fi_mask = ds_mask["FI_mask"].astype(bool) if ds_mask is not None else None
        if fi_mask is not None:
            aice, hi, fi_mask = xr.align(aice, hi, fi_mask, join="inner")
        si_mask = self._si_mask(aice)

        out = xr.Dataset()
        memo: dict[str, xr.DataArray] = {}

        def remember(name: str, da: xr.DataArray):
            memo[name] = da
            out[name] = da

        def get_or_compute(name: str) -> xr.DataArray | None:
            if name in memo:
                return memo[name]
            if name == "FIA" and fi_mask is not None:
                remember(name, self.compute_area_series(aice, area, fi_mask, name="FIA", long_name="Fast Ice Area", scale=self.metrics.area_scale))
            elif name == "FIV" and fi_mask is not None:
                remember(name, self.compute_volume_series(aice, hi, area, fi_mask, name="FIV", long_name="Fast Ice Volume", scale=self.metrics.volume_scale))
            elif name == "FIT" and fi_mask is not None:
                remember(name, self.compute_thickness_series(aice, hi, area, fi_mask, name="FIT", long_name="Fast Ice Thickness"))
            elif name == "FIP" and fi_mask is not None:
                remember(name, self.compute_persistence_mask(fi_mask, name="FIP", long_name="Fast Ice Persistence"))
            elif name == "FIS" and fi_mask is not None and "strength" in ds:
                remember(name, self.compute_strength_series(aice, hi, ds["strength"], area, fi_mask, name="FIS", long_name="Fast Ice Strength"))
            elif name == "FITVR" and fi_mask is not None and "dvidtt" in ds:
                remember(name, self.compute_volume_rate(ds["dvidtt"], aice, area, fi_mask, name="FITVR", long_name="Fast Ice Thermodynamic Volume Rate"))
            elif name == "FIMVR" and fi_mask is not None and "dvidtd" in ds:
                remember(name, self.compute_volume_rate(ds["dvidtd"], aice, area, fi_mask, name="FIMVR", long_name="Fast Ice Dynamic Volume Rate"))
            elif name == "FITAR" and fi_mask is not None and "daidtt" in ds:
                remember(name, self.compute_area_rate(ds["daidtt"], area, fi_mask, name="FITAR", long_name="Fast Ice Thermodynamic Area Rate"))
            elif name == "FIMAR" and fi_mask is not None and "daidtd" in ds:
                remember(name, self.compute_area_rate(ds["daidtd"], area, fi_mask, name="FIMAR", long_name="Fast Ice Dynamic Area Rate"))

            elif name == "SIA":
                remember(name, self.compute_area_series(aice, area, None, name="SIA", long_name="Sea Ice Area", scale=self.metrics.area_scale))
            elif name == "SIV":
                remember(name, self.compute_volume_series(aice, hi, area, None, name="SIV", long_name="Sea Ice Volume", scale=self.metrics.volume_scale))
            elif name == "SIT":
                remember(name, self.compute_thickness_series(aice, hi, area, None, name="SIT", long_name="Sea Ice Thickness"))
            elif name == "SIP":
                remember(name, self.compute_temporal_mean(aice, name="SIP", long_name="Sea Ice Mean Concentration"))
            elif name == "SIS" and "strength" in ds:
                remember(name, self.compute_strength_series(aice, hi, ds["strength"], area, si_mask, name="SIS", long_name="Sea Ice Strength"))
            elif name == "SITVR" and "dvidtt" in ds:
                remember(name, self.compute_volume_rate(ds["dvidtt"], aice, area, si_mask, name="SITVR", long_name="Sea Ice Thermodynamic Volume Rate"))
            elif name == "SIMVR" and "dvidtd" in ds:
                remember(name, self.compute_volume_rate(ds["dvidtd"], aice, area, si_mask, name="SIMVR", long_name="Sea Ice Dynamic Volume Rate"))
            elif name == "SITAR" and "daidtt" in ds:
                remember(name, self.compute_area_rate(ds["daidtt"], area, si_mask, name="SITAR", long_name="Sea Ice Thermodynamic Area Rate"))
            elif name == "SIMAR" and "daidtd" in ds:
                remember(name, self.compute_area_rate(ds["daidtd"], area, si_mask, name="SIMAR", long_name="Sea Ice Dynamic Area Rate"))

            elif name == "FIHI" and fi_mask is not None:
                remember(name, self.compute_temporal_mean(hi.where(fi_mask), name="FIHI", long_name="Fast Ice Mean Thickness"))
            elif name == "SIHI":
                remember(name, self.compute_temporal_mean(hi.where(si_mask), name="SIHI", long_name="Sea Ice Mean Thickness"))
            elif name == "FIST" and fi_mask is not None and "strength" in ds:
                sfield = xr.where(fi_mask & (hi > 0), ds["strength"] / hi.where(hi > 0) / 1e6, np.nan)
                remember(name, sfield.sum(dim="time").rename("FIST"))
                memo[name].attrs.update({"long_name": "Fast Ice Temporal Sum Strength", "units": "MPa"})
                out[name] = memo[name]
            elif name == "SIST" and "strength" in ds:
                sfield = xr.where(si_mask & (hi > 0), ds["strength"] / hi.where(hi > 0) / 1e6, np.nan)
                remember(name, sfield.sum(dim="time").rename("SIST"))
                memo[name].attrs.update({"long_name": "Sea Ice Temporal Sum Strength", "units": "MPa"})
                out[name] = memo[name]
            elif name == "FITVR_YR" and fi_mask is not None and "dvidtt" in ds:
                remember(name, self.compute_spatial_rate_year(ds["dvidtt"], fi_mask, name="FITVR_YR", long_name="Fast Ice Thermodynamic Volume Rate Climatology"))
            elif name == "FIMVR_YR" and fi_mask is not None and "dvidtd" in ds:
                remember(name, self.compute_spatial_rate_year(ds["dvidtd"], fi_mask, name="FIMVR_YR", long_name="Fast Ice Dynamic Volume Rate Climatology"))
            elif name == "FITAR_YR" and fi_mask is not None and "daidtt" in ds:
                remember(name, self.compute_spatial_rate_year(ds["daidtt"], fi_mask, name="FITAR_YR", long_name="Fast Ice Thermodynamic Area Rate Climatology", area=area))
            elif name == "FIMAR_YR" and fi_mask is not None and "daidtd" in ds:
                remember(name, self.compute_spatial_rate_year(ds["daidtd"], fi_mask, name="FIMAR_YR", long_name="Fast Ice Dynamic Area Rate Climatology", area=area))
            elif name == "SITVR_YR" and "dvidtt" in ds:
                remember(name, self.compute_spatial_rate_year(ds["dvidtt"], si_mask, name="SITVR_YR", long_name="Sea Ice Thermodynamic Volume Rate Climatology"))
            elif name == "SIMVR_YR" and "dvidtd" in ds:
                remember(name, self.compute_spatial_rate_year(ds["dvidtd"], si_mask, name="SIMVR_YR", long_name="Sea Ice Dynamic Volume Rate Climatology"))
            elif name == "SITAR_YR" and "daidtt" in ds:
                remember(name, self.compute_spatial_rate_year(ds["daidtt"], si_mask, name="SITAR_YR", long_name="Sea Ice Thermodynamic Area Rate Climatology", area=area))
            elif name == "SIMAR_YR" and "daidtd" in ds:
                remember(name, self.compute_spatial_rate_year(ds["daidtd"], si_mask, name="SIMAR_YR", long_name="Sea Ice Dynamic Area Rate Climatology", area=area))

            elif name == "FIA_by_region" and fi_mask is not None:
                fia_reg, fit_reg = self.compute_region_series(
                    aice, hi, area, region_mask, fi_mask,
                    area_name="FIA_by_region",
                    thickness_name="FIT_by_region",
                    area_long_name="Fast Ice Area by Antarctic sector",
                    thickness_long_name="Fast Ice Thickness by Antarctic sector",
                )
                remember("FIA_by_region", fia_reg)
                remember("FIT_by_region", fit_reg)
            elif name == "SIA_by_region":
                sia_reg, sit_reg = self.compute_region_series(
                    aice, hi, area, region_mask, None,
                    area_name="SIA_by_region",
                    thickness_name="SIT_by_region",
                    area_long_name="Sea Ice Area by Antarctic sector",
                    thickness_long_name="Sea Ice Thickness by Antarctic sector",
                )
                remember("SIA_by_region", sia_reg)
                remember("SIT_by_region", sit_reg)

            return memo.get(name)

        for primary in [
            "FIA", "FIV", "FIT", "FIP", "FIS", "FITVR", "FIMVR", "FITAR", "FIMAR",
            "SIA", "SIV", "SIT", "SIP", "SIS", "SITVR", "SIMVR", "SITAR", "SIMAR",
            "FIHI", "FIST", "FITVR_YR", "FIMVR_YR", "FITAR_YR", "FIMAR_YR",
            "SIHI", "SIST", "SITVR_YR", "SIMVR_YR", "SITAR_YR", "SIMAR_YR",
            "FIA_by_region", "FIT_by_region", "SIA_by_region", "SIT_by_region",
        ]:
            if primary in requested:
                get_or_compute(primary)

        seasonal_requests = {
            "FIA": self.FIA_SEASONAL_NAMES,
            "FIT": self.FIT_SEASONAL_NAMES,
            "SIA": self.SIA_SEASONAL_NAMES,
            "SIT": self.SIT_SEASONAL_NAMES,
        }
        for base, names in seasonal_requests.items():
            if requested & names:
                base_da = get_or_compute(base)
                if base_da is not None:
                    seasonal = self.compute_seasonal_summary(base_da, base)
                    for nm, da in seasonal.items():
                        if nm in requested:
                            out[nm] = da

        if requested & self.FIPSI_NAMES and fi_mask is not None:
            fipsi = self.persistence_stability_index(fi_mask, area)
            for nm, da in fipsi.items():
                if nm in requested:
                    out[nm] = da

        if requested & (self.FIA_SKILL_NAMES | self.FIT_SKILL_NAMES):
            base_ds = xr.Dataset()
            if requested & self.FIA_SKILL_NAMES:
                fia = get_or_compute("FIA")
                if fia is not None:
                    base_ds["FIA"] = fia
            if requested & self.FIT_SKILL_NAMES:
                fit = get_or_compute("FIT")
                if fit is not None:
                    base_ds["FIT"] = fit
            if len(base_ds.data_vars) > 0:
                skill = self._obs_skill_dataset(base_ds)
                for nm in skill.data_vars:
                    if nm in requested:
                        out[nm] = skill[nm]

        def maybe_stress(prefix: str, mask: xr.DataArray | None):
            area_e = self._ensure_2d_static(ds["earea"]) if "earea" in ds else area
            area_n = self._ensure_2d_static(ds["narea"]) if "narea" in ds else area
            mapping = [
                ("KuxE", area_e, f"{prefix}KuxE"),
                ("KuyE", area_e, f"{prefix}KuyE"),
                ("KuxN", area_n, f"{prefix}KuxN"),
                ("KuyN", area_n, f"{prefix}KuyN"),
            ]
            for varname, weights, base in mapping:
                needed = {f"{base}_mean", f"{base}_abs_mean", f"{base}_valid_area_m2"}
                if requested & needed and varname in ds:
                    dsi = self.compute_area_weighted_stress(ds[varname], weights, mask, base_name=base)
                    for nm in dsi.data_vars:
                        if nm in requested:
                            out[nm] = dsi[nm]
            mag_specs = [
                ("KuxE", "KuyE", area_e, f"{prefix}KuE_mag"),
                ("KuxN", "KuyN", area_n, f"{prefix}KuN_mag"),
            ]
            for xname, yname, weights, base in mag_specs:
                needed = {f"{base}_mean", f"{base}_abs_mean", f"{base}_valid_area_m2"}
                if requested & needed and xname in ds and yname in ds:
                    mag = xr.apply_ufunc(np.hypot, ds[xname], ds[yname], dask="allowed")
                    mag.attrs["units"] = ds[xname].attrs.get("units", "Pa")
                    dsi = self.compute_area_weighted_stress(mag, weights, mask, base_name=base)
                    for nm in dsi.data_vars:
                        if nm in requested:
                            out[nm] = dsi[nm]

        if requested & set(self.STRESS):
            if fi_mask is not None:
                maybe_stress("FI", fi_mask)
            maybe_stress("SI", si_mask)

        return out

    def _prepare_output_dataset(self, ds_out: xr.Dataset) -> xr.Dataset:
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking metrics output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)
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
        ds_out = _sanitize_for_zarr_write(ds_out)
        return ds_out

    def compute_metrics(
        self,
        method: str,
        *,
        overwrite: bool = False,
        metric_names: str | Iterable[str] | None = None,
        metric_groups: str | Iterable[str] | None = None,
        update_missing_only: bool = True,
    ) -> str:
        norm = normalize_method(method)
        requested = set(self._expand_metric_names(metric_names=metric_names, metric_groups=metric_groups))
        self.logger.info("Resolved class store for %s: %s", norm, self.paths.classification_store(norm))
        self.logger.info("Resolved metrics store for %s: %s", norm, self.paths.metrics_store(norm))
        if not requested:
            raise ValueError("No metrics requested.")
        store = self.paths.metrics_store(norm)
        # existing = None if overwrite else self._open_existing_metrics(norm)
        # if existing is None:
        #     to_compute = requested
        # elif update_missing_only:
        #     to_compute = {name for name in requested if name not in existing.data_vars}
        # else:
        #     to_compute = requested
        # if not to_compute and existing is not None:
        #     self.logger.info("All requested metrics already present for %s; nothing to do.", norm)
        #     return str(store)

        # self.logger.info("Requested metrics (%d): %s", len(requested), ", ".join(sorted(requested)))
        # self.logger.info("Computing metrics (%d): %s", len(to_compute), ", ".join(sorted(to_compute)))
        # ds_new = self._compute_requested_metrics(norm, to_compute)

        # if existing is not None and not overwrite:
        #     ds_out = xr.merge([existing, ds_new], compat="override", combine_attrs="override")
        # else:
        #     ds_out = ds_new
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
        ds_new = self._compute_requested_metrics(norm, to_compute)
        ds_new.attrs.update({"sim_name"   : self.run.sim_name,
                             "start_date" : self.run.start_date,
                             "end_date"   : self.run.end_date,
                             "hemisphere" : self.run.hemisphere,
                             "ice_type"   : self.classify.ice_type,
                             "grid_type"  : self.classify.grid_type,
                             "method"     : norm}) #attrs.update({...})
        ds_new = self._prepare_output_dataset(ds_new)
        store.parent.mkdir(parents=True, exist_ok=True)
        if existing is not None and update_missing_only and not overwrite:
            overlap = set(ds_new.data_vars) & set(existing.data_vars)
            if overlap:
                raise ValueError(f"Refusing to append overlapping metric names: {sorted(overlap)}")
            self._assert_same_indexes(existing, ds_new, dims=("time", "region"))
            self.logger.info("Appending new metrics to %s", store)
            ds_new.to_zarr(store,
                           mode="a",
                           consolidated=False,
                           encoding=self._encoding_from_dataset(ds_new),
                           zarr_format=2)
            self._metrics_cache.pop(norm, None)
            return str(store)
        # only use full rewrite path for overwrite/full-recompute
        ds_out = ds_new if (existing is None or overwrite) else xr.merge([existing, ds_new],
                                                                         compat        = "override",
                                                                         combine_attrs = "override",
                                                                         join          = "exact")
        # ds_out.attrs.update(
        #     {
        #         "sim_name": self.run.sim_name,
        #         "start_date": self.run.start_date,
        #         "end_date": self.run.end_date,
        #         "hemisphere": self.run.hemisphere,
        #         "ice_type": self.classify.ice_type,
        #         "grid_type": self.classify.grid_type,
        #         "method": norm,
        #     }
        # )
        # ds_out = self._prepare_output_dataset(ds_out)
        # store.parent.mkdir(parents=True, exist_ok=True)
        tmp_store = store.with_name(store.name + ".tmp")
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        if overwrite and store.exists():
            shutil.rmtree(store)
        encoding = {}
        for name, var in ds_out.variables.items():
            chunks = getattr(var.data, "chunks", None)
            if chunks is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in chunks)}
        self.logger.info("Writing metrics to %s", tmp_store)
        ds_out.to_zarr(tmp_store,
                       mode="w",
                       consolidated=False,
                       encoding=encoding,
                       zarr_format=2)
        if store.exists():
            shutil.rmtree(store)
        tmp_store.rename(store)
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
