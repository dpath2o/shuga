from __future__ import annotations
import shutil
from importlib.resources   import files
from pathlib               import Path
from typing                import Mapping, Sequence
import numpy               as np
import pandas              as pd
import xarray              as xr
from shuga.core.logging    import build_file_logger
from shuga.core.naming     import normalize_method
from shuga.core.paths      import ShugaPaths
from shuga.core.regions    import ANTARCTIC_8_REGIONS
from shuga.core.types      import ClassificationSpec, MetricsSpec, RunSpec
from shuga.io.zarr_loading import open_cice_history

# FIP plotting helper functions
def _region_dict_from_source(regions: Mapping[str, Sequence[float]] | None = None) -> dict[str, tuple[float, float, float, float]]:
    if regions is not None:
        return {k: tuple(map(float, v)) for k, v in regions.items()}
    out: dict[str, tuple[float, float, float, float]] = {}
    for name, meta in ANTARCTIC_8_REGIONS.items():
        if isinstance(meta, Mapping):
            bounds = meta.get("geo_region", meta.get("region", None))
        else:
            bounds = meta
        if bounds is None:
            raise KeyError(f"Region {name!r} does not define 'geo_region' or equivalent bounds")
        out[name] = tuple(map(float, bounds))
    return out

def _normalize_lon180(x: float | np.ndarray) -> float | np.ndarray:
    return ((np.asarray(x) + 180.0) % 360.0) - 180.0

def _region_mask_and_plot_lon(lon_flat_180: np.ndarray, lat_flat: np.ndarray, bounds: Sequence[float]) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    lon_min, lon_max, lat_min, lat_max = map(float, bounds)
    lon_min                            = float(_normalize_lon180(lon_min))
    lon_max                            = float(_normalize_lon180(lon_max))
    wraps                              = lon_min > lon_max
    plot_lon                           = lon_flat_180.copy()
    if wraps:
        # Example: [160, -130] after 0..360 -> -180..180 normalization.
        # Shift points west of lon_min by +360 so the plotted domain is continuous.
        plot_lon     = np.where(plot_lon < lon_min, plot_lon + 360.0, plot_lon)
        lon_max_plot = lon_max + 360.0
        inside_lon   = (lon_flat_180 >= lon_min) | (lon_flat_180 <= lon_max)
        region_plot  = [lon_min, lon_max_plot, lat_min, lat_max]
        mc           = lon_min + 0.5 * (lon_max_plot - lon_min)
    else:
        inside_lon  = (lon_flat_180 >= lon_min) & (lon_flat_180 <= lon_max)
        region_plot = [lon_min, lon_max, lat_min, lat_max]
        mc          = lon_min + 0.5 * (lon_max - lon_min)
    inside_lat = (lat_flat >= lat_min) & (lat_flat <= lat_max)
    mask       = inside_lon & inside_lat
    mc         = float(_normalize_lon180(mc))
    return mask, plot_lon, region_plot, mc

def _region_projection(bounds: Sequence[float], fig_size: float = 20.0) -> str:
    lon_min, lon_max, lat_min, lat_max = map(float, bounds)
    lat_c                              = 0.5 * (lat_min + lat_max)
    lon_min_n                          = float(_normalize_lon180(lon_min))
    lon_max_n                          = float(_normalize_lon180(lon_max))
    if lon_min_n > lon_max_n:
        lon_max_n += 360.0
    mc   = float(_normalize_lon180(lon_min_n + 0.5 * (lon_max_n - lon_min_n)))
    hemi = "S" if lat_c < 0 else "N"
    pole = "-90" if lat_c < 0 else "90"
    return f"{hemi}{mc:g}/{pole}/{fig_size:g}c"

def _default_fip_cpt() -> str:
    return str(files("shuga").joinpath("cpt", "FIP.cpt"))

def _default_fip_root(self) -> Path:
    # Preferred: a graphics root stored on the paths object.
    for attr in ("graphics_root", "plot_root", "figure_root"):
        val = getattr(self.paths, attr, None)
        if val:
            return Path(val).expanduser()
    # HPC-style fallback if project/user exist.
    project = getattr(self.run, "project", None)
    user = getattr(self.run, "user", None)
    if project and user:
        return Path("/g/data") / str(project) / str(user) / "GRAPHICAL" / "AFIM"
    # Final local fallback.
    return Path.home() / "GRAPHICAL" / "AFIM"

class CICEMetrics:
    """Compute and load fast-ice metrics for raw, binary-days, and rolling-mean masks."""

    def __init__(self, run: RunSpec, classify: ClassificationSpec,
                 metrics : MetricsSpec | None = None,
                 paths   : ShuggaPaths | None = None, *,
                 chunks  : dict | None        = None,
                 logger                       = None) -> None:
        self.run         = run
        self.classify    = classify
        self.metrics     = metrics or MetricsSpec()
        self.paths       = paths or ShugaPaths(run=run, classify=classify)
        self.chunks      = chunks or {"time": 31}
        self.logger      = logger or build_file_logger("shuga.metrics", self.paths.metrics_log_path())
        self.region_defs = ANTARCTIC_8_REGIONS
        self._cice_cache: xr.Dataset | None = None
        self._metrics_cache: dict[str, xr.Dataset] = {}

    @property
    def mask_var_name(self) -> str:
        return f"{self.classify.ice_type}_mask"

    def load_cice(self, variables: list[str] | None = None) -> xr.Dataset:
        if self._cice_cache is None:
            vars_keep = variables or ["aice", "hi", "tarea", "TLON", "TLAT"]
            self.logger.info("Resolved CICE store: %s", self.paths.resolve_cice_store())
            static_store = self.paths.resolve_static_store()
            if static_store is not None:
                self.logger.info("Resolved static store: %s", static_store)
            self._cice_cache = open_cice_history(self.paths,
                                                 variables   = vars_keep,
                                                 extend_days = 0,
                                                 chunks      = self.chunks,
                                                 logger      = self.logger)
        ds = self._cice_cache
        if variables is not None:
            keep = [v for v in variables if v in ds.variables or v in ds.coords]
            ds = ds[keep]
        return ds

    def load_classification(self, method: str) -> xr.DataArray:
        store = self.paths.classification_store(method)
        if not store.exists():
            raise FileNotFoundError(f"Classification store does not exist: {store}")
        ds = xr.open_zarr(store, consolidated=False, chunks=self.chunks)
        if self.mask_var_name in ds.data_vars:
            out = ds[self.mask_var_name]
        elif len(ds.data_vars) == 1:
            out = next(iter(ds.data_vars.values()))
        else:
            raise KeyError(f"Could not find a classification mask in {store}")
        return out.astype(bool)

    def load_metrics(self, method: str) -> xr.Dataset:
        norm = normalize_method(method)
        if norm not in self._metrics_cache:
            store = self.paths.metrics_store(norm)
            if not store.exists():
                raise FileNotFoundError(f"Metrics store does not exist: {store}")
            self._metrics_cache[norm] = xr.open_zarr(store, consolidated=False, chunks=self.chunks)
        return self._metrics_cache[norm]

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

    def _region_mask(self, template: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        lon180       = self._lon_to_180(lon)
        region_masks = []
        names        = []
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
        da      = (aice.where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice)) / self.metrics.area_scale
        da.name = "FIA"
        da.attrs.update({"long_name": "Fast Ice Area", "units": "10^3 km^2"})
        return da

    def compute_fiv(self, aice: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        da      = (aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice)) / self.metrics.volume_scale
        da.name = "FIV"
        da.attrs.update({"long_name": "Fast Ice Volume", "units": "10^3 km^3"})
        return da

    def compute_fit(self, aice: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        volume   = (aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice))
        area_num = (aice.where(mask, 0.0) * area).sum(dim=self._spatial_dims(aice))
        da = xr.where(area_num > 0, volume / area_num, np.nan)
        da.name = "FIT"
        da.attrs.update({"long_name": "Fast Ice Thickness", "units": "m"})
        return da

    def compute_fip(self, mask: xr.DataArray) -> xr.DataArray:
        da      = mask.astype("float32").mean(dim="time")
        da.name = "FIP"
        da.attrs.update({"long_name": "Fast Ice Persistence", "units": "1"})
        return da

    def compute_regional_fia_fit(self, aice: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, mask: xr.DataArray,
                                 region_mask: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        weighted_area = aice.where(mask, 0.0) * area
        weighted_vol  = aice.where(mask, 0.0) * hi.fillna(0.0).where(mask, 0.0) * area
        spatial_dims  = self._spatial_dims(aice)
        region_area   = weighted_area.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        region_vol    = weighted_vol.expand_dims(region=region_mask.region).where(region_mask, 0.0)
        fia_reg       = (region_area.sum(dim=spatial_dims) / self.metrics.area_scale).transpose("time", "region")
        fit_num       = region_vol.sum(dim=spatial_dims).transpose("time", "region")
        fit_den       = region_area.sum(dim=spatial_dims).transpose("time", "region")
        fit_reg       = xr.where(fit_den > 0, fit_num / fit_den, np.nan)
        fia_reg.name  = "FIA_by_region"
        fit_reg.name  = "FIT_by_region"
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
            rows.append({"year"    : year,
                         "max"     : float(grp.max()),
                         "min"     : float(grp.min()),
                         "doy_max" : float(pd.Timestamp(grp.idxmax()).dayofyear),
                         "doy_min" : float(pd.Timestamp(grp.idxmin()).dayofyear)})
        if not rows:
            return {}
        df = pd.DataFrame(rows)
        out = {}
        for col in ("max", "min", "doy_max", "doy_min"):
            vals                        = df[col].to_numpy(dtype=float)
            out[f"{prefix}_{col}_mean"] = xr.DataArray(np.nanmean(vals))
            out[f"{prefix}_{col}_std"]  = xr.DataArray(np.nanstd(vals, ddof=0))
        return out

    def persistence_stability_index(self, mask: xr.DataArray, area: xr.DataArray,
                                    persistence_threshold: float = 0.8,
                                    winter_months: tuple[int, ...] = (5, 6, 7, 8, 9, 10)) -> dict[str, xr.DataArray]:
        winter = mask.sel(time=mask.time.dt.month.isin(winter_months))
        if winter.sizes.get("time", 0) == 0:
            return {"FIPSI"                 : xr.DataArray(np.nan),
                    "persistent_winter_area": xr.DataArray(np.nan),
                    "ever_winter_area"      : xr.DataArray(np.nan)}
        persistence     = winter.astype("float32").mean("time")
        persistent_area = xr.where(persistence >= persistence_threshold, area, 0.0).sum()
        ever_area       = xr.where(winter.any("time"), area, 0.0).sum()
        ratio           = xr.where(ever_area > 0, persistent_area / ever_area, np.nan)
        return {"FIPSI"                  : ratio,
                "persistent_winter_area" : persistent_area / self.metrics.area_scale,
                "ever_winter_area"       : ever_area / self.metrics.area_scale}

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

    def _output_chunk_map(self, ds: xr.Dataset) -> dict[str, int]:
        chunk_map: dict[str, int] = {}
        if "time" in ds.dims:
            chunk_map["time"] = int(self.chunks.get("time", 31))
        if "region" in ds.dims:
            chunk_map["region"] = -1
        for dim in ("nj", "ni", "nlat", "nlon", "y", "x"):
            if dim in ds.dims:
                chunk_map[dim] = -1
        return chunk_map

    def compute_metrics(self, method: str, *, overwrite: bool = False) -> str:
        norm = normalize_method(method)
        self.logger.info("Resolved class store for %s: %s", norm, self.paths.classification_store(norm))
        self.logger.info("Resolved metrics store for %s: %s", norm, self.paths.metrics_store(norm))
        ds = self.load_cice(["aice", "hi", "tarea", "TLON", "TLAT"])
        mask = self.load_classification(norm)
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
        chunk_map = self._output_chunk_map(ds_out)
        if chunk_map:
            self.logger.info("Rechunking metrics output with chunks: %s", chunk_map)
            ds_out = ds_out.chunk(chunk_map)

        encoding = {}
        for name, var in ds_out.data_vars.items():
            if getattr(var.data, "chunks", None) is not None:
                encoding[name] = {"chunks": tuple(int(c[0]) for c in var.chunks)}

        self.logger.info("Writing metrics to %s", store)
        ds_out.to_zarr(store, mode="w", consolidated=False, encoding=encoding)
        self._metrics_cache[norm] = ds_out
        return str(store)

    def _require_pygmt(self):
        try:
            import pygmt
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyGMT is required for plotting methods.") from exc
        return pygmt

    def plot_fip(self, method: str,
                 output_path: str | None = None,
                 output_root: str | Path | None = None,
                 region_name: str | None = None,
                 region: Sequence[float] | None = None,
                 regions: Mapping[str, Sequence[float]] | None = None,
                 fig_size: float = 20.0,
                 cmap: str | Path | None = None,
                 title: str | None = None,
                 shorelines: str = "0.25p,black",
                 land: str = "lightgray",
                 water: str = "white",
                 grid_style: str = "s0.05c",
                 colorbar_position: str = "JMR+w8c/0.4c+v+o0.8c/0c",
                 colorbar_xlabel: str = "Fast Ice Persistence",
                 colorbar_ylabel: str | None = None) -> str | dict[str, str]:
        """
        Plot FIP for one region or a collection of regions.

        Behaviour
        ---------
        - If `region` is supplied, plot only that region.
        - Else if `region_name` is supplied, look it up in `regions` or the default
          Antarctic 8-region dictionary.
        - Else plot all regions in `regions` if supplied, otherwise all default
          Antarctic 8 regions.

        Output paths default to:
            <graphics_root>/<SIM_NAME>/<REGION>/FIP/<YYYY-MM-DD>_<YYYY-MM-DD>_<method>_FIP.png
        unless `output_path` is supplied for a single-region plot.
        """
        pygmt             = self._require_pygmt()
        norm              = normalize_method(method)
        ds                = self.load_metrics(norm)
        cice              = self.load_cice(["TLON", "TLAT"])
        lon, lat          = self._detect_lonlat(cice)
        fip               = ds["FIP"].load()
        lon_flat          = _normalize_lon180(lon.values.ravel())
        lat_flat          = lat.values.ravel()
        z_flat            = fip.values.ravel()
        finite            = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(z_flat)
        available_regions = _region_dict_from_source(regions)
        if region is not None:
            plot_regions = {region_name or "custom": tuple(map(float, region))}
        elif region_name is not None:
            if region_name not in available_regions:
                raise KeyError(f"Unknown region {region_name!r}. Available: {list(available_regions)}")
            plot_regions = {region_name: available_regions[region_name]}
        else:
            plot_regions = available_regions
        cmap_path = str(Path(cmap).expanduser()) if cmap is not None else _default_fip_cpt()
        root      = Path(output_root).expanduser() if output_root is not None else _default_fip_root(self)
        if output_path is not None and len(plot_regions) != 1:
            raise ValueError("output_path can only be used when plotting a single region")
        outputs: dict[str, str] = {}
        for rname, bounds in plot_regions.items():
            region_good, plot_lon, plot_region, mc = _region_mask_and_plot_lon(lon_flat, lat_flat, bounds)
            good = finite & region_good
            if not np.any(good):
                self.logger.warning("No finite FIP points found inside region %s; skipping", rname)
                continue
            data       = pd.DataFrame({"lon": plot_lon[good],
                                       "lat": lat_flat[good],
                                       "z"  : z_flat[good]})
            projection = _region_projection(bounds, fig_size=fig_size)
            frame      = ["af"]
            if title is not None:
                frame.append(f'+t"{title}"')
            if output_path is not None:
                path = Path(output_path).expanduser()
            else:
                fname = f"{self.run.start_date}_{self.run.end_date}_{norm}_FIP.png"
                path = root / self.run.sim_name / rname / "FIP" / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            cbar_frame: list[str] = []
            cbar_frame.append(f'xaf+l"{colorbar_xlabel}"' if colorbar_xlabel else "xaf")
            if colorbar_ylabel:
                cbar_frame.append(f'y+l"{colorbar_ylabel}"')
            self.logger.info("Plotting FIP for region=%s bounds=%s projection=%s MC=%s -> %s", rname, list(map(float, bounds)), projection, mc, path)
            fig = pygmt.Figure()
            fig.basemap(region=plot_region, projection=projection, frame=frame)
            fig.coast(shorelines=shorelines, land=land, water=water)
            fig.plot(x=data["lon"], y=data["lat"], style=grid_style, fill=data["z"], cmap=cmap_path)
            fig.colorbar(cmap=cmap_path, position=colorbar_position, frame=cbar_frame)
            fig.savefig(path)
            outputs[rname] = str(path)
        if region is not None or region_name is not None:
            return next(iter(outputs.values())) if outputs else ""
        return outputs

    def plot_timeseries(self, variable: str, method: str, region: str = "total", output_path: str | None = None) -> str:
        pygmt = self._require_pygmt()
        norm = normalize_method(method)
        ds = self.load_metrics(norm)
        var = variable.upper()
        if region.lower() == "total":
            if var not in ds:
                raise KeyError(f"Variable {var!r} not available in metrics dataset.")
            series = ds[var]
            title = f"{self.run.sim_name} {var} ({norm})"
        else:
            key = f"{var}_by_region"
            if key not in ds:
                raise KeyError(f"Regional variable {key!r} not available in metrics dataset.")
            series = ds[key].sel(region=region)
            title = f"{self.run.sim_name} {var} {region} ({norm})"
        x = pd.to_datetime(series["time"].values)
        y = series.values.astype(float)
        good = np.isfinite(y)
        if not np.any(good):
            raise ValueError(f"No finite values available for {var} and region={region!r}.")
        x = x[good]
        y = y[good]
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if ymin == ymax:
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
            ymin -= pad
            ymax += pad
        else:
            pad = (ymax - ymin) * 0.08
            ymin -= pad
            ymax += pad
        path = Path(output_path).expanduser() if output_path else self.paths.timeseries_plot_path(var, norm, region)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig = pygmt.Figure()
        region_spec = [x.min(), x.max(), ymin, ymax]
        fig.basemap(region=region_spec, projection="X16c/6c", frame=["WSen", "xaf", f'yaf+l"{series.attrs.get("units", "")}"', f'+t"{title}"'])
        fig.plot(x=x, y=y, pen="1.2p,black")
        fig.savefig(path)
        return str(path)
