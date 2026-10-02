from __future__ import annotations
import logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)
DEFAULT_SEAICE_ROOT = Path("/g/data/gv90/da1339/SeaIce")
ESA_ROOT = DEFAULT_SEAICE_ROOT / "ESA" / "CCI"
AWI_ROOT = DEFAULT_SEAICE_ROOT / "AWI"
PROCESSED_ROOT = DEFAULT_SEAICE_ROOT / "SIT" / "processed"
HEM_MAP = {"SH":"SH","sh":"SH","NH":"NH","nh":"NH"}
SIT_CANDIDATES = ("sea_ice_thickness","sithick","sit","thickness")
SIC_CANDIDATES = ("sea_ice_concentration","sea_ice_area_fraction","siconc","sic")
TIME_CANDIDATES = ("time","Time","date")

@dataclass(frozen=True)
class SITSourceSpec:
    source: str
    level: str
    sensor: str
    hemisphere: str
    priority: int = 0

def normalise_hemisphere(value: str) -> str:
    if value not in HEM_MAP:
        raise ValueError(f"Unsupported hemisphere {value!r}; use SH or NH.")
    return HEM_MAP[value]

def _date_from_filename(path: Path):
    m = re.search(r"((?:19|20)\d{6})", path.name)
    if m:
        return pd.Timestamp(m.group(1))
    m = re.search(r"((?:19|20)\d{4})", path.name)
    return pd.Timestamp(m.group(1)+"15") if m else None

def _find_var(ds, candidates):
    return next((n for n in candidates if n in ds.data_vars), None)

def _find_sit_var(ds):
    hit = _find_var(ds, SIT_CANDIDATES)
    if hit:
        return hit
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name","")).lower()
        long = str(da.attrs.get("long_name","")).lower()
        if "sea_ice_thickness" in std or ("sea ice" in long and "thickness" in long):
            return name
    raise KeyError(f"Could not identify SIT variable. Available={list(ds.data_vars)}")

def _extract_time(ds, path):
    for name in TIME_CANDIDATES:
        if name in ds.coords or name in ds.variables:
            try:
                vals = pd.to_datetime(np.ravel(ds[name].values))
                vals = vals[~pd.isna(vals)]
                if len(vals):
                    return pd.Timestamp(vals[0])
            except Exception:
                pass
    ts = _date_from_filename(path)
    if ts is None:
        raise ValueError(f"Could not infer timestamp from {path}")
    return ts

def _standardise_file(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
    sit_name = _find_sit_var(ds)
    sit = ds[sit_name].astype("float32")
    ts = _extract_time(ds, path)
    if "time" not in sit.dims:
        sit = sit.expand_dims(time=[ts])
    else:
        sit = sit.assign_coords(time=pd.to_datetime(sit.time.values))
    sit = sit.where(np.isfinite(sit) & (sit > 0))

    out = xr.Dataset({"SIT_grid": sit})
    out["SIT_grid"].attrs = {"long_name":"Gridded Sea Ice Thickness","units":"m","source_variable":sit_name}

    sic_name = _find_var(ds, SIC_CANDIDATES)
    if sic_name:
        sic = ds[sic_name].astype("float32")
        if "time" not in sic.dims:
            sic = sic.expand_dims(time=out.time)
        else:
            sic = sic.assign_coords(time=pd.to_datetime(sic.time.values))
        out["SIC_grid"] = sic
    return out

def _weighted_mean(sit: xr.DataArray, sic: xr.DataArray | None = None) -> xr.DataArray:
    dims = tuple(d for d in sit.dims if d != "time")
    valid = np.isfinite(sit) & (sit > 0)
    weights = xr.where(valid & np.isfinite(sic) & (sic > 0), sic, 0.0) if sic is not None else xr.where(valid,1.0,0.0)
    num = (xr.where(valid,sit,0.0)*weights).sum(dim=dims, skipna=True)
    den = weights.sum(dim=dims, skipna=True)
    out = (num/den.where(den>0)).rename("SIT")
    out.attrs = {"long_name":"Sea Ice Thickness","units":"m"}
    return out

def discover_esa_l3c(root: str | Path = ESA_ROOT, *, hemisphere: str,
                     sensors: Sequence[str] = ("envisat","cryosat2","sentinel3a","sentinel3b"),
                     version: str = "v4.0") -> list[Path]:
    hem = normalise_hemisphere(hemisphere)
    root = Path(root).expanduser()
    files = []
    for sensor in sensors:
        p = root/"thickness"/"L3C"/sensor/version/hem
        if p.exists():
            files.extend(sorted(p.rglob("*.nc")))
    return sorted(set(files))

def discover_awi_l3cp(root: str | Path = AWI_ROOT, *, hemisphere: str,
                      sensors: Sequence[str] = ("envisat","cryosat2","sentinel3a","sentinel3b")) -> list[Path]:
    hem = normalise_hemisphere(hemisphere).lower()
    root = Path(root).expanduser()
    files = []
    for sensor in sensors:
        p = root/"l3cp_release"/hem/sensor
        if p.exists():
            files.extend(sorted(p.rglob("*.nc")))
    return sorted(set(files))

def _sensor_from_path(path: Path) -> str:
    lower = str(path).lower()
    for sensor in ("envisat","cryosat2","sentinel3a","sentinel3b"):
        if sensor in lower:
            return sensor
    return "unknown"

def _combine_sensor_series(sensor_series: dict[str, xr.DataArray], source_name: str) -> xr.Dataset:
    members = []
    for sensor, da in sorted(sensor_series.items()):
        members.append(da.expand_dims(sensor_member=[sensor]))
    aligned = xr.concat(members, dim="sensor_member", join="outer")
    sit = aligned.median("sensor_member", skipna=True).rename("SIT")
    spread = aligned.std("sensor_member", skipna=True, ddof=0).rename("SIT_sensor_spread")
    n = np.isfinite(aligned).sum("sensor_member").astype("int16").rename("n_sensors")
    ds = xr.Dataset({"SIT":sit,"SIT_sensor_spread":spread,"n_sensors":n})
    ds.attrs["sensor_overlap_method"] = "median across valid sensor-level hemispheric SIT estimates"
    return ds

def build_source_store(files: Sequence[Path], *, output: str | Path, source_name: str,
                       hemisphere: str, overwrite: bool = False) -> Path:
    output = Path(output)
    if output.exists() and not overwrite:
        LOGGER.info("Source output exists; skipping: %s", output)
        return output
    if not files:
        raise FileNotFoundError(f"No files found for {source_name} {hemisphere}")

    by_sensor = {}
    for i, path in enumerate(files,1):
        try:
            ds = _standardise_file(path)
            by_sensor.setdefault(_sensor_from_path(path), []).append(ds)
            if i % 50 == 0:
                LOGGER.info("%s indexed %d/%d", source_name, i, len(files))
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", path, exc)

    output.mkdir(parents=True, exist_ok=True)
    sensor_series = {}

    for sensor, parts in sorted(by_sensor.items()):
        merged = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal",
                           compat="override", join="override", combine_attrs="override").sortby("time")
        _, idx = np.unique(pd.DatetimeIndex(merged.time.values), return_index=True)
        merged = merged.isel(time=np.sort(idx))
        sic = merged["SIC_grid"] if "SIC_grid" in merged else None
        ts = _weighted_mean(merged["SIT_grid"], sic=sic)

        merged.chunk({"time":1}).to_zarr(output/f"{sensor}.zarr", mode="w", consolidated=True, zarr_format=2)
        ts.to_dataset(name="SIT").chunk({"time":min(31,ts.sizes["time"])}).to_zarr(
            output/f"{sensor}_SIT_timeseries.zarr", mode="w", consolidated=True, zarr_format=2
        )
        sensor_series[sensor] = ts

    composite = _combine_sensor_series(sensor_series, source_name)
    composite.attrs.update(source=source_name, hemisphere=normalise_hemisphere(hemisphere),
                           sensors=",".join(sorted(sensor_series)))
    composite.chunk({"time":min(31,composite.sizes["time"])}).to_zarr(
        output/"SIT_timeseries.zarr", mode="w", consolidated=True, zarr_format=2
    )
    return output
