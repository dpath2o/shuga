from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import xarray as xr

LOGGER              = logging.getLogger(__name__)
DEFAULT_SEAICE_ROOT = Path("/g/data/gv90/da1339/SeaIce")
ESA_ROOT            = DEFAULT_SEAICE_ROOT / "ESA" / "CCI"
AWI_ROOT            = DEFAULT_SEAICE_ROOT / "AWI"
PROCESSED_ROOT      = DEFAULT_SEAICE_ROOT / "SIT" / "processed"
HEM_MAP             = {"SH": "SH", "sh": "SH", "NH": "NH", "nh": "NH"}
SIT_CANDIDATES      = ("sea_ice_thickness", "sithick", "sit", "thickness")
SIC_CANDIDATES      = ("sea_ice_concentration", "sea_ice_area_fraction", "siconc", "sic")
TIME_CANDIDATES     = ("time", "Time", "date")

@dataclass(frozen=True)
class SITSourceSpec:
    source: str
    level: str
    sensor: str
    hemisphere: str
    priority: int = 0

def normalise_hemisphere(value: str) -> str:
    try:
        return HEM_MAP[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported hemisphere {value!r}; use SH or NH.") from exc

def _date_from_filename(path: Path) -> pd.Timestamp | None:
    m = re.search(r"((?:19|20)\d{6})", path.name)
    if m:
        return pd.Timestamp(m.group(1))
    m = re.search(r"((?:19|20)\d{4})", path.name)
    if m:
        return pd.Timestamp(m.group(1) + "15")
    return None

def _find_sit_var(ds: xr.Dataset) -> str:
    for name in SIT_CANDIDATES:
        if name in ds.data_vars:
            return name
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).lower()
        long_name = str(da.attrs.get("long_name", "")).lower()
        if "sea_ice_thickness" in std or ("sea ice" in long_name and "thickness" in long_name):
            return name
    raise KeyError(f"Could not identify SIT variable. Available: {list(ds.data_vars)}")

def _find_sic_var(ds: xr.Dataset) -> str | None:
    for name in SIC_CANDIDATES:
        if name in ds.data_vars:
            return name
    return None

def _extract_time(ds: xr.Dataset, path: Path) -> pd.Timestamp:
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
    out["SIT_grid"].attrs = {"long_name"       : "Gridded Sea Ice Thickness",
                             "units"           : "m",
                             "source_variable" : sit_name}
    sic_name              = _find_sic_var(ds)
    if sic_name is not None:
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
    if sic is not None:
        weights = xr.where(valid & np.isfinite(sic) & (sic > 0), sic, 0.0)
    else:
        weights = xr.where(valid, 1.0, 0.0)
    num = (xr.where(valid, sit, 0.0) * weights).sum(dim=dims, skipna=True)
    den = weights.sum(dim=dims, skipna=True)
    out = (num / den.where(den > 0)).rename("SIT")
    out.attrs = {"long_name": "Sea Ice Thickness", "units": "m"}
    return out

def discover_esa_l3c(root: str | Path = ESA_ROOT, *,
                     hemisphere: str,
                     sensors: Sequence[str] = ("envisat", "cryosat2", "sentinel3a", "sentinel3b"),
                     version: str = "v4.0") -> list[Path]:
    """
    Discover ESA CCI v4.0 L3C SIT files in the corrected local mirror.

    Expected layout:
      <root>/thickness/L3C/<sensor>/<version>/<HEM>/.../*.nc
    """
    hem = normalise_hemisphere(hemisphere)
    root = Path(root).expanduser()
    out: list[Path] = []
    for sensor in sensors:
        p = root / "thickness" / "L3C" / sensor.lower() / version / hem
        if not p.exists():
            LOGGER.debug("ESA L3C path not present: %s", p)
            continue
        out.extend(sorted(p.rglob("*.nc")))
    return sorted(set(out))

def discover_esa_l2p(root: str | Path = ESA_ROOT, *,
                     hemisphere: str,
                     sensors: Sequence[str] = ("envisat", "cryosat2", "sentinel3a", "sentinel3b"),
                     version: str = "v4.0") -> list[Path]:
    """Discover ESA CCI v4.0 L2P files for future swath processing."""
    hem = normalise_hemisphere(hemisphere)
    root = Path(root).expanduser()
    out: list[Path] = []
    for sensor in sensors:
        p = root / "thickness" / "L2P" / sensor.lower() / version / hem
        if p.exists():
            out.extend(sorted(p.rglob("*.nc")))
    return sorted(set(out))

def discover_awi_l3cp(root: str | Path = AWI_ROOT, *,
                      hemisphere: str,
                      sensors: Sequence[str] = ("envisat", "cryosat2", "sentinel3a", "sentinel3b")) -> list[Path]:
    hem = normalise_hemisphere(hemisphere).lower()
    root = Path(root)
    out: list[Path] = []
    for sensor in sensors:
        p = root / "l3cp_release" / hem / sensor.lower()
        if p.exists():
            out.extend(sorted(p.rglob("*.nc")))
    return sorted(set(out))

def _sensor_from_path(path: Path) -> str:
    lower = str(path).lower()
    for sensor in ("envisat", "cryosat2", "sentinel3a", "sentinel3b"):
        if sensor in lower:
            return sensor
    return "unknown"

def build_source_store(files: Sequence[Path], *,
                       output: str | Path,
                       source_name: str,
                       hemisphere: str,
                       overwrite: bool = False) -> Path:
    output = Path(output)
    if output.exists() and not overwrite:
        return output
    if not files:
        raise FileNotFoundError(f"No files found for {source_name} {hemisphere}")
    by_sensor: dict[str, list[xr.Dataset]] = {}
    for i, path in enumerate(files, 1):
        try:
            ds = _standardise_file(path)
            sensor = _sensor_from_path(path)
            by_sensor.setdefault(sensor, []).append(ds)
            if i % 50 == 0:
                LOGGER.info("%s indexed %d/%d", source_name, i, len(files))
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", path, exc)
    output.mkdir(parents=True, exist_ok=True)
    all_series: list[xr.DataArray] = []
    for sensor, parts in by_sensor.items():
        merged = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override", combine_attrs="override").sortby("time")
        _, idx = np.unique(pd.DatetimeIndex(merged.time.values), return_index=True)
        merged = merged.isel(time=np.sort(idx))
        sic    = merged["SIC_grid"] if "SIC_grid" in merged else None
        ts     = _weighted_mean(merged["SIT_grid"], sic=sic)
        ts     = ts.assign_coords(source=("time", [source_name] * ts.sizes["time"]), sensor=("time", [sensor] * ts.sizes["time"]))
        all_series.append(ts)
        sensor_store = output / f"{sensor}.zarr"
        merged.chunk({"time": 1}).to_zarr(sensor_store, mode="w", consolidated=True)
    ts_all = xr.concat(all_series, dim="time").sortby("time")
    _, idx = np.unique(pd.DatetimeIndex(ts_all.time.values), return_index=True)
    ts_all = ts_all.isel(time=np.sort(idx))
    ts_ds = ts_all.to_dataset(name="SIT")
    ts_ds.attrs.update(source=source_name, hemisphere=normalise_hemisphere(hemisphere))
    ts_ds.chunk({"time": min(31, ts_ds.sizes["time"])}).to_zarr(output / "SIT_timeseries.zarr", mode="w", consolidated=True)
    return output

def build_continuous_sit(
    *,
    esa_timeseries: str | Path | None,
    awi_timeseries: str | Path | None,
    output: str | Path,
    hemisphere: str,
    prefer: str = "AWI",
    overwrite: bool = False,
) -> Path:
    """
    Build a combined hemispheric SIT time series from available sources.

    If both source stores exist, the preferred source is used first and the
    other fills missing source timestamps. If only one source exists, that
    source is written directly. Genuine observation gaps are not interpolated.
    """
    output = Path(output)
    if output.exists() and not overwrite:
        return output

    esa_path = Path(esa_timeseries) if esa_timeseries is not None else None
    awi_path = Path(awi_timeseries) if awi_timeseries is not None else None

    esa = (
        xr.open_zarr(esa_path, consolidated=True)["SIT"]
        if esa_path is not None and esa_path.exists()
        else None
    )
    awi = (
        xr.open_zarr(awi_path, consolidated=True)["SIT"]
        if awi_path is not None and awi_path.exists()
        else None
    )

    if esa is None and awi is None:
        raise FileNotFoundError("Neither ESA nor AWI SIT time-series store exists.")

    prefer = prefer.upper()

    if esa is not None and awi is not None:
        union = pd.DatetimeIndex(
            sorted(
                set(pd.DatetimeIndex(esa.time.values))
                | set(pd.DatetimeIndex(awi.time.values))
            )
        )
        esa = esa.reindex(time=union)
        awi = awi.reindex(time=union)

        if prefer == "AWI":
            sit = awi.combine_first(esa)
            source = xr.where(
                np.isfinite(awi), "AWI",
                xr.where(np.isfinite(esa), "ESA", "")
            )
        elif prefer == "ESA":
            sit = esa.combine_first(awi)
            source = xr.where(
                np.isfinite(esa), "ESA",
                xr.where(np.isfinite(awi), "AWI", "")
            )
        else:
            raise ValueError("prefer must be AWI or ESA")

    elif awi is not None:
        LOGGER.warning("ESA unavailable: building continuous SIT from AWI only.")
        sit = awi
        source = xr.full_like(awi, "AWI", dtype=str)

    else:
        LOGGER.warning("AWI unavailable: building continuous SIT from ESA only.")
        sit = esa
        source = xr.full_like(esa, "ESA", dtype=str)

    sit = sit.rename("SIT")
    sit.attrs = {"long_name": "Sea Ice Thickness", "units": "m"}

    ds = xr.Dataset({"SIT": sit, "source": source.astype(str)})
    ds.attrs.update(
        hemisphere=normalise_hemisphere(hemisphere),
        preferred_source=prefer,
        note=(
            "Preferred source plus fallback where available; "
            "no interpolation across genuine observation gaps."
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    ds.chunk({"time": min(31, ds.sizes["time"])}).to_zarr(
        output, mode="w", consolidated=True
    )
    return output
