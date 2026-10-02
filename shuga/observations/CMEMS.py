from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ROOT = Path("/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily")
DEFAULT_PATTERN = "*_CMEMS_0p083_org.nc"

_SOURCE_TO_SHUGA = {
    "siconc": "aice",
    "sithick": "hi",
    "usi": "uice",
    "vsi": "vice",
}

_FILE_RE = re.compile(
    r"(?P<start>\d{8})_(?P<end>\d{8})_CMEMS_0p083_org\.nc$"
)


def _normalise_hemisphere(value: str) -> str:
    token = str(value).strip().lower()
    mapping = {
        "s": "SH", "sh": "SH", "south": "SH", "southern": "SH",
        "n": "NH", "nh": "NH", "north": "NH", "northern": "NH",
    }
    if token not in mapping:
        raise ValueError(f"Unsupported hemisphere={value!r}; use SH or NH.")
    return mapping[token]


def discover_cmems_files(
    root: str | Path = DEFAULT_ROOT,
    *,
    start_date: str,
    end_date: str,
    padding_days: int = 0,
    pattern: str = DEFAULT_PATTERN,
) -> list[Path]:
    """Return annual CMEMS files intersecting the requested time interval."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"CMEMS root does not exist: {root}")

    dt0 = pd.Timestamp(start_date) - pd.Timedelta(days=int(padding_days))
    dtN = pd.Timestamp(end_date) + pd.Timedelta(days=int(padding_days))

    selected: list[Path] = []
    for path in sorted(root.glob(pattern)):
        match = _FILE_RE.search(path.name)
        if match is None:
            continue
        f0 = pd.Timestamp(match.group("start"))
        fN = pd.Timestamp(match.group("end"))
        if fN >= dt0 and f0 <= dtN:
            selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"No CMEMS files intersect {dt0.date()}..{dtN.date()} under {root}"
        )
    return selected


def _preprocess_factory(
    *,
    hemisphere: str,
    variables: Sequence[str],
):
    hemisphere = _normalise_hemisphere(hemisphere)
    source_vars = [name for name in variables if name in _SOURCE_TO_SHUGA]

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        keep = [name for name in source_vars if name in ds]
        ds = ds[keep]

        if hemisphere == "SH":
            ds = ds.sel(latitude=slice(-90.0, 0.0))
        else:
            ds = ds.sel(latitude=slice(0.0, 90.0))

        rename = {name: _SOURCE_TO_SHUGA[name] for name in keep}
        ds = ds.rename(rename)
        return ds

    return _preprocess


def open_cmems(
    *,
    root: str | Path = DEFAULT_ROOT,
    start_date: str,
    end_date: str,
    hemisphere: str = "SH",
    variables: Sequence[str] = ("siconc", "sithick", "usi", "vsi"),
    padding_days: int = 0,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Open the annual CMEMS 0.083-degree daily files lazily with xarray/dask.

    Source fields are normalised to shuga-style names:
      siconc  -> aice
      sithick -> hi
      usi     -> uice
      vsi     -> vice
    """
    chunks = chunks or {"time": 31, "latitude": 256, "longitude": 540}
    files = discover_cmems_files(
        root,
        start_date=start_date,
        end_date=end_date,
        padding_days=padding_days,
    )

    dt0 = pd.Timestamp(start_date) - pd.Timedelta(days=int(padding_days))
    dtN = pd.Timestamp(end_date) + pd.Timedelta(days=int(padding_days))

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=_preprocess_factory(
            hemisphere=hemisphere,
            variables=variables,
        ),
        chunks=chunks,
        parallel=True,
        decode_cf=True,
        mask_and_scale=True,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        join="exact",
    )

    ds = ds.sel(time=slice(dt0, dtN))
    ds.attrs.update(
        {
            "source": "Copernicus Marine CMEMS 0.083-degree daily sea-ice product",
            "hemisphere": _normalise_hemisphere(hemisphere),
        }
    )
    return ds


def spherical_cell_area(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    *,
    earth_radius_m: float = 6_371_000.0,
) -> xr.DataArray:
    """
    Exact spherical area for a regular/rectilinear latitude-longitude grid.

    Cell edges are inferred from coordinate midpoints. Poleward edges are
    clipped to +/-90 degrees.
    """
    lat = np.asarray(latitude.values, dtype=np.float64)
    lon = np.asarray(longitude.values, dtype=np.float64)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("CMEMS cell-area calculation expects 1-D lat/lon coordinates.")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("At least two latitude and longitude points are required.")

    lat_edges = np.empty(lat.size + 1, dtype=np.float64)
    lat_edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    lat_edges[0] = lat[0] - 0.5 * (lat[1] - lat[0])
    lat_edges[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    lat_edges = np.clip(lat_edges, -90.0, 90.0)

    # CMEMS is globally regular in longitude. Use the median separation so the
    # dateline does not produce an anomalous edge width.
    dlon_deg = float(np.median(np.diff(lon)))
    dlon = np.deg2rad(abs(dlon_deg))

    sin_term = np.abs(
        np.sin(np.deg2rad(lat_edges[1:]))
        - np.sin(np.deg2rad(lat_edges[:-1]))
    )
    area = (earth_radius_m ** 2) * dlon * sin_term[:, None]
    area = np.broadcast_to(area, (lat.size, lon.size)).copy()

    da = xr.DataArray(
        area.astype(np.float64),
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name="tarea",
    )
    da.attrs.update(
        {
            "long_name": "CMEMS spherical grid-cell area",
            "units": "m2",
            "earth_radius_m": float(earth_radius_m),
        }
    )
    return da


def static_store_path(
    root: str | Path = DEFAULT_ROOT,
    hemisphere: str = "SH",
) -> Path:
    return Path(root).expanduser() / _normalise_hemisphere(hemisphere) / "static.zarr"


def ensure_static_store(
    *,
    root: str | Path = DEFAULT_ROOT,
    hemisphere: str = "SH",
    overwrite: bool = False,
) -> Path:
    """
    Build a small CMEMS static Zarr store containing latitude, longitude,
    TLAT/TLON aliases and tarea.
    """
    root = Path(root).expanduser()
    out = static_store_path(root, hemisphere)
    if out.exists() and not overwrite:
        return out

    # Any annual file is sufficient to obtain the native grid.
    candidates = sorted(root.glob(DEFAULT_PATTERN))
    if not candidates:
        raise FileNotFoundError(f"No CMEMS annual files found under {root}")

    with xr.open_dataset(candidates[0], decode_cf=True) as src:
        if _normalise_hemisphere(hemisphere) == "SH":
            lat = src["latitude"].sel(latitude=slice(-90.0, 0.0)).load()
        else:
            lat = src["latitude"].sel(latitude=slice(0.0, 90.0)).load()
        lon = src["longitude"].load()

    tarea = spherical_cell_area(lat, lon)
    lon2d, lat2d = xr.broadcast(lon, lat)
    # xr.broadcast follows input ordering; transpose to latitude,longitude.
    tlon = lon2d.transpose("latitude", "longitude").rename("TLON")
    tlat = lat2d.transpose("latitude", "longitude").rename("TLAT")

    ds = xr.Dataset(
        {
            "tarea": tarea,
            "TLON": tlon.astype(np.float32),
            "TLAT": tlat.astype(np.float32),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    ds.attrs.update(
        {
            "source": "CMEMS 0.083-degree native grid",
            "hemisphere": _normalise_hemisphere(hemisphere),
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out, mode="w", consolidated=True)
    return out
