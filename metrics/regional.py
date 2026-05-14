# shuga/metrics/regional.py
from __future__ import annotations
import pandas as pd
import xarray as xr

def lon_to_180(lon: xr.DataArray) -> xr.DataArray:
    """
    Convert longitudes to [-180, 180).
    """
    return ((lon + 180.0) % 360.0) - 180.0

def ensure_2d_static(da: xr.DataArray) -> xr.DataArray:
    """
    Drop a singleton time dimension from a static field if present.
    """
    return da.isel(time=0, drop=True) if "time" in da.dims else da

def spatial_dims(
    da: xr.DataArray,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Return dimensions treated as spatial by area/volume reductions.
    """
    exclude = exclude or {"time", "region"}
    return [dim for dim in da.dims if dim not in exclude]

def detect_lonlat(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Find longitude/latitude fields in a CICE-like dataset.
    """
    lon_name = next(
        (n for n in ("TLON", "ULON", "lon", "longitude") if n in ds.variables or n in ds.coords),
        None,
    )
    lat_name = next(
        (n for n in ("TLAT", "ULAT", "lat", "latitude") if n in ds.variables or n in ds.coords),
        None,
    )

    if lon_name is None or lat_name is None:
        raise KeyError("Could not find longitude/latitude fields in the CICE dataset.")

    return ds[lon_name], ds[lat_name]

def region_mask(
    template: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    region_defs: dict,
) -> xr.DataArray:
    """
    Build a boolean region mask with leading region dimension.

    region_defs entries must contain geo_region:
        (lon_min, lon_max, lat_min, lat_max)
    """
    lon180 = lon_to_180(lon)
    region_masks = []
    names = []

    for region_name, spec in region_defs.items():
        lon_min, lon_max, lat_min, lat_max = spec["geo_region"]

        if lon_min <= lon_max:
            lon_sel = (lon180 >= lon_min) & (lon180 <= lon_max)
        else:
            lon_sel = (lon180 >= lon_min) | (lon180 <= lon_max)

        lat_sel = (lat >= lat_min) & (lat <= lat_max)

        region_masks.append((lon_sel & lat_sel).astype(bool))
        names.append(region_name)

    out = xr.concat(region_masks, dim=pd.Index(names, name="region"))
    return out.transpose("region", *template.dims)
