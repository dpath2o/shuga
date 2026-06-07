from __future__ import annotations
from typing import Iterable
import numpy as np
import xarray as xr

def dim_coords(da: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Keep only dimension coordinates from a DataArray.

    This avoids accidentally carrying large/static auxiliary coordinates into
    small output products.
    """
    return {dim: da.coords[dim] for dim in da.dims if dim in da.coords}

def normalise_longitudes(lon, to: str = "-180-180", eps: float = 1e-12):
    """
    Normalise longitudes to either '-180-180' or '0-360'.

    Works with numpy arrays and xarray DataArrays.
    """
    lon_wrapped = ((lon % 360.0) + 360.0) % 360.0

    if to == "0-360":
        if isinstance(lon_wrapped, xr.DataArray):
            return xr.where(np.isclose(lon_wrapped, 360.0, atol=eps), 0.0, lon_wrapped)
        return np.where(np.isclose(lon_wrapped, 360.0, atol=eps), 0.0, lon_wrapped)
    if to == "-180-180":
        out = ((lon_wrapped + 180.0) % 360.0) - 180.0
        if isinstance(out, xr.DataArray):
            return xr.where(np.isclose(out, 180.0, atol=eps), -180.0, out)
        return np.where(np.isclose(out, 180.0, atol=eps), -180.0, out)
    raise ValueError("to must be either '-180-180' or '0-360'.")

def latlon_to_degrees(da: xr.DataArray, *, target: str, lon_type: str = "-180-180") -> xr.DataArray:
    values = np.asarray(da.values, dtype="float64")
    units  = str(da.attrs.get("units", "")).strip().lower()
    finite = np.isfinite(values)
    if "radian" in units or (finite.any() and np.nanmax(np.abs(values[finite])) <= 2.0 * np.pi + 1e-6):
        values = np.rad2deg(values)
    if target.upper().endswith("LON"):
        values = normalise_longitudes(values, to=lon_type)
    attrs          = dict(da.attrs)
    attrs["units"] = "degrees"
    attrs.setdefault("standard_name", "longitude" if target.upper().endswith("LON") else "latitude")
    return xr.DataArray(values, dims = da.dims, coords = dim_coords(da),  attrs = attrs, name = target)

def angle_to_radians(da: xr.DataArray, *, target: str) -> xr.DataArray:
    values = np.asarray(da.values, dtype="float64")
    units  = str(da.attrs.get("units", "")).strip().lower()
    finite = np.isfinite(values)
    if "degree" in units or (finite.any() and np.nanmax(np.abs(values[finite])) > 2.0 * np.pi + 1e-6):
        values = np.deg2rad(values)
    attrs          = dict(da.attrs)
    attrs["units"] = "radians"
    return xr.DataArray(values, dims = da.dims, coords = dim_coords(da), attrs = attrs, name = target)

def metric_to_meters(da: xr.DataArray, *, target: str) -> xr.DataArray:
    values = np.asarray(da.values, dtype="float64")
    units  = str(da.attrs.get("units", "")).strip().lower()
    if units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        values = values / 100.0
    elif units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        values = values * 1000.0
    attrs          = dict(da.attrs)
    attrs["units"] = "m"
    return xr.DataArray(values, dims = da.dims, coords = dim_coords(da), attrs = attrs, name = target)

def area_to_m2(da: xr.DataArray, *, target: str) -> xr.DataArray:
    values = np.asarray(da.values, dtype="float64")
    units  = str(da.attrs.get("units", "")).strip().lower().replace(" ", "")
    if units in {"cm^2", "cm2"}:
        values = values / 10_000.0
    elif units in {"km^2", "km2"}:
        values = values * 1_000_000.0
    attrs          = dict(da.attrs)
    attrs["units"] = "m^2"
    return xr.DataArray(values, dims = da.dims, coords = dim_coords(da), attrs = attrs, name = target)

def pick_variable(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None

def coerce_2d_dims_to_nj_ni(da: xr.DataArray) -> xr.DataArray:
    """
    Rename the last two dimensions of a 2D DataArray to nj/ni.

    This is intended for history-compatible static fields. Do not use it for
    true corner/edge grids that intentionally use nj_b/ni_b.
    """
    if da.ndim != 2:
        return da
    rename: dict[str, str] = {}
    if da.dims[-2] != "nj":
        rename[da.dims[-2]] = "nj"
    if da.dims[-1] != "ni":
        rename[da.dims[-1]] = "ni"
    return da.rename(rename) if rename else da
