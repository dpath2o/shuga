from __future__ import annotations
import numpy as np
import xarray as xr
from shuga.metrics.regional import ensure_2d_static, spatial_dims

# ------------------------------------------------------------------
def _align_mask(mask: xr.DataArray | None, field: xr.DataArray) -> xr.DataArray | None:
    if mask is None:
        return None
    try:
        return mask.broadcast_like(field)
    except Exception:
        return None

# ------------------------------------------------------------------
def _align_time_inner(*arrays: xr.DataArray | None) -> tuple[xr.DataArray | None, ...]:
    """
    Align DataArrays on their common time coordinate, preserving None entries.

    This is needed because classified masks and CICE tendency fields can have
    slightly different time labels or lengths after rolling/binary filtering,
    grouped monthly zarr loading, or legacy store creation.
    """
    indexed = [(i, arr) for i, arr in enumerate(arrays)
               if arr is not None and "time" in arr.dims]
    if len(indexed) < 2:
        return arrays
    positions, valid = zip(*indexed)
    aligned = xr.align(*valid, join="inner")
    out = list(arrays)
    for pos, arr in zip(positions, aligned):
        out[pos] = arr
    return tuple(out)

# ------------------------------------------------------------------
def _masked_weighted_field(field: xr.DataArray, area: xr.DataArray,
                            mask: xr.DataArray | None = None) -> xr.DataArray:
    area2d   = ensure_2d_static(area)
    mask_eff = _align_mask(mask, field)
    if mask_eff is not None:
        field = field.where(mask_eff, 0.0)
    return field * area2d

# ------------------------------------------------------------------
def compute_area_series(sic: xr.DataArray, area: xr.DataArray,
                        mask: xr.DataArray | None = None, *, name: str, long_name: str,
                        scale: float | None = None) -> xr.DataArray:
    """
    Area time series from concentration and cell area.

    If mask is supplied, concentration is zeroed outside the mask before area
    integration. This preserves the existing shuga convention where FIA/SIA are
    concentration-weighted areas.
    """
    weighted = sic.where(mask, 0.0) if mask is not None else sic
    da       = (weighted * ensure_2d_static(area)).sum(dim=spatial_dims(sic))
    if scale is not None:
        da = da / scale
        units = "10^3 km^2"
    else:
        units = "m^2"
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    return da

# ------------------------------------------------------------------
def compute_volume_series(sic: xr.DataArray, hi: xr.DataArray, area: xr.DataArray,
                          mask: xr.DataArray | None = None, *, name: str, long_name: str,
                          scale: float | None = None) -> xr.DataArray:
    """
    Volume time series from concentration, thickness, and cell area.
    """
    field = sic * hi
    if mask is not None:
        field = field.where(mask, 0.0)
    da = (field * ensure_2d_static(area)).sum(dim=spatial_dims(field))
    if scale is not None:
        da = da / scale
        units = "10^3 km^3"
    else:
        units = "m^3"
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    return da

# ------------------------------------------------------------------
def compute_thickness_series(sic: xr.DataArray, hi: xr.DataArray, area: xr.DataArray,
                             mask: xr.DataArray | None = None, *, name: str, long_name: str) -> xr.DataArray:
    """
    Concentration/area-weighted mean thickness time series.
    """
    area2d             = ensure_2d_static(area)
    weighted_area      = sic
    weighted_thickness = sic * hi
    if mask is not None:
        weighted_area      = weighted_area.where(mask, 0.0)
        weighted_thickness = weighted_thickness.where(mask, 0.0)
    dims        = spatial_dims(weighted_thickness)
    numerator   = (weighted_thickness * area2d).sum(dim=dims)
    denominator = (weighted_area * area2d).sum(dim=dims)
    da          = (numerator / denominator.where(denominator > 0)).rename(name)
    da.attrs.update({"long_name": long_name, "units": "m"})
    return da

# ------------------------------------------------------------------
def compute_strength_series(sic: xr.DataArray, hi: xr.DataArray, strength: xr.DataArray, area: xr.DataArray,
                            mask: xr.DataArray | None = None, *, name: str, long_name: str) -> xr.DataArray:
    """
    Area-weighted mean compressive strength over sea-ice/fast-ice cells.

    Strength is normalised by thickness and converted to MPa, matching the
    existing FIS/SIS diagnostic convention used elsewhere in shuga.
    """
    valid = hi > 0
    if mask is not None:
        valid = valid & mask
    field       = xr.where(valid, strength / hi.where(hi > 0) / 1e6, np.nan)
    area2d      = ensure_2d_static(area)
    weights     = xr.where(np.isfinite(field), sic * area2d, 0.0)
    dims        = spatial_dims(field)
    numerator   = (field * weights).sum(dim=dims)
    denominator = weights.sum(dim=dims)
    da          = (numerator / denominator.where(denominator > 0)).rename(name)
    da.attrs.update({"long_name": long_name, "units": "MPa"})
    return da

# ------------------------------------------------------------------
def compute_persistence_mask(mask: xr.DataArray, *, name: str, long_name: str,
                             percent: bool = True) -> xr.DataArray:
    """
    Temporal fast-/sea-ice persistence.

    By default returns percent of valid time steps. Set percent=False for a
    0--1 fraction.
    """
    da = mask.astype("float32").mean(dim="time", skipna=True)
    if percent:
        da    = da * 100.0
        units = "%"
    else:
        units = "1"
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    return da

# ------------------------------------------------------------------
def compute_temporal_mean(da: xr.DataArray, *, name: str, long_name: str) -> xr.DataArray:
    """
    Temporal mean over the time axis.
    """
    out = da.mean(dim="time", skipna=True).rename(name)
    out.attrs.update({"long_name": long_name,
                      "units"    : da.attrs.get("units", "1")})
    return out

# ------------------------------------------------------------------
def convert_thickness_tendency_to_m_per_day(da: xr.DataArray) -> xr.DataArray:
    """
    Convert CICE tendency-like fields to m day^-1 where possible.

    Historical CICE output often stores dvidtt/dvidtd as cm day^-1. Unknown
    units are treated as cm day^-1 for backwards compatibility.
    """
    units = str(da.attrs.get("units", "")).lower().replace(" ", "")
    if units in {"cm/day", "cmday-1", "cmd-1"}:
        out = da / 100.0
    elif units in {"m/day", "mday-1", "md-1"}:
        out = da
    elif units in {"m/s", "ms-1"}:
        out = da * 86400.0
    else:
        out = da / 100.0
    out.attrs.update(da.attrs)
    out.attrs["units"] = "m day^-1"
    return out

# ------------------------------------------------------------------
def compute_volume_rate(tendency: xr.DataArray, sic: xr.DataArray, area: xr.DataArray,
                        mask: xr.DataArray | None = None, *, name: str, long_name: str) -> xr.DataArray:
    """
    Integrated volume tendency time series.

    The tendency is converted to m day^-1, multiplied by concentration and
    area, then summed spatially.
    """
    rate = convert_thickness_tendency_to_m_per_day(tendency)
    if mask is not None:
        rate, sic, mask = _align_time_inner(rate, sic, mask)
    else:
        rate, sic = xr.align(rate, sic, join="inner")
    field = rate * sic
    if mask is not None:
        field = field.where(mask, 0.0)
    da = (field * ensure_2d_static(area)).sum(dim=spatial_dims(field))
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": "m^3 day^-1"})
    return da

# ------------------------------------------------------------------
def compute_area_rate(tendency: xr.DataArray, area: xr.DataArray,
                      mask: xr.DataArray | None = None, *, name: str, long_name: str) -> xr.DataArray:
    """
    Integrated area tendency time series.

    This assumes the incoming tendency is an areal concentration tendency per
    day or equivalent area-fraction tendency.
    """
    field = tendency
    if mask is not None:
        field, mask = _align_time_inner(field, mask)
        field       = field.where(mask, 0.0)
    da = (field * ensure_2d_static(area)).sum(dim=spatial_dims(field))
    da = da.rename(name)
    da.attrs.update({"long_name": long_name, "units": "m^2 day^-1"})
    return da

# ------------------------------------------------------------------
def compute_spatial_rate_year(tendency: xr.DataArray, mask: xr.DataArray, *, name: str, long_name: str,
                              area: xr.DataArray | None = None) -> xr.DataArray:
    """
    Spatial annual/climatological rate diagnostic.

    For thickness/volume-rate fields, the output is a temporal mean tendency.
    For area-rate fields, pass area to weight the tendency by cell area before
    averaging over time.
    """
    field = convert_thickness_tendency_to_m_per_day(tendency)
    if mask is not None:
        field, mask = _align_time_inner(field, mask)
        field       = field.where(mask)
    if area is not None:
        field = field * ensure_2d_static(area)
        units = "m^2 day^-1"
    else:
        units = "m day^-1"
    da = field.mean(dim="time", skipna=True).rename(name)
    da.attrs.update({"long_name": long_name, "units": units})
    return da

# ------------------------------------------------------------------
def compute_region_series(sic: xr.DataArray, hi: xr.DataArray, area: xr.DataArray, region_mask: xr.DataArray,
                          mask: xr.DataArray | None = None, *, area_name: str, thickness_name: str, area_long_name: str,
                          thickness_long_name: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Regional area and concentration/area-weighted thickness time series.
    """
    area2d = ensure_2d_static(area)
    if mask is not None:
        sic_eff = sic.where(mask, 0.0)
        hi_eff  = hi.where(mask)
    else:
        sic_eff = sic
        hi_eff  = hi
    sic_reg     = sic_eff * region_mask
    hi_reg      = hi_eff * region_mask
    dims        = spatial_dims(sic_reg)
    area_da     = (sic_reg * area2d).sum(dim=dims) / 1e9
    area_da     = area_da.rename(area_name)
    area_da.attrs.update({"long_name": area_long_name, "units": "10^3 km^2"})
    numerator   = (sic_reg * hi_reg * area2d).sum(dim=dims)
    denominator = (sic_reg * area2d).sum(dim=dims)
    thick_da    = numerator / denominator.where(denominator > 0)
    thick_da    = thick_da.rename(thickness_name)
    thick_da.attrs.update({"long_name": thickness_long_name, "units": "m"})
    return area_da, thick_da

# ------------------------------------------------------------------
def compute_area_weighted_stress(stress: xr.DataArray, area: xr.DataArray,
                                 mask: xr.DataArray | None = None, *, base_name: str) -> xr.Dataset:
    """
    Area-weighted stress mean, absolute mean, and valid area.

    Returns:
    - {base_name}_mean
    - {base_name}_abs_mean
    - {base_name}_valid_area_m2
    """
    area2d   = ensure_2d_static(area)
    mask_eff = _align_mask(mask, stress)
    valid    = np.isfinite(stress)
    if mask_eff is not None:
        valid = valid & mask_eff
    weights    = xr.where(valid, area2d, 0.0)
    total_area = weights.sum(dim=spatial_dims(stress))
    mean       = (stress.where(valid, 0.0) * weights).sum(dim=spatial_dims(stress))
    mean       = mean / total_area.where(total_area > 0)
    mean       = mean.rename(f"{base_name}_mean")
    mean.attrs.update({"long_name": f"{base_name} area-weighted mean",
                       "units"    : stress.attrs.get("units", "Pa")})
    abs_mean   = (abs(stress).where(valid, 0.0) * weights).sum(dim=spatial_dims(stress))
    abs_mean   = abs_mean / total_area.where(total_area > 0)
    abs_mean   = abs_mean.rename(f"{base_name}_abs_mean")
    abs_mean.attrs.update({"long_name": f"{base_name} area-weighted absolute mean",
                           "units"    : stress.attrs.get("units", "Pa")})
    valid_area = total_area.rename(f"{base_name}_valid_area_m2")
    valid_area.attrs.update({"long_name": f"{base_name} valid area",
                             "units": "m^2"})
    return xr.Dataset({mean.name: mean, abs_mean.name: abs_mean, valid_area.name: valid_area})

