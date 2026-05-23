from __future__ import annotations
import numpy as np
import xarray as xr
from shuga.metrics.cice         import _publish_matching
from shuga.metrics.regional     import ensure_2d_static
from shuga.metrics.calculations import compute_area_weighted_stress

def stress_requested(requested: set[str], prefix: str) -> bool:
    """
    Return True if any requested metric starts with the stress prefix.

    Examples
    --------
    prefix='FI' matches FIKuxE_mean, FIKuE_mag_mean, etc.
    prefix='SI' matches SIKuxE_mean, SIKuE_mag_mean, etc.
    """
    return any(name.startswith(f"{prefix}K") for name in requested)

# def _publish_matching(out: xr.Dataset, dsi: xr.Dataset, requested: set[str]) -> xr.Dataset:
#     for name in dsi.data_vars:
#         if name in requested:
#             out[name] = dsi[name]
#     return out

def compute_stress_dataset(*, ds: xr.Dataset, area: xr.DataArray, requested: set[str], prefix: str, mask: xr.DataArray | None,
                           calculator = compute_area_weighted_stress) -> xr.Dataset:
    """
    Compute requested area-weighted CICE stress diagnostics.

    Parameters
    ----------
    ds : xr.Dataset
        CICE history dataset containing any of KuxE, KuyE, KuxN, KuyN.
    area : xr.DataArray
        T-grid fallback area.
    requested : set[str]
        Requested metric names.
    prefix : {'FI', 'SI'}
        Metric prefix.
    mask : xr.DataArray | None
        Fast-ice or sea-ice mask.
    calculator : callable, optional
        Stress calculator. Kept injectable so CICEMetrics can pass its
        backwards-compatible alias if needed.

    Returns
    -------
    xr.Dataset
        Dataset containing only requested stress diagnostics that could be
        computed from available inputs.
    """
    out             = xr.Dataset()
    area_e          = ensure_2d_static(ds["earea"]) if "earea" in ds else area
    area_n          = ensure_2d_static(ds["narea"]) if "narea" in ds else area
    component_specs = [("KuxE", area_e, f"{prefix}KuxE"),
                       ("KuyE", area_e, f"{prefix}KuyE"),
                       ("KuxN", area_n, f"{prefix}KuxN"),
                       ("KuyN", area_n, f"{prefix}KuyN")]
    for varname, weights, base in component_specs:
        needed = {f"{base}_mean", f"{base}_abs_mean", f"{base}_valid_area_m2"}
        if not (requested & needed):
            continue
        if varname not in ds:
            continue
        dsi = calculator(ds[varname], weights, mask, base_name = base)
        out = _publish_matching(out, dsi, requested)
    magnitude_specs = [("KuxE", "KuyE", area_e, f"{prefix}KuE_mag"),
                       ("KuxN", "KuyN", area_n, f"{prefix}KuN_mag")]
    for xname, yname, weights, base in magnitude_specs:
        needed = {f"{base}_mean", f"{base}_abs_mean", f"{base}_valid_area_m2"}
        if not (requested & needed):
            continue
        if xname not in ds or yname not in ds:
            continue
        mag = xr.apply_ufunc(np.hypot, ds[xname], ds[yname], dask="allowed")
        mag.attrs["units"] = ds[xname].attrs.get("units", "Pa")
        dsi = calculator(mag, weights, mask, base_name = base)
        out = _publish_matching(out, dsi, requested)
    return out
