# shuga/metrics/diagnostics.py
from __future__ import annotations
import numpy  as np
import xarray as xr
from shuga.metrics.calculations import compute_area_weighted_stress

RHO_ICE_DEFAULT       = 917.0
EPS_DEFAULT           = 1.0e-12
DIAG_BASE_NAMES       = ["ice_speed",
                         "rel_ice_ocean_speed",
                         "strain_invariant",
                         "tau_air",
                         "tau_ocean",
                         "tau_internal",
                         "tau_coriolis",
                         "tau_tilt",
                         "ld_mag_proxy",
                         "tau_ld_est",
                         "R_ld_budget",
                         "P_ld_est"]
DIAG_PREFIXES         = ("FI", "PI", "SI")
DIAGNOSTIC_NAMES      = ["ice_speed",
                         "rel_ice_ocean_speed",
                         "strain_invariant",
                         "tau_air",
                         "tau_ocean",
                         "tau_internal",
                         "tau_coriolis",
                         "tau_tilt",
                         "ld_x_proxy",
                         "ld_y_proxy",
                         "ld_mag_proxy",
                         "tau_ld_est",
                         "R_ld_budget",
                         "P_ld_est"]
DIAGNOSTIC_INPUT_VARS = ["uvel", "vvel", "uocn", "vocn", "aice", "hi",
                         # strain / deformation
                         "sigP", "divu", "shear",
                         # dynamic stress terms
                         "strairx", "strairy",
                         "strocnx", "strocny",
                         "strintx", "strinty",
                         "strcorx", "strcory",
                         "strtltx", "strtlty",
                         # lateral-drag edge/proxy terms
                         "KuxE", "KuxN",
                         "KuyE", "KuyN"]

def prefixed_diag_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}_mean" for name in DIAG_BASE_NAMES]

def prefixed_diags_requested(requested: set[str], prefix: str | None = None) -> bool:
    if prefix is not None:
        return bool(set(prefixed_diag_names(prefix)) & set(requested))
    return any(bool(set(prefixed_diag_names(pfx)) & set(requested)) for pfx in DIAG_PREFIXES)

def compute_prefixed_diagnostic_dataset(*, ds: xr.Dataset, area: xr.DataArray, requested: set[str],
                                        prefix: str,
                                        mask: xr.DataArray | None,
                                        rho_ice: float = RHO_ICE_DEFAULT,
                                        eps: float = EPS_DEFAULT) -> xr.Dataset:
    """
    Compute area-weighted diagnostic time series for one ice-type mask.

    Output names follow:
    - FI_<base>_mean
    - PI_<base>_mean
    - SI_<base>_mean
    """
    wanted = {base: f"{prefix}_{base}_mean" for base in DIAG_BASE_NAMES if f"{prefix}_{base}_mean" in requested}
    if not wanted:
        return xr.Dataset()
    base_ds = compute_diagnostic_terms(ds, requested = set(wanted), rho_ice = rho_ice, eps = eps)
    out     = xr.Dataset()
    for base_name, out_name in wanted.items():
        if base_name not in base_ds:
            continue
        dsi       = compute_area_weighted_stress(base_ds[base_name], area, mask, base_name = f"{prefix}_{base_name}")
        mean_name = f"{prefix}_{base_name}_mean"
        if mean_name in dsi:
            out[out_name] = dsi[mean_name]
            out[out_name].attrs.update({"long_name"        : f"{prefix} area-weighted mean {base_name}",
                                        "source_diagnostic": base_name,
                                        "ice_mask_prefix"  : prefix})
    return out

def diagnostics_requested(requested: set[str]) -> bool:
    return bool(set(DIAGNOSTIC_NAMES) & set(requested))

def _publish_requested(ds: xr.Dataset, requested: set[str]) -> xr.Dataset:
    keep = [name for name in DIAGNOSTIC_NAMES if name in requested and name in ds]
    return ds[keep] if keep else xr.Dataset()

def mag(ds: xr.Dataset, x: str, y: str, name: str) -> xr.DataArray | None:
    if x not in ds or y not in ds:
        return None
    out     = xr.apply_ufunc(np.hypot, ds[x], ds[y], dask="allowed").rename(name)
    x_units = ds[x].attrs.get("units", "")
    y_units = ds[y].attrs.get("units", "")
    out.attrs.update(units=x_units if x_units == y_units else x_units or y_units or "unknown",
                     long_name = name, source_x = x, source_y = y)
    return out

def ice_speed(ds: xr.Dataset) -> xr.DataArray | None:
    if not all(v in ds for v in ("uvel", "vvel")):
        return None
    out = xr.apply_ufunc(np.hypot, ds["uvel"], ds["vvel"], dask="allowed").rename("ice_speed")
    out.attrs.update(units = "m s-1", long_name = "Sea-ice speed", source_x = "uvel", source_y = "vvel")
    return out

def rel_ice_ocean_speed(ds: xr.Dataset) -> xr.DataArray | None:
    if not all(v in ds for v in ("uvel", "vvel", "uocn", "vocn")):
        return None
    out = xr.apply_ufunc(np.hypot, ds["uvel"] - ds["uocn"], ds["vvel"] - ds["vocn"], dask="allowed").rename("rel_ice_ocean_speed")
    out.attrs.update(units = "m s-1", long_name = "Relative ice-ocean speed", note = "Computed as hypot(uvel-uocn, vvel-vocn).")
    return out

def strain_invariant(ds: xr.Dataset) -> xr.DataArray | None:
    """
    Prefer existing CICE invariant if present.

    Fallback hierarchy:
    1. sigP
    2. hypot(divu, shear)
    3. abs(shear)
    """
    if "sigP" in ds:
        out = ds["sigP"].rename("strain_invariant")
        out.attrs.update(long_name = "Strain-rate invariant", note = "Renamed from existing CICE sigP field.")
        return out
    if all(v in ds for v in ("divu", "shear")):
        out = xr.apply_ufunc(np.hypot, ds["divu"], ds["shear"], dask="allowed").rename("strain_invariant")
        out.attrs.update(units     = ds["divu"].attrs.get("units", "s-1"),
                         long_name = "Strain-rate invariant proxy",
                         note      = "Computed as hypot(divu, shear).")
        return out
    if "shear" in ds:
        out = abs(ds["shear"]).rename("strain_invariant")
        out.attrs.update(units     = ds["shear"].attrs.get("units", "s-1"),
                         long_name = "Shear-only strain-rate proxy",
                         note      = "Fallback computed as abs(shear).")
        return out
    return None

def ku_proxy(ds: xr.Dataset) -> xr.Dataset:
    """
    Build T-grid-like lateral-drag proxy fields from available edge terms.

    Notes
    -----
    The CICE branch history fields expose edge-oriented variables such as
    KuxE/KuyE and KuxN/KuyN. This routine averages available edge fields into
    compact x/y proxies. Treat stress estimates derived from these fields as
    diagnostic proxies until the exact sign/unit convention is confirmed in the
    branch history metadata and source code.
    """
    out      = xr.Dataset()
    x_fields = [v for v in ("KuxE", "KuxN") if v in ds]
    y_fields = [v for v in ("KuyE", "KuyN") if v in ds]
    if x_fields:
        out["ld_x_proxy"] = xr.concat([ds[v] for v in x_fields], dim="ld_edge_x").mean("ld_edge_x")
        out["ld_x_proxy"].attrs.update(long_name     = "Lateral-drag x proxy from Kux edge fields",
                                       source_fields = ",".join(x_fields),
                                       units         = ds[x_fields[0]].attrs.get("units", "unknown"))
    if y_fields:
        out["ld_y_proxy"] = xr.concat([ds[v] for v in y_fields], dim="ld_edge_y").mean("ld_edge_y")
        out["ld_y_proxy"].attrs.update(long_name     = "Lateral-drag y proxy from Kuy edge fields",
                                       source_fields = ",".join(y_fields),
                                       units         = ds[y_fields[0]].attrs.get("units", "unknown"))
    if "ld_x_proxy" in out and "ld_y_proxy" in out:
        out["ld_mag_proxy"] = xr.apply_ufunc(np.hypot, out["ld_x_proxy"], out["ld_y_proxy"], dask="allowed")
    elif "ld_x_proxy" in out:
        out["ld_mag_proxy"] = abs(out["ld_x_proxy"])
    elif "ld_y_proxy" in out:
        out["ld_mag_proxy"] = abs(out["ld_y_proxy"])
    if "ld_mag_proxy" in out:
        out["ld_mag_proxy"] = out["ld_mag_proxy"].rename("ld_mag_proxy")
        out["ld_mag_proxy"].attrs.update(
            long_name="Lateral-drag magnitude proxy from Kux/Kuy history fields",
            units=out["ld_x_proxy"].attrs.get("units", out["ld_y_proxy"].attrs.get("units", "unknown") if "ld_y_proxy" in out else "unknown"))
    return out

def compute_diagnostic_terms(ds: xr.Dataset, *,
                             requested: set[str] | None = None,
                             rho_ice  : float = RHO_ICE_DEFAULT,
                             eps      : float = EPS_DEFAULT) -> xr.Dataset:
    """
    Build diagnostic CICE dynamic fields for form-function comparison.

    These are intentionally returned as spatial/time fields. Domain/mask means
    should be computed downstream using classified masks, e.g. FI vs mobile pack.
    """
    requested = set(requested or DIAGNOSTIC_NAMES)
    out       = xr.Dataset()
    speed     = ice_speed(ds)
    if speed is not None:
        out["ice_speed"] = speed
    rel = rel_ice_ocean_speed(ds)
    if rel is not None:
        out["rel_ice_ocean_speed"] = rel
    strain = strain_invariant(ds)
    if strain is not None:
        out["strain_invariant"] = strain
    stress_specs = [("tau_air",      "strairx", "strairy", "air-ocean/atmospheric stress magnitude"),
                    ("tau_ocean",    "strocnx", "strocny", "ocean stress magnitude"),
                    ("tau_internal", "strintx", "strinty", "internal ice stress magnitude"),
                    ("tau_coriolis", "strcorx", "strcory", "Coriolis stress magnitude"),
                    ("tau_tilt",     "strtltx", "strtlty", "sea-surface tilt stress magnitude")]
    for label, x, y, long_name in stress_specs:
        z = mag(ds, x, y, label)
        if z is not None:
            z.attrs.update(units = z.attrs.get("units", "nominal Pa"), long_name = long_name)
            out[label] = z
    ku = ku_proxy(ds)
    for name in ku.data_vars:
        out[name] = ku[name]
    if "ld_mag_proxy" in out and "hi" in ds:
        h_eff             = ds["hi"] * ds["aice"] if "aice" in ds else ds["hi"]
        out["tau_ld_est"] = (rho_ice * h_eff * out["ld_mag_proxy"]).rename("tau_ld_est")
        out["tau_ld_est"].attrs.update(units     = "Pa as Kux/Kuy are accelerations",
                                       long_name = "Estimated lateral-drag stress magnitude",
                                       note      = "Computed as rho_ice * hi * aice * ld_mag_proxy. Treat as proxy unless Kux/Kuy units are confirmed.",
                                       rho_ice   = float(rho_ice))
    if all(v in out for v in ("tau_ld_est", "tau_air", "tau_ocean", "tau_internal")):
        denom              = out["tau_air"] + out["tau_ocean"] + out["tau_internal"] + eps
        out["R_ld_budget"] = (out["tau_ld_est"] / denom).rename("R_ld_budget")
        out["R_ld_budget"].attrs.update(units     = "1",
                                        long_name = "Lateral-drag stress ratio",
                                        note      = "tau_ld_est / (tau_air + tau_ocean + tau_internal)",
                                        eps       = float(eps))

    if (all(v in out for v in ("ld_x_proxy", "ld_y_proxy")) and all(v in ds for v in ("uvel", "vvel", "hi"))):
        h_eff           = ds["hi"] * ds["aice"] if "aice" in ds else ds["hi"]
        tau_x           = rho_ice * h_eff * out["ld_x_proxy"]
        tau_y           = rho_ice * h_eff * out["ld_y_proxy"]
        out["P_ld_est"] = (tau_x * ds["uvel"] + tau_y * ds["vvel"]).rename("P_ld_est")
        out["P_ld_est"].attrs.update(units     = "W m-2 if Kux/Kuy signs are stress accelerations",
                                     long_name = "Estimated lateral-drag power tau_ld · u",
                                     note      = "Sign depends on Kux/Kuy convention. Negative values should indicate damping only if K proxies oppose ice velocity.",
                                     rho_ice   = float(rho_ice))
    return _publish_requested(out, requested)
