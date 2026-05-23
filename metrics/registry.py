from __future__ import annotations
from collections.abc import Iterable
CORE_FI            = ["FIA",
                      "FIV",
                      "FIT",
                      "FIP",
                      "FIS",
                      "FITVR",
                      "FIMVR",
                      "FITAR",
                      "FIMAR"]
CORE_SI            = ["SIA",
                      "SIV",
                      "SIT",
                      "SIP",
                      "SIS",
                      "SITVR",
                      "SIMVR",
                      "SITAR",
                      "SIMAR"]
REGIONAL           = ["FIA_by_region",
                      "FIT_by_region",
                      "SIA_by_region",
                      "SIT_by_region"]
SPATIAL            = ["FIHI",
                      "FIST",
                      "FITVR_YR",
                      "FIMVR_YR",
                      "FITAR_YR",
                      "FIMAR_YR",
                      "SIHI",
                      "SIST",
                      "SITVR_YR",
                      "SIMVR_YR",
                      "SITAR_YR",
                      "SIMAR_YR"]
SUMMARY            = ["FIA_max_mean",
                      "FIA_max_std",
                      "FIA_min_mean",
                      "FIA_min_std",
                      "FIA_doy_max_mean",
                      "FIA_doy_max_std",
                      "FIA_doy_min_mean",
                      "FIA_doy_min_std",
                      "FIT_max_mean",
                      "FIT_max_std",
                      "FIT_min_mean",
                      "FIT_min_std",
                      "FIT_doy_max_mean",
                      "FIT_doy_max_std",
                      "FIT_doy_min_mean",
                      "FIT_doy_min_std",
                      "SIA_max_mean",
                      "SIA_max_std",
                      "SIA_min_mean",
                      "SIA_min_std",
                      "SIA_doy_max_mean",
                      "SIA_doy_max_std",
                      "SIA_doy_min_mean",
                      "SIA_doy_min_std",
                      "SIT_max_mean",
                      "SIT_max_std",
                      "SIT_min_mean",
                      "SIT_min_std",
                      "SIT_doy_max_mean",
                      "SIT_doy_max_std",
                      "SIT_doy_min_mean",
                      "SIT_doy_min_std",
                      "FIPSI",
                      "persistent_winter_area",
                      "ever_winter_area",
                      "FIA_Bias",
                      "FIA_RMSE",
                      "FIA_MAE",
                      "FIA_Corr",
                      "FIT_Bias",
                      "FIT_RMSE",
                      "FIT_MAE",
                      "FIT_Corr"]
STRESS             = ["FIKuxE_mean",
                      "FIKuxE_abs_mean",
                      "FIKuxE_valid_area_m2",
                      "FIKuyE_mean",
                      "FIKuyE_abs_mean",
                      "FIKuyE_valid_area_m2",
                      "FIKuxN_mean",
                      "FIKuxN_abs_mean",
                      "FIKuxN_valid_area_m2",
                      "FIKuyN_mean",
                      "FIKuyN_abs_mean",
                      "FIKuyN_valid_area_m2",
                      "FIKuE_mag_mean",
                      "FIKuE_mag_abs_mean",
                      "FIKuE_mag_valid_area_m2",
                      "FIKuN_mag_mean",
                      "FIKuN_mag_abs_mean",
                      "FIKuN_mag_valid_area_m2",
                      "SIKuxE_mean",
                      "SIKuxE_abs_mean",
                      "SIKuxE_valid_area_m2",
                      "SIKuyE_mean",
                      "SIKuyE_abs_mean",
                      "SIKuyE_valid_area_m2",
                      "SIKuxN_mean",
                      "SIKuxN_abs_mean",
                      "SIKuxN_valid_area_m2",
                      "SIKuyN_mean",
                      "SIKuyN_abs_mean",
                      "SIKuyN_valid_area_m2",
                      "SIKuE_mag_mean",
                      "SIKuE_mag_abs_mean",
                      "SIKuE_mag_valid_area_m2",
                      "SIKuN_mag_mean",
                      "SIKuN_mag_abs_mean",
                      "SIKuN_mag_valid_area_m2"]
DIAGS              = ["ice_speed",
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
METRIC_GROUPS      = {"fi_core" : CORE_FI,
                      "si_core" : CORE_SI,
                      "regional": REGIONAL,
                      "spatial" : SPATIAL,
                      "summary" : SUMMARY,
                      "stress"  : STRESS,
                      "diags"   : DIAGS,
                      "default" : CORE_FI + SPATIAL + STRESS,
                      "all"     : CORE_FI + CORE_SI + REGIONAL + SPATIAL + SUMMARY + STRESS + DIAGS}
FIPSI_NAMES        = {"FIPSI", "persistent_winter_area", "ever_winter_area"}
FIA_SKILL_NAMES    = {"FIA_Bias", "FIA_RMSE", "FIA_MAE", "FIA_Corr"}
FIT_SKILL_NAMES    = {"FIT_Bias", "FIT_RMSE", "FIT_MAE", "FIT_Corr"}
FIA_SEASONAL_NAMES = {"FIA_max_mean",
                      "FIA_max_std",
                      "FIA_min_mean",
                      "FIA_min_std",
                      "FIA_doy_max_mean",
                      "FIA_doy_max_std",
                      "FIA_doy_min_mean",
                      "FIA_doy_min_std"}
FIT_SEASONAL_NAMES = {"FIT_max_mean",
                      "FIT_max_std",
                      "FIT_min_mean",
                      "FIT_min_std",
                      "FIT_doy_max_mean",
                      "FIT_doy_max_std",
                      "FIT_doy_min_mean",
                      "FIT_doy_min_std"}
SIA_SEASONAL_NAMES = {"SIA_max_mean",
                      "SIA_max_std",
                      "SIA_min_mean",
                      "SIA_min_std",
                      "SIA_doy_max_mean",
                      "SIA_doy_max_std",
                      "SIA_doy_min_mean",
                      "SIA_doy_min_std"}
SIT_SEASONAL_NAMES = {"SIT_max_mean",
                      "SIT_max_std",
                      "SIT_min_mean",
                      "SIT_min_std",
                      "SIT_doy_max_mean",
                      "SIT_doy_max_std",
                      "SIT_doy_min_mean",
                      "SIT_doy_min_std"}

def as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]

def expand_metric_names(metric_names=None, metric_groups=None, *, default_group: str = "default") -> list[str]:
    explicit = as_list(metric_names)
    groups   = as_list(metric_groups)
    if not explicit and not groups:
        groups = [default_group]
    out: list[str] = []
    seen: set[str] = set()
    def _add_many(names: Iterable[str]) -> None:
        for name in names:
            token = str(name).strip()
            if not token or token in seen:
                continue
            out.append(token)
            seen.add(token)
    _add_many(explicit)
    for group in groups:
        key = group.strip().lower()
        if key not in METRIC_GROUPS:
            raise ValueError(f"Unknown metric group {group!r}. Valid groups: {sorted(METRIC_GROUPS)}")
        _add_many(METRIC_GROUPS[key])
    return out
