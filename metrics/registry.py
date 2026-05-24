from __future__ import annotations
from collections.abc import Iterable
CORE_BASE_NAMES    = ["IA",      #Ice Area
                      "IV",      #Ice Volume
                      "IT",      #Ice Thickness
                      "IP",      #Ice Persistence
                      "IS",      #Ice Strength
                      "ITVR",    #Ice Thermodynamic Volume Rate
                      "IMVR",    #Ice Dynamic/Mechanical Volume Rate
                      "ITAR",    #Ice Thermodynamic Area Rate
                      "IMAR"]    #Ice Dynamic/Mechanical Area Rate]
CORE_FI            = [f"F{name}" for name in CORE_BASE_NAMES]
CORE_PI            = [f"P{name}" for name in CORE_BASE_NAMES]
CORE_SI            = [f"S{name}" for name in CORE_BASE_NAMES]
CORES              = CORE_FI + CORE_PI + CORE_SI
REGIONAL           = ["FIA_by_region",
                      "FIT_by_region",
                      "PIA_by_region",
                      "PIT_by_region",
                      "SIA_by_region",
                      "SIT_by_region"]
SPATIAL_BASE_NAMES = ["IHI",
                      "IST",
                      "ITVR_YR",
                      "IMVR_YR",
                      "ITAR_YR",
                      "IMAR_YR"]
SPATIAL_FI         = [f"F{name}" for name in SPATIAL_BASE_NAMES]
SPATIAL_PI         = [f"P{name}" for name in SPATIAL_BASE_NAMES]
SPATIAL_SI         = [f"S{name}" for name in SPATIAL_BASE_NAMES]
SPATIAL            = SPATIAL_FI + SPATIAL_PI + SPATIAL_SI
SUMMARY_BASE_NAMES = ["IA_max_mean",
                      "IA_max_std",
                      "IA_min_mean",
                      "IA_min_std",
                      "IA_doy_max_mean",
                      "IA_doy_max_std",
                      "IA_doy_min_mean",
                      "IA_doy_min_std",
                      "IT_max_mean",
                      "IT_max_std",
                      "IT_min_mean",
                      "IT_min_std",
                      "IT_doy_max_mean",
                      "IT_doy_max_std",
                      "IT_doy_min_mean",
                      "IT_doy_min_std"]
SUMMARY_FI         = [f"F{name}" for name in SUMMARY_BASE_NAMES]
SUMMARY_PI         = [f"P{name}" for name in SUMMARY_BASE_NAMES]
SUMMARY_SI         = [f"S{name}" for name in SUMMARY_BASE_NAMES]
SUMMARY            = SUMMARY_FI + SUMMARY_PI + SUMMARY_SI
STRESS_BASE_NAMES  = ["IKuxE_mean",
                      "IKuxE_abs_mean",
                      "IKuxE_valid_area_m2",
                      "IKuyE_mean",
                      "IKuyE_abs_mean",
                      "IKuyE_valid_area_m2",
                      "IKuxN_mean",
                      "IKuxN_abs_mean",
                      "IKuxN_valid_area_m2",
                      "IKuyN_mean",
                      "IKuyN_abs_mean",
                      "IKuyN_valid_area_m2",
                      "IKuE_mag_mean",
                      "IKuE_mag_abs_mean",
                      "IKuE_mag_valid_area_m2",
                      "IKuN_mag_mean",
                      "IKuN_mag_abs_mean",
                      "IKuN_mag_valid_area_m2"]
STRESS_FI          = [f"F{name}" for name in STRESS_BASE_NAMES]
STRESS_PI          = [f"P{name}" for name in STRESS_BASE_NAMES]
STRESS_SI          = [f"S{name}" for name in STRESS_BASE_NAMES]
STRESS             = STRESS_FI + STRESS_PI + STRESS_SI
DIAG_BASE_NAMES    = ["ice_speed",
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
FI_DIAGS           = [f"FI_{name}_mean" for name in DIAG_BASE_NAMES]
PI_DIAGS           = [f"PI_{name}_mean" for name in DIAG_BASE_NAMES]
SI_DIAGS           = [f"SI_{name}_mean" for name in DIAG_BASE_NAMES]
DIAGS              = FI_DIAGS + PI_DIAGS + SI_DIAGS
FI_SPECIFIC        = ["FIPSI",
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
METRIC_GROUPS      = {"fi_core"    : CORE_FI,
                      "pi_core"    : CORE_PI,
                      "si_core"    : CORE_SI,
                      "fi_regional": ["FIA_by_region", "FIT_by_region"],
                      "pi_regional": ["PIA_by_region", "PIT_by_region"],
                      "si_regional": ["SIA_by_region", "SIT_by_region"],
                      "regional"   : REGIONAL,
                      "fi_spatial" : SPATIAL_FI,
                      "pi_spatial" : SPATIAL_PI,
                      "si_spatial" : SPATIAL_SI,
                      "spatial"    : SPATIAL,
                      "fi_summary" : SUMMARY_FI,
                      "pi_summary" : SUMMARY_PI,
                      "si_summary" : SUMMARY_SI,
                      "summary"    : SUMMARY,
                      "fi_stress"  : STRESS_FI,
                      "pi_stress"  : STRESS_PI,
                      "si_stress"  : STRESS_SI,
                      "stress"     : STRESS,
                      "fi_diags"   : FI_DIAGS,
                      "pi_diags"   : PI_DIAGS,
                      "si_diags"   : SI_DIAGS,
                      "diags"      : DIAGS,
                      "default"    : CORE_FI,
                      "fi_all"     : CORE_FI + ["FIA_by_region", "FIT_by_region"] + SPATIAL_FI + SUMMARY_FI + STRESS_FI + FI_DIAGS + FI_SPECIFIC,
                      "pi_all"     : CORE_PI + ["PIA_by_region", "PIT_by_region"] + SPATIAL_PI + SUMMARY_PI + STRESS_PI + PI_DIAGS,
                      "si_all"     : CORE_SI + ["SIA_by_region", "SIT_by_region"] + SPATIAL_SI + SUMMARY_SI + STRESS_SI + SI_DIAGS,
                      "all"        : CORE_FI + CORE_PI + CORE_SI + REGIONAL + SPATIAL + SUMMARY + STRESS + DIAGS + FI_SPECIFIC}
FIPSI_NAMES        = {"FIPSI", "persistent_winter_area", "ever_winter_area"}
FIA_SKILL_NAMES    = {"FIA_Bias", "FIA_RMSE", "FIA_MAE", "FIA_Corr"}
FIT_SKILL_NAMES    = {"FIT_Bias", "FIT_RMSE", "FIT_MAE", "FIT_Corr"}
IA_SEASONAL_NAMES  = {"IA_max_mean",
                      "IA_max_std",
                      "IA_min_mean",
                      "IA_min_std",
                      "IA_doy_max_mean",
                      "IA_doy_max_std",
                      "IA_doy_min_mean",
                      "IA_doy_min_std"}
FIA_SEASONAL_NAMES = {f"F{name}" for name in IA_SEASONAL_NAMES}
PIA_SEASONAL_NAMES = {f"P{name}" for name in IA_SEASONAL_NAMES}
SIA_SEASONAL_NAMES = {f"S{name}" for name in IA_SEASONAL_NAMES}
IT_SEASONAL_NAMES  = {"IT_max_mean",
                      "IT_max_std",
                      "IT_min_mean",
                      "IT_min_std",
                      "IT_doy_max_mean",
                      "IT_doy_max_std",
                      "IT_doy_min_mean",
                      "IT_doy_min_std"}
FIT_SEASONAL_NAMES = {f"F{name}" for name in IT_SEASONAL_NAMES}
PIT_SEASONAL_NAMES = {f"P{name}" for name in IT_SEASONAL_NAMES}
SIT_SEASONAL_NAMES = {f"S{name}" for name in IT_SEASONAL_NAMES}

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
