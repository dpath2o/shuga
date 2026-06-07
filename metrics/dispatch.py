# shuga/metrics/dispatch.py
from __future__      import annotations
from dataclasses     import dataclass, field
from collections.abc import Iterable
import numpy         as np
import xarray        as xr

PRIMARY_METRIC_NAMES = ["FIA", "FIV", "FIT", "FIP", "FIS", "FITVR", "FIMVR", "FITAR", "FIMAR",
                        "PIA", "PIV", "PIT", "PIP", "PIS", "PITVR", "PIMVR", "PITAR", "PIMAR",
                        "SIA", "SIV", "SIT", "SIP", "SIS", "SITVR", "SIMVR", "SITAR", "SIMAR",
                        "FIHI", "FIST", "FITVR_YR", "FIMVR_YR", "FITAR_YR", "FIMAR_YR",
                        "PIHI", "PIST", "PITVR_YR", "PIMVR_YR", "PITAR_YR", "PIMAR_YR",
                        "SIHI", "SIST", "SITVR_YR", "SIMVR_YR", "SITAR_YR", "SIMAR_YR",
                        "FIA_by_region", "FIT_by_region",
                        "PIA_by_region", "PIT_by_region",
                        "SIA_by_region", "SIT_by_region"]
PRIMARY_METRIC_SET   = set(PRIMARY_METRIC_NAMES)

# ------------------------------------------------------------------
def needs_classified_masks(requested: Iterable[str], fipsi_names: set[str]) -> bool:
    """
    Return True when requested metrics require classified FI/PI masks.
    """
    return any(name.startswith("FI") or name.startswith("PI")
               or name.startswith("FIA_") or name.startswith("FIT_")
               or name in fipsi_names for name in requested)

# ------------------------------------------------------------------
def needs_fast_ice_mask(requested: Iterable[str], fipsi_names: set[str]) -> bool:
    """
    Return True when the requested metric set needs mask is loaded 
    """
    return needs_classified_masks(requested, fipsi_names)

@dataclass(slots=True)
class MetricDispatchContext:
    """
    Shared metric-dispatch inputs for one simulation/method context.
    """
    ds          : xr.Dataset
    aice        : xr.DataArray
    hi          : xr.DataArray
    area        : xr.DataArray
    region_mask : xr.DataArray
    fi_mask     : xr.DataArray | None
    pi_mask     : xr.DataArray | None
    si_mask     : xr.DataArray
    area_scale  : float
    volume_scale: float

@dataclass
class MetricDispatcher:
    """
    Compute primary shuga metrics on demand with memoisation.

    This replaces the large if/elif block previously embedded inside
    CICEMetrics._compute_requested_metrics().
    """
    context   : MetricDispatchContext
    calculator: object
    memo      : dict[str, xr.DataArray] = field(default_factory=dict)

    def get(self, name: str) -> xr.DataArray | None:
        if name in self.memo:
            return self.memo[name]
        da = self._compute_one(name)
        if da is not None:
            self.memo[name] = da
        return da

    def dataset_for(self, names: Iterable[str]) -> xr.Dataset:
        out = xr.Dataset()
        for name in names:
            da = self.get(name)
            if da is not None:
                out[name] = da
        return out

    def _remember_many(self, values: dict[str, xr.DataArray]) -> None:
        for name, da in values.items():
            self.memo[name] = da

    def _compute_one(self, name: str) -> xr.DataArray | None:
        ctx         = self.context
        calc        = self.calculator
        ds          = ctx.ds
        aice        = ctx.aice
        hi          = ctx.hi
        area        = ctx.area
        fi_mask     = ctx.fi_mask
        pi_mask     = ctx.pi_mask
        si_mask     = ctx.si_mask
        region_mask = ctx.region_mask
        #-----------------------------------------------------------------------------------
        # fast ice (primaries)
        #----------------------------------------------------------------------------------
        if name == "FIA" and fi_mask is not None:
            return calc.compute_area_series(aice, area, fi_mask, name = "FIA", long_name = "Fast Ice Area", scale = ctx.area_scale)
        if name == "FIV" and fi_mask is not None:
            return calc.compute_volume_series(aice, hi, area, fi_mask, name = "FIV", long_name = "Fast Ice Volume", scale = ctx.volume_scale)
        if name == "FIT" and fi_mask is not None:
            return calc.compute_thickness_series(aice, hi, area, fi_mask, name = "FIT", long_name = "Fast Ice Thickness")
        if name == "FIP" and fi_mask is not None:
            return calc.compute_persistence_mask(fi_mask, name = "FIP", long_name = "Fast Ice Persistence")
        if name == "FIS" and fi_mask is not None and "strength" in ds:
            return calc.compute_strength_series(aice, hi, ds["strength"], area, fi_mask, name = "FIS", long_name = "Fast Ice Strength")
        if name == "FITVR" and fi_mask is not None and "dvidtt" in ds:
            return calc.compute_volume_rate(ds["dvidtt"], aice, area, fi_mask, name = "FITVR", long_name = "Fast Ice Thermodynamic Volume Rate")
        if name == "FIMVR" and fi_mask is not None and "dvidtd" in ds:
            return calc.compute_volume_rate(ds["dvidtd"], aice, area, fi_mask, name = "FIMVR", long_name = "Fast Ice Dynamic Volume Rate")
        if name == "FITAR" and fi_mask is not None and "daidtt" in ds:
            return calc.compute_area_rate(ds["daidtt"], area, fi_mask, name = "FITAR", long_name = "Fast Ice Thermodynamic Area Rate")
        if name == "FIMAR" and fi_mask is not None and "daidtd" in ds:
            return calc.compute_area_rate(ds["daidtd"], area, fi_mask, name = "FIMAR", long_name = "Fast Ice Dynamic Area Rate")
        #-----------------------------------------------------------------------------------
        # pack ice (primaries)
        #----------------------------------------------------------------------------------
        if name == "PIA" and pi_mask is not None:
            return calc.compute_area_series(aice, area, pi_mask, name = "PIA", long_name = "Pack Ice Area", scale = ctx.area_scale)
        if name == "PIV" and pi_mask is not None:
            return calc.compute_volume_series(aice, hi, area, pi_mask, name = "PIV", long_name = "Pack Ice Volume", scale = ctx.volume_scale)
        if name == "PIT" and pi_mask is not None:
            return calc.compute_thickness_series(aice, hi, area, pi_mask, name = "PIT", long_name = "Pack Ice Thickness")
        if name == "PIP" and pi_mask is not None:
            return calc.compute_persistence_mask(pi_mask, name = "PIP", long_name = "Pack Ice Persistence")
        if name == "PIS" and pi_mask is not None and "strength" in ds:
            return calc.compute_strength_series(aice, hi, ds["strength"], area, pi_mask, name = "PIS", long_name = "Pack Ice Strength")
        if name == "PITVR" and pi_mask is not None and "dvidtt" in ds:
            return calc.compute_volume_rate(ds["dvidtt"], aice, area, pi_mask, name = "PITVR", long_name = "Pack Ice Thermodynamic Volume Rate")
        if name == "PIMVR" and pi_mask is not None and "dvidtd" in ds:
            return calc.compute_volume_rate(ds["dvidtd"], aice, area, pi_mask, name = "PIMVR", long_name = "Pack Ice Dynamic Volume Rate")
        if name == "PITAR" and pi_mask is not None and "daidtt" in ds:
            return calc.compute_area_rate(ds["daidtt"], area, pi_mask, name = "PITAR", long_name = "Pack Ice Thermodynamic Area Rate")
        if name == "PIMAR" and pi_mask is not None and "daidtd" in ds:
            return calc.compute_area_rate(ds["daidtd"], area, pi_mask, name = "PIMAR", long_name = "Pack Ice Dynamic Area Rate")
        #-----------------------------------------------------------------------------------
        # all sea ice (primaries)
        #----------------------------------------------------------------------------------
        if name == "SIA":
            return calc.compute_area_series(aice, area, None, name = "SIA", long_name = "Sea Ice Area", scale = ctx.area_scale)
        if name == "SIV":
            return calc.compute_volume_series(aice, hi, area, None, name = "SIV", long_name = "Sea Ice Volume", scale = ctx.volume_scale)
        if name == "SIT":
            return calc.compute_thickness_series(aice, hi, area, None, name = "SIT", long_name = "Sea Ice Thickness")
        if name == "SIP":
            return calc.compute_persistence_mask(None, name = "SIP", long_name = "Sea Ice Persistence")
        if name == "SIS" and "strength" in ds:
            return calc.compute_strength_series(aice, hi, ds["strength"], area, None, name = "SIS", long_name = "Sea Ice Strength")
        if name == "SITVR" and "dvidtt" in ds:
            return calc.compute_volume_rate(ds["dvidtt"], aice, area, None, name = "SITVR", long_name = "Sea Ice Thermodynamic Volume Rate")
        if name == "SIMVR" and "dvidtd" in ds:
            return calc.compute_volume_rate(ds["dvidtd"], aice, area, None, name = "SIMVR", long_name = "Sea Ice Dynamic Volume Rate")
        if name == "SITAR" and "daidtt" in ds:
            return calc.compute_area_rate(ds["daidtt"], area, None, name = "SITAR", long_name = "Sea Ice Thermodynamic Area Rate")
        if name == "SIMAR" and "daidtd" in ds:
            return calc.compute_area_rate(ds["daidtd"], area, None, name = "SIMAR", long_name = "Sea Ice Dynamic Area Rate")
        #-----------------------------------------------------------------------------------
        # fast ice (spatials -- 2D)
        #----------------------------------------------------------------------------------
        if name == "FIHI" and fi_mask is not None:
            return calc.compute_temporal_mean(hi.where(fi_mask), name = "FIHI", long_name = "Fast Ice Mean Thickness")
        if name == "FIST" and fi_mask is not None and "strength" in ds:
            return calc.compute_temporal_mean(ds['strength'].where(fi_mask), name = "FIST", long_name = "Fast Ice Mean Strength")
        if name == "FITVR_YR" and fi_mask is not None and "dvidtt" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtt"], fi_mask, name = "FITVR_YR", long_name = "Fast Ice Thermodynamic Volume Rate Climatology")
        if name == "FIMVR_YR" and fi_mask is not None and "dvidtd" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtd"], fi_mask, name = "FIMVR_YR", long_name = "Fast Ice Dynamic Volume Rate Climatology")
        if name == "FITAR_YR" and fi_mask is not None and "daidtt" in ds:
            return calc.compute_spatial_rate_year(ds["daidtt"], fi_mask, name = "FITAR_YR", long_name = "Fast Ice Thermodynamic Area Rate Climatology", area = area)
        if name == "FIMAR_YR" and fi_mask is not None and "daidtd" in ds:
            return calc.compute_spatial_rate_year(ds["daidtd"], fi_mask, name = "FIMAR_YR", long_name = "Fast Ice Dynamic Area Rate Climatology", area = area)
        #-----------------------------------------------------------------------------------
        # pack ice (spatials -- 2D)
        #----------------------------------------------------------------------------------
        if name == "PIHI" and pi_mask is not None:
            return calc.compute_temporal_mean(hi.where(pi_mask), name = "PIHI", long_name = "Pack Ice Mean Thickness")
        if name == "PIST" and pi_mask is not None and "strength" in ds:
            return calc.compute_temporal_mean(ds['strength'].where(pi_mask), name = "PIST", long_name = "Pack Ice Mean Strength")
        if name == "PITVR_YR" and pi_mask is not None and "dvidtt" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtt"], pi_mask, name = "PITVR_YR", long_name = "Pack Ice Thermodynamic Volume Rate Climatology")
        if name == "PIMVR_YR" and pi_mask is not None and "dvidtd" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtd"], pi_mask, name = "PIMVR_YR", long_name = "Pack Ice Dynamic Volume Rate Climatology")
        if name == "PITAR_YR" and pi_mask is not None and "daidtt" in ds:
            return calc.compute_spatial_rate_year(ds["daidtt"], pi_mask, name = "PITAR_YR", long_name = "Pack Ice Thermodynamic Area Rate Climatology", area = area)
        if name == "PIMAR_YR" and pi_mask is not None and "daidtd" in ds:
            return calc.compute_spatial_rate_year(ds["daidtd"], pi_mask, name = "PIMAR_YR", long_name = "Pack Ice Dynamic Area Rate Climatology", area = area)
        #-----------------------------------------------------------------------------------
        # all sea ice (spatials -- 2D)
        #----------------------------------------------------------------------------------
        if name == "SIHI":
            return calc.compute_temporal_mean(hi.where(si_mask), name = "SIHI", long_name = "Sea Ice Mean Thickness")
        if name == "SIST" and "strength" in ds:
            return calc.compute_temporal_mean(ds['strength'].where(si_mask), name = "SIST", long_name = "Sea Ice Mean Strength")
        if name == "SITVR_YR" and "dvidtt" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtt"], si_mask, name = "SITVR_YR", long_name = "Sea Ice Thermodynamic Volume Rate Climatology")
        if name == "SIMVR_YR" and "dvidtd" in ds:
            return calc.compute_spatial_rate_year(ds["dvidtd"], si_mask, name = "SIMVR_YR", long_name = "Sea Ice Dynamic Volume Rate Climatology")
        if name == "SITAR_YR" and "daidtt" in ds:
            return calc.compute_spatial_rate_year(ds["daidtt"], si_mask, name = "SITAR_YR", long_name = "Sea Ice Thermodynamic Area Rate Climatology", area = area)
        if name == "SIMAR_YR" and "daidtd" in ds:
            return calc.compute_spatial_rate_year(ds["daidtd"], si_mask, name = "SIMAR_YR", long_name = "Sea Ice Dynamic Area Rate Climatology", area = area)
        #-----------------------------------------------------------------------------------
        # IA/IT by region for FI/PI/SI
        #----------------------------------------------------------------------------------
        if name in {"FIA_by_region", "FIT_by_region"} and fi_mask is not None:
            fia_reg, fit_reg = calc.compute_region_series(aice, hi, area, region_mask, fi_mask,
                                                          area_name           = "FIA_by_region",
                                                          thickness_name      = "FIT_by_region",
                                                          area_long_name      = "Fast Ice Area by Antarctic sector",
                                                          thickness_long_name = "Fast Ice Thickness by Antarctic sector")
            self._remember_many({"FIA_by_region": fia_reg, "FIT_by_region": fit_reg})
            return self.memo.get(name)
        if name in {"PIA_by_region", "PIT_by_region"} and pi_mask is not None:
            pia_reg, pit_reg = calc.compute_region_series(aice, hi, area, region_mask, pi_mask,
                                                          area_name           = "PIA_by_region",
                                                          thickness_name      = "PIT_by_region",
                                                          area_long_name      = "Pack Ice Area by Antarctic sector",
                                                          thickness_long_name = "Pack Ice Thickness by Antarctic sector")
            self._remember_many({"PIA_by_region": pia_reg, "PIT_by_region": pit_reg})
            return self.memo.get(name)
        if name in {"SIA_by_region", "SIT_by_region"}:
            sia_reg, sit_reg = calc.compute_region_series(aice, hi, area, region_mask, None,
                                                          area_name           = "SIA_by_region",
                                                          thickness_name      = "SIT_by_region",
                                                          area_long_name      = "Sea Ice Area by Antarctic sector",
                                                          thickness_long_name = "Sea Ice Thickness by Antarctic sector")
            self._remember_many({"SIA_by_region": sia_reg, "SIT_by_region": sit_reg})
            return self.memo.get(name)
        return None

