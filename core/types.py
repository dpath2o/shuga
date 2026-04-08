from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(slots=True)
class RunSpec:
    sim_name: str
    start_date: str
    end_date: str
    hemisphere: str = "SH"
    project: str = "gv90"
    user: str = "da1339"

@dataclass(slots=True)
class ClassificationSpec:
    ice_type: str = "FI"
    grid_type: str | Sequence[str] = "Tc"
    ispd_thresh: float = 5e-4
    methods: tuple[str, ...] = ("raw", "binary-days", "rolling-mean")
    bin_window: int = 11
    bin_min_days: int = 9
    roll_window: int = 15
    # B-grid / already-on-native-u-v outputs
    speed_var_u: str = "uvel"
    speed_var_v: str = "vvel"
    # C-grid east/north edge outputs
    uvelE_var: str = "uvelE"
    uvelN_var: str = "uvelN"
    vvelE_var: str = "vvelE"
    vvelN_var: str = "vvelN"
    # common thresholds / behavior
    aice_var: str = "aice"
    aice_thresh: float = 0.15
    wrap_x: bool = True
    cgrid_combine: str = "mean"
    @classmethod
    def from_methods(cls, methods: Iterable[str] | None = None, **kwargs) -> "ClassificationSpec":
        obj = cls(**kwargs)
        if methods is not None:
            obj.methods = tuple(methods)
        return obj


@dataclass(slots=True)
class MetricsSpec:
    methods: tuple[str, ...] = ("binary-days", "rolling-mean")
    obs_metrics_store: str | None = None
    obs_fia_var: str = "FIA"
    obs_fit_var: str = "FIT"
    coast_distance_var: str | None = None
    area_scale: float = 1e9
    volume_scale: float = 1e12

@dataclass(slots=True)
class PlottingSpec:
    fig_size: float = 20.0
    fip_fig_size: float = 20.0
    split_fig_size: float = 14.0
    region_fig_size: float = 20.0
    shorelines: str = "0.25p,black"
    land: str = "lightgray"
    water: str = "white"
    grid_style: str = "s0.1c"
    nsidc_pen: str = "1p,green"
    fip_cmap: str | Path | None = None
    colorbar_position: str = "JMB+w8c/0.4c+v+o0.8c/0c"
    colorbar_xlabel: str | None = None #"Fast Ice Persistence"
    colorbar_ylabel: str | None = None


@dataclass(slots=True)
class ObservationSpec:
    seaice_root: str | Path | None = None
    nsidc_root: str | Path | None = None
    nsidc_version: str = "G02202_V6"
    nsidc_cellarea_root: str | Path | None = None
    nsidc_cellarea_product: str = "NSIDC0771"
    nsidc_resolution_km: float = 25.0
    nsidc_sic_var: str = "cdr_seaice_conc"
    nsidc_threshold: float = 0.15
    af2020_root: str | Path | None = None
    af2020_fia_daily_file: str = "FIA_AF2020_daily.nc"
    af2020_fia_daily_var: str = "AF2020"
    af2020_regridded_store: str = "AF-FI-2020db_org-timestep_reG.zarr"
    af2020_climatology_store: str = "AF-FI-2020db_18yrAvg_gridded.zarr"
    af2020_regridded_var: str = "FI"
    af2020_climatology_var: str = "FI_OBS_GRD"
    af2020_time_var: str = "t_FI_obs"
    af2020_doy_var: str = "doy"
