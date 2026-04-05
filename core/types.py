
from __future__ import annotations

from dataclasses import dataclass
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
    obs_metrics_store: str | None = None
    obs_fia_var: str = "FIA"
    obs_fit_var: str = "FIT"
    coast_distance_var: str | None = None
    area_scale: float = 1e9
    volume_scale: float = 1e12
