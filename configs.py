"""
Configuration/specification namespace for shuga.

Examples
--------
from shuga import configs

run_cfg = configs.RunSpec(...)
pth_cfg = configs.ShugaPaths(...)
"""

from .core.types import (RunSpec,
                         ClassificationSpec,
                         MetricsSpec,
                         PlottingSpec,
                         ObservationSpec,
                         CICEGridSpec,
                         WaveForcingSpec,
                         LateralDragSpec)
from .core.paths import ShugaPaths

__all__ = ["RunSpec",
           "ClassificationSpec",
           "MetricsSpec",
           "PlottingSpec",
           "ObservationSpec",
           "CICEGridSpec",
           "WaveForcingSpec",
           "LateralDragSpec",
           "ShugaPaths"]
