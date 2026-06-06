"""
shuga: Antarctic fast-ice classification, metrics, plotting, observations,
regridding, and forcing workflows for CICE Zarr output.

The top-level package intentionally avoids eager imports of heavy optional
submodules such as plotting, waves, pyresample, and observations. Import those
objects directly from their submodules when possible, e.g.

    from shuga.plotting.cice import CICEPlotter
    from shuga.waves.cawcr import CAWCRRegridConfig, CAWCRRegridder
    from shuga.observations import AF2020Observations, NSIDCObservations
"""

__version__ = "0.3.0"

from .core.types import (RunSpec,
                         ClassificationSpec,
                         MetricsSpec,
                         PlottingSpec,
                         ObservationSpec,
                         CICEGridSpec,
                         WaveForcingSpec,
                         LateralDragSpec)
from .core.paths import ShugaPaths
from .core.logging import build_file_logger

__all__ = ["__version__",
           "RunSpec",
           "ClassificationSpec",
           "MetricsSpec",
           "PlottingSpec",
           "ObservationSpec",
           "CICEGridSpec",
           "WaveForcingSpec",
           "LateralDragSpec",
           "ShugaPaths",
           "build_file_logger",
           "CICEStoreLocator",
           "IceHistoryLoader",
           "CICEClassifier",
           "CICEMetrics",
           "CICEGridwork",
           "FormFactors",
           "CAWCRRegridConfig",
           "CAWCRRegridder",
           "CICEPlotter",
           "SeaIceObservations",
           "AF2020Observations",
           "NSIDCObservations",
           "load_ice_history",
           "load_cice",
           "load_classified",
           "load_metrics",
           "open_cice_history",
           "report_sim_status"]


def __getattr__(name):
    """
    Lazy top-level imports for backwards compatibility.

    This keeps `import shuga` cheap and prevents scripts that only need one
    lightweight submodule from importing plotting, waves, observations, etc.
    """
    if name == "CICEStoreLocator":
        from .io.store_locator import CICEStoreLocator
        return CICEStoreLocator

    if name == "IceHistoryLoader":
        from .io import IceHistoryLoader
        return IceHistoryLoader

    if name in {"load_ice_history", "load_cice", "load_classified", "load_metrics", "open_cice_history"}:
        from . import io
        return getattr(io, name)

    if name == "CICEClassifier":
        from .classify.cice import CICEClassifier
        return CICEClassifier

    if name == "CICEMetrics":
        from .metrics.cice import CICEMetrics
        return CICEMetrics

    if name == "CICEGridwork":
        from .grid.cice import CICEGridwork
        return CICEGridwork

    if name == "FormFactors":
        from .grid.lateral_drag import FormFactors
        return FormFactors

    if name in {"CAWCRRegridConfig", "CAWCRRegridder"}:
        from .waves.cawcr import CAWCRRegridConfig, CAWCRRegridder
        return {"CAWCRRegridConfig": CAWCRRegridConfig, "CAWCRRegridder": CAWCRRegridder}[name]

    if name == "CICEPlotter":
        from .plotting.cice import CICEPlotter
        return CICEPlotter

    if name in {"SeaIceObservations", "AF2020Observations", "NSIDCObservations", "SeaIceObs", "AF2020Obs", "NSIDCObs"}:
        from .observations import SeaIceObservations, AF2020Observations, NSIDCObservations
        return {
            "SeaIceObservations": SeaIceObservations,
            "AF2020Observations": AF2020Observations,
            "NSIDCObservations": NSIDCObservations,
        }[name]

    if name == "report_sim_status":
        from .core.reporting import report_sim_status
        return report_sim_status

    raise AttributeError(f"module 'shuga' has no attribute {name!r}")
