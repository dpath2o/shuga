"""shuga: Antarctic fast-ice classification, metrics, plotting, and observations workflows for CICE Zarr output."""

from .core.types           import (RunSpec,
                                   ClassificationSpec,
                                   MetricsSpec,
                                   PlottingSpec,
                                   ObservationSpec,
                                   CICEGridSpec,
                                   WaveForcingSpec,
                                   LateralDragSpec)
from .core.paths           import ShugaPaths
from .core.logging         import build_file_logger
from .io                   import load_cice, load_classified, load_metrics, open_cice_history
from .io.store_locator     import CICEStoreLocator
from .classify.cice        import CICEClassifier
from .metrics.cice         import CICEMetrics
from .grid.cice            import CICEGridwork
from .grid.lateral_drag    import FormFactors
from .waves.cawcr          import CAWCRRegridConfig, CAWCRRegridder
from .plotting             import CICEPlotter
from .observations         import SeaIceObservations
__all__     = ["RunSpec",
               "ClassificationSpec",
               "MetricsSpec",
               "PlottingSpec",
               "ObservationSpec",
               "CICEGridSpec",
               "WaveForcingSpec",
               "LateralDragSpec",
               "ShugaPaths",
               "build_file_logger",
               "load_cice",
               "load_classified",
               "load_metrics",
               "open_cice_history",
               "CICEStoreLocator",
               "CICEClassifier",
               "CICEMetrics",
               "CICEGridwork",
               "FormFactors",
               "CAWCRRegridConfig",
               "CAWCRRegridder",
               "CICEPlotter",
               "SeaIceObservations"]
__version__ = "0.1.3"
