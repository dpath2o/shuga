"""shuga: Antarctic fast-ice classification, metrics, plotting, and observations workflows for CICE Zarr output."""

from .core.types           import RunSpec, ClassificationSpec, MetricsSpec, PlottingSpec, ObservationSpec
from .core.paths           import ShugaPaths
from .core.store_selection import StoreSelection, ResolvedStore
from .io.store_locator     import CICEStoreLocator
from .classify.cice        import CICEClassifier
from .metrics.cice         import CICEMetrics
from .plotting             import CICEPlotter
from .observations         import SeaIceObservations
__all__     = ["RunSpec",
               "ClassificationSpec",
               "MetricsSpec",
               "PlottingSpec",
               "ObservationSpec",
               "ShugaPaths",
               "CICEClassifier",
               "CICEMetrics",
               "CICEPlotter",
               "SeaIceObservations"]
__version__ = "0.2.0"
