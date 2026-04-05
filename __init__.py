"""shugga: Antarctic fast-ice classification and metrics workflows for CICE Zarr output."""

from .core.types import RunSpec, ClassificationSpec, MetricsSpec
from .core.paths import ShugaPaths
from .classify.cice import CICEClassifier
from .metrics.cice import CICEMetrics

__all__ = [
    "RunSpec",
    "ClassificationSpec",
    "MetricsSpec",
    "ShuggaPaths",
    "CICEClassifier",
    "CICEMetrics",
]

__version__ = "0.1.0"
