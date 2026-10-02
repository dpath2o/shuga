"""
Loader namespace for shuga.

Examples
--------
from shuga import loaders

ds = loaders.load_cice(...)
mets = loaders.load_metrics(...)
"""

from .io import (IceHistoryLoader,
                 load_ice_history,
                 load_cice,
                 load_classified,
                 load_metrics,
                 open_cice_history)

__all__ = ["IceHistoryLoader",
           "load_ice_history",
           "load_cice",
           "load_classified",
           "load_metrics",
           "open_cice_history"]
