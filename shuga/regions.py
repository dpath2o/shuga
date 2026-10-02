"""
Region namespace for shuga.

Examples
--------
from shuga import regions

reg_plot = regions.ANTARCTIC_8_REGIONS["EIO"]["plot_region"]
"""
from .core.regions import ANTARCTIC_8_REGIONS

__all__ = ["ANTARCTIC_8_REGIONS"]
