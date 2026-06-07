from __future__ import annotations
from pathlib import Path
import xarray as xr
from shuga.core.paths import ShugaPaths
from shuga.core.types import CICEGridSpec
from shuga.grid.cice import CICEGridwork

def load_cice_tgrid_for_xesmf(cice_grid_file: str | Path | None = None, *, lon_type: str = "0-360", logger = None) -> xr.Dataset:
    """
    Load a CICE/ACCESS-OM3 T grid using shuga.grid and expose it in xESMF form.

    Parameters
    ----------
    cice_grid_file
        Full path to the destination CICE grid file. If omitted, the default
        grid file is resolved from shuga.core.paths.ShugaPaths.

    Returns
    -------
    xr.Dataset
        Dataset with 2-D xESMF destination coordinates:
            lon(ny, nx)
            lat(ny, nx)

    Notes
    -----
    The returned dimension names intentionally match the legacy CICE forcing
    file convention used by the existing annual ERA5 products.
    """
    pth_cfg   = ShugaPaths()
    grid_path = pth_cfg.resolve_cice_grid_file(cice_grid_file)
    grid_spec = CICEGridSpec(lon_type = lon_type)
    gridwork  = CICEGridwork(pth_cfg = pth_cfg, grid_spec = grid_spec, logger = logger)
    bundle    = gridwork.load_cice_grid(P_grid = grid_path, build_faces = False)
    lon       = xr.DataArray(bundle.tgrid["TLON"].values, dims = ("ny", "nx"), name = "lon", attrs = {"units": "degrees_east"})
    lat       = xr.DataArray(bundle.tgrid["TLAT"].values, dims = ("ny", "nx"), name = "lat", attrs = {"units": "degrees_north"})
    return xr.Dataset(data_vars = {"lon": lon, "lat": lat},
                      attrs     = { "source_path" : str(grid_path), "grid_kind" : bundle.grid_kind})
