from __future__ import annotations
import xarray as xr

def sanitise_for_zarr_write(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove backend encodings that commonly poison fresh Zarr writes.
    """
    out = ds.copy()
    for name in out.variables:
        out[name].encoding = {}
    out.encoding = {}
    return out

def strip_to_time_coord(da: xr.DataArray, *,
                        name: str | None = None) -> xr.DataArray:
    """
    Return a DataArray with only the time coordinate retained.

    This is useful for classification/metrics outputs where static spatial
    coordinates should not be duplicated into every product store.
    """
    coords = {"time": da["time"]} if "time" in da.coords else None
    return xr.DataArray(da.data,
                        dims   = da.dims,
                        coords = coords,
                        name   = name or da.name,
                        attrs  = dict(da.attrs))
