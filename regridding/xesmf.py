from __future__ import annotations
import logging, re
from dataclasses import dataclass
from pathlib import Path
import xarray as xr
import xesmf as xe

LOGGER = logging.getLogger(__name__)

def sanitise_weight_token(value: str) -> str:
    """
    Make a string safe and readable for use in a weight filename.
    """
    value = str(value).strip()
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")

def format_era5_to_cice_weight_filename(cice_grid_file: str | Path,
                                        regrid_method : str,
                                        extrap_method : str) -> str:
    """
    Format ERA5 -> CICE xESMF weight filename.

    Pattern:
        map_ERA5_to_{CICE_GRID_FILENAME_STEM}_{regrid_method}_{extrap_method}.nc

    Example:
        ACCESS-OM3-025_Cgrid.nc, patch, nearest_s2d
        -> map_ERA5_to_ACCESS-OM3-025_Cgrid_patch_nearest_s2d.nc
    """
    grid_stem = Path(cice_grid_file).name
    if grid_stem.endswith(".nc"):
        grid_stem = grid_stem[:-3]
    grid_stem = sanitise_weight_token(grid_stem)
    method    = sanitise_weight_token(regrid_method)
    extrap    = sanitise_weight_token(extrap_method)
    return f"map_ERA5_to_{grid_stem}_{method}_{extrap}.nc"

def normalise_cice_forcing_dims(da: xr.DataArray) -> xr.DataArray:
    """
    Normalise xESMF output dimensions to the legacy CICE forcing convention.

    CICE forcing files historically use:
        variable(time, ny, nx)

    xESMF may return:
        variable(time, nj, ni)

    This function renames nj/ni -> ny/nx and keeps time first.
    """
    rename = {}
    if "nj" in da.dims:
        rename["nj"] = "ny"
    if "ni" in da.dims:
        rename["ni"] = "nx"
    if rename:
        da = da.rename(rename)
    if "time" in da.dims:
        spatial_dims = [d for d in da.dims if d != "time"]
        da = da.transpose("time", *spatial_dims)
    return da

@dataclass(frozen=True)
class XESMFRegridSpec:
    method         : str = "patch"
    extrap_method  : str = "nearest_s2d"
    weight_file    : Path | str | None = None
    rebuild_weights: bool = False
    reuse_weights  : bool = True

def build_xesmf_regridder(src: xr.Dataset | xr.DataArray,
                          dst: xr.Dataset,
                          spec: XESMFRegridSpec, *, logger = None) -> xe.Regridder:
    """
    Build or reuse an xESMF regridder.

    Weight-file logic:
      - rebuild_weights=True removes an existing weight file first.
      - otherwise, reuse existing weights if present.
      - if no weight file exists, xESMF builds it.
    """
    log      = logger or LOGGER
    filename = None
    reuse    = spec.reuse_weights
    if spec.weight_file is not None:
        weight_file = Path(spec.weight_file)
        weight_file.parent.mkdir(parents=True, exist_ok=True)
        if spec.rebuild_weights and weight_file.exists():
            log.warning("Removing existing xESMF weight file: %s", weight_file)
            weight_file.unlink()
        filename = str(weight_file)
        reuse    = weight_file.exists() and not spec.rebuild_weights
        log.info("xESMF weight file     : %s", weight_file)
        log.info("xESMF reuse weights   : %s", reuse)
        log.info("xESMF rebuild weights : %s", spec.rebuild_weights)
    log.info("xESMF method          : %s", spec.method)
    log.info("xESMF extrap_method   : %s", spec.extrap_method)
    return xe.Regridder(src, dst, spec.method, extrap_method = spec.extrap_method, reuse_weights = reuse, filename = filename)

def regrid_dataarray_to_cice_tgrid(da       : xr.DataArray,
                                   regridder: xe.Regridder,
                                   name     : str,
                                   long_name: str | None = None,
                                   units    : str | None = None,
                                   dtype    : str = "float32") -> xr.DataArray:
    """
    Apply an xESMF regridder and normalise result to CICE forcing convention.
    """
    out      = regridder(da).astype(dtype)
    out.name = name
    out      = normalise_cice_forcing_dims(out)
    if long_name is not None:
        out.attrs["long_name"] = long_name
    elif "long_name" in da.attrs:
        out.attrs["long_name"] = da.attrs["long_name"]
    if units is not None:
        out.attrs["units"] = units
    elif "units" in da.attrs:
        out.attrs["units"] = da.attrs["units"]
    return out
