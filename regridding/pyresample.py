from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import xarray as xr

@dataclass(frozen=True, slots=True)
class PyresampleSpec:
    """Configuration for AF2020/CICE common-grid pyresample workflows."""
    pixel_size_m         : float = 5_000.0
    radius_of_influence_m: float = 10_000.0
    buffer_m             : float = 20_000.0
    area_id              : str = "epsg3031_5km_union"
    projection           : str = "EPSG:3031"
    fill_value           : float = np.nan

# -----------------------------------------------------------------------------
# Longitude/projection/grid geometry helpers
# -----------------------------------------------------------------------------
def normalise_longitudes(lon, to: str = "-180-180"):
    """Normalise longitudes to either ``-180-180`` or ``0-360``."""
    arr = np.asarray(lon)
    token = str(to).replace("..", "-")
    if token in {"0-360", "0_360"}:
        return arr % 360.0
    if token in {"-180-180", "180", "pm180"}:
        return ((arr + 180.0) % 360.0) - 180.0
    raise ValueError("to must be '-180-180' or '0-360'.")

def _as_numpy2d(value) -> np.ndarray:
    arr = value.values if hasattr(value, "values") else np.asarray(value)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D lon/lat array, got shape={arr.shape}")
    return arr

def to_3031_extent(lat2d, lon2d, *, buffer_m: float = 20_000.0) -> list[float]:
    """
    Project a swath's lat/lon coordinates to EPSG:3031 and return a buffered extent.

    Returns ``[xmin, ymin, xmax, ymax]`` in metres. Longitudes are wrapped to
    ``[-180, 180)`` before projection, matching the AFIM pyresample workflow.
    """
    from pyproj import Transformer
    lat = _as_numpy2d(lat2d)
    lon = normalise_longitudes(_as_numpy2d(lon2d), to="-180-180")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = transformer.transform(lon.ravel(), lat.ravel())
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("No finite projected points found while building EPSG:3031 extent.")
    xmin, xmax = x[finite].min(), x[finite].max()
    ymin, ymax = y[finite].min(), y[finite].max()
    return [float(xmin - buffer_m), float(ymin - buffer_m), float(xmax + buffer_m), float(ymax + buffer_m)]

def union_extents(extents: Sequence[Sequence[float]]) -> list[float]:
    """Return the bounding union of multiple EPSG:3031 extents."""
    if not extents:
        raise ValueError("At least one extent is required.")
    xs = [float(e[0]) for e in extents] + [float(e[2]) for e in extents]
    ys = [float(e[1]) for e in extents] + [float(e[3]) for e in extents]
    return [min(xs), min(ys), max(xs), max(ys)]

def snap_extent_to_grid(extent: Sequence[float], pixel_size: float) -> list[float]:
    """Expand an EPSG:3031 extent so all edges lie on a pixel-size grid."""
    xmin, ymin, xmax, ymax = [float(v) for v in extent]
    p = float(pixel_size)
    return [float(np.floor(xmin / p) * p),
            float(np.floor(ymin / p) * p),
            float(np.ceil(xmax / p) * p),
            float(np.ceil(ymax / p) * p)]

def make_area_definition(extent     : Sequence[float], *,
                         pixel_size : float = 5_000.0,
                         area_id    : str = "epsg3031_5km_union",
                         description: str | None = None):
    """Create a pyresample ``AreaDefinition`` for a regular EPSG:3031 grid."""
    from pyresample.geometry import AreaDefinition
    xmin, ymin, xmax, ymax = [float(v) for v in extent]
    p = float(pixel_size)
    width = int(round((xmax - xmin) / p))
    height = int(round((ymax - ymin) / p))
    xmax = xmin + width * p
    ymax = ymin + height * p
    return AreaDefinition(area_id     = area_id,
                          description = description or f"Common EPSG:3031 grid, {p:g} m pixels",
                          proj_id     = "epsg3031",
                          projection  = "EPSG:3031",
                          width       = width,
                          height      = height,
                          area_extent = (xmin, ymin, xmax, ymax))

def grid_coords_from_area(area_def, *, pixel_size: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D x/y cell-centre coordinates from an AreaDefinition."""
    xmin, ymin, xmax, ymax = [float(v) for v in area_def.area_extent]
    width, height = int(area_def.width), int(area_def.height)
    if pixel_size is None:
        pixel_size = (xmax - xmin) / width
    p = float(pixel_size)
    x = xmin + (np.arange(width) + 0.5) * p
    y = ymax - (np.arange(height) + 0.5) * p
    return x, y

def lonlat_from_area(area_def, *, pixel_size: float | None = None, wrap: str = "0-360") -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lon/lat arrays for a regular EPSG:3031 AreaDefinition."""
    from pyproj import Transformer

    x, y = grid_coords_from_area(area_def, pixel_size=pixel_size)
    X, Y = np.meshgrid(x, y)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(X, Y)
    lon = normalise_longitudes(lon, to=wrap)
    return lon.astype("float32"), np.asarray(lat, dtype="float32")

def area_definition_from_lonlat_pairs(pairs: Sequence[tuple[object, object]], *, spec: PyresampleSpec = PyresampleSpec()):
    """
    Build a common EPSG:3031 grid from multiple ``(lat2d, lon2d)`` inputs.

    This reproduces the AFIM notebook pattern:
    ``extent = snap(union([to_3031_extent(model), to_3031_extent(obs)]), PIXEL)``.
    """
    extents = [to_3031_extent(lat, lon, buffer_m=spec.buffer_m) for lat, lon in pairs]
    extent = snap_extent_to_grid(union_extents(extents), spec.pixel_size_m)
    return make_area_definition(extent, pixel_size=spec.pixel_size_m, area_id=spec.area_id)

def area_definition_from_xy(x, y, *, pixel_size: float = 5_000.0, area_id: str = "epsg3031_common", description: str | None = None):
    """
    Reconstruct a pyresample AreaDefinition from 1-D EPSG:3031 x/y centre coordinates.

    This is useful when a stored AF2020 common-grid zarr already contains x/y,
    and simulation FIP later needs to be resampled onto that exact grid.
    """
    from pyresample.geometry import AreaDefinition
    x = np.asarray(x.values if hasattr(x, "values") else x, dtype="float64")
    y = np.asarray(y.values if hasattr(y, "values") else y, dtype="float64")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D coordinate arrays.")
    p    = float(pixel_size)
    xmin = float(np.nanmin(x) - 0.5 * p)
    xmax = float(np.nanmax(x) + 0.5 * p)
    ymin = float(np.nanmin(y) - 0.5 * p)
    ymax = float(np.nanmax(y) + 0.5 * p)
    return AreaDefinition(area_id     = area_id,
                          description = description or f"EPSG:3031 common grid reconstructed from x/y, {p:g} m pixels",
                          proj_id     = "epsg3031",
                          projection  = "EPSG:3031",
                          width       = int(x.size),
                          height      = int(y.size),
                          area_extent = (xmin, ymin, xmax, ymax))

# -----------------------------------------------------------------------------
# Pyresample application helpers
# -----------------------------------------------------------------------------
def resample_swath_to_area(src_da: xr.DataArray, lat2d, lon2d, area_def, *,
                           radius     : float = 10_000.0,
                           fill_value = np.nan,
                           pixel_size : float | None = None,
                           name       : str | None = None) -> xr.DataArray:
    """
    Nearest-neighbour resample a 2-D swath field to an EPSG:3031 AreaDefinition grid.

    This is the direct shuga equivalent of the AFIM ``resample_swath_to_area`` helper.
    It is intentionally nearest-neighbour: the FIP/FIC products are categorical or
    fractional occupancy products and should not be smoothed by bilinear interpolation.
    """
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest
    lat = _as_numpy2d(lat2d)
    lon = normalise_longitudes(_as_numpy2d(lon2d), to="-180-180")
    data = src_da.values
    if data.ndim != 2:
        raise ValueError(f"resample_swath_to_area expects a 2-D DataArray, got dims={src_da.dims}")
    swath = SwathDefinition(lons = lon, lats = lat)
    out2d = resample_nearest(source_geo_def      = swath,
                             data                = data,
                             target_geo_def      = area_def,
                             radius_of_influence = float(radius),
                             fill_value          = fill_value,
                             nprocs              = 0,
                             reduce_data         = True)
    x, y = grid_coords_from_area(area_def, pixel_size=pixel_size)
    da_out = xr.DataArray(out2d,
                          dims   = ("y", "x"),
                          coords = {"x": ("x", x, {"units": "m", "standard_name": "projection_x_coordinate"}),
                                    "y": ("y", y, {"units": "m", "standard_name": "projection_y_coordinate"})},
                          name   = name or src_da.name,
                          attrs  = {"crs": "EPSG:3031", "grid_mapping": "spstereo", **dict(src_da.attrs)})
    return da_out

def resample_dataarray_to_area(da: xr.DataArray, lat2d, lon2d, area_def, *,
                               radius     : float = 10_000.0,
                               fill_value = np.nan,
                               pixel_size : float | None = None,
                               time_dim   : str = "time",
                               name       : str | None = None) -> xr.DataArray:
    """Resample either a 2-D field or a time series of 2-D fields to an AreaDefinition."""
    if time_dim not in da.dims:
        return resample_swath_to_area(da, lat2d, lon2d, area_def, radius = radius, fill_value = fill_value, pixel_size = pixel_size, name = name)
    pieces: list[xr.DataArray] = []
    times = pd.to_datetime(da[time_dim].values)
    for t in times:
        src   = da.sel({time_dim: t})
        piece = resample_swath_to_area(src, lat2d, lon2d, area_def, radius = radius, fill_value = fill_value, pixel_size = pixel_size, name = name or da.name).expand_dims({time_dim: [t]})
        pieces.append(piece)
    out      = xr.concat(pieces, dim=time_dim)
    out.name = name or da.name
    out.attrs.update(da.attrs)
    return out

def add_lonlat_from_epsg3031(ds: xr.Dataset | xr.DataArray, *, wrap: str = "0-360", out_dtype: str = "float32"):
    """Attach 2-D lon/lat coordinates to an x/y EPSG:3031 dataset or data array."""
    is_da = isinstance(ds, xr.DataArray)
    obj = ds.to_dataset(name=ds.name or "field") if is_da else ds
    if "x" not in obj.dims or "y" not in obj.dims:
        raise ValueError("Expected EPSG:3031 object with dims ('y', 'x').")
    from pyproj import Transformer
    x = obj["x"].values
    y = obj["y"].values
    X, Y = np.meshgrid(x, y)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_np, lat_np = transformer.transform(X, Y)
    lon_np = normalise_longitudes(lon_np, to=wrap)
    lon = xr.DataArray(lon_np, dims=("y", "x"), coords={"y": obj["y"], "x": obj["x"]})
    lat = xr.DataArray(lat_np, dims=("y", "x"), coords={"y": obj["y"], "x": obj["x"]})
    if out_dtype:
        lon = lon.astype(out_dtype)
        lat = lat.astype(out_dtype)
    obj = obj.assign_coords(lon=lon, lat=lat)
    return obj[next(iter(obj.data_vars))] if is_da else obj

# -----------------------------------------------------------------------------
# Fast-ice persistence/FIC comparison helpers
# -----------------------------------------------------------------------------
def fip_difference_dataset(mod_fip: xr.DataArray, obs_fip: xr.DataArray, *,
                           tol               : float = 1e-8,
                           category_threshold: float = 0.5,
                           name_mod          : str = "mod",
                           name_obs          : str = "obs") -> xr.Dataset:
    """
    Return a continuous and categorical FIP difference dataset on a shared grid.

    This intentionally follows the AFIM notebook convention:

        diff = mod - obs

    with:
      - FIP['diff'] continuous in [-1, 1];
      - cells where both model and observation are zero masked before differencing;
      - FIP['diff_cat'] retained as an optional plotting/statistics aid.

    Category codes:
      - 0: agreement,           -threshold <= diff <= threshold
      - 1: simulation-dominant,  diff > threshold
      - 2: observation-dominant, diff < -threshold
      - NaN: both zero, non-finite, or exactly outside (-1, 1) category range.
    """
    mod, obs  = xr.align(mod_fip.astype("float32").rename(name_mod), obs_fip.astype("float32").rename(name_obs), join = "exact")
    both_zero = (xr.apply_ufunc(np.isclose, mod, 0.0, kwargs={"atol": tol}) & xr.apply_ufunc(np.isclose, obs, 0.0, kwargs={"atol": tol}))
    finite    = np.isfinite(mod) & np.isfinite(obs)
    diff      = (mod - obs)
    diff      = diff.where(~both_zero)
    diff      = diff.where(finite)
    diff      = diff.clip(min=-1.0, max=1.0)
    diff      = diff.astype("float32").rename("diff")
    diff.attrs.update(long_name       = "Fast ice persistence difference",
                      units           = "1",
                      sign_convention = "model minus observation",
                      valid_min       = -1.0,
                      valid_max       = 1.0,
                      comment         = "Continuous FIP difference. Cells where model and observation are both zero are masked before differencing.")
    th  = float(category_threshold)
    cat = xr.full_like(diff, np.nan, dtype="float32")
    valid_for_cat = (diff > -1.0) & (diff < 1.0)
    cat = xr.where(valid_for_cat & (diff < -th), 2.0, cat)
    cat = xr.where(valid_for_cat & (diff >= -th) & (diff <= th), 0.0, cat)
    cat = xr.where(valid_for_cat & (diff > th), 1.0, cat)
    cat = cat.rename("diff_cat")
    cat.attrs.update(long_name          = "FIP difference category",
                     units              = "1",
                     flag_values        = "0 1 2",
                     flag_meanings      = "agreement simulation_dominant observation_dominant",
                     category_threshold = th,
                     comment            = ("Edges at -threshold and +threshold are included in agreement. "
                                           "Values exactly <= -1 or >= 1 remain NaN, matching the AFIM notebook."))
    ds = xr.Dataset({name_mod: mod, name_obs: obs, "diff": diff, "diff_cat": cat})
    for coord in ("lon", "lat", "x", "y"):
        if coord in mod.coords and coord not in ds.coords:
            ds = ds.assign_coords({coord: mod.coords[coord]})
        elif coord in obs.coords and coord not in ds.coords:
            ds = ds.assign_coords({coord: obs.coords[coord]})
    ds.attrs.update(description           = "Fast ice persistence comparison on a shared grid",
                    difference_convention = "diff = mod - obs",
                    diff_is_continuous    = "true",
                    category_codes        = "0 agreement; 1 simulation_dominant; 2 observation_dominant")
    return ds

def fip_weight(FIP: xr.Dataset, *, mode: str = "max", t: float = 0.10, gamma: float = 1.2) -> xr.DataArray:
    """Compute an opacity/weight field for FIP-difference plotting."""
    if "obs" not in FIP:
        raise KeyError("FIP['obs'] is required")
    obs = FIP["obs"]
    mod = FIP["mod"] if "mod" in FIP else None
    if mode == "max" and mod is not None:
        cov = xr.apply_ufunc(np.maximum, obs, mod)
    elif mode == "mean" and mod is not None:
        cov = 0.5 * (obs + mod)
    elif mode == "prod" and mod is not None:
        cov = xr.apply_ufunc(np.sqrt, obs * mod)
    else:
        cov = obs
    w = ((cov - float(t)) / (1.0 - float(t))).clip(0.0, 1.0)
    if gamma != 1.0:
        w = w ** float(gamma)
    w.name = "diff_weight"
    w.attrs.update(long_name="FIP plotting opacity/weight", units="1")
    return w

def _region_mask(lon, lat, region: Sequence[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = [float(v) for v in region]
    lon_m = xr.apply_ufunc(normalise_longitudes, lon, kwargs={"to": "-180-180"}, dask="allowed")
    if lon_min <= lon_max:
        mlon = (lon_m >= lon_min) & (lon_m <= lon_max)
    else:
        mlon = (lon_m >= lon_min) | (lon_m <= lon_max)
    return mlon & (lat >= lat_min) & (lat <= lat_max)


def compute_fipdiff_stats_weighted(FIP: xr.Dataset, *,
                                   pixel_size_m         : float = 5_000.0,
                                   regions              : dict | None = None,
                                   gi_mask              : xr.DataArray | None = None,
                                   clip01               : bool = True,
                                   threecat_percent_only: bool = True) -> pd.DataFrame:
    """Compute AFIM-style regional agreement/model-dominant/obs-dominant FIP statistics."""
    if regions is None:
        from shuga.core.regions import ANTARCTIC_8_REGIONS
        regions = ANTARCTIC_8_REGIONS
    for k in ("mod", "obs", "lon", "lat"):
        if k not in FIP and k not in FIP.coords:
            raise ValueError("FIP must contain 'mod', 'obs', 'lon', and 'lat'.")
    mod = FIP["mod"].astype("float64")
    obs = FIP["obs"].astype("float64")
    if clip01:
        mod = mod.clip(0.0, 1.0)
        obs = obs.clip(0.0, 1.0)
    overlap = xr.apply_ufunc(np.minimum, mod, obs)
    model_excess = xr.apply_ufunc(np.maximum, mod - obs, 0.0)
    obs_excess = xr.apply_ufunc(np.maximum, obs - mod, 0.0)
    lon = FIP["lon"] if "lon" in FIP else FIP.coords["lon"]
    lat = FIP["lat"] if "lat" in FIP else FIP.coords["lat"]
    cell_area_km2 = (float(pixel_size_m) * float(pixel_size_m)) / 1e6
    if gi_mask is not None:
        valid = ~gi_mask.astype(bool).broadcast_like(mod)
    else:
        valid = xr.ones_like(mod, dtype=bool)
    rows = []
    for rname, rdict in regions.items():
        geo = rdict.get("plot_region", rdict.get("geo_region"))
        R = _region_mask(lon, lat, geo) & valid
        A_km2 = float((overlap * R).sum(skipna=True)) * cell_area_km2
        M_km2 = float((model_excess * R).sum(skipna=True)) * cell_area_km2
        O_km2 = float((obs_excess * R).sum(skipna=True)) * cell_area_km2
        tot = A_km2 + M_km2 + O_km2
        MOD_km2 = float((mod * R).sum(skipna=True)) * cell_area_km2
        OBS_km2 = float((obs * R).sum(skipna=True)) * cell_area_km2
        rows.append(dict(region                    = rname,
                         pct_agreement             = 100.0 * A_km2 / tot if tot > 0 else np.nan,
                         pct_model_dominant        = 100.0 * M_km2 / tot if tot > 0 else np.nan,
                         pct_observation_dominant  = 100.0 * O_km2 / tot if tot > 0 else np.nan,
                         model_FIA_km2             = MOD_km2,
                         obs_FIA_km2               = OBS_km2,
                         agreement_overlap_km2     = A_km2,
                         model_excess_km2          = M_km2,
                         obs_excess_km2            = O_km2,
                         agreement_pct_of_model    = 100.0 * A_km2 / MOD_km2 if MOD_km2 > 0 else np.nan,
                         model_excess_pct_of_model = 100.0 * M_km2 / MOD_km2 if MOD_km2 > 0 else np.nan,
                         agreement_pct_of_obs      = 100.0 * A_km2 / OBS_km2 if OBS_km2 > 0 else np.nan,
                         obs_excess_pct_of_obs     = 100.0 * O_km2 / OBS_km2 if OBS_km2 > 0 else np.nan))
    df = pd.DataFrame(rows).set_index("region")
    A_sum = df["agreement_overlap_km2"].sum()
    M_sum = df["model_excess_km2"].sum()
    O_sum = df["obs_excess_km2"].sum()
    tot_sum = A_sum + M_sum + O_sum
    ant_row = dict(pct_agreement             = 100.0 * A_sum / tot_sum if tot_sum > 0 else np.nan,
                   pct_model_dominant        = 100.0 * M_sum / tot_sum if tot_sum > 0 else np.nan,
                   pct_observation_dominant  = 100.0 * O_sum / tot_sum if tot_sum > 0 else np.nan,
                   model_FIA_km2             = df["model_FIA_km2"].sum(),
                   obs_FIA_km2               = df["obs_FIA_km2"].sum(),
                   agreement_overlap_km2     = A_sum,
                   model_excess_km2          = M_sum,
                   obs_excess_km2            = O_sum,
                   agreement_pct_of_model    = np.nan,
                   model_excess_pct_of_model = np.nan,
                   agreement_pct_of_obs      = np.nan,
                   obs_excess_pct_of_obs     = np.nan)
    df_all = pd.concat([df, pd.DataFrame(ant_row, index=["ANT"])])
    if threecat_percent_only:
        return df_all[["pct_agreement", "pct_model_dominant", "pct_observation_dominant"]]
    return df_all

def regrid_bool_to_common(mask: xr.DataArray, lat2d, lon2d, like: xr.Dataset | xr.DataArray, *,
                          radius    : float = 10_000.0,
                          pixel_size: float = 5_000.0,
                          name      : str = "mask") -> xr.DataArray:
    """Nearest-neighbour regrid a 2-D boolean/categorical mask to an existing common grid."""
    from pyresample.geometry import AreaDefinition
    if "x" not in like.coords or "y" not in like.coords:
        raise ValueError("like must have x/y coordinates from the common EPSG:3031 grid.")
    x = like["x"].values
    y = like["y"].values
    xmin = float(x.min() - 0.5 * pixel_size)
    xmax = float(x.max() + 0.5 * pixel_size)
    ymin = float(y.min() - 0.5 * pixel_size)
    ymax = float(y.max() + 0.5 * pixel_size)
    area_def = AreaDefinition(area_id     = "common_like",
                              description = "EPSG:3031 grid reconstructed from x/y coordinates",
                              proj_id     = "epsg3031",
                              projection  = "EPSG:3031",
                              width       = len(x),
                              height      = len(y),
                              area_extent = (xmin, ymin, xmax, ymax))
    out = resample_swath_to_area(mask.astype("float32"), lat2d, lon2d, area_def, radius=radius, fill_value=0.0, pixel_size=pixel_size, name=name)
    return (out >= 0.5).rename(name)
