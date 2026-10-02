from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Sequence
import numpy as np
import xarray as xr

try:
    import dask.array as da
except Exception:  # pragma: no cover
    da = None

GRID_TOKENS = ("Ta", "Tb", "Tx", "Tc")

def parse_grid_selection(grid_type: str | Iterable[str] | None) -> tuple[str, ...]:
    if grid_type is None:
        return ("Tc",)
    if isinstance(grid_type, str):
        raw = grid_type.replace(",", " ").split()
    else:
        raw = [str(v) for v in grid_type]
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        token = token.strip()
        if not token:
            continue
        token = token[0].upper() + token[1:]
        if token not in GRID_TOKENS:
            raise ValueError(f"Unsupported grid_type token {token!r}. Expected one of {GRID_TOKENS}.")
        if token not in seen:
            out.append(token)
            seen.add(token)
    if not out:
        return ("Tc",)
    if "Tc" in seen and len(seen) > 1:
        raise ValueError("grid_type token 'Tc' is exclusive and cannot be combined with Ta/Tb/Tx.")
    return tuple(out)

def _uses_dask(arr) -> bool:
    return da is not None and isinstance(arr, da.Array)

def _nan_to_zero(arr):
    if _uses_dask(arr):
        return da.where(da.isfinite(arr), arr, 0.0)
    return np.nan_to_num(arr, nan=0.0)

def _pad(arr, pad_width):
    if _uses_dask(arr):
        return da.pad(arr, pad_width, constant_values=np.nan)
    return np.pad(arr, pad_width, constant_values=np.nan)

def _roll(arr, shift: int, axis: int):
    if _uses_dask(arr):
        return da.roll(arr, shift=shift, axis=axis)
    return np.roll(arr, shift=shift, axis=axis)

def _coord_or_range(template: xr.DataArray, dim: str) -> xr.DataArray | np.ndarray:
    if dim in template.coords:
        return template.coords[dim]
    return np.arange(template.sizes[dim], dtype="int32")

def _wrap_last_equals_first(data):
    if data.shape[-1] <= 1:
        return data
    if _uses_dask(data):
        return da.concatenate([data[..., :-1], data[..., 0:1]], axis=-1)
    out = np.array(data, copy=True)
    out[..., -1] = out[..., 0]
    return out

def _dataarray_from_spatial_data(data, source: xr.DataArray, target: xr.DataArray, name: str | None = None) -> xr.DataArray:
    ydim, xdim = target.dims[-2], target.dims[-1]
    dims = (*source.dims[:-2], ydim, xdim)
    coords = {}
    for dim in source.dims[:-2]:
        if dim in source.coords:
            coords[dim] = source.coords[dim]
    coords[ydim] = _coord_or_range(target, ydim)
    coords[xdim] = _coord_or_range(target, xdim)
    return xr.DataArray(data, dims=dims, coords=coords, name=name)

def bavg_to_t(component: xr.DataArray, target: xr.DataArray, *, nan_to_zero: bool, wrap_x: bool = True, name: str | None = None) -> xr.DataArray:
    arr = component.data
    if nan_to_zero:
        arr = _nan_to_zero(arr)
    y_len, x_len = target.sizes[target.dims[-2]], target.sizes[target.dims[-1]]
    out          = 0.25 * (arr[..., :-1, :-1] +
                           arr[..., :-1, 1: ] +
                           arr[..., 1: , :-1] +
                           arr[..., 1: , 1: ])
    pad_y        = max(int(y_len - out.shape[-2]), 0)
    pad_x        = max(int(x_len - out.shape[-1]), 0)
    if pad_y or pad_x:
        out = _pad(out, [(0, 0)] * (out.ndim - 2) + [(0, pad_y), (0, pad_x)])
    out = out[..., :y_len, :x_len]
    if wrap_x:
        out = _wrap_last_equals_first(out)
    return _dataarray_from_spatial_data(out.astype(np.float32), component, target, name=name)

def b2t_speed(u_b: xr.DataArray, v_b: xr.DataArray, target: xr.DataArray, *, method: str = "Tb", wrap_x: bool = True) -> xr.DataArray:
    method = method[0].upper() + method[1:]
    if method not in ("Ta", "Tb"):
        raise ValueError("b2t_speed only supports Ta or Tb. Use an explicit regridder for Tx.")
    nan_to_zero = method == "Tb"
    u_t         = bavg_to_t(u_b, target, nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="uT")
    v_t         = bavg_to_t(v_b, target, nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="vT")
    out         = xr.apply_ufunc(np.hypot, u_t, v_t, dask="parallelized", output_dtypes=[np.float32])
    return out.rename(f"ispd_{method}")

def cavg_to_t(component: xr.DataArray, target: xr.DataArray, *, direction: str, nan_to_zero: bool = True, wrap_x: bool = True, name: str | None = None) -> xr.DataArray:
    direction = direction.lower().strip()
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    arr = component.data
    if nan_to_zero:
        arr = _nan_to_zero(arr)
    y_len, x_len = target.sizes[target.dims[-2]], target.sizes[target.dims[-1]]
    if direction == "x":
        nx = int(arr.shape[-1])
        if nx == x_len + 1:
            out = 0.5 * (arr[..., :x_len] + arr[..., 1:x_len + 1])
        else:
            if wrap_x:
                shifted = _roll(arr, -1, axis=-1)
                out     = 0.5 * (arr + shifted)
            else:
                core = 0.5 * (arr[..., :-1] + arr[..., 1:])
                out  = _pad(core, [(0, 0)] * (core.ndim - 1) + [(0, 1)])
    else:
        ny = int(arr.shape[-2])
        if ny == y_len + 1:
            out = 0.5 * (arr[..., :y_len, :] + arr[..., 1:y_len + 1, :])
        else:
            core = 0.5 * (arr[..., :-1, :] + arr[..., 1:, :])
            out  = _pad(core, [(0, 0)] * (core.ndim - 2) + [(0, 1), (0, 0)])
    pad_y = max(int(y_len - out.shape[-2]), 0)
    pad_x = max(int(x_len - out.shape[-1]), 0)
    if pad_y or pad_x:
        out = _pad(out, [(0, 0)] * (out.ndim - 2) + [(0, pad_y), (0, pad_x)])
    out = out[..., :y_len, :x_len]
    if wrap_x:
        out = _wrap_last_equals_first(out)
    return _dataarray_from_spatial_data(out.astype(np.float32), component, target, name=name)

def c2t(uvelE      : xr.DataArray,
        uvelN      : xr.DataArray,
        vvelE      : xr.DataArray,
        vvelN      : xr.DataArray,
        target     : xr.DataArray, *,
        wrap_x     : bool = True,
        combine    : str = "mean",
        nan_to_zero: bool = True) -> tuple[xr.DataArray, xr.DataArray]:
    e_from_u = cavg_to_t(uvelE, target, direction="x", nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="velE_from_U")
    n_from_u = cavg_to_t(uvelN, target, direction="x", nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="velN_from_U")
    e_from_v = cavg_to_t(vvelE, target, direction="y", nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="velE_from_V")
    n_from_v = cavg_to_t(vvelN, target, direction="y", nan_to_zero=nan_to_zero, wrap_x=wrap_x, name="velN_from_V")
    combine = (combine or "mean").lower().strip()
    if combine == "uv":
        velE_T = e_from_u
        velN_T = n_from_v
    elif combine == "mean":
        velE_T = 0.5 * (e_from_u + e_from_v)
        velN_T = 0.5 * (n_from_u + n_from_v)
    else:
        raise ValueError("combine must be 'mean' or 'uv'")
    return velE_T.astype(np.float32), velN_T.astype(np.float32)

def compute_tgrid_speed(ds: xr.Dataset, target: xr.DataArray, *,
                        grid_type    : str | Sequence[str],
                        u_var        : str = "uvel",
                        v_var        : str = "vvel",
                        uvelE_var    : str = "uvelE",
                        uvelN_var    : str = "uvelN",
                        vvelE_var    : str = "vvelE",
                        vvelN_var    : str = "vvelN",
                        wrap_x       : bool = True,
                        cgrid_combine: str = "mean",
                        regridder    : Callable[[xr.DataArray], xr.DataArray] | None = None,
                        logger       = None) -> xr.DataArray:
    selection = parse_grid_selection(grid_type)
    if "Tc" in selection:
        required = (uvelE_var, uvelN_var, vvelE_var, vvelN_var)
        missing = [name for name in required if name not in ds]
        if missing:
            raise KeyError(f"Tc requested but missing required C-grid variables: {missing}")
        if logger is not None:
            logger.info("Computing T-grid speed using Tc reconstruction from C-grid east/north components")
        velE_T, velN_T = c2t(ds[uvelE_var], ds[uvelN_var], ds[vvelE_var], ds[vvelN_var], target,
                             wrap_x      = wrap_x,
                             combine     = cgrid_combine,
                             nan_to_zero = True)
        return xr.apply_ufunc(np.hypot, velE_T, velN_T, dask="parallelized", output_dtypes=[np.float32]).rename("ispd_Tc")
    missing = [name for name in (u_var, v_var) if name not in ds]
    if missing:
        raise KeyError(f"B-grid/T-grid speed requested but missing velocity variables: {missing}")
    members: list[xr.DataArray] = []
    if "Ta" in selection:
        if logger is not None:
            logger.info("Computing T-grid speed using Ta 2x2 mean (NaNs propagate)")
        members.append(b2t_speed(ds[u_var], ds[v_var], target, method="Ta", wrap_x=wrap_x))
    if "Tb" in selection:
        if logger is not None:
            logger.info("Computing T-grid speed using Tb 2x2 mean (NaNs->0 no-slip)")
        members.append(b2t_speed(ds[u_var], ds[v_var], target, method="Tb", wrap_x=wrap_x))
    if "Tx" in selection:
        if regridder is None:
            raise NotImplementedError("Tx requested but no explicit B-grid->T-grid regridder was supplied. "
                                      "Pass regridder=... to CICEClassifier or use Ta/Tb/Tc.")
        if logger is not None:
            logger.info("Computing T-grid speed using explicit Tx regridder")
        u_t = regridder(ds[u_var].fillna(0.0))
        v_t = regridder(ds[v_var].fillna(0.0))
        members.append(xr.apply_ufunc(np.hypot, u_t, v_t, dask="parallelized", output_dtypes=[np.float32]).rename("ispd_Tx"))
    if not members:
        raise ValueError("No supported grid_type tokens resolved for speed reconstruction.")
    if len(members) == 1:
        return members[0].astype(np.float32)
    return xr.concat(members, dim="__grid_method__").mean("__grid_method__", skipna=True).astype(np.float32).rename("ice_speed")
