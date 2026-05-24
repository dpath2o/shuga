from __future__ import annotations
from collections.abc import Callable
import xarray as xr
from shuga.core.naming import normalize_method

def compute_seasonal_summary_dataset(
    *,
    requested: set[str],
    dispatcher,
    output: xr.Dataset,
    seasonal_requests: dict[str, set[str]],
    compute_seasonal_summary: Callable[[xr.DataArray, str], dict[str, xr.DataArray]],
) -> xr.Dataset:
    """
    Compute requested seasonal scalar summaries from primary 1-D metrics.

    dispatcher must expose get(name).
    """
    out = xr.Dataset()
    for base, names in seasonal_requests.items():
        names = set(names)
        if not (requested & names):
            continue

        if base in output:
            base_da = output[base]
        else:
            base_da = dispatcher.get(base)

        if base_da is None:
            continue

        seasonal = compute_seasonal_summary(base_da, base)

        for name, da in seasonal.items():
            if name in requested:
                out[name] = da

    return out


def compute_fipsi_dataset(
    *,
    requested: set[str],
    fipsi_names: set[str],
    fi_mask: xr.DataArray | None,
    area: xr.DataArray,
    persistence_stability_index: Callable[[xr.DataArray, xr.DataArray], dict[str, xr.DataArray]],
) -> xr.Dataset:
    """
    Compute requested fast-ice persistence-stability diagnostics.
    """
    out = xr.Dataset()

    if not (requested & fipsi_names):
        return out

    if fi_mask is None:
        return out

    fipsi = persistence_stability_index(fi_mask, area)

    for name, da in fipsi.items():
        if name in requested:
            out[name] = da

    return out


def compute_obs_skill_dataset(
    *,
    requested: set[str],
    dispatcher,
    output: xr.Dataset,
    fia_skill_names: set[str],
    fit_skill_names: set[str],
    obs_skill_dataset: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """
    Compute requested observation skill metrics.

    dispatcher must expose get(name). output is used to avoid recomputing
    primary metrics already published.
    """
    out = xr.Dataset()

    if not (requested & (fia_skill_names | fit_skill_names)):
        return out

    base_ds = xr.Dataset()

    if requested & fia_skill_names:
        if "FIA" in output:
            base_ds["FIA"] = output["FIA"]
        else:
            fia = dispatcher.get("FIA")
            if fia is not None:
                base_ds["FIA"] = fia

    if requested & fit_skill_names:
        if "FIT" in output:
            base_ds["FIT"] = output["FIT"]
        else:
            fit = dispatcher.get("FIT")
            if fit is not None:
                base_ds["FIT"] = fit

    if not base_ds.data_vars:
        return out

    skill = obs_skill_dataset(base_ds)

    for name, da in skill.items():
        if name in requested:
            out[name] = da

    return out


def common_metrics_attrs(
    *,
    sim_name: str,
    start_date: str,
    end_date: str,
    hemisphere: str,
    ice_type: str,
    grid_type: str,
    method: str,
) -> dict[str, str]:
    """
    Common attrs for method-specific shuga metrics datasets.
    """
    return {
        "sim_name": sim_name,
        "start_date": start_date,
        "end_date": end_date,
        "hemisphere": hemisphere,
        "ice_type": ice_type,
        "grid_type": grid_type,
        "method": normalize_method(method),
    }


def attach_common_metrics_attrs(
    ds: xr.Dataset,
    *,
    sim_name: str,
    start_date: str,
    end_date: str,
    hemisphere: str,
    ice_type: str,
    grid_type: str,
    method: str,
) -> xr.Dataset:
    """
    Attach common metadata attrs to a metrics dataset.
    """
    ds = ds.copy()
    ds.attrs.update(
        common_metrics_attrs(
            sim_name=sim_name,
            start_date=start_date,
            end_date=end_date,
            hemisphere=hemisphere,
            ice_type=ice_type,
            grid_type=grid_type,
            method=method,
        )
    )
    return ds
