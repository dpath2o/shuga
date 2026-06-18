#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga.core.regions import ANTARCTIC_8_REGIONS
from shuga.core.types import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec
from shuga.core.paths import ShugaPaths
from shuga.io import load_metrics

CIRCUM_REGION = "circum_antarctic"
REGION_ORDER = (CIRCUM_REGION, *ANTARCTIC_8_REGIONS.keys())
STAT_ORDER = ("mean", "max", "min", "p95")
GROWTH_MONTHS = (4, 5, 6)
RETREAT_MONTHS = (11, 12, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build AF2020/simulation FIA statistics and simulation-minus-AF2020 "
            "difference tables for the circum-Antarctic domain and eight Antarctic sectors."
        )
    )
    p.add_argument("-s", "--sim-name", action="append", dest="sim_name", default=[])
    p.add_argument("--sim-names", default=None, dest="sim_names", help="Optional comma/space separated simulation names.")
    p.add_argument("-b", "--start-date", required=True)
    p.add_argument("-e", "--end-date", required=True)
    p.add_argument("-m", "--classification", default="binary-days")
    p.add_argument("--grid-type", default="Tc")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--ispd-thresh", type=float, default=5.0e-4)
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--af2020-fia-store", default="/g/data/gv90/da1339/SeaIce/FI_obs/AF-FI-2020db_FIA_from_original_dataset.zarr")
    p.add_argument("--table-root", default="/g/data/gv90/da1339/graphical/LD-pub-workspace/tables")
    p.add_argument("--tag", default=None)
    p.add_argument("--min-edge-trim-days", type=int, default=None,
                   help="Days trimmed from both ends before computing minimum FIA. Default: bin_window//2 for binary-days, else 0.")
    p.add_argument("--growth-months", nargs="+", type=int, default=list(GROWTH_MONTHS))
    p.add_argument("--retreat-months", nargs="+", type=int, default=list(RETREAT_MONTHS))
    p.add_argument("--min-rate-points", type=int, default=3)
    p.add_argument("--min-rate-months", type=int, default=2)
    p.add_argument("--chunks-time", type=int, default=366)
    return p.parse_args()



def _region_order_present(da: xr.DataArray) -> xr.DataArray:
    if "region" not in da.dims:
        raise ValueError(f"Expected a region dimension on {da.name!r}; dims={da.dims}")
    labels = [str(v) for v in da["region"].values]
    da = da.assign_coords(region=labels)
    ordered = [r for r in REGION_ORDER if r in labels]
    ordered.extend([r for r in labels if r not in ordered])
    return da.sel(region=ordered)


def _coerce_fia_time_region(ds: xr.Dataset, *, source_name: str, require_regional: bool = True) -> xr.DataArray:
    pieces: list[xr.DataArray] = []
    if "FIA" in ds:
        ca = ds["FIA"].squeeze(drop=True)
        if "region" in ca.dims:
            pieces.append(ca)
        else:
            pieces.append(ca.expand_dims(region=pd.Index([CIRCUM_REGION], name="region")))
    if "FIA_by_region" in ds:
        reg = ds["FIA_by_region"].squeeze(drop=True)
        if "region" not in reg.dims:
            raise ValueError(f"{source_name}: FIA_by_region does not have a region dimension; dims={reg.dims}")
        if "region" not in reg.coords or len(reg["region"].values) == len(ANTARCTIC_8_REGIONS):
            labels = [str(v) for v in reg["region"].values] if "region" in reg.coords else []
            if not labels or all(s.isdigit() for s in labels):
                reg = reg.assign_coords(region=list(ANTARCTIC_8_REGIONS.keys()))
        pieces.append(reg)
    if not pieces:
        raise KeyError(f"{source_name}: no FIA or FIA_by_region variable found.")
    if require_regional and not any("region" in p.dims and set(str(v) for v in p["region"].values) & set(ANTARCTIC_8_REGIONS) for p in pieces):
        raise KeyError(f"{source_name}: regional FIA_by_region is required for the requested table.")
    da = xr.concat(pieces, dim="region") if len(pieces) > 1 else pieces[0]
    da = da.rename("FIA")
    if "time" not in da.dims:
        raise ValueError(f"{source_name}: FIA must have a time dimension; dims={da.dims}")
    da = _region_order_present(da).transpose("time", "region")
    da.attrs.setdefault("units", "10^3 km^2")
    return da


def _open_af2020_fia(path: Path, start_date: str, end_date: str, chunks_time: int) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=False, chunks={"time": int(chunks_time)})
    if "FIA" not in ds:
        raise KeyError(f"AF2020 FIA store does not contain FIA: {path}")
    da = ds["FIA"].sel(time=slice(start_date, end_date))
    return _region_order_present(da).transpose("time", "region")


def _open_sim_fia(sim_name: str, args: argparse.Namespace) -> xr.DataArray:
    run_cfg = RunSpec(
        sim_name=sim_name,
        start_date=args.start_date,
        end_date=args.end_date,
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )
    cls_cfg = ClassificationSpec(
        ice_type=args.ice_type,
        grid_type=args.grid_type,
        ispd_thresh=args.ispd_thresh,
        methods=(args.classification,),
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
    )
    met_cfg = MetricsSpec(methods=(args.classification,))
    plt_cfg = PlottingSpec()
    obs_cfg = ObservationSpec()
    pth_cfg = ShugaPaths(run_cfg=run_cfg, cls_cfg=cls_cfg, met_cfg=met_cfg, plt_cfg=plt_cfg, obs_cfg=obs_cfg)
    ds = load_metrics(
        run_cfg=run_cfg,
        cls_cfg=cls_cfg,
        met_cfg=met_cfg,
        plt_cfg=plt_cfg,
        obs_cfg=obs_cfg,
        pth_cfg=pth_cfg,
        classification=args.classification,
        variables=["FIA", "FIA_by_region"],
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
        grid_type=args.grid_type,
        ice_type=args.ice_type,
        ispd_thresh=args.ispd_thresh,
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
        chunks={"time": int(args.chunks_time)},
    )
    da = _coerce_fia_time_region(ds, source_name=sim_name, require_regional=True)
    return da.sel(time=slice(args.start_date, args.end_date))


def _trim_time_edges(da: xr.DataArray, days: int) -> xr.DataArray:
    if days <= 0 or da.sizes.get("time", 0) == 0:
        return da
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    t0 = times.min() + pd.Timedelta(days=int(days))
    t1 = times.max() - pd.Timedelta(days=int(days))
    if t1 < t0:
        return da.isel(time=slice(0, 0))
    return da.sel(time=slice(t0, t1))


def _safe_float(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def summarise_fia(da: xr.DataArray, *, source: str, sim_name: str, start_date: str, end_date: str,
                  min_edge_trim_days: int) -> pd.DataFrame:
    rows: list[dict] = []
    da = _region_order_present(da).sel(time=slice(start_date, end_date)).load()
    for region in [str(v) for v in da["region"].values]:
        series = da.sel(region=region).dropna("time", how="all")
        min_series = _trim_time_edges(series, int(min_edge_trim_days)).dropna("time", how="all")
        n_time = int(series.count().values) if series.sizes.get("time", 0) else 0
        n_time_min = int(min_series.count().values) if min_series.sizes.get("time", 0) else 0
        values = {
            "mean": _safe_float(series.mean(skipna=True).values) if n_time else np.nan,
            "max": _safe_float(series.max(skipna=True).values) if n_time else np.nan,
            "min": _safe_float(min_series.min(skipna=True).values) if n_time_min else np.nan,
            "p95": _safe_float(series.quantile(0.95, skipna=True).values) if n_time else np.nan,
        }
        for stat in STAT_ORDER:
            rows.append({
                "source": source,
                "sim_name": sim_name,
                "region": region,
                "statistic": stat,
                "value": values[stat],
                "units": da.attrs.get("units", "10^3 km^2"),
                "start_date": start_date,
                "end_date": end_date,
                "n_time": n_time,
                "n_time_for_min": n_time_min,
                "min_edge_trim_days": int(min_edge_trim_days),
            })
    return pd.DataFrame(rows)


def _season_year(times: pd.DatetimeIndex, *, season: str) -> np.ndarray:
    if season == "growth":
        return times.year.to_numpy()
    if season == "retreat":
        return np.where(times.month == 1, times.year - 1, times.year)
    raise ValueError(season)


def _linear_slope_per_day(times: Iterable[pd.Timestamp], values: Iterable[float]) -> float:
    t = pd.DatetimeIndex(pd.to_datetime(list(times)))
    y = np.asarray(list(values), dtype="float64")
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x = (t[ok] - t[ok][0]).days.astype("float64")
    if np.unique(x).size < 2:
        return float("nan")
    return float(np.polyfit(x, y[ok], 1)[0])


def seasonal_rate_statistics(da: xr.DataArray, *, source: str, sim_name: str, start_date: str, end_date: str,
                             growth_months: Iterable[int], retreat_months: Iterable[int],
                             min_rate_points: int, min_rate_months: int) -> pd.DataFrame:
    rows: list[dict] = []
    da = _region_order_present(da).sel(time=slice(start_date, end_date)).load()
    season_months = {"growth": tuple(int(m) for m in growth_months),
                     "retreat": tuple(int(m) for m in retreat_months)}
    for region in [str(v) for v in da["region"].values]:
        series = da.sel(region=region).dropna("time", how="all")
        times_all = pd.DatetimeIndex(pd.to_datetime(series["time"].values))
        vals_all = np.asarray(series.values, dtype="float64")
        for season, months in season_months.items():
            keep = np.isin(times_all.month, months) & np.isfinite(vals_all)
            if keep.sum() < int(min_rate_points):
                slopes = np.array([], dtype="float64")
                season_ids = np.array([], dtype="int64")
            else:
                times = times_all[keep]
                vals = vals_all[keep]
                season_ids_all = _season_year(times, season=season)
                slopes_list: list[float] = []
                season_list: list[int] = []
                for sid in np.unique(season_ids_all):
                    idx = season_ids_all == sid
                    if idx.sum() < int(min_rate_points):
                        continue
                    if len(set(times[idx].month)) < int(min_rate_months):
                        continue
                    slope = _linear_slope_per_day(times[idx], vals[idx])
                    if np.isfinite(slope):
                        slopes_list.append(slope)
                        season_list.append(int(sid))
                slopes = np.asarray(slopes_list, dtype="float64")
                season_ids = np.asarray(season_list, dtype="int64")
            values = {
                "mean": float(np.nanmean(slopes)) if slopes.size else np.nan,
                "max": float(np.nanmax(slopes)) if slopes.size else np.nan,
                "min": float(np.nanmin(slopes)) if slopes.size else np.nan,
                "p95": float(np.nanpercentile(slopes, 95)) if slopes.size else np.nan,
            }
            for stat in STAT_ORDER:
                rows.append({
                    "source": source,
                    "sim_name": sim_name,
                    "region": region,
                    "season": season,
                    "statistic": stat,
                    "value": values[stat],
                    "units": "10^3 km^2 day^-1",
                    "start_date": start_date,
                    "end_date": end_date,
                    "months": ",".join(str(m) for m in months),
                    "n_seasons": int(slopes.size),
                    "first_season_year": int(season_ids.min()) if season_ids.size else np.nan,
                    "last_season_year": int(season_ids.max()) if season_ids.size else np.nan,
                    "min_rate_points": int(min_rate_points),
                    "min_rate_months": int(min_rate_months),
                })
    return pd.DataFrame(rows)


def difference_table(df: pd.DataFrame, *, join_cols: list[str]) -> pd.DataFrame:
    obs = df[df["sim_name"] == "AF2020"].copy()
    sim = df[df["sim_name"] != "AF2020"].copy()
    keep_obs = join_cols + ["value"]
    merged = sim.merge(obs[keep_obs], on=join_cols, how="left", suffixes=("_sim", "_af2020"))
    merged = merged.rename(columns={"value_sim": "sim_value", "value_af2020": "af2020_value"})
    merged["diff"] = merged["sim_value"] - merged["af2020_value"]
    merged["diff_pct_af2020"] = np.where(
        np.isfinite(merged["af2020_value"]) & (merged["af2020_value"] != 0.0),
        100.0 * merged["diff"] / np.abs(merged["af2020_value"]),
        np.nan,
    )
    cols_front = ["sim_name", *join_cols, "sim_value", "af2020_value", "diff", "diff_pct_af2020"]
    rest = [c for c in merged.columns if c not in cols_front and c not in {"source"}]
    return merged[cols_front + rest]


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main() -> None:
    args = parse_args()
    sim_names = list(args.sim_name or [])
    if args.sim_names:
        sim_names.extend(re.split(r"[\s,]+", args.sim_names.strip()))
    sim_names = [n for n in dict.fromkeys(str(n).strip() for n in sim_names) if n]
    if not sim_names:
        raise ValueError("No simulations supplied. Use --sim-name SIM one or more times, or --sim-names 'SIM1 SIM2'.")

    if args.min_edge_trim_days is None:
        min_edge_trim_days = int(args.bin_window // 2) if args.classification == "binary-days" else 0
    else:
        min_edge_trim_days = int(args.min_edge_trim_days)

    af_path = Path(args.af2020_fia_store).expanduser()
    table_root = Path(args.table_root).expanduser()
    table_root.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"{args.classification}_{args.grid_type}_{args.start_date}_{args.end_date}"
    tag = _safe_tag(tag)

    af_fia = _open_af2020_fia(af_path, args.start_date, args.end_date, args.chunks_time)
    stats_frames = [summarise_fia(
        af_fia,
        source="observation",
        sim_name="AF2020",
        start_date=args.start_date,
        end_date=args.end_date,
        min_edge_trim_days=0,
    )]
    rate_frames = [seasonal_rate_statistics(
        af_fia,
        source="observation",
        sim_name="AF2020",
        start_date=args.start_date,
        end_date=args.end_date,
        growth_months=args.growth_months,
        retreat_months=args.retreat_months,
        min_rate_points=args.min_rate_points,
        min_rate_months=args.min_rate_months,
    )]

    for sim_name in sim_names:
        print(f"[load] {sim_name}")
        sim_fia = _open_sim_fia(sim_name, args)
        stats_frames.append(summarise_fia(
            sim_fia,
            source="simulation",
            sim_name=sim_name,
            start_date=args.start_date,
            end_date=args.end_date,
            min_edge_trim_days=min_edge_trim_days,
        ))
        rate_frames.append(seasonal_rate_statistics(
            sim_fia,
            source="simulation",
            sim_name=sim_name,
            start_date=args.start_date,
            end_date=args.end_date,
            growth_months=args.growth_months,
            retreat_months=args.retreat_months,
            min_rate_points=args.min_rate_points,
            min_rate_months=args.min_rate_months,
        ))

    stats = pd.concat(stats_frames, ignore_index=True)
    rates = pd.concat(rate_frames, ignore_index=True)
    stats_diff = difference_table(stats, join_cols=["region", "statistic"])
    rates_diff = difference_table(rates, join_cols=["region", "season", "statistic"])

    outputs = {
        "stats": table_root / f"FIA_statistics_{tag}.csv",
        "stats_diff": table_root / f"FIA_statistics_sim_minus_AF2020_{tag}.csv",
        "rates": table_root / f"FIA_growth_retreat_rates_{tag}.csv",
        "rates_diff": table_root / f"FIA_growth_retreat_rates_sim_minus_AF2020_{tag}.csv",
    }
    stats.to_csv(outputs["stats"], index=False)
    stats_diff.to_csv(outputs["stats_diff"], index=False)
    rates.to_csv(outputs["rates"], index=False)
    rates_diff.to_csv(outputs["rates_diff"], index=False)

    for label, path in outputs.items():
        print(f"[done] {label}: {path}")


if __name__ == "__main__":
    main()
