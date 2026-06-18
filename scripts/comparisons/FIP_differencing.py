#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.core.types import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec
from shuga.core.paths import ShugaPaths
from shuga.io import load_classified, load_metrics
from shuga.plotting.cice import CICEPlotter, compute_fip
from shuga.regridding.pyresample import (add_lonlat_from_epsg3031,
                                        area_definition_from_xy,
                                        compute_fipdiff_stats_weighted,
                                        fip_difference_dataset,
                                        fip_weight,
                                        resample_swath_to_area)
AF2020_MIN = pd.Timestamp("2000-03-01")
AF2020_MAX = pd.Timestamp("2018-02-15")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and plot continuous/categorical FIP differences: simulation - AF2020.")
    p.add_argument("-s", "--sim-name", required=True)
    p.add_argument("-b", "--start-date", required=True, help="Simulation run/context start date.")
    p.add_argument("-e", "--end-date", required=True, help="Simulation run/context end date.")
    p.add_argument("--fip-start", default=None, help="FIP comparison start. Defaults to max(start-date, AF2020_MIN).")
    p.add_argument("--fip-end", default=None, help="FIP comparison end. Defaults to min(end-date, AF2020_MAX).")
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
    p.add_argument("--af2020-store", default="/g/data/gv90/da1339/SeaIce/FI_obs/AF-FI-2020db_common-5km_pyresample.zarr")
    p.add_argument("--out-store", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--plot-root", default=None)
    p.add_argument("--stats-csv", default=None)
    p.add_argument("--pixel-size-m", type=float, default=5000.0)
    p.add_argument("--radius-of-influence-m", type=float, default=10000.0)
    p.add_argument("--cice-lon-shift-deg", type=float, default=0.25)
    p.add_argument("--sim-fip-source", choices=("auto", "metrics", "classification"), default="auto")
    p.add_argument("--category-threshold", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot", action="store_true", help="Plot FIP['diff'] for all 8 regions after writing numerics.")
    p.add_argument("--plot-categorical", action="store_true", help="Plot FIP['diff_cat'] instead of FIP['diff'].")
    p.add_argument("--chunks-time", type=int, default=31)
    return p.parse_args()

def _comparison_window(args: argparse.Namespace) -> tuple[str, str]:
    req0 = pd.Timestamp(args.fip_start or args.start_date)
    req1 = pd.Timestamp(args.fip_end or args.end_date)
    use0 = max(req0, AF2020_MIN)
    use1 = min(req1, AF2020_MAX)
    if use1 < use0:
        raise ValueError(f"No overlap with AF2020 period: requested {req0.date()}..{req1.date()}")
    return use0.strftime("%Y-%m-%d"), use1.strftime("%Y-%m-%d")

def _open_static_grid(sim_name: str) -> xr.Dataset:
    path = Path.home() / "AFIM_archive" / "CICE_0p25_Cgrid_coords.zarr"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_zarr(path, consolidated=False)
    return ds[["TLON", "TLAT"]]

def _metric_fip_is_compatible(da: xr.DataArray, fip_start: str, fip_end: str) -> bool:
    attrs = da.attrs
    t0    = attrs.get("time_start") or attrs.get("start_date") or attrs.get("dt0_str")
    t1    = attrs.get("time_end") or attrs.get("end_date") or attrs.get("dtN_str")
    if t0 is None or t1 is None:
        return False
    return pd.Timestamp(t0).date() == pd.Timestamp(fip_start).date() and pd.Timestamp(t1).date() == pd.Timestamp(fip_end).date()

def _load_or_compute_sim_fip(args        : argparse.Namespace,
                             run_cfg     : RunSpec,
                             cls_cfg     : ClassificationSpec,
                             met_cfg     : MetricsSpec,
                             plt_cfg     : PlottingSpec,
                             obs_cfg     : ObservationSpec,
                             pth_cfg     : ShugaPaths,
                             fip_start   : str,
                             fip_end     : str) -> xr.DataArray:
    if args.sim_fip_source in ("auto", "metrics"):
        try:
            ds_met = load_metrics(run_cfg        = run_cfg,
                                  cls_cfg        = cls_cfg,
                                  met_cfg        = met_cfg,
                                  plt_cfg        = plt_cfg,
                                  obs_cfg        = obs_cfg,
                                  pth_cfg        = pth_cfg,
                                  classification = args.classification,
                                  variables      = ["FIP"],
                                  hemisphere     = args.hemisphere,
                                  grid_type      = args.grid_type,
                                  chunks         = {"time": args.chunks_time})
            if "FIP" in ds_met:
                da = ds_met["FIP"].squeeze(drop=True)
                if args.sim_fip_source == "metrics" or _metric_fip_is_compatible(da, fip_start, fip_end):
                    da = da.rename("SIM_FIP")
                    da.attrs.update(source="shuga metrics store")
                    return da
                print("[info] metrics FIP date attrs do not match requested AF2020 overlap; recomputing from classification.")
        except Exception as exc:
            if args.sim_fip_source == "metrics":
                raise
            print(f"[info] could not use metrics FIP; recomputing from classification: {exc}")
    cls = load_classified(run_cfg        = run_cfg,
                          cls_cfg        = cls_cfg,
                          met_cfg        = met_cfg,
                          plt_cfg        = plt_cfg,
                          obs_cfg        = obs_cfg,
                          pth_cfg        = pth_cfg,
                          classification = args.classification,
                          variables      = [f"{args.ice_type.upper()}_mask"],
                          hemisphere     = args.hemisphere,
                          grid_type      = args.grid_type,
                          chunks         = {"time": args.chunks_time})
    mask_name = f"{args.ice_type.upper()}_mask"
    if mask_name not in cls:
        # fallback for older stores
        if "FI_mask" in cls:
            mask_name = "FI_mask"
        else:
            raise KeyError(f"No {args.ice_type.upper()}_mask/FI_mask found in classification store.")
    fi = cls[mask_name].sel(time = slice(fip_start, fip_end))
    if fi.sizes.get("time", 0) == 0:
        raise ValueError(f"No simulation FI_mask data in {fip_start}..{fip_end}.")
    out = compute_fip(fi, name = "SIM_FIP")
    out.attrs.update(source = "computed from classification", time_start = fip_start, time_end = fip_end)
    return out

def main() -> None:
    args               = parse_args()
    fip_start, fip_end = _comparison_window(args)
    run_cfg            = RunSpec(sim_name   = args.sim_name,
                                 start_date = args.start_date,
                                 end_date   = args.end_date,
                                 hemisphere = args.hemisphere,
                                 project    = args.project,
                                 user       = args.user)
    cls_cfg            = ClassificationSpec(ice_type     = args.ice_type,
                                            grid_type    = args.grid_type,
                                            ispd_thresh  = args.ispd_thresh,
                                            methods      = (args.classification,),
                                            bin_window   = args.bin_window,
                                            bin_min_days = args.bin_min_days,
                                            roll_window  = args.roll_window)
    met_cfg            = MetricsSpec(methods = (args.classification,))
    plt_cfg            = PlottingSpec()
    obs_cfg            = ObservationSpec()
    pth_cfg            = ShugaPaths(run_cfg = run_cfg, cls_cfg = cls_cfg, met_cfg = met_cfg, plt_cfg = plt_cfg, obs_cfg = obs_cfg)
    af_store           = Path(args.af2020_store).expanduser()
    af                 = xr.open_zarr(af_store, consolidated = False, chunks = {"time": args.chunks_time})
    if "FIC" not in af:
        raise KeyError(f"AF2020 store must contain FIC to compute period-specific FIP: {af_store}")
    obs_fip = af["FIC"].sel(time=slice(fip_start, fip_end)).mean("time", skipna=True).astype("float32").rename("obs")
    obs_fip.attrs.update(source="AF2020 common-grid FIC", time_start=fip_start, time_end=fip_end)
    sim_fip_native = _load_or_compute_sim_fip(args, run_cfg, cls_cfg, met_cfg, plt_cfg, obs_cfg, pth_cfg, fip_start, fip_end)
    cice           = _open_static_grid(args.sim_name)
    area_def       = area_definition_from_xy(af["x"], af["y"], pixel_size = float(args.pixel_size_m), area_id = "AF2020_common_for_FIP_diff")
    mod_fip        = resample_swath_to_area(sim_fip_native, cice["TLAT"].values, cice["TLON"].values + float(args.cice_lon_shift_deg), area_def,
                                            radius     = float(args.radius_of_influence_m),
                                            fill_value = np.nan,
                                            pixel_size = float(args.pixel_size_m),
                                            name       = "mod")
    FIP            = fip_difference_dataset(mod_fip, obs_fip, category_threshold = float(args.category_threshold), name_mod = "mod", name_obs = "obs")
    FIP            = FIP.assign_coords(lon = af["lon"], lat = af["lat"])
    FIP.attrs.update(sim_name              = args.sim_name,
                     classification        = args.classification,
                     grid_type             = args.grid_type,
                     fip_start             = fip_start,
                     fip_end               = fip_end,
                     af2020_store          = str(af_store),
                     pixel_size_m          = float(args.pixel_size_m),
                     radius_of_influence_m = float(args.radius_of_influence_m),
                     cice_lon_shift_deg    = float(args.cice_lon_shift_deg))
    if args.out_store is not None:
        out_store = Path(args.out_store).expanduser()
    else:
        out_root = Path(args.out_root).expanduser() if args.out_root else Path.home() / "AFIM_archive" / args.sim_name / "zarr" / "comparisons"
        out_store = out_root / f"FIPdiff_{args.sim_name}_minus_AF2020_{args.classification}_{args.grid_type}_{fip_start}_{fip_end}.zarr"
    if out_store.exists() and args.overwrite:
        shutil.rmtree(out_store)
    if out_store.exists() and not args.overwrite:
        raise FileExistsError(f"{out_store} exists. Pass --overwrite to replace it.")
    out_store.parent.mkdir(parents = True, exist_ok = True)
    FIP.to_zarr(out_store, mode="w", consolidated = False)
    print(f"[done] wrote FIP difference store: {out_store}")
    stats     = compute_fipdiff_stats_weighted(FIP, pixel_size_m = float(args.pixel_size_m), threecat_percent_only = False)
    stats_csv = Path(args.stats_csv).expanduser() if args.stats_csv else out_store.with_suffix(".regional_stats.csv")
    stats.to_csv(stats_csv)
    print(f"[done] wrote stats: {stats_csv}")
    if args.plot:
        plotter   = CICEPlotter(run_cfg = run_cfg, cls_cfg = cls_cfg, met_cfg = met_cfg, plt_cfg = plt_cfg, obs_cfg =obs_cfg, pth_cfg = pth_cfg)
        plot_root = Path(args.plot_root).expanduser() if args.plot_root else out_store.parent / "figures"
        field     = "diff_cat" if args.plot_categorical else "diff"
        plotter.plot_fip(method      = args.classification,
                         source      = "dataset",
                         field       = field,
                         dataset     = FIP,
                         output_root = plot_root,
                         title       = f"{args.sim_name} - AF2020 {fip_start} to {fip_end}")
        print(f"[done] wrote regional plots under: {plot_root}")

if __name__ == "__main__":
    main()
