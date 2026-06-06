#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.io import load_cice, load_classified
from shuga.observations.AF2020 import AF2020Naming, AF2020Observations
from shuga.regridder.pyresample import (PyresampleSpec,
                                        add_lonlat_from_epsg3031,
                                        area_definition_from_lonlat_pairs,
                                        compute_fipdiff_stats_weighted,
                                        fip_difference_dataset,
                                        fip_weight,
                                        resample_dataarray_to_area)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description = ("Build AF2020-vs-CICE fast-ice comparison products on a shared EPSG:3031 pyresample grid. "
                                               "FIP differencing is always performed on this shared grid. Optional FIC sample dates are also "
                                               "resampled to the shared grid for side-by-side AF2020/model plotting."))
    p.add_argument("-s", "--sim-name", required=True)
    p.add_argument("-b", "--start-date", required=True, help="Overall comparison start date, also FIC default start.")
    p.add_argument("-e", "--end-date", required=True, help="Overall comparison end date, also FIC default end.")
    p.add_argument("--fip-start", default=None, help="FIP accumulation start. Defaults to --start-date.")
    p.add_argument("--fip-end", default=None, help="FIP accumulation end. Defaults to --end-date.")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--iceh-frequency", default="daily")
    p.add_argument("-m", "--classification", default="binary-days", choices=("raw", "binary-days", "rolling-mean"))
    p.add_argument("--grid-type", default="Tc")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--ispd-thresh", type=float, default=5.0e-4)
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--af2020-raw-root", default=None, help="Directory containing raw FastIce_70_YYYY.nc files.")
    p.add_argument("--af2020-var", default="Fast_Ice_Time_series")
    p.add_argument("--af2020-lon-var", default="longitude")
    p.add_argument("--af2020-lat-var", default="latitude")
    p.add_argument("--af2020-area-var", default="area")
    p.add_argument("--af2020-time-var", default="time")
    p.add_argument("--af2020-thresh", type=float, default=4.0)
    p.add_argument("--fip-sampling", default = "af2020-native", choices = ("af2020-native", "daily"),
                   help=("Temporal basis for FIP. 'af2020-native' matches the notebook: AF2020 native 15-day masks, "
                         "model FI_mask reindexed to AF2020 times. 'daily' uses daily interpolated AF2020 FIC and daily model mask."))
    p.add_argument("--af2020-daily-method", default="linear", choices=("linear", "nearest"))
    p.add_argument("--max-gap-days", type=int, default=35)
    p.add_argument("--pixel-size-m", type=float, default=5_000.0)
    p.add_argument("--radius-of-influence-m", type=float, default=10_000.0)
    p.add_argument("--buffer-m", type=float, default=20_000.0)
    p.add_argument("--cice-lon-shift-deg", type=float, default=0.25,
                   help="Longitude shift applied to CICE TLON before pyresample, reproducing the AFIM notebook's TLON+0.25 step.")
    p.add_argument("--fill-value", type=float, default=np.nan)
    p.add_argument("--fic-dates", default = None,
                   help = "Comma-separated dates to write as FIC side-by-side common-grid samples. Avoids writing huge daily whole-Antarctic stores.")
    p.add_argument("--write-daily-fic",
                   action = "store_true",
                   help = "Write every daily FIC field between --start-date and --end-date. This can be very large.")
    p.add_argument("--include-weight", action="store_true", help="Add FIP diff plotting weight field.")
    p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--D-out", default=None)
    p.add_argument("--name-prefix", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def _date_list(args: argparse.Namespace) -> list[pd.Timestamp]:
    if args.write_daily_fic:
        return list(pd.date_range(args.start_date, args.end_date, freq="D"))
    if not args.fic_dates:
        return []
    return [pd.Timestamp(tok.strip()) for tok in args.fic_dates.split(",") if tok.strip()]

def _get_2d(da: xr.DataArray) -> xr.DataArray:
    return da.isel(time=0, drop=True) if "time" in da.dims else da

def _load_model_fields(args    : argparse.Namespace,
                       run_cfg : RunSpec,
                       cls_cfg : ClassificationSpec,
                       obs_cfg : ObservationSpec,
                       pth_cfg : ShugaPaths,
                       chunks  : dict):
    # Load CICE history/static fields required for FIC and geometry.
    cice      = load_cice(run = run_cfg, classify = cls_cfg, observations = obs_cfg, paths = pth_cfg, variables = ["aice", "TLON", "TLAT"],
                     hemisphere = args.hemisphere, chunks = chunks)
    mask_name = f"{args.ice_type.upper()}_mask"
    cls       = load_classified(run = run_cfg, classify = cls_cfg, observations = obs_cfg, paths = pth_cfg, variables=[mask_name],
                                classification = args.classification , hemisphere = args.hemisphere, chunks = chunks)
    if mask_name not in cls:
        raise KeyError(f"{mask_name} was not found in classified store. Variables: {list(cls.data_vars)}")
    return cice, cls[mask_name].astype("float32").rename("SIM_FI_mask")


def _model_fip_for_sampling(model_mask: xr.DataArray, af_mask: xr.DataArray, args: argparse.Namespace) -> xr.DataArray:
    fip_start = args.fip_start or args.start_date
    fip_end   = args.fip_end or args.end_date
    if args.fip_sampling == "af2020-native":
        af_times = af_mask.sel(time=slice(fip_start, fip_end)).time.values
        if len(af_times) == 0:
            raise ValueError("No AF2020 native times in requested FIP window.")
        # nearest is deliberate: this reproduces the notebook logic where daily CICE masks
        # are sampled at AF2020 native output times before persistence is computed.
        mod_t = model_mask.sel(time=slice(fip_start, fip_end)).reindex(time=af_times, method="nearest")
        out   = mod_t.mean("time", skipna=True).clip(0.0, 1.0).astype("float32").rename("SIM_FIP_native")
    else:
        mod_t = model_mask.sel(time=slice(fip_start, fip_end))
        out   = mod_t.mean("time", skipna=True).clip(0.0, 1.0).astype("float32").rename("SIM_FIP_native")
    out.attrs.update(long_name="model fast-ice persistence on native CICE grid", units="1", fip_sampling=args.fip_sampling)
    return out

def _obs_fip_for_sampling(af: AF2020Observations, af_mask: xr.DataArray, args: argparse.Namespace) -> xr.DataArray:
    fip_start = args.fip_start or args.start_date
    fip_end   = args.fip_end or args.end_date
    if args.fip_sampling == "af2020-native":
        return af.persistence(af_mask, start_date = fip_start, end_date = fip_end, name="AF_FIP_native")
    af_daily = af.daily_fic(af_mask, start_date = fip_start, end_date = fip_end,
                            method = args.af2020_daily_method, max_gap_days = args.max_gap_days,
                            name = "AF_FIC_daily_for_FIP")
    return af.persistence(af_daily, name="AF_FIP_native")

def main() -> None:
    args   = parse_args()
    chunks = {"time": args.chunks_time}
    run_cfg = RunSpec(sim_name       = args.sim_name,
                      start_date     = args.start_date,
                      end_date       = args.end_date,
                      hemisphere     = args.hemisphere,
                      project        = args.project,
                      user           = args.user,
                      iceh_frequency = args.iceh_frequency)
    cls_cfg = ClassificationSpec(ice_type     = args.ice_type,
                                 grid_type    = args.grid_type,
                                 ispd_thresh  = args.ispd_thresh,
                                 methods      = (args.classification,),
                                 bin_window   = args.bin_window,
                                 bin_min_days = args.bin_min_days,
                                 roll_window  = args.roll_window)
    obs_cfg = ObservationSpec()
    pth_cfg = ShugaPaths(run = run_cfg, classify = cls_cfg, observations = obs_cfg)
    AF20_nm = AF2020Naming(variable  = args.af2020_var,
                           lon       = args.af2020_lon_var,
                           lat       = args.af2020_lat_var,
                           area      = args.af2020_area_var,
                           time      = args.af2020_time_var,
                           threshold = args.af2020_thresh)
    af      = AF2020Observations(run = run_cfg, obs = obs_cfg, paths = pth_cfg, raw_root = args.af2020_raw_root, chunks = chunks, naming = AF20_nm)
    fip0    = args.fip_start or args.start_date
    fipN    = args.fip_end or args.end_date
    af_ds   = af.open_raw(fip0, fipN)
    af_mask = af.native_mask(af_ds)
    cice, model_mask = _load_model_fields(args, run, classify, observations, paths, chunks)
    TLAT    = _get_2d(cice["TLAT"])
    TLON    = _get_2d(cice["TLON"]) + float(args.cice_lon_shift_deg)
    pyr_cfg = PyresampleSpec(pixel_size_m          = args.pixel_size_m,
                             radius_of_influence_m = args.radius_of_influence_m,
                             buffer_m              = args.buffer_m,
                             fill_value            = args.fill_value)
    area_def = area_definition_from_lonlat_pairs([(TLAT.values, TLON.values), (af_ds[AF20_nm.lat].values, af_ds[AF20_nm.lon].values)], spec = pyr_cfg)
    # FIP on native grids, then common EPSG:3031 grid.
    obs_fip_native = _obs_fip_for_sampling(af, af_mask, args)
    sim_fip_native = _model_fip_for_sampling(model_mask, af_mask, args)
    obs_fip_common = resample_dataarray_to_area(obs_fip_native, af_ds[AF20_nm.lat].values, af_ds[AF20_nm.lon].values, area_def,
                                                radius     = args.radius_of_influence_m,
                                                fill_value = np.nan,
                                                pixel_size = args.pixel_size_m,
                                                name       = "obs")
    sim_fip_common = resample_dataarray_to_area(sim_fip_native, TLAT.values, TLON.values, area_def,
                                                radius     = args.radius_of_influence_m,
                                                fill_value = np.nan,
                                                pixel_size = args.pixel_size_m,
                                                name       = "mod")
    fip = fip_difference_dataset(sim_fip_common, obs_fip_common, name_mod="mod", name_obs="obs")
    fip = add_lonlat_from_epsg3031(fip, wrap="0-360")
    fip.attrs.update(sim_name              = args.sim_name,
                     classification        = args.classification,
                     grid_type             = args.grid_type,
                     fip_start             = fip0,
                     fip_end               = fipN,
                     fip_sampling          = args.fip_sampling,
                     pyresample_method     = "nearest",
                     pixel_size_m          = float(args.pixel_size_m),
                     radius_of_influence_m = float(args.radius_of_influence_m),
                     cice_lon_shift_deg    = float(args.cice_lon_shift_deg),
                     description           = "CICE and AF2020 FIP resampled to common EPSG:3031 pyresample grid; diff = model - AF2020.")
    if args.include_weight:
        fip["diff_weight"] = fip_weight(fip, mode="max", t=0.10, gamma=1.2)
    D_out = Path(args.D_out).expanduser() if args.D_out else Path.home() / "AFIM_archive" / args.sim_name / "zarr" / "comparisons"
    name_prefix     = args.name_prefix or f"af2020_pyresample_{args.sim_name}_{args.classification}_{fip0}_{fipN}"
    D_out.mkdir(parents=True, exist_ok=True)
    fip_store = D_out / f"{name_prefix}_FIP.zarr"
    if fip_store.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {fip_store}. Pass --overwrite to replace.")
    fip.to_zarr(fip_store, mode="w", consolidated=False)
    print(f"Wrote FIP common-grid store: {fip_store}")
    # Native AF2020 FIP is kept separately so AF2020-only plots can use native grid when desired.
    native_store = D_out / f"{name_prefix}_AF2020_native_FIP.zarr"
    native_ds    = xr.Dataset({"AF_FIP_native": obs_fip_native})
    area         = af.native_area(af_ds)
    if area is not None:
        native_ds["AF_cell_area"] = area
    native_ds.to_zarr(native_store, mode="w", consolidated=False)
    print(f"Wrote AF2020 native FIP store: {native_store}")
    if not args.skip_stats:
        stats    = compute_fipdiff_stats_weighted(fip, pixel_size_m = args.pixel_size_m, threecat_percent_only = False)
        csv_path = D_out / f"{name_prefix}_FIP_diff_stats.csv"
        tex_path = D_out / f"{name_prefix}_FIP_diff_stats.tex"
        stats.to_csv(csv_path)
        with open(tex_path, "w") as f:
            f.write(stats.to_latex(float_format=lambda x: f"{x:,.2f}", na_rep="--"))
        print(f"Wrote FIP stats CSV: {csv_path}")
        print(f"Wrote FIP stats TeX: {tex_path}")
    # Optional FIC sample products. This does not write daily fields unless requested.
    fic_dates = _date_list(args)
    if fic_dates:
        print(f"Building {len(fic_dates)} FIC common-grid sample(s).")
        af_fic_daily   = af.daily_fic(af_mask,
                                      start_date   = str(min(fic_dates).date()),
                                      end_date     = str(max(fic_dates).date()),
                                      method       = args.af2020_daily_method,
                                      max_gap_days = args.max_gap_days,
                                      name         = "AF_FIC_native_daily")
        sim_fic_native = (model_mask * cice["aice"].astype("float32")).rename("SIM_FIC_native")
        obs_pieces     = []
        sim_pieces     = []
        for t in fic_dates:
            td         = pd.Timestamp(t)
            obs2       = af_fic_daily.sel(time=td, method="nearest")
            sim2       = sim_fic_native.sel(time=td, method="nearest")
            obs_common = resample_dataarray_to_area(obs2, af_ds[AF20_nm.lat].values, af_ds[AF20_nm.lon].values, area_def,
                                                    radius     = args.radius_of_influence_m,
                                                    fill_value = np.nan,
                                                    pixel_size = args.pixel_size_m,
                                                    name       = "AF_FIC").expand_dims(time=[td])
            sim_common = resample_dataarray_to_area(sim2, TLAT.values, TLON.values, area_def,
                                                    radius     = args.radius_of_influence_m,
                                                    fill_value = np.nan,
                                                    pixel_size = args.pixel_size_m,
                                                    name       = "SIM_FIC").expand_dims(time=[td])
            obs_pieces.append(obs_common)
            sim_pieces.append(sim_common)
        fic = xr.Dataset({"AF_FIC": xr.concat(obs_pieces, dim="time"), "SIM_FIC": xr.concat(sim_pieces, dim="time")})
        fic = add_lonlat_from_epsg3031(fic, wrap="0-360")
        fic.attrs.update(sim_name               = args.sim_name,
                         classification         = args.classification,
                         pyresample_method      = "nearest",
                         temporal_interpolation = args.af2020_daily_method,
                         cice_lon_shift_deg     = float(args.cice_lon_shift_deg),
                         description            = "Sampled daily FIC fields on the shared EPSG:3031 grid: AF2020 daily interpolation and CICE FI_mask*aice.")
        fic_store = D_out / f"{name_prefix}_FIC_samples.zarr"
        fic.to_zarr(fic_store, mode="w", consolidated=False)
        print(f"Wrote FIC sample common-grid store: {fic_store}")

if __name__ == "__main__":
    main()
