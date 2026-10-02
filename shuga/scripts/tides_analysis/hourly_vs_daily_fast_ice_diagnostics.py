#!/usr/bin/env python3
"""
Hourly-vs-daily fast-ice mobility diagnostics, using shuga for CICE history I/O.

Workflow
--------
1. Optionally convert hourly CICE NetCDF history files into a grouped shuga Zarr
   store: iceh_hourly.zarr/YYYY_MM_DD.
2. Load each day's hourly history via shuga.io.IceHistoryLoader, including static
   grid fields merged from iceh_static.zarr.
3. Compute daily-vector and hourly-derived mobility diagnostics.

This intentionally keeps the diagnostic calculations local and explicit, but
delegates history discovery, NetCDF-to-Zarr conversion, static-store creation,
and static/dynamic merging to shuga.
"""
from __future__ import annotations
import argparse
import logging
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga import (CICEGridSpec,
                   ClassificationSpec,
                   IceHistoryLoader,
                   RunSpec,
                   ShugaPaths)
from shuga.core.data_conversion import NC2Zarr

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description     = ("Compare daily-vector fast-ice classification with hourly-derived "
                                                   "mobility diagnostics using shuga hourly Zarr stores."),
                                formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="Inclusive end date, YYYY-MM-DD")
    p.add_argument("--hemisphere", default="SH", choices=["SH", "NH"])
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--hourly-root", "--hist-dir", dest="hourly_root", default=None, type=Path,
                   help=("Directory containing hourly CICE NetCDF files. The legacy name "
                         "--hist-dir is accepted as an alias. This can be the directory "
                         "containing iceh_01h.YYYY-MM-DD-SSSSS.nc directly."))
    p.add_argument("--archive-root", default=None, type=Path)
    p.add_argument("--afim-output-root", default=None, type=Path)
    p.add_argument("--cice-store", default=None, type=Path, help="Target/source iceh_hourly.zarr path")
    p.add_argument("--static-store", "--static-zarr", dest="static_store", default=None, type=Path,
                   help="Target/source iceh_static.zarr path. The legacy --static-zarr alias is accepted.")
    p.add_argument("--classification-root", default=None, type=Path)
    p.add_argument("--logs-root", default=None, type=Path)
    # Optional grid assets used by shuga static-store fallback if static fields
    # are absent from the history files.
    p.add_argument("--grid-file", default=None, type=Path)
    p.add_argument("--kmt-file", default=None, type=Path)
    p.add_argument("--bathymetry-file", default=None, type=Path)
    p.add_argument("--f2-file", default=None, type=Path)
    p.add_argument("--gridcpl-file", default=None, type=Path)
    p.add_argument("--ice-in-file", default=None, type=Path)
    # Conversion controls.
    p.add_argument("--skip-history-conversion", action="store_true",
                   help="Skip NetCDF-to-Zarr conversion and use an existing iceh_hourly.zarr store.")
    p.add_argument("--overwrite-history", action="store_true")
    p.add_argument("--overwrite-static", action="store_true")
    p.add_argument("--delete-original", action="store_true")
    p.add_argument("--netcdf-engine", default="scipy")
    p.add_argument("--chunks-time", default=24, type=int)
    # Diagnostic controls.
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--log-file", default=None, type=Path)
    p.add_argument("--thr", "--ispd-thresh", dest="thr", default=5.0e-4, type=float)
    p.add_argument("--near-mult", default=4.0, type=float)
    p.add_argument("--aice-thresh", default=0.15, type=float)
    p.add_argument("--hi-thresh", default=0.05, type=float)
    p.add_argument("--min-hourly-files", default=20, type=int)
    return p.parse_args()

def setup_logging(args: argparse.Namespace) -> tuple[logging.Logger, Path]:
    if args.log_file is None:
        log_file = args.out_csv.with_suffix(".log")
    else:
        log_file = args.log_file
    log_file = Path(log_file).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shuga.tide_analysis.hourly_vs_daily_fastice")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(fmt     = "%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt = "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("=" * 100)
    logger.info("Starting shuga hourly-vs-daily fast-ice diagnostic")
    logger.info("Log file: %s", log_file)
    return logger, log_file

def build_shuga_paths(args: argparse.Namespace) -> tuple[RunSpec, ClassificationSpec, ShugaPaths]:
    run = RunSpec(sim_name       = args.sim_name,
                  start_date     = args.start_date,
                  end_date       = args.end_date,
                  hemisphere     = args.hemisphere,
                  project        = args.project,
                  user           = args.user,
                  iceh_frequency = "hourly")
    classify = ClassificationSpec(ice_type    = "FI",
                                  grid_type   = "Tc",
                                  ispd_thresh = args.thr,
                                  aice_thresh = args.aice_thresh,
                                  speed_var_u = "uvel",
                                  speed_var_v = "vvel",
                                  aice_var    = "aice")
    cice_grid = CICEGridSpec(grid_file       = args.grid_file,
                             kmt_file        = args.kmt_file,
                             bathymetry_file = args.bathymetry_file,
                             f2_file         = args.f2_file,
                             gridcpl_file    = args.gridcpl_file,
                             ice_in_file     = args.ice_in_file)
    paths = ShugaPaths(run                 = run,
                       classify            = classify,
                       afim_output_root    = args.afim_output_root,
                       cice_store          = args.cice_store,
                       static_store        = args.static_store,
                       classification_root = args.classification_root,
                       archive_root        = args.archive_root,
                       logs_root           = args.logs_root,
                       cice_grid           = cice_grid)
    return run, classify, paths


def ensure_hourly_zarr(args: argparse.Namespace, paths: ShugaPaths, logger: logging.Logger) -> None:
    if args.skip_history_conversion:
        logger.info("--skip-history-conversion requested")
        logger.info("Resolved CICE hourly store: %s", paths.resolve_cice_store())
        logger.info("Resolved static store     : %s", paths.resolve_static_store())
        return
    chunks = {"time": args.chunks_time}
    converter = NC2Zarr(paths         = paths,
                        logger        = logger,
                        chunks        = chunks,
                        netcdf_engine = args.netcdf_engine)
    result = converter.ensure_iceh_stores(dt0_str          = args.start_date,
                                          dtN_str          = args.end_date,
                                          hourly_root      = args.hourly_root,
                                          overwrite        = args.overwrite_history,
                                          overwrite_static = args.overwrite_static,
                                          delete_original  = args.delete_original,
                                          netcdf_engine    = args.netcdf_engine)
    logger.info("Resolved/updated CICE hourly store: %s", result.cice_store)
    if result.static_store is not None:
        logger.info("Resolved/updated static store     : %s", result.static_store)
    logger.info("nc2zarr summary: groups_scanned=%d groups_written=%d groups_rewritten=%d "
                "groups_skipped=%d source_files_seen=%d source_files_used=%d",
                result.months_scanned,
                result.months_written,
                result.months_rewritten,
                result.months_skipped,
                result.daily_files_seen,
                result.daily_files_used)

def _hemisphere_mask(tlat: xr.DataArray, hemisphere: str) -> xr.DataArray:
    hemi = hemisphere.upper()
    if hemi == "SH":
        return tlat < 0.0
    if hemi == "NH":
        return tlat > 0.0
    raise ValueError(f"Unsupported hemisphere={hemisphere!r}")

def maybe_crop_to_best_hemisphere_half(ds: xr.Dataset, hemisphere: str, logger: logging.Logger) -> xr.Dataset:
    """
    Defensive crop for full/global stores.

    IceHistoryLoader applies a hemisphere mask, but it does not necessarily
    drop the other half of a global CICE grid. This keeps memory lower and
    also protects against global static stores being merged with SH-only
    dynamic history.
    """
    if "nj" not in ds.sizes or "TLAT" not in ds:
        return ds
    nj = int(ds.sizes["nj"])
    if nj < 2 or nj % 2 != 0:
        return ds
    half = nj // 2
    candidates = [("first", slice(0, half)),
                  ("last", slice(half, nj))]
    scored: list[tuple[float, float, str, slice]] = []
    for name, slc in candidates:
        tlat_block = ds["TLAT"].isel(nj=slc)
        hemi_score = float(_hemisphere_mask(tlat_block, hemisphere).mean().compute())
        valid_score = 0.0
        if "aice" in ds and "time" in ds["aice"].dims and ds.sizes.get("time", 0) > 0:
            valid_score = float(ds["aice"].isel(time=0, nj=slc).notnull().mean().compute())
        # Primary score is geographic. Valid-score only breaks ties when a
        # dynamic half has been padded with NaNs during static/dynamic merging.
        scored.append((hemi_score + 0.05 * valid_score, hemi_score, name, slc))
    scored.sort(reverse=True)
    best_total, best_hemi, best_name, best_slc = scored[0]
    other_total = scored[1][0]
    if best_hemi >= 0.50 and best_total > other_total:
        logger.info("Cropping nj=%d to %d using %s half for %s "
                    "(hemi_score=%.3f, total_score=%.3f)",
                    nj,
                    half,
                    best_name,
                    hemisphere,
                    best_hemi,
                    best_total)
        return ds.isel(nj=best_slc)
    return ds

def area_sum_1e3km2(area: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return area.where(mask).sum(skipna=True) / 1.0e9

def awmean(area: xr.DataArray, x: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    w     = area.where(mask)
    denom = w.sum(skipna=True)
    return (x.where(mask) * w).sum(skipna=True) / denom

def scalar(x) -> float:
    try:
        return float(x.compute())
    except Exception:
        return float(x)

def format_daily_row(row: dict) -> str:
    return (f"{row['day']}\n\t"
            f"n_hourly              = {row['n_hourly_files']:02d}\n\t"
            f"ice                   = {row['ice_area_1e3_km2']:10.2f}\n\t"
            f"daily_imm             = {row['daily_immobile_area_1e3_km2']:10.2f}\n\t"
            f"daily_mobile          = {row['daily_mobile_area_1e3_km2']:10.2f}\n\t"
            f"near                  = {row['daily_near_area_1e3_km2']:10.2f}\n\t"
            f"hidden_any            = {row['hidden_any_area_1e3_km2']:10.2f}\n\t"
            f"hidden_3h             = {row['hidden_3h_area_1e3_km2']:10.2f}\n\t"
            f"hidden_any_frac       = {row['daily_immobile_hidden_any_frac']:.4f}\n\t"
            f"hidden_3h_frac        = {row['daily_immobile_hidden_3h_frac']:.4f}\n\t"
            f"pulsed_near           = {row['pulsed_near_area_1e3_km2']:10.2f}\n\t"
            f"mean_frac_mobile_near = {row['mean_frac_mobile_immobile_or_near']:.4f}")

def load_day(loader: IceHistoryLoader, day: str, args: argparse.Namespace, logger: logging.Logger) -> xr.Dataset:
    variables = ["uvel", "vvel", "aice", "hi", "tarea", "tmask", "TLAT"]
    ds        = loader.load(dt0_str    = day,
                            dtN_str    = day,
                            variables  = variables,
                            hemisphere = args.hemisphere,
                            chunks     = {"time": args.chunks_time})
    ds        = maybe_crop_to_best_hemisphere_half(ds, args.hemisphere, logger)
    missing   = [v for v in variables if v not in ds]
    if missing:
        raise ValueError(f"{day}: shuga-loaded dataset missing required variables: {missing}")
    # Pull one day into local memory for the diagnostic calculations.
    ds = ds.load()
    return ds

def process_day(loader: IceHistoryLoader, day: str, args: argparse.Namespace, logger: logging.Logger) -> dict | None:
    logger.info("Loading %s through shuga IceHistoryLoader", day)
    ds = load_day(loader, day, args, logger)
    try:
        n_hourly_timesteps = int(ds.sizes.get("time", 0))
        n_source_files = int(ds.attrs.get("_nc2zarr_file_count", n_hourly_timesteps))
        if n_hourly_timesteps < args.min_hourly_files:
            logger.warning("Skipping %s: only %d hourly timesteps in shuga store", day, n_hourly_timesteps)
            return None
        logger.info("%s working grid: time=%d, nj=%d, ni=%d", day, n_hourly_timesteps, int(ds.sizes["nj"]), int(ds.sizes["ni"]))
        area  = ds["tarea"]
        tmask = ds["tmask"]
        tlat  = ds["TLAT"]
        # Remove accidental singleton time dimensions in static variables.
        if "time" in area.dims:
            area = area.isel(time=0, drop=True)
        if "time" in tmask.dims:
            tmask = tmask.isel(time=0, drop=True)
        if "time" in tlat.dims:
            tlat = tlat.isel(time=0, drop=True)
        static_mask = (tmask > 0) & _hemisphere_mask(tlat, args.hemisphere)
        aice        = ds["aice"].mean("time", skipna=True)
        hi          = ds["hi"].mean("time", skipna=True)
        ice         = static_mask & (aice > args.aice_thresh) & (hi > args.hi_thresh)
        u           = ds["uvel"]
        v           = ds["vvel"]
        spd_h       = np.hypot(u, v)
        # Daily vector speed implied by averaging u/v first.
        spd_vec_dly = np.hypot(u.mean("time", skipna=True), v.mean("time", skipna=True))
        # Hourly-derived scalar diagnostics.
        spd_scalar_mean = spd_h.mean("time", skipna=True)
        spd_rms         = np.sqrt((u * u + v * v).mean("time", skipna=True))
        spd_max         = spd_h.max("time", skipna=True)
        frac_mob        = (spd_h > args.thr).mean("time", skipna=True)
        any_mob         = (spd_h > args.thr).any("time")
        dly_imm         = ice & (spd_vec_dly < args.thr)
        dly_mob         = ice & (spd_vec_dly >= args.thr)
        dly_near        = ice & (spd_vec_dly >= args.thr) & (spd_vec_dly < args.near_mult * args.thr)
        dly_imm_or_near = ice & (spd_vec_dly < args.near_mult * args.thr)
        hid_any         = dly_imm & any_mob
        hid_1h          = dly_imm & (frac_mob >= 1 / 24)
        hid_3h          = dly_imm & (frac_mob >= 3 / 24)
        hid_6h          = dly_imm & (frac_mob >= 6 / 24)
        pulsed_near     = dly_imm_or_near & (frac_mob > 0) & (frac_mob < 0.5)
        ice_area        = area_sum_1e3km2(area, ice)
        dly_imm_area    = area_sum_1e3km2(area, dly_imm)
        row             = {"day"                                      : day,
                           "n_hourly_files"                           : n_source_files,
                           "n_hourly_timesteps"                       : n_hourly_timesteps,
                           "area_source"                              : "shuga:iceh_static.zarr:tarea",
                           "ice_area_1e3_km2"                         : scalar(ice_area),
                           "daily_immobile_area_1e3_km2"              : scalar(dly_imm_area),
                           "daily_mobile_area_1e3_km2"                : scalar(area_sum_1e3km2(area, dly_mob)),
                           "daily_near_area_1e3_km2"                  : scalar(area_sum_1e3km2(area, dly_near)),
                           "daily_immobile_or_near_area_1e3_km2"      : scalar(area_sum_1e3km2(area, dly_imm_or_near)),
                           "hidden_any_area_1e3_km2"                  : scalar(area_sum_1e3km2(area, hid_any)),
                           "hidden_1h_area_1e3_km2"                   : scalar(area_sum_1e3km2(area, hid_1h)),
                           "hidden_3h_area_1e3_km2"                   : scalar(area_sum_1e3km2(area, hid_3h)),
                           "hidden_6h_area_1e3_km2"                   : scalar(area_sum_1e3km2(area, hid_6h)),
                           "pulsed_near_area_1e3_km2"                 : scalar(area_sum_1e3km2(area, pulsed_near)),
                           "daily_immobile_hidden_any_frac"           : np.nan,
                           "daily_immobile_hidden_3h_frac"            : np.nan,
                           "mean_daily_vector_speed_immobile_or_near" : scalar(awmean(area, spd_vec_dly, dly_imm_or_near)),
                           "mean_hourly_scalar_speed_immobile_or_near": scalar(awmean(area, spd_scalar_mean, dly_imm_or_near)),
                           "mean_hourly_rms_speed_immobile_or_near"   : scalar(awmean(area, spd_rms, dly_imm_or_near)),
                           "mean_hourly_max_speed_immobile_or_near"   : scalar(awmean(area, spd_max, dly_imm_or_near)),
                           "mean_frac_mobile_immobile_or_near"        : scalar(awmean(area, frac_mob, dly_imm_or_near))}
        if row["daily_immobile_area_1e3_km2"] > 0:
            row["daily_immobile_hidden_any_frac"] = (row["hidden_any_area_1e3_km2"] / row["daily_immobile_area_1e3_km2"])
            row["daily_immobile_hidden_3h_frac"]  = (row["hidden_3h_area_1e3_km2"] / row["daily_immobile_area_1e3_km2"])
        return row
    finally:
        ds.close()

def write_outputs(rows: list[dict], out_csv: Path, logger: logging.Logger) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote daily diagnostics: %s", out_csv)
    monthly_csv = out_csv.with_name(out_csv.stem + "_monthly_summary.csv")
    if df.empty:
        logger.warning("No rows were processed; monthly summary not written")
        return
    df["month"]  = pd.to_datetime(df["day"]).dt.to_period("M").astype(str)
    numeric_cols = df.select_dtypes(include="number").columns
    monthly      = df.groupby("month")[numeric_cols].mean()
    monthly.to_csv(monthly_csv)
    logger.info("Wrote monthly diagnostics: %s", monthly_csv)

def main() -> None:
    args = parse_args()
    logger, log_file = setup_logging(args)
    logger.info("sim_name         : %s", args.sim_name)
    logger.info("start_date       : %s", args.start_date)
    logger.info("end_date         : %s", args.end_date)
    logger.info("hemisphere       : %s", args.hemisphere)
    logger.info("hourly_root      : %s", args.hourly_root)
    logger.info("cice_store       : %s", args.cice_store)
    logger.info("static_store     : %s", args.static_store)
    logger.info("out_csv          : %s", args.out_csv)
    logger.info("thr              : %.6e", args.thr)
    logger.info("near_mult        : %.3f", args.near_mult)
    logger.info("aice_thresh      : %.3f", args.aice_thresh)
    logger.info("hi_thresh        : %.3f", args.hi_thresh)
    logger.info("min_hourly_files : %d", args.min_hourly_files)
    run, classify, paths = build_shuga_paths(args)
    ensure_hourly_zarr(args, paths, logger)
    loader = IceHistoryLoader(paths, logger=logger)
    days   = pd.date_range(args.start_date, args.end_date, freq="D")
    rows: list[dict] = []
    for d in days:
        day = d.strftime("%Y-%m-%d")
        try:
            row = process_day(loader, day, args, logger)
            if row is None:
                continue
            rows.append(row)
            logger.info("DAILY_ROW | %s", format_daily_row(row))
            partial_csv = args.out_csv.with_name(args.out_csv.stem + "_partial.csv")
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            logger.info("Wrote partial CSV: %s", partial_csv)
        except Exception as exc:
            logger.error("Failed processing %s: %s", day, exc)
            logger.error(traceback.format_exc())
            raise
    write_outputs(rows, args.out_csv, logger)
    logger.info("Finished shuga hourly-vs-daily fast-ice diagnostic")
    logger.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
