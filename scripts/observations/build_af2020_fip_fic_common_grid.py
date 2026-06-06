#!/usr/bin/env python3
from __future__ import annotations

import argparse, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
#####################################################################
# make sure this reflects the correct location of mawsons-chest repo
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
#####################################################################
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.regridder.pyresample import (PyresampleSpec,
                                        add_lonlat_from_epsg3031,
                                        area_definition_from_lonlat_pairs,
                                        resample_dataarray_to_area,
                                        resample_swath_to_area)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description = ("Build a persistent AF2020 common-grid FIC/FIP store. "
                                               "AF2020 FIC is kept on the native 15-day AF2020 time coordinate; "
                                               "no daily temporal interpolation is performed."))
    p.add_argument("--af2020-raw-root", default="/g/data/gv90/da1339/SeaIce/FI_obs/org")
    p.add_argument("--out-store", default="/g/data/gv90/da1339/SeaIce/FI_obs/AF-FI-2020db_common-5km_pyresample.zarr")
    p.add_argument("--cice-static-store", default=None, help="Path to iceh_static.zarr. If omitted, use ~/AFIM_archive/SIM_NAME/zarr/iceh_static.zarr.")
    p.add_argument("-s", "--sim-name", default="LD-static-Cs1e-3", help="Simulation used only to locate a representative CICE static grid.")
    p.add_argument("--start-date", default="2000-03-01")
    p.add_argument("--end-date", default="2018-02-15")
    p.add_argument("--fip-start", default="2000-03-01")
    p.add_argument("--fip-end", default="2018-02-15")
    p.add_argument("--af2020-variable", default="Fast_Ice_Time_series")
    p.add_argument("--af2020-lon", default="longitude")
    p.add_argument("--af2020-lat", default="latitude")
    p.add_argument("--af2020-time", default="time")
    p.add_argument("--af2020-threshold", type=float, default=4.0)
    p.add_argument("--south-lat-max", type=float, default=-45.0)
    p.add_argument("--pixel-size-m", type=float, default=5000.0)
    p.add_argument("--radius-of-influence-m", type=float, default=10000.0)
    p.add_argument("--buffer-m", type=float, default=20000.0)
    p.add_argument("--cice-lon-shift-deg", type=float, default=0.25)
    p.add_argument("--time-batch", type=int, default=8)
    p.add_argument("--chunks-time", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def _raw_files(root: Path, start: str, end: str) -> list[Path]:
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    files = [root / f"FastIce_70_{yy:04d}.nc" for yy in range(t0.year, t1.year + 1)]
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError(f"No FastIce_70_YYYY.nc files found in {root}")
    return files

def _open_af2020(args: argparse.Namespace) -> xr.Dataset:
    files = _raw_files(Path(args.af2020_raw_root).expanduser(), args.start_date, args.end_date)
    ds    = xr.open_mfdataset(files, engine = "netcdf4", combine = "by_coords", chunks = {args.af2020_time: "auto"}, data_vars = "minimal", coords = "minimal", compat = "override")
    if args.af2020_time != "time" and args.af2020_time in ds:
        ds = ds.rename({args.af2020_time: "time"})
    ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values)).sortby("time")
    return ds.sel(time=slice(args.start_date, args.end_date))

def _open_cice_static(args: argparse.Namespace) -> xr.Dataset:
    if args.cice_static_store is not None:
        path = Path(args.cice_static_store).expanduser()
    else:
        path = Path.home() / "AFIM_archive" / args.sim_name / "zarr" / "iceh_static.zarr"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_zarr(path, consolidated=False)
    for v in ("TLON", "TLAT"):
        if v not in ds:
            raise KeyError(f"{v} not found in {path}")
    tlat = ds["TLAT"]
    ydim = tlat.dims[0]
    xdim = tlat.dims[1]
    row_mask = (tlat <= float(args.south_lat_max)).any(dim=xdim).compute()
    rows = np.where(row_mask.values)[0]
    if rows.size == 0:
        raise ValueError(f"No CICE rows found south of {args.south_lat_max}.")
    return ds[["TLON", "TLAT"]].isel({ydim: slice(int(rows.min()), int(rows.max()) + 1)})

def _mask_from_af2020(ds: xr.Dataset, args: argparse.Namespace) -> xr.DataArray:
    if args.af2020_variable not in ds:
        raise KeyError(f"{args.af2020_variable} not found. Available: {list(ds.data_vars)}")
    mask = xr.where(ds[args.af2020_variable] >= float(args.af2020_threshold), 1.0, 0.0).astype("float32")
    mask = mask.rename("FIC")
    mask.attrs.update(long_name="AF2020 fast ice concentration/occupancy",
                      units="1",
                      source_variable=args.af2020_variable,
                      threshold=args.af2020_threshold,
                      temporal_sampling="native AF2020 15-day")
    return mask

def _write_batch(ds_batch: xr.Dataset, out_store: Path, *, first: bool) -> None:
    encoding = {}
    if "FIC" in ds_batch:
        encoding["FIC"] = {"compressor": None}
    if first:
        ds_batch.to_zarr(out_store, mode="w", consolidated=False)
    else:
        ds_batch[["FIC"]].to_zarr(out_store, mode="a", append_dim="time", consolidated=False)

def main() -> None:
    args = parse_args()
    out_store = Path(args.out_store).expanduser()
    if out_store.exists() and not args.overwrite:
        print(f"[skip] AF2020 common-grid store already exists: {out_store}")
        print("       Pass --overwrite to rebuild.")
        return
    if args.dry_run:
        print(args)
        return
    if out_store.exists() and args.overwrite:
        shutil.rmtree(out_store)
    af = _open_af2020(args)
    cice = _open_cice_static(args)
    mask = _mask_from_af2020(af, args)
    spec = PyresampleSpec(
        pixel_size_m=float(args.pixel_size_m),
        radius_of_influence_m=float(args.radius_of_influence_m),
        buffer_m=float(args.buffer_m),
        area_id=f"AF2020_CICE_common_{int(args.pixel_size_m)}m",
    )
    area_def = area_definition_from_lonlat_pairs(
        [
            (af[args.af2020_lat].values, af[args.af2020_lon].values),
            (cice["TLAT"].values, cice["TLON"].values + float(args.cice_lon_shift_deg)),
        ],
        spec=spec,
    )
    # Lon/lat are attached once and then stored permanently.
    template = resample_swath_to_area(
        mask.isel(time=0),
        af[args.af2020_lat].values,
        af[args.af2020_lon].values,
        area_def,
        radius=float(args.radius_of_influence_m),
        fill_value=np.nan,
        pixel_size=float(args.pixel_size_m),
        name="FIC",
    )
    template_ds = add_lonlat_from_epsg3031(template.to_dataset(name="FIC"), wrap="0-360")
    lon = template_ds["lon"]
    lat = template_ds["lat"]
    ntime = mask.sizes["time"]
    first = True
    for i0 in range(0, ntime, int(args.time_batch)):
        i1 = min(i0 + int(args.time_batch), ntime)
        print(f"[AF2020 FIC] resampling time indices {i0}:{i1} of {ntime}")
        fic_common = resample_dataarray_to_area(
            mask.isel(time=slice(i0, i1)),
            af[args.af2020_lat].values,
            af[args.af2020_lon].values,
            area_def,
            radius=float(args.radius_of_influence_m),
            fill_value=np.nan,
            pixel_size=float(args.pixel_size_m),
            name="FIC",
        ).chunk({"time": int(args.chunks_time)})
        ds_batch = xr.Dataset({"FIC": fic_common})
        if first:
            ds_batch = ds_batch.assign_coords(lon=lon, lat=lat)
            ds_batch["pixel_area_m2"] = xr.DataArray(
                float(args.pixel_size_m) ** 2,
                attrs={"units": "m2", "description": "Nominal constant area of EPSG:3031 common-grid cells."},
            )
            ds_batch.attrs.update(
                title="AF2020 fast ice on common EPSG:3031 pyresample grid",
                crs="EPSG:3031",
                pixel_size_m=float(args.pixel_size_m),
                radius_of_influence_m=float(args.radius_of_influence_m),
                buffer_m=float(args.buffer_m),
                cice_lon_shift_deg=float(args.cice_lon_shift_deg),
                af2020_threshold=float(args.af2020_threshold),
                source_temporal_sampling="native AF2020 15-day",
                note="FIC is not temporally interpolated; it retains AF2020 native timestamps.",
            )
        _write_batch(ds_batch, out_store, first=first)
        first = False
    # FIP is computed over the requested AF2020 native-time period, then resampled once.
    fip_native = mask.sel(time=slice(args.fip_start, args.fip_end)).mean("time", skipna=True).astype("float32").rename("FIP")
    fip_common = resample_swath_to_area(
        fip_native,
        af[args.af2020_lat].values,
        af[args.af2020_lon].values,
        area_def,
        radius=float(args.radius_of_influence_m),
        fill_value=np.nan,
        pixel_size=float(args.pixel_size_m),
        name="FIP",
    )
    fip_common.attrs.update(
        long_name="AF2020 fast ice persistence",
        units="1",
        time_start=str(pd.Timestamp(args.fip_start).date()),
        time_end=str(pd.Timestamp(args.fip_end).date()),
    )
    xr.Dataset({"FIP": fip_common}).to_zarr(out_store, mode="a", consolidated=False)
    print(f"[done] wrote {out_store}")

if __name__ == "__main__":
    main()
