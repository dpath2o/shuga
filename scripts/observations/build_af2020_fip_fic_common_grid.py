#!/usr/bin/env python3
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga.core.paths import ShugaPaths
from shuga.grid.cice import CICEGridwork
from shuga.observations import AF2020Obs, AF2020Spec
from shuga.regridding.pyresample import (
    PyresampleSpec,
    add_lonlat_from_epsg3031,
    area_definition_from_lonlat_pairs,
    resample_dataarray_to_area,
    resample_swath_to_area,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a persistent AF2020 common-grid FIC/FIP store. "
            "AF2020 FIC is kept on the native 15-day AF2020 time coordinate; "
            "no daily temporal interpolation is performed."
        )
    )
    p.add_argument(
        "--out-store",
        default=None,
        help=(
            "Persistent AF2020 common-grid zarr output store. "
            "Default: AF2020Spec.D_reG / AF2020Spec.F_reG."
        ),
    )
    p.add_argument(
        "--cice-static-store",
        default=None,
        help=(
            "Path to CICE coordinate/static zarr store. "
            "Default: ShugaPaths.resolve_static_store(), normally "
            "~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr."
        ),
    )
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


def _write_batch(ds_batch: xr.Dataset, out_store: Path, *, first: bool) -> None:
    if first:
        ds_batch.to_zarr(out_store, mode="w", consolidated=False)
    else:
        ds_batch[["FIC"]].to_zarr(out_store, mode="a", append_dim="time", consolidated=False)


def main() -> None:
    args = parse_args()

    paths = ShugaPaths()
    gridwork = CICEGridwork(paths=paths)

    af20_cfg = AF2020Spec(
        variable=args.af2020_variable,
        lon=args.af2020_lon,
        lat=args.af2020_lat,
        time=args.af2020_time,
        threshold=args.af2020_threshold,
    )

    afobs = AF2020Obs(
        paths=paths,
        af20_cfg=af20_cfg,
    )

    if args.out_store is None:
        P_zarr = Path(afobs.D_reG).expanduser() / afobs.af20_cfg.F_reG
    else:
        P_zarr = Path(args.out_store).expanduser()

    if args.dry_run:
        print(args)
        print(f"CICE static store : {args.cice_static_store or paths.resolve_static_store()}")
        print(f"AF2020 origin dir : {afobs.D_org_nc}")
        print(f"Output zarr       : {P_zarr}")
        return

    if P_zarr.exists() and not args.overwrite:
        print(f"[skip] AF2020 common-grid store already exists: {P_zarr}")
        print("        Pass --overwrite to rebuild.")
        return

    if P_zarr.exists() and args.overwrite:
        shutil.rmtree(P_zarr)

    P_zarr.parent.mkdir(parents=True, exist_ok=True)

    af = afobs.open_org(
        start_date=args.start_date,
        end_date=args.end_date,
    )

    mask = afobs.native_mask(
        ds=af,
        threshold=args.af2020_threshold,
        name="FIC",
    )

    cice = gridwork.load_cice_static(
        P_cice_static_store=args.cice_static_store,
        variables=["TLON", "TLAT"],
        south_lat_max=args.south_lat_max,
        lon_type="0-360",
    )

    pyre_cfg = PyresampleSpec(
        pixel_size_m=float(args.pixel_size_m),
        radius_of_influence_m=float(args.radius_of_influence_m),
        buffer_m=float(args.buffer_m),
        area_id=f"AF2020_CICE_common_{int(args.pixel_size_m)}m",
    )

    area_def = area_definition_from_lonlat_pairs(
        [
            (af[args.af2020_lat].values, af[args.af2020_lon].values),
            (
                cice["TLAT"].values,
                cice["TLON"].values + float(args.cice_lon_shift_deg),
            ),
        ],
        spec=pyre_cfg,
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
                attrs={
                    "units": "m2",
                    "description": "Nominal constant area of EPSG:3031 common-grid cells.",
                },
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

        _write_batch(ds_batch, P_zarr, first=first)
        first = False

    fip_native = (
        mask.sel(time=slice(args.fip_start, args.fip_end))
        .mean("time", skipna=True)
        .astype("float32")
        .rename("FIP")
    )

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

    xr.Dataset({"FIP": fip_common}).to_zarr(P_zarr, mode="a", consolidated=False)
    print(f"[done] wrote {P_zarr}")


if __name__ == "__main__":
    main()
