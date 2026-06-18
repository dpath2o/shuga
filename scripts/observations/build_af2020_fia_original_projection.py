#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec
from shuga.observations import AF2020Obs, AF2020Spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build AF2020 FIA on the original AF2020 projection/time axis. "
            "No spatial regridding and no temporal interpolation are performed."
        )
    )
    p.add_argument("--start-date", default="2000-03-01")
    p.add_argument("--end-date", default="2018-02-15")
    p.add_argument("--org-root", default="/g/data/jk72/af1544/fraser2020_data")
    p.add_argument("--obs-root", default="/g/data/gv90/da1339/SeaIce/FI_obs")
    p.add_argument("--out-store", default="/g/data/gv90/da1339/SeaIce/FI_obs/AF-FI-2020db_FIA_from_original_dataset.zarr")
    p.add_argument("--af2020-variable", default="Fast_Ice_Time_series")
    p.add_argument("--af2020-lon", default="longitude")
    p.add_argument("--af2020-lat", default="latitude")
    p.add_argument("--af2020-area", default="area")
    p.add_argument("--af2020-time", default="time")
    p.add_argument("--af2020-threshold", type=float, default=4.0)
    p.add_argument("--scale", type=float, default=1.0e9, help="Area scaling. Default gives 10^3 km^2 from m^2.")
    p.add_argument("--chunks-time", type=int, default=48)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = RunSpec(
        sim_name="AF2020",
        start_date=args.start_date,
        end_date=args.end_date,
        hemisphere="SH",
        project="gv90",
        user="da1339",
    )
    obs_cfg = ObservationSpec(
        af2020_root=args.obs_root,
        af2020_fia_native_store=Path(args.out_store).name,
    )
    pth_cfg = ShugaPaths(run_cfg=run_cfg, obs_cfg=obs_cfg)
    af20_cfg = AF2020Spec(
        variable=args.af2020_variable,
        lon=args.af2020_lon,
        lat=args.af2020_lat,
        area=args.af2020_area,
        time=args.af2020_time,
        D_org_nc=args.org_root,
        D_reG=args.obs_root,
        threshold=args.af2020_threshold,
    )
    afobs = AF2020Obs(
        run_cfg=run_cfg,
        obs_cfg=obs_cfg,
        pth_cfg=pth_cfg,
        D_org_nc=args.org_root,
        D_reG=args.obs_root,
        chunks={"time": int(args.chunks_time)},
        af20_cfg=af20_cfg,
    )

    out_store = Path(args.out_store).expanduser()
    if args.dry_run:
        print(args)
        print(f"AF2020 origin dir : {afobs.D_org_nc}")
        print(f"Output zarr       : {out_store}")
        return

    ds = afobs.write_native_fia_timeseries(
        out_store=out_store,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.af2020_threshold,
        scale=args.scale,
        overwrite=args.overwrite,
    )

    print(f"[done] wrote {out_store}")
    print(ds)


if __name__ == "__main__":
    main()
