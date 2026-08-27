#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import pandas as pd
import xarray as xr

from shuga.core.paths import ShugaPaths
from shuga.core.types import CICEGridSpec, ObservationSpec, RunSpec, WaveForcingSpec
from shuga.plotting.cawcr import plot_whacs_daily_comparison
from shuga.waves.whacs import WHACS_SOURCE_ROOT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create daily 2x2 PyGMT WHACS regridding QC figures: native station Hs/Tp "
            "on the upper row and regridded CICE-grid Hs/Tp on the lower row."
        )
    )
    p.add_argument("year", type=int)
    p.add_argument("month", type=int)
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--sim-name", default="LD-waves-exp01")
    p.add_argument("--source-root", type=Path, default=WHACS_SOURCE_ROOT)
    p.add_argument("--regridded-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--north", type=float, default=-35.0)
    p.add_argument("--grid-stride", type=int, default=3)
    p.add_argument("--hs-max", type=float, default=None)
    p.add_argument("--tp-max", type=float, default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    return p


def whacs_file(source_root: Path, year: int, month: int) -> Path:
    last_day = calendar.monthrange(year, month)[1]
    return source_root / (
        f"efth_WHACS_hindcast_spec_GRID_1hr_"
        f"{year:04d}{month:02d}010000-{year:04d}{month:02d}{last_day:02d}2300.nc"
    )


def main() -> None:
    args = build_parser().parse_args()
    if not (1 <= args.month <= 12):
        raise SystemExit("month must be in the range 1..12")
    if args.grid_stride < 1:
        raise SystemExit("--grid-stride must be >= 1")

    year = int(args.year)
    month = int(args.month)

    regridded_root = args.regridded_root or Path(f"/g/data/{args.project}/{args.user}/afim_input/CAWCR")
    output_root = args.output_root or Path(f"/g/data/{args.project}/{args.user}/GRAPHICAL/LD-waves/WHACS/daily")

    src_file = whacs_file(args.source_root, year, month)
    reg_file = regridded_root / f"CAWCR_efreq_for_CICE6_{year:04d}{month:02d}.nc"
    if not src_file.exists():
        raise FileNotFoundError(src_file)
    if not reg_file.exists():
        raise FileNotFoundError(reg_file)

    run = RunSpec(
        sim_name=args.sim_name,
        start_date=f"{year:04d}-{month:02d}-01",
        end_date=f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}",
        hemisphere="SH",
        project=args.project,
        user=args.user,
    )
    paths = ShugaPaths(
        run_cfg=run,
        obs_cfg=ObservationSpec(),
        wave_frcg_cfg=WaveForcingSpec(regridded_wave_root=regridded_root),
        G_cice_cfg=CICEGridSpec(),
        graphics_root=output_root,
    )

    print(f"native WHACS : {src_file}")
    print(f"regridded    : {reg_file}")
    print(f"figures      : {output_root}")

    ds_raw = xr.open_dataset(
        src_file,
        chunks={"time": 24, "station": 512, "frequency": -1, "direction": -1},
    )
    ds_wave = xr.open_dataset(reg_file, chunks={"time": 24})

    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_stop = month_start + pd.offsets.MonthBegin(1)
    for day in pd.date_range(month_start, month_stop, freq="1D", inclusive="left"):
        out = output_root / f"{year:04d}" / f"{month:02d}" / f"WHACS_Hs_Tp_regrid_QC_{day:%Y%m%d}.png"
        if out.exists() and not args.overwrite:
            print(f"exists, skipping: {out}")
            continue
        print(f"plotting {day:%Y-%m-%d}")
        plot_whacs_daily_comparison(
            ds_raw,
            ds_wave,
            day,
            output=out,
            region=[-180.0, 180.0, -90.0, float(args.north)],
            grid_stride=args.grid_stride,
            hs_max=args.hs_max,
            tp_max=args.tp_max,
            paths=paths,
            source_var="efth",
            dpi=args.dpi,
        )

    ds_raw.close()
    ds_wave.close()


if __name__ == "__main__":
    main()
