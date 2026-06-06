#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shuga.core.types import ClassificationSpec, MetricsSpec, ObservationSpec, PlottingSpec, RunSpec
from shuga.core.paths import ShugaPaths
from shuga.io import load_classified, load_cice
from shuga.plotting.cice import CICEPlotter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot AF2020 FIC and simulation FIC side by side by region. "
            "AF2020 remains on the prebuilt common EPSG:3031 grid and native 15-day timestamps. "
            "Simulation FIC is not regridded; daily simulation FI_mask*aice is nearest-selected/downsampled "
            "to AF2020 timestamps for each frame."
        )
    )
    p.add_argument("-s", "--sim-name", required=True)
    p.add_argument("-b", "--start-date", default="2000-03-01")
    p.add_argument("-e", "--end-date", default="2031-12-31")
    p.add_argument("--classification", default="binary-days")
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
    p.add_argument("--output-root", default=None)
    p.add_argument("--regions", default=None, help="Comma-separated region names; default all 8 Antarctic regions.")
    p.add_argument("--max-frames", type=int, default=None, help="Optional debug limiter.")
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _mask_name(ice_type: str) -> str:
    return f"{ice_type.upper()}_mask"


def main() -> None:
    args = parse_args()

    run = RunSpec(
        sim_name=args.sim_name,
        start_date=args.start_date,
        end_date=args.end_date,
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )
    classify = ClassificationSpec(
        ice_type=args.ice_type,
        grid_type=args.grid_type,
        ispd_thresh=args.ispd_thresh,
        methods=(args.classification,),
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
    )
    metrics = MetricsSpec(methods=(args.classification,))
    plotting = PlottingSpec()
    observations = ObservationSpec()
    paths = ShugaPaths(run=run, classify=classify, metrics=metrics, plotting=plotting, observations=observations)
    plotter = CICEPlotter(run=run, classify=classify, metrics=metrics, plotting=plotting, observations=observations, paths=paths)

    af = xr.open_zarr(Path(args.af2020_store).expanduser(), consolidated=False, chunks={"time": args.chunks_time})
    af_fic = af["FIC"].sel(time=slice(args.start_date, args.end_date))

    cls = load_classified(
        run=run,
        classify=classify,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        classification=args.classification,
        variables=[_mask_name(args.ice_type)],
        hemisphere=args.hemisphere,
        grid_type=args.grid_type,
        chunks={"time": args.chunks_time},
    )
    cice = load_cice(
        run=run,
        classify=classify,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        variables=["aice", "TLON", "TLAT"],
        hemisphere=args.hemisphere,
        chunks={"time": args.chunks_time},
    )

    mname = _mask_name(args.ice_type)
    if mname not in cls and "FI_mask" in cls:
        mname = "FI_mask"

    sim_fic_daily = (cls[mname].astype("float32") * cice["aice"].astype("float32")).rename("SIM_FIC")
    sim_fic = sim_fic_daily.reindex(time=af_fic.time.values, method="nearest")

    from shuga.core.regions import ANTARCTIC_8_REGIONS
    if args.regions:
        region_names = [r.strip() for r in args.regions.split(",") if r.strip()]
    else:
        region_names = list(ANTARCTIC_8_REGIONS)

    out_root = Path(args.output_root).expanduser() if args.output_root else paths.figure_root() / "FIC_pair" / args.sim_name
    out_root.mkdir(parents=True, exist_ok=True)

    pygmt = plotter._require_pygmt()
    times = pd.to_datetime(af_fic.time.values)
    if args.max_frames is not None:
        times = times[: int(args.max_frames)]

    for t in times:
        tstr = pd.Timestamp(t).strftime("%Y-%m-%d")
        af_da = af_fic.sel(time=t)
        sim_da = sim_fic.sel(time=t, method="nearest")

        for rname in region_names:
            reg = ANTARCTIC_8_REGIONS[rname].get("plot_region", ANTARCTIC_8_REGIONS[rname]["geo_region"])
            proj = plotter.projection_from_region(reg, fig_size=10.0)

            af_df = plotter.pygmt_da_prep(af_da.where(af_da > 0), lon=af["lon"], lat=af["lat"], region=reg)
            sim_df = plotter.pygmt_da_prep(sim_da.where(sim_da > 0), lon=cice["TLON"], lat=cice["TLAT"], region=reg)

            fig = pygmt.Figure()
            with fig.subplot(nrows=1, ncols=2, figsize=("22c", "8c"), margins=["0.2c", "0.2c"], frame="lrtb"):
                with fig.set_panel(panel=0):
                    plotter.pygmt_base_layer(fig, reg, proj, title=f"AF2020 FIC {rname} {tstr}")
                    pygmt.makecpt(cmap=str(plotter._default_fip_cmap()), series=[0, 1, 0.05], continuous=True)
                    fig.plot(x=af_df["lon"], y=af_df["lat"], style=plotter.plotting.grid_style, fill=af_df["z"], cmap=True)
                    fig.coast(region=reg, projection=proj, shorelines=plotter.plotting.shorelines)
                with fig.set_panel(panel=1):
                    plotter.pygmt_base_layer(fig, reg, proj, title=f"{args.sim_name} FIC {rname} {tstr}")
                    pygmt.makecpt(cmap=str(plotter._default_fip_cmap()), series=[0, 1, 0.05], continuous=True)
                    fig.plot(x=sim_df["lon"], y=sim_df["lat"], style=plotter.plotting.grid_style, fill=sim_df["z"], cmap=True)
                    fig.coast(region=reg, projection=proj, shorelines=plotter.plotting.shorelines)

            out = out_root / rname / f"FIC_pair_{args.sim_name}_AF2020_{rname}_{tstr}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)
            if args.show:
                fig.show()
            print(out)


if __name__ == "__main__":
    main()
