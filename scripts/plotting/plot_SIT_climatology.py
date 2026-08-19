#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.plotting.cice import CICEPlotter

DEFAULT_EXPERIMENTS = {"no-slip-LFI": "LFI rheology without lateral drag",
                       "Cs-high": "static high Cs",
                       "Cq-high": "quadratic high Cq"}
MODEL_PALETTE       = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00", "#56B4E9", "#F0E442"]
REFERENCE_COLORS    = {"ESA-CCI/AWI": "#AA33CC",
                       "CMEMS"      : "#61D97B"}

def parse_experiments(value):
    if not value:
        return dict(DEFAULT_EXPERIMENTS)
    out = {}
    for item in value.split(","):
        if "=" in item:
            sim, label = item.split("=", 1)
        else:
            sim = label = item
        out[sim.strip()] = label.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--hemisphere", choices=("SH","NH"), default="SH")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--experiments")
    ap.add_argument("--obs-store", type=Path, required=True)
    ap.add_argument("--cmems-store", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--envelope", default="minmax", choices=("minmax","std","p10-p90"))
    ap.add_argument("--smooth-days", type=int, default=7)
    ap.add_argument("--y-max", type=float, default=4.0)
    args    = ap.parse_args()
    exps    = parse_experiments(args.experiments)
    run_cfg = RunSpec(sim_name = "SIT-comparison", start_date = args.start_date, end_date = args.end_date, hemisphere = args.hemisphere)
    cls_cfg = ClassificationSpec(ice_type = "SI", methods = ("raw",))
    obs_cfg = ObservationSpec()
    pth_cfg = ShugaPaths(run_cfg = run_cfg, cls_cfg = cls_cfg, obs_cfg = obs_cfg, afim_output_root = args.data_root, graphics_root = args.output.parent)
    plotter = CICEPlotter(run_cfg = run_cfg, cls_cfg = cls_cfg, obs_cfg = obs_cfg, pth_cfg = pth_cfg)
    series  = {}
    series["ESA-CCI/AWI"] = plotter.load_sit_store(args.obs_store, label = "ESA-CCI/AWI", start_date = args.start_date, end_date = args.end_date, variable = "SIT")
    series["CMEMS"] = plotter.load_sit_store(args.cmems_store, label = "CMEMS", start_date = args.start_date, end_date = args.end_date, variable = "SIT")
    for sim_name, label in exps.items():
        sim_run   = RunSpec(sim_name = sim_name, start_date = args.start_date, end_date = args.end_date, hemisphere = args.hemisphere)
        sim_cls   = ClassificationSpec(ice_type = "SI", methods = ("raw",))
        sim_paths = ShugaPaths(run_cfg = sim_run, cls_cfg = sim_cls, afim_output_root = args.data_root)
        store     = sim_paths.output_root/"zarr"/args.hemisphere/"SI"/"mets.zarr"
        series[label] = plotter.load_sit_store(store, label = label, start_date = args.start_date, end_date = args.end_date, variable = "SIT")
    df           = pd.concat(series.values(), axis=1).sort_index()
    model_labels = list(exps.values())
    colors       = dict(REFERENCE_COLORS)
    colors.update({label: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, label in enumerate(model_labels)})
    plotter.plot_sit_daily_climatology_envelope(df, args.output,
                                                start_date  = args.start_date,
                                                end_date    = args.end_date,
                                                envelope    = args.envelope,
                                                smooth_days = args.smooth_days,
                                                order       = ["ESA-CCI/AWI","CMEMS"] + model_labels,
                                                colors      = colors,
                                                y_min       = 0.0,
                                                y_max       = args.y_max,
                                                title       = None,
                                                write_csv   = True)

if __name__ == "__main__":
    main()
