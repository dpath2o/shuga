#!/usr/bin/env python3
"""Plot Antarctic SIA daily climatology from precomputed observational/model stores."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.observations.NSIDC import NSIDCObservations
from shuga.plotting.cice import CICEPlotter

# EXPS_DICT = {"AOM2-ERA5" : "ACCESS-OM2-ERA5",
#              "notensnogi": "notens-nogi",
#              "ry93"      : "ry93",
#              "elps-min"  : "elps-min"}

EXPS_DICT = {"Cs-high"     : "static high Cs (1e-4)",
             "Cq-high"     : "quadratic high Cq (750)",
             "no-slip-LFI" : "LFI rheology without lateral drag "}

PLOT_ORDER = ["NSIDC", "OSI-SAF-450", "ORAS", "no-slip-LFI", "Cs-high", "Cq-high"]#"ACCESS-OM2-ERA5", "notens-nogi", "ry93", "elps-min"]

# Only model colours belong here. NSIDC, OSI-SAF-450 and ORAS are hard-wired
# in shuga.plotting.cice.DEFAULT_SIA_STYLES.
# SIA_COLORS = {"ACCESS-OM2-ERA5": "#E69F00",
#               "notens-nogi"    : "#FF99CC",
#               "ry93"           : "#D55E00",
#               "elps-min"       : "#0072B2"}
SIA_COLORS = {"Cs-high"     : "#E69F00",
              "Cq-high"     : "#FF99CC",
              "no-slip-LFI" : "#D55E00"}

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="1994-01-01")
    p.add_argument("--end-date", default="1999-12-31")
    p.add_argument("--model-root", type=Path, default=Path("/g/data/gv90/da1339/afim_output/paper1"))
    p.add_argument("--seaice-root", type=Path, default=Path("/g/data/gv90/da1339/SeaIce"))
    p.add_argument("--nsidc-store", type=Path)
    p.add_argument("--osisaf-store", type=Path)
    p.add_argument("--oras-store", type=Path, default=Path("/g/data/gv90/da1339/SeaIce/CMEMS/CMEMS_SIA_SIV_SIT_1994-1999.zarr"))
    p.add_argument("--output", type=Path, default=Path("/g/data/gv90/da1339/GRAPHICAL/paper1/SIA_SH_climatology_1994-1999_with_OSISAF450.png"))
    p.add_argument("--envelope", choices=("minmax","std","p10-p90"), default="minmax")
    p.add_argument("--smooth-days", type=int, default=1)
    p.add_argument("--y-max", type=float, default=18.0)
    return p

def main() -> None:
    args         = parser().parse_args()
    run_cfg      = RunSpec(sim_name="SIA-comparison", start_date=args.start_date, end_date=args.end_date, hemisphere="SH")
    cls_cfg      = ClassificationSpec(ice_type="SI", grid_type="Tb", methods=("raw",))
    obs_cfg      = ObservationSpec(seaice_root=args.seaice_root, nsidc_version="G02202_V6")
    pth_cfg      = ShugaPaths(run_cfg=run_cfg, cls_cfg=cls_cfg, obs_cfg=obs_cfg, afim_output_root=args.model_root, graphics_root=args.output.parent)
    plotter      = CICEPlotter(run_cfg=run_cfg, cls_cfg=cls_cfg, obs_cfg=obs_cfg, pth_cfg=pth_cfg)
    nsidc        = NSIDCObservations(run_cfg=run_cfg, obs_cfg=obs_cfg, pth_cfg=pth_cfg)
    nsidc_store  = args.nsidc_store or nsidc.processed_sia_sie_store("SH")
    osisaf_store = args.osisaf_store or (args.seaice_root / "OSI-SAF-450" / "processed" / "OSI-SAF-450_SH_SIA.zarr")
    series: dict[str, pd.Series] = {}
    series["NSIDC"]       = plotter.load_sia_store(nsidc_store, label="NSIDC", start_date=args.start_date, end_date=args.end_date, variable="SIA")
    series["OSI-SAF-450"] = plotter.load_sia_store(osisaf_store, label="OSI-SAF-450", start_date=args.start_date, end_date=args.end_date)
    series["ORAS"]        = plotter.load_sia_store(args.oras_store, label="ORAS", start_date=args.start_date, end_date=args.end_date)
    for sim_name, label in EXPS_DICT.items():
        sim_run       = RunSpec(sim_name=sim_name, start_date=args.start_date, end_date=args.end_date, hemisphere="SH")
        sim_cls       = ClassificationSpec(ice_type="SI", grid_type="Tb", methods=("raw",))
        sim_paths     = ShugaPaths(run_cfg=sim_run, cls_cfg=sim_cls, afim_output_root=args.model_root)
        store         = sim_paths.output_root / "zarr" / "SH" / "SI" / "mets.zarr"
        series[label] = plotter.load_sia_store(store, label=label, start_date=args.start_date, end_date=args.end_date, variable="SIA")
    df    = pd.concat(series.values(), axis=1).sort_index()
    order = [name for name in PLOT_ORDER if name in df.columns]
    plotter.plot_sia_daily_climatology_envelope(df, args.output,
                                                start_date  = args.start_date,
                                                end_date    = args.end_date,
                                                envelope    = args.envelope,
                                                smooth_days = args.smooth_days,
                                                order       = order,
                                                colors      = SIA_COLORS,
                                                y_min       = 0.0,
                                                y_max       = args.y_max,
                                                title       = None,
                                                write_csv   = True)

if __name__ == "__main__":
    main()
