#!/usr/bin/env python3
"""Plot Antarctic SIA daily climatology against NSIDC, OSI-SAF-450 and CMEMS."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.observations.NSIDC import NSIDCObservations
from shuga.plotting.cice import CICEPlotter


DEFAULT_EXPERIMENTS = {
    "no-slip-LFI": "LFI rheology without lateral drag",
    "Cs-high": "static high Cs",
    "Cq-high": "quadratic high Cq",
}

REFERENCE_ORDER = ["NSIDC", "OSI-SAF-450", "CMEMS"]

REFERENCE_COLORS = {
    "CMEMS": "#61D97B",
}

MODEL_COLORS = {
    "no-slip-LFI": "#D55E00",
    "Cs-high": "#E69F00",
    "Cq-high": "#FF99CC",
}


def _parse_experiments(value: str | None) -> dict[str, str]:
    """
    Parse comma-separated SIM=LABEL entries.

    Examples
    --------
    --experiments "ry93=ry93,elps-min=elps-min"
    --experiments "no-slip-LFI=LFI rheology,Cs-high=static high Cs"
    """
    if value is None or not value.strip():
        return dict(DEFAULT_EXPERIMENTS)

    out: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            sim, label = item.split("=", 1)
            sim = sim.strip()
            label = label.strip()
        else:
            sim = item
            label = item
        if not sim:
            raise ValueError(f"Invalid experiment specification: {item!r}")
        out[sim] = label
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot SIA climatology for standalone CICE experiments against "
            "NSIDC, OSI-SAF-450 and CMEMS."
        )
    )
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)

    p.add_argument(
        "--paper-root",
        type=Path,
        required=True,
        help=(
            "Root containing experiment directories, e.g. "
            "/g/data/gv90/da1339/afim_output/paper1 or "
            "/g/data/gv90/da1339/afim_output"
        ),
    )
    p.add_argument(
        "--experiments",
        default=None,
        help=(
            "Comma-separated SIM=LABEL entries. If omitted, defaults to "
            "no-slip-LFI,Cs-high,Cq-high."
        ),
    )

    p.add_argument(
        "--seaice-root",
        type=Path,
        default=Path("/g/data/gv90/da1339/SeaIce"),
    )
    p.add_argument("--nsidc-store", type=Path)
    p.add_argument("--osisaf-store", type=Path)
    p.add_argument(
        "--cmems-store",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily/SH/SI/mets.zarr"
        ),
    )

    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--envelope",
        choices=("minmax", "std", "p10-p90"),
        default="minmax",
    )
    p.add_argument("--smooth-days", type=int, default=1)
    p.add_argument("--y-max", type=float, default=18.0)
    return p


def main() -> None:
    args = parser().parse_args()
    experiments = _parse_experiments(args.experiments)

    run_cfg = RunSpec(
        sim_name="SIA-comparison",
        start_date=args.start_date,
        end_date=args.end_date,
        hemisphere="SH",
    )
    cls_cfg = ClassificationSpec(
        ice_type="SI",
        grid_type="Tb",
        methods=("raw",),
    )
    obs_cfg = ObservationSpec(
        seaice_root=args.seaice_root,
        nsidc_version="G02202_V6",
    )
    pth_cfg = ShugaPaths(
        run_cfg=run_cfg,
        cls_cfg=cls_cfg,
        obs_cfg=obs_cfg,
        afim_output_root=args.paper_root,
        graphics_root=args.output.parent,
    )

    plotter = CICEPlotter(
        run_cfg=run_cfg,
        cls_cfg=cls_cfg,
        obs_cfg=obs_cfg,
        pth_cfg=pth_cfg,
    )
    nsidc = NSIDCObservations(
        run_cfg=run_cfg,
        obs_cfg=obs_cfg,
        pth_cfg=pth_cfg,
    )

    nsidc_store = (
        args.nsidc_store
        or nsidc.processed_sia_sie_store("SH")
    )
    osisaf_store = (
        args.osisaf_store
        or args.seaice_root
        / "OSI-SAF-450"
        / "processed"
        / "OSI-SAF-450_SH_SIA.zarr"
    )

    series: dict[str, pd.Series] = {}

    series["NSIDC"] = plotter.load_sia_store(
        nsidc_store,
        label="NSIDC",
        start_date=args.start_date,
        end_date=args.end_date,
        variable="SIA",
    )

    series["OSI-SAF-450"] = plotter.load_sia_store(
        osisaf_store,
        label="OSI-SAF-450",
        start_date=args.start_date,
        end_date=args.end_date,
        variable="sia",
    )

    series["CMEMS"] = plotter.load_sia_store(
        args.cmems_store,
        label="CMEMS",
        start_date=args.start_date,
        end_date=args.end_date,
        variable="SIA",
    )

    for sim_name, label in experiments.items():
        sim_run = RunSpec(
            sim_name=sim_name,
            start_date=args.start_date,
            end_date=args.end_date,
            hemisphere="SH",
        )
        sim_cls = ClassificationSpec(
            ice_type="SI",
            grid_type="Tb",
            methods=("raw",),
        )
        sim_paths = ShugaPaths(
            run_cfg=sim_run,
            cls_cfg=sim_cls,
            afim_output_root=args.paper_root,
        )
        store = sim_paths.output_root / "zarr" / "SH" / "SI" / "mets.zarr"

        series[label] = plotter.load_sia_store(
            store,
            label=label,
            start_date=args.start_date,
            end_date=args.end_date,
            variable="SIA",
        )

    df = pd.concat(series.values(), axis=1).sort_index()

    model_labels = list(experiments.values())
    order = [
        name
        for name in (REFERENCE_ORDER + model_labels)
        if name in df.columns
    ]

    colors = dict(MODEL_COLORS)
    colors.update(REFERENCE_COLORS)

    plotter.plot_sia_daily_climatology_envelope(
        df,
        args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        envelope=args.envelope,
        smooth_days=args.smooth_days,
        order=order,
        colors=colors,
        y_min=0.0,
        y_max=args.y_max,
        title=None,
        write_csv=True,
    )


if __name__ == "__main__":
    main()
