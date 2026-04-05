#!/usr/bin/env python3
from __future__ import annotations

import argparse

from shugga import CICEMetrics, ClassificationSpec, MetricsSpec, RunSpec, ShuggaPaths
from shugga.core.logging import build_file_logger


def _comma_split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute fast-ice metrics from shugga classification outputs.")
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--BorC2T-type", default="Tc")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--methods", default="binary-days,rolling-mean")
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--cice-store", default=None)
    p.add_argument("--static-store", default=None)
    p.add_argument("--classification-root", default=None)
    p.add_argument("--afim-output-root", default=None)
    p.add_argument("--graphics-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--obs-metrics-store", default=None)
    p.add_argument("--obs-fia-var", default="FIA")
    p.add_argument("--obs-fit-var", default="FIT")
    p.add_argument("--coast-distance-var", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot-fip", action="store_true")
    p.add_argument("--plot-fia", action="store_true")
    p.add_argument("--plot-fit", action="store_true")
    p.add_argument("--plot-region", default="total")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    methods = _comma_split(args.methods)

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
        grid_type=args.BorC2T_type,
        ispd_thresh=args.ispd_thresh,
        methods=tuple(methods),
        bin_window=args.bin_window,
        bin_min_days=args.bin_min_days,
        roll_window=args.roll_window,
    )
    metrics = MetricsSpec(
        obs_metrics_store=args.obs_metrics_store,
        obs_fia_var=args.obs_fia_var,
        obs_fit_var=args.obs_fit_var,
        coast_distance_var=args.coast_distance_var,
    )
    paths = ShuggaPaths(
        run=run,
        classify=classify,
        afim_output_root=args.afim_output_root,
        graphics_root=args.graphics_root,
        cice_store=args.cice_store,
        static_store=args.static_store,
        classification_root=args.classification_root,
        logs_root=args.logs_root,
    )

    logger = build_file_logger("shugga.metrics", paths.metrics_log_path(), level=args.log_level)
    logger.info("Logging to: %s", paths.metrics_log_path())
    logger.info("Resolved CICE store: %s", paths.resolve_cice_store())
    static_store = paths.resolve_static_store()
    if static_store is not None:
        logger.info("Resolved static store: %s", static_store)
    logger.info("Resolved classification root: %s", paths.classification_root_path)

    runner = CICEMetrics(run=run, classify=classify, metrics=metrics, paths=paths, logger=logger)
    for method in methods:
        logger.info("Processing class method: %s", method)
        runner.compute_metrics(method, overwrite=args.overwrite)
        if args.plot_fip:
            logger.info("Wrote FIP plot: %s", runner.plot_fip(method))
        if args.plot_fia:
            logger.info("Wrote FIA plot: %s", runner.plot_timeseries("FIA", method, region=args.plot_region))
        if args.plot_fit:
            logger.info("Wrote FIT plot: %s", runner.plot_timeseries("FIT", method, region=args.plot_region))


if __name__ == "__main__":
    main()
