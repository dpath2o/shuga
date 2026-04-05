#!/usr/bin/env python3
from __future__ import annotations

import argparse

from shugga import ClassificationSpec, CICEClassifier, RunSpec, ShuggaPaths
from shugga.core.logging import build_file_logger


def _comma_split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify CICE fast ice into raw, binary-days, and rolling-mean masks.")
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--BorC2T-type", default="Tc")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--methods", default="raw,binary-days,rolling-mean")
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--aice-thresh", type=float, default=0.15)
    p.add_argument("--speed-var-u", default="uvel")
    p.add_argument("--speed-var-v", default="vvel")
    p.add_argument("--cice-store", default=None)
    p.add_argument("--static-store", default=None)
    p.add_argument("--classification-root", default=None)
    p.add_argument("--afim-output-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--overwrite", action="store_true")
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
        speed_var_u=args.speed_var_u,
        speed_var_v=args.speed_var_v,
        aice_thresh=args.aice_thresh,
    )
    paths = ShuggaPaths(
        run=run,
        classify=classify,
        afim_output_root=args.afim_output_root,
        cice_store=args.cice_store,
        static_store=args.static_store,
        classification_root=args.classification_root,
        logs_root=args.logs_root,
    )
    logger = build_file_logger("shugga.classify", paths.classification_log_path(), level=args.log_level)
    logger.info("Logging to: %s", paths.classification_log_path())
    logger.info("Resolved CICE store: %s", paths.resolve_cice_store())
    static_store = paths.resolve_static_store()
    if static_store is not None:
        logger.info("Resolved static store: %s", static_store)
    logger.info("Resolved classification root: %s", paths.classification_root_path)

    runner = CICEClassifier(run=run, classify=classify, paths=paths, logger=logger)
    outputs = runner.run_methods(methods=methods, overwrite=args.overwrite)
    for method, path in outputs.items():
        logger.info("Wrote %s classification: %s", method, path)


if __name__ == "__main__":
    main()
