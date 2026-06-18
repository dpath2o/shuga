#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib    import Path
#####################################################################
# make sure this reflects the correct location of mawsons-chest repo
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
#####################################################################
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga              import (CICEMetrics,
                                CICEPlotter,
                                ClassificationSpec,
                                MetricsSpec,
                                ObservationSpec,
                                PlottingSpec,
                                RunSpec,
                                ShugaPaths)
from shuga.core.naming  import normalize_method
from shuga.core.logging import build_file_logger

def _comma_split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute or update shuga metrics from classification outputs and/or CICE history.")
    p.add_argument("--sim-name", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH")
    p.add_argument("--project", default="jk72")
    p.add_argument("--user", default="da1339")
    p.add_argument("--ice-type", default="FI")
    p.add_argument("--grid-type", "--BorC2T-type", dest="grid_type", default="Tc")
    p.add_argument("--ispd-thresh", type=float, default=5e-4)
    p.add_argument("--methods", default="binary-days,rolling-mean")
    p.add_argument("--bin-window", type=int, default=11)
    p.add_argument("--bin-min-days", type=int, default=9)
    p.add_argument("--roll-window", type=int, default=15)
    p.add_argument("--iceh-frequency", choices = ["daily", "hourly"], default = "daily",
                   help="CICE history frequency. daily -> iceh_daily.zarr/YYYY-MM; hourly -> iceh_hourly.zarr/YYYY_MM_DD.")
    p.add_argument("--cice-store", default=None)
    p.add_argument("--static-store", default=None)
    p.add_argument("--classification-root", default=None)
    p.add_argument("--afim-output-root", default=None)
    p.add_argument("--graphics-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--obs-metrics-store", default=None)
    p.add_argument("--obs-fia-var", default="FIA")
    p.add_argument("--obs-fit-var", default="FIT")
    p.add_argument("--rebuild-on-index-mismatch", action="store_true")
    p.add_argument("--coast-distance-var", default=None)
    p.add_argument("--seaice-root", default=None)
    p.add_argument("--nsidc-root", default=None)
    p.add_argument("--nsidc-cellarea-root", default=None)
    p.add_argument("--af2020-root", default=None)
    p.add_argument("--metric-groups",
                   default="default",
                   help=("Comma-separated metric groups to compute/update. "
                         "Use domain-specific groups with the new split stores: "
                         "fi_core, fi_regional, fi_spatial, fi_summary, fi_stress, fi_diags, fi_all; "
                         "pi_core, pi_regional, pi_spatial, pi_summary, pi_stress, pi_diags, pi_all; "
                         "si_core, si_regional, si_spatial, si_summary, si_stress, si_diags, si_all. "
                         "Cross-domain groups regional, spatial, summary, stress, diags, all are for diagnostics only "
                         "and should not be used with split-domain metrics writing."))
    p.add_argument("--metric-names",
                   default=None,
                   help="Comma-separated explicit metric names to compute/update in addition to metric-groups.")
    p.add_argument("--update-missing-only",
                   action="store_true",
                   help="Only compute requested metrics that are absent from the existing mets.zarr store.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild the metrics store from scratch for the requested set.")
    p.add_argument("--plot-fip", action="store_true")
    p.add_argument("--plot-fia", action="store_true")
    p.add_argument("--plot-fit", action="store_true")
    p.add_argument("--plot-sia", action="store_true")
    p.add_argument("--plot-sit", action="store_true")
    p.add_argument("--plot-region", default="total")
    p.add_argument("--fip-region", default=None)
    p.add_argument("--fig-size", type=float, default=20.0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p

def main() -> None:
    args          = build_parser().parse_args()
    methods       = [normalize_method(m) for m in _comma_split(args.methods)]
    metric_groups = _comma_split(args.metric_groups)
    metric_names  = _comma_split(args.metric_names)
    run_cfg       = RunSpec(sim_name       = args.sim_name,
                            start_date     = args.start_date,
                            end_date       = args.end_date,
                            hemisphere     = args.hemisphere,
                            project        = args.project,
                            user           = args.user,
                            iceh_frequency = args.iceh_frequency)
    cls_cfg       = ClassificationSpec(ice_type     = args.ice_type,
                                       grid_type    = args.grid_type,
                                       ispd_thresh  = args.ispd_thresh,
                                       methods      = tuple(methods),
                                       bin_window   = args.bin_window,
                                       bin_min_days = args.bin_min_days,
                                       roll_window  = args.roll_window)
    met_cfg       = MetricsSpec(methods            = tuple(methods),
                                obs_metrics_store  = args.obs_metrics_store,
                                obs_fia_var        = args.obs_fia_var,
                                obs_fit_var        = args.obs_fit_var,
                                coast_distance_var = args.coast_distance_var)
    obs_cfg       = ObservationSpec(seaice_root         = args.seaice_root,
                                    nsidc_root          = args.nsidc_root,
                                    nsidc_cellarea_root = args.nsidc_cellarea_root,
                                    af2020_root         = args.af2020_root)
    plt_cfg       = PlottingSpec(fig_size        = args.fig_size,
                                 fip_fig_size    = args.fig_size,
                                 region_fig_size = args.fig_size)
    pth_cfg       = ShugaPaths(run_cfg                 = run_cfg,
                               cls_cfg            = cls_cfg,
                               obs_cfg        = obs_cfg,
                               afim_output_root    = args.afim_output_root,
                               graphics_root       = args.graphics_root,
                               cice_store          = args.cice_store,
                               static_store        = args.static_store,
                               classification_root = args.classification_root,
                               logs_root           = args.logs_root)
    logger        = build_file_logger("shuga.met_cfg", pth_cfg.metrics_log_path(), level=args.log_level)
    logger.info("Logging to: %s", pth_cfg.metrics_log_path())
    logger.info("Resolved CICE store: %s", pth_cfg.resolve_cice_store())
    static_store  = pth_cfg.resolve_static_store()
    if static_store is None:
        logger.warning("No CICE static store resolved. Metrics requiring tarea/TLON/TLAT or face areas may fail.")
    logger.info("Resolved classification root: %s", pth_cfg.classification_root_path)
    runner              = CICEMetrics(run_cfg = run_cfg, cls_cfg = cls_cfg, met_cfg = met_cfg, pth_cfg = pth_cfg, logger = logger)
    plotter             = CICEPlotter(run_cfg = run_cfg,
                                      cls_cfg = cls_cfg,
                                      met_cfg = met_cfg,
                                      plt_cfg = plt_cfg,
                                      obs_cfg = obs_cfg,
                                      pth_cfg = pth_cfg,
                                      logger  = logger)
    update_missing_only = args.update_missing_only or (not args.overwrite)
    for method in methods:
        logger.info("Processing class method: %s", method)
        runner.compute_metrics(method,
                               overwrite                 = args.overwrite,
                               metric_groups             = metric_groups,
                               metric_names              = metric_names,
                               update_missing_only       = args.update_missing_only or not args.overwrite,
                               rebuild_on_index_mismatch = args.rebuild_on_index_mismatch)
        if args.plot_fip:
            logger.info("Wrote FIP plot(s): %s", plotter.plot_fip(method, region_name=args.fip_region if args.fip_region not in {None, "", "total"} else None))
        if args.plot_fia:
            logger.info("Wrote FIA plot: %s", plotter.plot_timeseries("FIA", method, region=args.plot_region))
        if args.plot_fit:
            logger.info("Wrote FIT plot: %s", plotter.plot_timeseries("FIT", method, region=args.plot_region, add_f2020=False))
        if args.plot_sia:
            logger.info("Wrote SIA plot: %s", plotter.plot_timeseries("SIA", method, region=args.plot_region))
        if args.plot_sit:
            logger.info("Wrote SIT plot: %s", plotter.plot_timeseries("SIT", method, region=args.plot_region, add_f2020=False))

if __name__ == "__main__":
    main()
