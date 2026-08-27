#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import CICEGridSpec, ObservationSpec, RunSpec, WaveForcingSpec
from shuga.waves.cawcr import CAWCRRegridConfig
from shuga.waves.whacs import WHACSRegridder, WHACS_SOURCE_ROOT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Regrid one monthly WHACS directional-spectrum file to hourly CICE25 "
            "wave forcing on the native CICE T grid. No NSIDC/model-ice mask is applied."
        )
    )
    p.add_argument("year", type=int)
    p.add_argument("month", type=int)

    p.add_argument("--project", default="gv90")
    p.add_argument("--user", default="da1339")
    p.add_argument("--sim-name", default="LD-waves-exp01")
    p.add_argument("--hemisphere", default="SH", choices=["SH", "NH"])

    p.add_argument("--source-root", type=Path, default=WHACS_SOURCE_ROOT)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--weights-root", type=Path, default=None)

    p.add_argument("--target-lat-max", type=float, default=-35.0,
                   help="Southern Hemisphere forcing-domain northern limit; cells north of this are zero. Default: -35")
    p.add_argument("--k-nearest", type=int, default=8)
    p.add_argument("--idw-power", type=float, default=2.5)
    p.add_argument("--radius-km", type=float, default=1000.0)
    p.add_argument("--time-chunk", type=int, default=4)
    p.add_argument("--compression-level", type=int, default=3)

    p.add_argument("--overwrite-nc", action="store_true",
                   help="Rebuild even if a completed output exists. Legacy/incomplete files are replaced automatically.")
    p.add_argument("--overwrite-weights", action="store_true")
    p.add_argument("--log-file", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (1 <= args.month <= 12):
        raise SystemExit("month must be in the range 1..12")
    if args.time_chunk < 1:
        raise SystemExit("--time-chunk must be >= 1")
    if not (0 <= args.compression_level <= 9):
        raise SystemExit("--compression-level must be 0..9")

    year = int(args.year)
    month = int(args.month)

    run = RunSpec(
        sim_name=args.sim_name,
        start_date=f"{year:04d}-01-01",
        end_date=f"{year:04d}-12-31",
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )
    obs = ObservationSpec()
    grid = CICEGridSpec()

    output_root = args.output_root or Path(f"/g/data/{args.project}/{args.user}/afim_input/CAWCR")
    weights_root = args.weights_root or Path(f"/g/data/{args.project}/{args.user}/grids/weights")
    waves = WaveForcingSpec(regridded_wave_root=output_root, weights_root=weights_root)

    paths = ShugaPaths(
        run_cfg=run,
        obs_cfg=obs,
        wave_frcg_cfg=waves,
        G_cice_cfg=grid,
    )

    output_path = output_root / f"CAWCR_efreq_for_CICE6_{year:04d}{month:02d}.nc"
    station_weights = weights_root / f"map_WHACSstations_to_ACCESS-OM3-025_idw_k{args.k_nearest}.npz"

    log_file = args.log_file or (
        paths.logs_root_path / "waves" / f"whacs_regrid_{year:04d}{month:02d}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = build_file_logger("shuga.waves.whacs", log_file)

    cfg = CAWCRRegridConfig(
        output_path=output_path,
        source_var="efth",
        station_lon_name="longitude",
        station_lat_name="latitude",
        time_dim="time",
        station_dim="station",
        frequency_dim="frequency",
        direction_dim="direction",
        frequency_lo_name="frequency1",
        frequency_hi_name="frequency2",
        k_nearest=args.k_nearest,
        idw_power=args.idw_power,
        radius_km=args.radius_km,
        hemisphere=args.hemisphere,
        target_lat_max=args.target_lat_max,
        fill_value=0.0,
        weights_path=station_weights,
    )

    worker = WHACSRegridder(cfg, source_root=args.source_root, logger=logger)
    out = worker.prepare_month(
        year,
        month,
        paths=paths,
        overwrite_weights=args.overwrite_weights,
        overwrite_output=args.overwrite_nc,
        time_chunk=args.time_chunk,
        complevel=args.compression_level,
    )
    print(out)


if __name__ == "__main__":
    main()
