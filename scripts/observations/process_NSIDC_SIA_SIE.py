#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.observations.NSIDC import NSIDCObservations

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute NSIDC daily SIA/SIE and write a compact Zarr store.")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--hemisphere", default="SH", choices=("SH", "NH"))
    p.add_argument("--seaice-root", type=Path, default=Path("/g/data/gv90/da1339/SeaIce"))
    p.add_argument("--nsidc-version", default="G02202_V6")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--output-store", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p

def main() -> None:
    args    = parser().parse_args()
    run_cfg = RunSpec(sim_name = "NSIDC", start_date = args.start_date, end_date = args.end_date, hemisphere = args.hemisphere)
    cls_cfg = ClassificationSpec(ice_type = "SI", grid_type = "Tb", methods = ("raw",))
    obs_cfg = ObservationSpec(seaice_root = args.seaice_root, nsidc_version = args.nsidc_version, nsidc_threshold = args.threshold)
    pth_cfg = ShugaPaths(run_cfg = run_cfg, cls_cfg = cls_cfg, obs_cfg = obs_cfg)
    nsidc   = NSIDCObservations(run_cfg = run_cfg, obs_cfg = obs_cfg, pth_cfg = pth_cfg, chunks = {"time": args.chunks_time})
    path    = nsidc.process_sia_sie(start_date   = args.start_date,
                                    end_date     = args.end_date,
                                    hemisphere   = args.hemisphere,
                                    threshold    = args.threshold,
                                    output_store = args.output_store,
                                    overwrite    = args.overwrite)
    print(path)

if __name__ == "__main__":
    main()
