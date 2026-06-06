#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick-look side-by-side AF2020/model FIC plot from a common-grid FIC sample store.")
    p.add_argument("fic_store", help="Zarr store written by build_af2020_fip_fic_common_grid.py ending in _FIC_samples.zarr")
    p.add_argument("--date", required=True)
    p.add_argument("--region", nargs=4, type=float, default=None, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--out", default=None)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    return p.parse_args()

def _norm_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def _region_mask(ds: xr.Dataset, region):
    lon = _norm_lon(ds["lon"])
    lat = ds["lat"]
    lon_min, lon_max, lat_min, lat_max = region
    if lon_min <= lon_max:
        mlon = (lon >= lon_min) & (lon <= lon_max)
    else:
        mlon = (lon >= lon_min) | (lon <= lon_max)
    return mlon & (lat >= lat_min) & (lat <= lat_max)

def main() -> None:
    args = parse_args()
    ds = xr.open_zarr(args.fic_store, consolidated=False)
    t = pd.Timestamp(args.date)
    ds = ds.sel(time=t, method="nearest")
    if args.region is not None:
        mask = _region_mask(ds, args.region)
        ds = ds.where(mask)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, var, title in zip(axes, ["AF_FIC", "SIM_FIC"], ["AF2020 FIC", "Model FIC"]):
        da = ds[var]
        im = ax.pcolormesh(ds["x"] / 1000.0, ds["y"] / 1000.0, da, vmin=args.vmin, vmax=args.vmax, shading="auto")
        ax.set_title(f"{title}  {pd.to_datetime(str(ds.time.values)).date()}")
        ax.set_xlabel("EPSG:3031 x (km)")
        ax.set_ylabel("EPSG:3031 y (km)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="FIC")
    out = Path(args.out).expanduser() if args.out else Path(args.fic_store).with_suffix(f".{t:%Y%m%d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
