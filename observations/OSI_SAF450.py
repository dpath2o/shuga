"""Download and process Antarctic sea-ice area from OSI-SAF / Copernicus Marine.

This module targets Copernicus Marine product
SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009 and, by default, the southern
hemisphere OSI-450-a1 CDR dataset exposed as
OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS.

The public API mirrors the small, script-friendly style used by shuga
observation modules: download files, open them with xarray, and reduce daily
sea-ice concentration to a pan-Antarctic SIA time series.
"""
from __future__ import annotations
import argparse, glob, logging, os, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

PRODUCT_ID = "SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009"
DEFAULT_DATASET_ID = "OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS"
DEFAULT_CONT_DATASET_ID = "OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-SH-LA-OBS"
DEFAULT_AMSR_CDR_SH_DATASET_ID = "osisaf_obs-si_glo_phy_sic-south_my_amsr_cdr_P1D-m"
DEFAULT_AMSR_ICDR_SH_DATASET_ID = "osisaf_obs-si_glo_phy_sic-south_my_amsr_icdr_P1D-m"

CONC_CANDIDATES = ("ice_conc",
                   "sea_ice_area_fraction",
                   "siconc",
                   "sic",
                   "concentration",
                   "cdr_seaice_conc")
LAT_CANDIDATES  = ("lat", "latitude", "TLAT")
LON_CANDIDATES  = ("lon", "longitude", "TLON")
AREA_CANDIDATES = ("cell_area", "area", "areacello", "tarea", "cellarea")
X_CANDIDATES    = ("xc", "x", "projection_x_coordinate")
Y_CANDIDATES    = ("yc", "y", "projection_y_coordinate")


@dataclass(frozen=True)
class OSISAF450Config:
    raw_dir: Path
    processed_dir: Path
    product_id: str = PRODUCT_ID
    dataset_id: str = DEFAULT_DATASET_ID
    hemisphere: str = "SH"
    concentration_threshold: float = 15.0
    overwrite: bool = False

    @property
    def sia_store(self) -> Path:
        hemi = self.hemisphere.upper()
        return self.processed_dir / f"OSI-SAF-450_{hemi}_SIA.zarr"

def _first_name(ds: xr.Dataset, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in ds:
            return name
        if name in ds.coords:
            return name
    lower = {name.lower(): name for name in list(ds.data_vars) + list(ds.coords)}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None

def _find_concentration_name(ds: xr.Dataset) -> str:
    name = _first_name(ds, CONC_CANDIDATES)
    if name is not None:
        return name
    for var in ds.data_vars:
        attrs = {k.lower(): str(v).lower() for k, v in ds[var].attrs.items()}
        std = attrs.get("standard_name", "")
        long = attrs.get("long_name", "")
        units = attrs.get("units", "")
        if "sea_ice_area_fraction" in std or "sea ice concentration" in long:
            return var
        if ("ice" in long and "concentration" in long) or units in {"%", "percent", "1"}:
            dims = set(ds[var].dims)
            if "time" in dims or ds[var].ndim >= 2:
                return var
    raise KeyError("Could not identify an OSI-SAF sea-ice concentration variable. Available variables: {list(ds.data_vars)}")

def _normalise_fraction(conc: xr.DataArray) -> xr.DataArray:
    """Return concentration as 0--1 fraction, preserving NaNs/masks."""
    units = str(conc.attrs.get("units", "")).strip().lower()
    valid = conc.where(np.isfinite(conc))
    vmax = float(valid.max(skipna=True).compute()) if valid.size else np.nan
    if units in {"%", "percent", "percentage"} or vmax > 1.5:
        valid = valid / 100.0
    return valid.clip(min=0.0, max=1.0)

def _infer_cell_area(ds: xr.Dataset, conc: xr.DataArray) -> xr.DataArray:
    """Infer grid-cell area in m2.

    Preference order:
      1. existing cell-area variable;
      2. product grid x/y spacing in metres;
      3. OSI-SAF nominal 25 km grid-cell area fallback.
    """
    area_name = _first_name(ds, AREA_CANDIDATES)
    if area_name is not None:
        area = ds[area_name]
        units = str(area.attrs.get("units", "")).lower()
        if units in {"km2", "km^2", "square_kilometers", "square kilometres"}:
            area = area * 1.0e6
        return area.rename("cell_area")
    x_name = _first_name(ds, X_CANDIDATES)
    y_name = _first_name(ds, Y_CANDIDATES)
    if x_name and y_name and ds[x_name].ndim == 1 and ds[y_name].ndim == 1:
        x = ds[x_name].astype(float)
        y = ds[y_name].astype(float)
        dx = float(abs(x.diff(x.dims[0]).median(skipna=True)))
        dy = float(abs(y.diff(y.dims[0]).median(skipna=True)))
        # Coordinates should be metres for LAEA polar grids. If they look like km,
        # promote to metres.
        if dx < 1_000 and dy < 1_000:
            dx *= 1_000.0
            dy *= 1_000.0
        spatial_dims = [d for d in conc.dims if d != "time"]
        if len(spatial_dims) >= 2:
            template = conc.isel({"time": 0}, drop=True) if "time" in conc.dims else conc
            return xr.ones_like(template, dtype="float64").rename("cell_area") * dx * dy
    LOGGER.warning("Falling back to nominal OSI-SAF 25 km x 25 km cell area.")
    template = conc.isel({"time": 0}, drop=True) if "time" in conc.dims else conc
    return xr.ones_like(template, dtype="float64").rename("cell_area") * (25_000.0**2)

def _hemisphere_mask(ds: xr.Dataset, conc: xr.DataArray, hemisphere: str) -> xr.DataArray | None:
    lat_name = _first_name(ds, LAT_CANDIDATES)
    if lat_name is None:
        return None
    lat = ds[lat_name]
    hemi = hemisphere.upper()
    if hemi == "SH":
        return lat < 0
    if hemi == "NH":
        return lat > 0
    return xr.ones_like(lat, dtype=bool)

def open_osisaf_files(path_pattern: str | Path | Sequence[str | Path], chunks: dict | None = None) -> xr.Dataset:
    """Open downloaded OSI-SAF NetCDF files as one xarray Dataset."""
    if isinstance(path_pattern, (str, Path)):
        paths = sorted(glob.glob(str(path_pattern)))
    else:
        paths = sorted(str(p) for p in path_pattern)
    if not paths:
        raise FileNotFoundError(f"No OSI-SAF NetCDF files matched {path_pattern!r}")
    return xr.open_mfdataset(paths, combine="by_coords", chunks=chunks or {"time": 31})

def compute_sia(ds: xr.Dataset, hemisphere: str = "SH", concentration_threshold: float = 15.0) -> xr.Dataset:
    """Compute daily sea-ice area from OSI-SAF concentration.

    Parameters
    ----------
    ds
        OSI-SAF/Copernicus xarray dataset.
    hemisphere
        ``SH`` by default. If latitude is available, the opposite hemisphere is
        masked. For southern-only datasets this has no practical effect.
    concentration_threshold
        Ice-edge threshold in percent. SIA is the sum of fractional ice-covered
        cell area for grid cells whose concentration is at least this threshold.

    Returns
    -------
    xarray.Dataset
        ``sia`` in 10^6 km^2 and ``sia_m2`` in m^2 indexed by time.
    """
    conc_name = _find_concentration_name(ds)
    conc = _normalise_fraction(ds[conc_name])
    area = _infer_cell_area(ds, conc)
    mask = xr.ones_like(conc, dtype=bool)
    hemi_mask = _hemisphere_mask(ds, conc, hemisphere)
    if hemi_mask is not None:
        mask = mask & hemi_mask
    threshold = float(concentration_threshold) / 100.0
    ice_area = conc.where(mask & (conc >= threshold), 0.0) * area
    spatial_dims = [d for d in ice_area.dims if d != "time"]
    sia_m2 = ice_area.sum(dim=spatial_dims, skipna=True).rename("sia_m2")
    sia = (sia_m2 / 1.0e12).rename("sia")
    sia.attrs.update(units="10^6 km^2", long_name="sea ice area")
    sia_m2.attrs.update(units="m2", long_name="sea ice area")
    out = xr.Dataset({"sia": sia, "sia_m2": sia_m2})
    out.attrs.update(source                          = "OSI-SAF/Copernicus Marine SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009",
                     concentration_variable          = conc_name,
                     concentration_threshold_percent = float(concentration_threshold),
                     hemisphere                      = hemisphere.upper())
    return out

def process_downloaded_files(raw_dir: str | Path,
                             processed_dir: str | Path,
                             hemisphere: str = "SH",
                             start_date: str | None = None,
                             end_date: str | None = None,
                             concentration_threshold: float = 15.0,
                             overwrite: bool = False,
                             chunks_time: int = 31) -> Path:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_store = processed_dir / f"OSI-SAF-450_{hemisphere.upper()}_SIA.zarr"
    if out_store.exists() and not overwrite:
        LOGGER.info("Output exists and overwrite=False: %s", out_store)
        return out_store
    ds = open_osisaf_files(raw_dir / "**" / "*.nc", chunks={"time": chunks_time})
    if start_date or end_date:
        ds = ds.sel(time=slice(start_date, end_date))
    sia = compute_sia(ds, hemisphere=hemisphere, concentration_threshold=concentration_threshold)
    sia.to_zarr(out_store, mode="w", consolidated=False)
    LOGGER.info("Wrote %s", out_store)
    return out_store

def download_with_copernicusmarine( output_dir: str | Path, start_date: str, end_date: str,
                                    dataset_id: str = DEFAULT_DATASET_ID,
                                    product_id: str = PRODUCT_ID,
                                    variables: Sequence[str] | None = None,
                                    username: str | None = None,
                                    password: str | None = None,
                                    overwrite: bool = False) -> None:
    """Download OSI-SAF files using the Copernicus Marine Toolbox CLI.

    Credentials are optional here. If omitted, the CLI uses its configured login
    or COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD environment variables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["copernicusmarine", "subset", "--product-id", product_id, "--dataset-id", dataset_id,
           "--start-datetime",f"{start_date}T00:00:00", "--end-datetime", f"{end_date}T23:59:59",
           "--minimum-latitude", "-90", "--maximum-latitude", "0",
           "--minimum-longitude", "-180", "--maximum-longitude", "180",
           "--output-directory", str(output_dir)]
    for var in variables or []:
        cmd.extend(["--variable", var])
    if overwrite:
        cmd.append("--force-download")
    if username:
        cmd.extend(["--username", username])
    if password:
        cmd.extend(["--password", password])
    LOGGER.info("Running: %s", " ".join(cmd[:12]) + " ...")
    subprocess.run(cmd, check=True)

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download/process OSI-SAF-450 Antarctic SIA.")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--product-id", default=PRODUCT_ID)
    p.add_argument("--hemisphere", default="SH", choices=["SH", "NH", "both", "global"])
    p.add_argument("--concentration-threshold", type=float, default=15.0)
    p.add_argument("--download", action="store_true")
    p.add_argument("--process", action="store_true")
    p.add_argument("--variable", action="append", default=[])
    p.add_argument("--username", default=os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"))
    p.add_argument("--password", default=os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--log-level", default="INFO")
    return p

def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    if not args.download and not args.process:
        args.download = True
        args.process = True
    if args.download:
        download_with_copernicusmarine(
            output_dir=args.raw_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            dataset_id=args.dataset_id,
            product_id=args.product_id,
            variables=args.variable,
            username=args.username,
            password=args.password,
            overwrite=args.overwrite,
        )
    if args.process:
        process_downloaded_files(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            hemisphere=args.hemisphere,
            start_date=args.start_date,
            end_date=args.end_date,
            concentration_threshold=args.concentration_threshold,
            overwrite=args.overwrite,
            chunks_time=args.chunks_time,
        )

if __name__ == "__main__":
    main()
