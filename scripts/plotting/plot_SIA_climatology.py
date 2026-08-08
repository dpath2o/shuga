#!/usr/bin/env python3
"""Plot the 1994–1999 Antarctic SIA climatology with Shuga and PyGMT."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, ObservationSpec, RunSpec
from shuga.observations.NSIDC import NSIDCObservations
from shuga.plotting.sia import (
    dataarray_to_series,
    plot_sia_daily_climatology_envelope_pygmt,
    sia_to_million_km2,
)

LOGGER = logging.getLogger("shuga.plot_SIA_climatology")

EARTH_RADIUS_M = 6_371_008.8

MODEL_NAMES = {
    "AOM2-ERA5": "ACCESS-OM2-ERA5",
    "notensnogi": "notens-nogi",
    "ry93": "ry93",
    "elps-min": "elps-min",
}

PLOT_ORDER = [
    "NSIDC",
    "OSI-SAF-450",
    "ORAS",
    "ACCESS-OM2-ERA5",
    "notens-nogi",
    "ry93",
    "elps-min",
]


def _open_dataset(
    path: Path,
    *,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir() or path.suffix == ".zarr":
        try:
            return xr.open_zarr(
                path,
                consolidated=True,
                chunks=chunks,
            )
        except (KeyError, ValueError, FileNotFoundError):
            return xr.open_zarr(
                path,
                consolidated=False,
                chunks=chunks,
            )

    return xr.open_dataset(path, chunks=chunks)


def _find_series_var(
    ds: xr.Dataset,
    preferred: str | None,
    candidates: tuple[str, ...],
) -> str:
    if preferred:
        if preferred not in ds:
            raise KeyError(
                f"Variable {preferred!r} not found. "
                f"Available: {list(ds.data_vars)}"
            )
        return preferred

    for name in candidates:
        if name in ds:
            return name

    raise KeyError(
        "Could not identify an SIA variable. "
        f"Available: {list(ds.data_vars)}"
    )


def _load_sia_store(
    path: Path,
    *,
    label: str,
    variable: str | None = None,
    candidates: tuple[str, ...] = ("SIA", "sia", "sea_ice_area"),
    units_override: str | None = None,
) -> pd.Series:
    ds = _open_dataset(path)
    var = _find_series_var(ds, variable, candidates)
    da = ds[var]

    if units_override:
        da = da.copy()
        da.attrs["units"] = units_override

    da = sia_to_million_km2(da)
    return dataarray_to_series(da, label)


def _coordinate_edges_1d(
    values: xr.DataArray,
    *,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    centres = np.asarray(values.values, dtype=float)

    if centres.ndim != 1 or centres.size < 2:
        raise ValueError(
            "Coordinate centres must be one-dimensional "
            "with at least two values."
        )

    differences = np.diff(centres)
    if not np.all(differences > 0):
        raise ValueError("Coordinate centres must be strictly increasing.")

    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * differences[0]
    edges[-1] = centres[-1] + 0.5 * differences[-1]

    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])

    return edges


def _regular_latlon_cell_area(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Return spherical rectilinear grid-cell area in m2."""
    lat_edges = np.deg2rad(
        _coordinate_edges_1d(
            latitude,
            clip=(-90.0, 90.0),
        )
    )
    lon_edges = np.deg2rad(
        _coordinate_edges_1d(longitude)
    )

    lat_strip = np.abs(
        np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    )
    dlon = np.abs(np.diff(lon_edges))

    values = (
        EARTH_RADIUS_M ** 2
        * lat_strip[:, None]
        * dlon[None, :]
    )

    return xr.DataArray(
        values,
        dims=(latitude.dims[0], longitude.dims[0]),
        coords={
            latitude.dims[0]: latitude,
            longitude.dims[0]: longitude,
        },
        name="cell_area",
        attrs={
            "long_name": "spherical grid-cell area",
            "units": "m2",
            "earth_radius_m": EARTH_RADIUS_M,
        },
    )


def _load_oras_series(
    path: Path,
    *,
    label: str,
    start_date: str,
    end_date: str,
    threshold: float = 0.15,
    chunks_time: int = 8,
    cache: Path | None = None,
) -> pd.Series:
    """
    Load ORAS SIA from an existing one-dimensional store, or calculate it
    from native gridded siconc(time, latitude, longitude).

    Processing order
    ----------------
    1. Reuse a valid derived SIA cache when available.
    2. Otherwise open native ORAS siconc.
    3. Calculate concentration-weighted Southern Hemisphere SIA.
    4. Materialise the daily SIA series.
    5. Write the derived SIA cache as Zarr v2.
    6. Return the pandas series used by the plotting workflow.
    """
    import shutil

    # ------------------------------------------------------------
    # Reuse an existing valid derived SIA cache
    # ------------------------------------------------------------
    if cache is not None and cache.exists():
        try:
            LOGGER.info(
                "Checking existing ORAS SIA cache: %s",
                cache,
            )

            cache_ds = _open_dataset(cache)

            if "SIA" not in cache_ds:
                raise KeyError(
                    f"Existing ORAS cache does not contain SIA: {cache}"
                )

            cached_sia = cache_ds["SIA"].sel(
                time=slice(start_date, end_date)
            )

            if cached_sia.sizes.get("time", 0) == 0:
                raise ValueError(
                    "Existing ORAS cache has no data in the requested period."
                )

            LOGGER.info(
                "Using existing ORAS SIA cache with %d samples",
                cached_sia.sizes["time"],
            )

            return dataarray_to_series(
                sia_to_million_km2(cached_sia),
                label,
            )

        except Exception as exc:
            LOGGER.warning(
                "Existing ORAS cache is incomplete or invalid: %s",
                exc,
            )
            LOGGER.warning(
                "Removing invalid ORAS cache: %s",
                cache,
            )
            shutil.rmtree(cache, ignore_errors=True)

    # ------------------------------------------------------------
    # Open the source ORAS store
    # ------------------------------------------------------------
    LOGGER.info(
        "Opening ORAS source store: %s",
        path,
    )

    ds = _open_dataset(
        path,
        chunks={"time": int(chunks_time)},
    )

    # ------------------------------------------------------------
    # Source may already contain a one-dimensional SIA series
    # ------------------------------------------------------------
    for variable in ("SIA", "sia", "sea_ice_area"):
        if variable in ds:
            source_sia = ds[variable].sel(
                time=slice(start_date, end_date)
            )

            LOGGER.info(
                "ORAS source already contains %s; no spatial integration required",
                variable,
            )

            return dataarray_to_series(
                sia_to_million_km2(source_sia),
                label,
            )

    # ------------------------------------------------------------
    # Otherwise calculate SIA from native siconc
    # ------------------------------------------------------------
    if "siconc" not in ds:
        raise KeyError(
            f"Could not find ORAS SIA or siconc in {path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    required_coords = (
        "time",
        "latitude",
        "longitude",
    )

    missing_coords = [
        coordinate
        for coordinate in required_coords
        if coordinate not in ds.coords
    ]

    if missing_coords:
        raise KeyError(
            "ORAS native-grid calculation requires coordinates "
            f"{required_coords}; missing {missing_coords}. "
            f"Available coordinates: {list(ds.coords)}"
        )

    LOGGER.info(
        "Computing ORAS SIA from native siconc: %s",
        path,
    )
    LOGGER.info(
        "ORAS SIA period: %s to %s",
        start_date,
        end_date,
    )
    LOGGER.info(
        "ORAS concentration threshold: %.3f",
        threshold,
    )

    concentration = ds["siconc"].sel(
        time=slice(start_date, end_date)
    )

    if concentration.sizes.get("time", 0) == 0:
        raise ValueError(
            f"ORAS source contains no data between "
            f"{start_date} and {end_date}"
        )

    # Southern Hemisphere only.
    concentration = concentration.where(
        ds["latitude"] < 0.0,
        drop=True,
    )

    # Xarray decodes the packed short values and _FillValue. Retain valid
    # concentration fractions and constrain minor numerical excursions.
    concentration = concentration.where(
        np.isfinite(concentration)
    ).clip(
        min=0.0,
        max=1.0,
    )

    LOGGER.info(
        "ORAS subset dimensions: %s",
        dict(concentration.sizes),
    )

    # Latitude-dependent area for the native regular 0.083-degree grid.
    area = _regular_latlon_cell_area(
        concentration["latitude"],
        concentration["longitude"],
    )

    # Apply the same 15% inclusion threshold used for NSIDC SIA.
    concentration_for_sia = concentration.where(
        concentration >= float(threshold),
        0.0,
    )

    ice_area = concentration_for_sia * area

    spatial_dims = [
        dimension
        for dimension in ice_area.dims
        if dimension != "time"
    ]

    sia = (
        ice_area.sum(
            dim=spatial_dims,
            skipna=True,
        )
        / 1.0e12
    ).rename("SIA")

    sia.attrs.update(
        long_name="ORAS Southern Hemisphere sea ice area",
        units="10^6 km^2",
        concentration_threshold=float(threshold),
        source=str(path),
        time_start=start_date,
        time_end=end_date,
        grid="native 0.083 degree regular latitude-longitude",
    )

    # This is the expensive Dask computation.
    LOGGER.info(
        "Materialising %d daily ORAS SIA values",
        sia.sizes["time"],
    )

    sia = sia.compute()

    LOGGER.info(
        "Finished ORAS SIA calculation: min=%.3f, max=%.3f 10^6 km^2",
        float(sia.min()),
        float(sia.max()),
    )

    # ------------------------------------------------------------
    # Write the newly calculated one-dimensional cache
    # ------------------------------------------------------------
    if cache is not None:
        cache.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cache.exists():
            shutil.rmtree(cache)

        cache_ds = xr.Dataset({"SIA": sia})

        # Remove source compressor/filter metadata inherited through Xarray.
        for variable in cache_ds.variables:
            cache_ds[variable].encoding.clear()

        chunk_time = min(
            365,
            cache_ds.sizes["time"],
        )

        LOGGER.info(
            "Writing newly computed ORAS SIA as Zarr v2: %s",
            cache,
        )

        cache_ds.to_zarr(
            cache,
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding={
                "SIA": {
                    "chunks": (chunk_time,),
                },
                "time": {
                    "chunks": (chunk_time,),
                },
            },
        )

        LOGGER.info(
            "Finished writing ORAS SIA cache: %s",
            cache,
        )

    return dataarray_to_series(
        sia,
        label,
    )


def _load_model_series(
    root: Path,
    simulation: str,
    label: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    run_cfg = RunSpec(
        sim_name=simulation,
        start_date=start_date,
        end_date=end_date,
        hemisphere="SH",
        project="gv90",
        user="da1339",
    )
    cls_cfg = ClassificationSpec(
        ice_type="SI",
        grid_type="Tb",
        methods=("raw",),
    )
    paths = ShugaPaths(
        run_cfg=run_cfg,
        cls_cfg=cls_cfg,
        afim_output_root=root,
    )

    store = (
        paths.output_root
        / "zarr"
        / "SH"
        / "SI"
        / "mets.zarr"
    )

    LOGGER.info("Loading %s from %s", label, store)
    return _load_sia_store(
        store,
        label=label,
        variable="SIA",
    )

def _load_nsidc(
    *,
    seaice_root: Path,
    start_date: str,
    end_date: str,
) -> pd.Series:
    run_cfg = RunSpec(
        sim_name="NSIDC",
        start_date=start_date,
        end_date=end_date,
        hemisphere="SH",
        project="gv90",
        user="da1339",
    )

    cls_cfg = ClassificationSpec(
        ice_type="SI",
        grid_type="Tb",
        methods=("raw",),
    )

    obs_cfg = ObservationSpec(
        seaice_root=seaice_root,
        nsidc_version="G02202_V6",
    )

    paths = ShugaPaths(
        run_cfg=run_cfg,
        cls_cfg=cls_cfg,
        obs_cfg=obs_cfg,
    )

    LOGGER.info(
        "Resolved NSIDC root: %s",
        paths.nsidc_root_path,
    )
    LOGGER.info(
        "Resolved NSIDC auxiliary root: %s",
        paths.nsidc_aux_root_path,
    )

    obs = NSIDCObservations(
        run_cfg=run_cfg,
        obs_cfg=obs_cfg,
        pth_cfg=paths,
        chunks={"time": 31},
    )

    ds = obs.compute_sia_sie(
        start_date=start_date,
        end_date=end_date,
        hemisphere="SH",
        threshold=0.15,
    )

    return dataarray_to_series(
        sia_to_million_km2(ds["SIA"]),
        "NSIDC",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the 1994–1999 Antarctic SIA climatology "
            "using Shuga and PyGMT."
        )
    )

    parser.add_argument(
        "--model-root",
        "--model_root",
        "--model",
        dest="model_root",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/afim_output/paper1"
        ),
    )
    parser.add_argument(
        "--seaice-root",
        "--seaice_root",
        dest="seaice_root",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/SeaIce"
        ),
    )
    parser.add_argument(
        "--osisaf-store",
        "--osisaf_store",
        dest="osisaf_store",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/SeaIce/OSI-SAF-450/"
            "processed/OSI-SAF-450_SH_SIA.zarr"
        ),
    )
    parser.add_argument(
        "--oras-store",
        "--oras_store",
        dest="oras_store",
        type=Path,
        required=True,
        help=(
            "NetCDF or Zarr containing either one-dimensional ORAS SIA "
            "or native gridded ORAS siconc."
        ),
    )
    parser.add_argument(
        "--oras-sia-cache",
        "--oras_sia_cache",
        dest="oras_sia_cache",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily/"
            "ORAS_SH_SIA_1994-1999.zarr"
        ),
    )
    parser.add_argument(
        "--oras-threshold",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--oras-chunks-time",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--start-date",
        default="1994-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="1999-12-31",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/g/data/gv90/da1339/GRAPHICAL/paper1/"
            "SIA_SH_climatology_1994-1999_with_OSISAF450.png"
        ),
    )
    parser.add_argument(
        "--envelope",
        choices=("minmax", "std", "p10-p90"),
        default="minmax",
    )
    parser.add_argument(
        "--smooth-days",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--exclude-osisaf",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level.upper(),
        ),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    series: dict[str, pd.Series] = {}

    series["NSIDC"] = _load_nsidc(
        seaice_root=args.seaice_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if not args.exclude_osisaf:
        series["OSI-SAF-450"] = _load_sia_store(
            args.osisaf_store,
            label="OSI-SAF-450",
            candidates=("sia", "SIA", "sea_ice_area"),
        )

    series["ORAS"] = _load_oras_series(
        args.oras_store,
        label="ORAS",
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.oras_threshold,
        chunks_time=args.oras_chunks_time,
        cache=args.oras_sia_cache,
    )

    for simulation, label in MODEL_NAMES.items():
        series[label] = _load_model_series(
            args.model_root,
            simulation,
            label,
            args.start_date,
            args.end_date,
        )

    df = pd.concat(
        series.values(),
        axis=1,
    ).sort_index()

    order = [
        name
        for name in PLOT_ORDER
        if name in df.columns
    ]

    LOGGER.info(
        "Plotting series: %s",
        ", ".join(order),
    )
    LOGGER.info(
        "Climatology period: %s to %s",
        args.start_date,
        args.end_date,
    )

    plot_sia_daily_climatology_envelope_pygmt(
        df,
        args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        envelope=args.envelope,
        smooth_days=args.smooth_days,
        order=order,
        y_min=0.0,
        y_max=args.y_max,
        title=None,
        write_csv=True,
    )

    LOGGER.info("Wrote figure: %s", args.output)
    LOGGER.info(
        "Wrote climatology table: %s",
        args.output.with_suffix(".csv"),
    )


if __name__ == "__main__":
    main()
