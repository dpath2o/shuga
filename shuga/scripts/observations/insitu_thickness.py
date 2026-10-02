#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger("shuga.observations.insitu_thickness")


DEFAULT_INPUT = Path(
    "/g/data/gv90/da1339/SeaIce/InSitu/raw/"
    "Zenodo/BEPSII_TS/DATA/BEPSII_TS_v1.0.nc"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/g/data/gv90/da1339/SeaIce/InSitu/processed/BEPSII_TS"
)


# Independently documented Antarctic fast-ice campaigns from BEPSII-TS Table S3.
# These are retained only as an independent QA check against i_type.
DOCUMENTED_SH_FAST_CRUISES = {
    "ScottBase 03",
    "PNRA XX 04-05",
    "Casey 09",
    "ICELIPIDS",
    "YROSIAE 11-12",
    "Rothera RS 13-15",
    "Filchner 14",
    "JARE 16-19",
}


# Operational interpretation established empirically by cross-checking i_type
# against the independently documented fast-ice campaigns:
#
#   i_type == 1 -> fast ice
#   i_type == 2 -> pack ice
#   i_type == 0 -> unknown / unclassified / other
#
# This mapping is explicitly recorded in output metadata.
BEPSII_TYPE_FAST = 1
BEPSII_TYPE_PACK = 2
BEPSII_TYPE_UNKNOWN = 0


def _decode_strings(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values)

    if values.dtype.kind == "S":
        return np.char.decode(
            values,
            "utf-8",
            errors="replace",
        ).astype(str)

    return np.char.strip(
        values.astype(str)
    )


def _parse_bepsii_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    """
    Parse BEPSII Date, documented as YYYYMMDD.

    Fall back to year + day-of-year where Date is missing/invalid.
    """
    date = np.asarray(ds["Date"].values, dtype=float)

    date_text = np.full(
        date.shape,
        "",
        dtype=object,
    )

    finite = np.isfinite(date)

    if finite.any():
        rounded = np.rint(
            date[finite]
        ).astype(np.int64)

        date_text[finite] = [
            f"{value:08d}"
            for value in rounded
        ]

    parsed = pd.to_datetime(
        pd.Series(date_text),
        format="%Y%m%d",
        errors="coerce",
    )

    if "year" in ds and "doy" in ds:
        year = np.asarray(
            ds["year"].values,
            dtype=float,
        )
        doy = np.asarray(
            ds["doy"].values,
            dtype=float,
        )

        missing = parsed.isna().to_numpy()
        fallback_ok = (
            missing
            & np.isfinite(year)
            & np.isfinite(doy)
        )

        if fallback_ok.any():
            fallback = (
                pd.to_datetime(
                    pd.Series(
                        np.rint(
                            year[fallback_ok]
                        ).astype(int)
                    ).astype(str),
                    format="%Y",
                    errors="coerce",
                )
                + pd.to_timedelta(
                    np.rint(
                        doy[fallback_ok]
                    ).astype(int) - 1,
                    unit="D",
                )
            )

            parsed.loc[
                fallback_ok
            ] = fallback.to_numpy()

    return pd.DatetimeIndex(parsed)


def _canonical_key(
    core_id: str,
    cruise: str,
    time: pd.Timestamp,
    lat: float,
    lon: float,
) -> str:
    date = (
        "NaT"
        if pd.isna(time)
        else time.strftime("%Y%m%d")
    )

    lat_text = (
        "nan"
        if not np.isfinite(lat)
        else f"{lat:.5f}"
    )

    lon_text = (
        "nan"
        if not np.isfinite(lon)
        else f"{lon:.5f}"
    )

    return "|".join(
        (
            str(core_id).strip(),
            str(cruise).strip(),
            date,
            lat_text,
            lon_text,
        )
    )


def _max_core_length(
    cl_s: np.ndarray,
    cl_t: np.ndarray,
) -> np.ndarray:
    stacked = np.vstack(
        (
            cl_s,
            cl_t,
        )
    )

    finite_any = np.isfinite(
        stacked
    ).any(axis=0)

    out = np.full(
        cl_s.shape,
        np.nan,
        dtype=float,
    )

    out[finite_any] = np.nanmax(
        stacked[:, finite_any],
        axis=0,
    )

    return out


def _build_point_dataset(
    source: xr.Dataset,
    *,
    hemisphere: str | None,
    valid_thickness_only: bool,
    deduplicate: bool,
) -> xr.Dataset:
    n = source.sizes["n_cores"]

    time = _parse_bepsii_time(
        source
    )

    lat = np.asarray(
        source["lat"].values,
        dtype=float,
    )
    lon = np.asarray(
        source["lon"].values,
        dtype=float,
    )
    hi = np.asarray(
        source["hi"].values,
        dtype=float,
    )
    hs = np.asarray(
        source["hs"].values,
        dtype=float,
    )

    i_type = np.asarray(
        source["i_type"].values,
        dtype=float,
    )

    core_id = _decode_strings(
        source["CoreID"]
    )
    cruise = _decode_strings(
        source["Cruise"]
    )

    keep = np.ones(
        n,
        dtype=bool,
    )

    if hemisphere == "SH":
        keep &= (
            np.isfinite(lat)
            & (lat < 0.0)
        )
    elif hemisphere == "NH":
        keep &= (
            np.isfinite(lat)
            & (lat > 0.0)
        )

    if valid_thickness_only:
        keep &= (
            np.isfinite(hi)
            & (hi > 0.0)
        )

    indices = np.flatnonzero(
        keep
    )

    keys = np.array(
        [
            _canonical_key(
                core_id[i],
                cruise[i],
                time[i],
                lat[i],
                lon[i],
            )
            for i in indices
        ],
        dtype=object,
    )

    duplicate_key = np.zeros(
        len(indices),
        dtype=np.int8,
    )

    if len(indices):
        _, inverse, counts = np.unique(
            keys,
            return_inverse=True,
            return_counts=True,
        )

        duplicate_key = (
            counts[inverse] > 1
        ).astype(np.int8)

    if deduplicate and len(indices):
        _, first = np.unique(
            keys,
            return_index=True,
        )

        first = np.sort(
            first
        )

        indices = indices[first]
        keys = keys[first]
        duplicate_key = duplicate_key[first]

    obs = np.arange(
        len(indices),
        dtype=np.int64,
    )

    keep_s = (
        np.asarray(
            source["i_keep_S"].values,
            dtype=float,
        )
        if "i_keep_S" in source
        else np.full(n, np.nan)
    )

    keep_t = (
        np.asarray(
            source["i_keep_T"].values,
            dtype=float,
        )
        if "i_keep_T" in source
        else np.full(n, np.nan)
    )

    unique_st_record = (
        (
            np.isfinite(keep_s)
            & (keep_s == 1)
        )
        | (
            np.isfinite(keep_t)
            & (keep_t == 1)
        )
    )

    is_fast = (
        np.isfinite(i_type)
        & (
            np.rint(i_type).astype(
                np.int64
            )
            == BEPSII_TYPE_FAST
        )
    ).astype(np.int8)

    is_pack = (
        np.isfinite(i_type)
        & (
            np.rint(i_type).astype(
                np.int64
            )
            == BEPSII_TYPE_PACK
        )
    ).astype(np.int8)

    is_unknown_type = (
        ~np.isfinite(i_type)
        | (
            np.rint(
                np.where(
                    np.isfinite(i_type),
                    i_type,
                    -999,
                )
            ).astype(np.int64)
            == BEPSII_TYPE_UNKNOWN
        )
    ).astype(np.int8)

    cruise_clean = np.char.strip(
        cruise.astype(str)
    )

    is_documented_fast_campaign = np.array(
        [
            name
            in DOCUMENTED_SH_FAST_CRUISES
            for name in cruise_clean
        ],
        dtype=np.int8,
    )

    type_campaign_consistent = (
        (
            is_documented_fast_campaign
            == 0
        )
        | (
            is_fast
            == 1
        )
    ).astype(np.int8)

    cl_s = (
        np.asarray(
            source["cl_S"].values,
            dtype=float,
        )
        if "cl_S" in source
        else np.full(n, np.nan)
    )

    cl_t = (
        np.asarray(
            source["cl_T"].values,
            dtype=float,
        )
        if "cl_T" in source
        else np.full(n, np.nan)
    )

    max_core_length = _max_core_length(
        cl_s,
        cl_t,
    )

    core_length_fraction = np.full(
        n,
        np.nan,
        dtype=float,
    )

    good_ratio = (
        np.isfinite(max_core_length)
        & np.isfinite(hi)
        & (hi > 0.0)
    )

    core_length_fraction[
        good_ratio
    ] = (
        max_core_length[
            good_ratio
        ]
        / hi[
            good_ratio
        ]
    )

    partial_core_lt80pct = (
        np.isfinite(
            core_length_fraction
        )
        & (
            core_length_fraction
            < 0.8
        )
    ).astype(np.int8)

    thickness_gt_3m = (
        np.isfinite(hi)
        & (hi > 3.0)
    ).astype(np.int8)

    thickness_gt_5m = (
        np.isfinite(hi)
        & (hi > 5.0)
    ).astype(np.int8)

    missing_snow = (
        ~np.isfinite(hs)
    ).astype(np.int8)

    data_vars: dict[str, tuple] = {
        "time": (
            "obs",
            time.to_numpy()[
                indices
            ].astype(
                "datetime64[ns]"
            ),
        ),
        "latitude": (
            "obs",
            lat[
                indices
            ].astype(
                np.float64
            ),
        ),
        "longitude": (
            "obs",
            lon[
                indices
            ].astype(
                np.float64
            ),
        ),
        "ice_thickness": (
            "obs",
            hi[
                indices
            ].astype(
                np.float32
            ),
        ),
        "snow_thickness": (
            "obs",
            hs[
                indices
            ].astype(
                np.float32
            ),
        ),
        "core_id": (
            "obs",
            core_id[
                indices
            ].astype(str),
        ),
        "cruise": (
            "obs",
            cruise[
                indices
            ].astype(str),
        ),
        "observation_key": (
            "obs",
            keys.astype(str),
        ),
        "bepsii_record_index": (
            "obs",
            indices.astype(
                np.int64
            ),
        ),
        "bepsii_type_index": (
            "obs",
            i_type[
                indices
            ].astype(
                np.float32
            ),
        ),
        "is_fast_ice": (
            "obs",
            is_fast[
                indices
            ].astype(
                np.int8
            ),
        ),
        "is_pack_ice": (
            "obs",
            is_pack[
                indices
            ].astype(
                np.int8
            ),
        ),
        "is_unknown_ice_type": (
            "obs",
            is_unknown_type[
                indices
            ].astype(
                np.int8
            ),
        ),
        "is_documented_fast_campaign": (
            "obs",
            is_documented_fast_campaign[
                indices
            ].astype(
                np.int8
            ),
        ),
        "type_campaign_consistent": (
            "obs",
            type_campaign_consistent[
                indices
            ].astype(
                np.int8
            ),
        ),
        "unique_ST_record": (
            "obs",
            unique_st_record[
                indices
            ].astype(
                np.int8
            ),
        ),
        "duplicate_observation_key": (
            "obs",
            duplicate_key.astype(
                np.int8
            ),
        ),
        "max_core_length": (
            "obs",
            max_core_length[
                indices
            ].astype(
                np.float32
            ),
        ),
        "core_length_fraction_of_thickness": (
            "obs",
            core_length_fraction[
                indices
            ].astype(
                np.float32
            ),
        ),
        "partial_core_lt80pct": (
            "obs",
            partial_core_lt80pct[
                indices
            ].astype(
                np.int8
            ),
        ),
        "thickness_gt_3m": (
            "obs",
            thickness_gt_3m[
                indices
            ].astype(
                np.int8
            ),
        ),
        "thickness_gt_5m": (
            "obs",
            thickness_gt_5m[
                indices
            ].astype(
                np.int8
            ),
        ),
        "missing_snow": (
            "obs",
            missing_snow[
                indices
            ].astype(
                np.int8
            ),
        ),
    }

    numeric_map = {
        "i_age": "bepsii_age_index",
        "i_ref": "bepsii_reference_index",
        "i_keep_S": "keep_salinity_record",
        "i_keep_T": "keep_temperature_record",
        "cl_S": "salinity_core_length",
        "cl_T": "temperature_core_length",
        "tsl_S": "salinity_sampled_length",
        "tsl_T": "temperature_sampled_length",
        "S_mean": "mean_core_salinity",
        "T_mean": "mean_core_temperature",
        "etopo_depth": "etopo_depth",
        "year": "sampling_year",
        "mon": "sampling_month",
        "doy": "sampling_day_of_year",
    }

    for original, target in (
        numeric_map.items()
    ):
        if (
            original in source
            and source[
                original
            ].dims
            == ("n_cores",)
        ):
            data_vars[
                target
            ] = (
                "obs",
                np.asarray(
                    source[
                        original
                    ].values
                )[
                    indices
                ],
            )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "obs": obs,
        },
    )

    ds["time"].attrs.update(
        long_name="Sampling time",
    )

    ds["latitude"].attrs.update(
        standard_name="latitude",
        units="degrees_north",
    )

    ds["longitude"].attrs.update(
        standard_name="longitude",
        units="degrees_east",
    )

    ds["ice_thickness"].attrs.update(
        long_name=(
            "Sea ice thickness from BEPSII core metadata"
        ),
        units="m",
        source_variable="hi",
    )

    ds["snow_thickness"].attrs.update(
        long_name="Snow depth from BEPSII core metadata",
        units="m",
        source_variable="hs",
    )

    ds["bepsii_type_index"].attrs.update(
        long_name="Original BEPSII i_type index",
        operational_mapping=(
            "0=unknown_or_other, 1=fast_ice, 2=pack_ice"
        ),
        mapping_basis=(
            "Empirically validated against independently documented "
            "BEPSII Southern Hemisphere fast-ice campaigns: all 190 "
            "records in the campaign QA subset have i_type=1 and none "
            "have i_type=2."
        ),
    )

    for name, meaning in (
        (
            "is_fast_ice",
            "BEPSII i_type == 1",
        ),
        (
            "is_pack_ice",
            "BEPSII i_type == 2",
        ),
        (
            "is_unknown_ice_type",
            "BEPSII i_type == 0 or missing",
        ),
    ):
        ds[
            name
        ].attrs.update(
            long_name=meaning,
            flag_values=[0, 1],
            flag_meanings="false true",
        )

    ds[
        "is_documented_fast_campaign"
    ].attrs.update(
        long_name=(
            "Independent campaign-level fast-ice QA flag "
            "derived from BEPSII Table S3"
        ),
        flag_values=[0, 1],
        flag_meanings=(
            "not_in_documented_fast_campaign "
            "documented_fast_campaign"
        ),
    )

    ds[
        "type_campaign_consistent"
    ].attrs.update(
        long_name=(
            "Consistency between operational i_type fast-ice "
            "classification and independent campaign-level fast-ice QA"
        ),
        flag_values=[0, 1],
        flag_meanings="inconsistent consistent",
    )

    ds[
        "core_length_fraction_of_thickness"
    ].attrs.update(
        long_name=(
            "Maximum available core length divided by metadata ice thickness"
        ),
        units="1",
    )

    ds.attrs.update(
        title=(
            "BEPSII-TS point observations prepared for shuga"
        ),
        source="BEPSII-TS v1.0",
        source_dimension="n_cores",
        source_records=int(n),
        hemisphere=(
            hemisphere
            or "both"
        ),
        valid_thickness_only=bool(
            valid_thickness_only
        ),
        duplicate_strategy=(
            "canonical CoreID+Cruise+Date+lat+lon key; "
            "first occurrence retained"
            if deduplicate
            else "none"
        ),
        thickness_definition=(
            "hi: ice thickness from metadata; cl_S/cl_T retained "
            "separately as physical core lengths and never substituted "
            "for thickness"
        ),
        operational_ice_type_mapping=(
            "i_type 1=fast, 2=pack, 0=unknown/other"
        ),
        operational_ice_type_mapping_evidence=(
            "Independent documented SH fast-ice campaign cross-tab: "
            "190/190 records i_type=1, 0 records i_type=2."
        ),
        qa_policy=(
            "Thickness/core-length QA fields are diagnostic only; "
            "no plausible thick observations are removed by default."
        ),
    )

    return ds


def _subset_by_type(
    ds: xr.Dataset,
    flag_name: str,
) -> xr.Dataset:
    mask = (
        ds[flag_name]
        .compute()
        .values
        .astype(bool)
    )

    idx = np.flatnonzero(mask)

    out = ds.isel(
        obs=idx
    )

    return out.assign_coords(
        obs=np.arange(
            out.sizes["obs"],
            dtype=np.int64,
        )
    )


def _write_zarr(
    ds: xr.Dataset,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if (
        path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"{path} exists. "
            "Use --overwrite to replace it."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk = min(
        512,
        max(
            1,
            ds.sizes["obs"],
        ),
    )

    ds.chunk(
        {
            "obs": chunk,
        }
    ).to_zarr(
        path,
        mode="w",
        consolidated=True,
        zarr_format=2,
    )


def _summary(
    ds: xr.Dataset,
) -> dict:
    hi = np.asarray(
        ds["ice_thickness"].values,
        dtype=float,
    )

    finite_hi = np.isfinite(
        hi
    )

    time = pd.to_datetime(
        ds["time"].values
    )

    good_time = ~pd.isna(
        time
    )

    return {
        "n_observations": int(
            ds.sizes["obs"]
        ),
        "n_finite_thickness": int(
            finite_hi.sum()
        ),
        "n_fast_ice": int(
            np.asarray(
                ds[
                    "is_fast_ice"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_pack_ice": int(
            np.asarray(
                ds[
                    "is_pack_ice"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_unknown_type": int(
            np.asarray(
                ds[
                    "is_unknown_ice_type"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_documented_fast_campaign": int(
            np.asarray(
                ds[
                    "is_documented_fast_campaign"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_type_campaign_inconsistent": int(
            (
                np.asarray(
                    ds[
                        "type_campaign_consistent"
                    ].values,
                    dtype=int,
                )
                == 0
            ).sum()
        ),
        "n_thickness_gt_3m": int(
            np.asarray(
                ds[
                    "thickness_gt_3m"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_thickness_gt_5m": int(
            np.asarray(
                ds[
                    "thickness_gt_5m"
                ].values,
                dtype=int,
            ).sum()
        ),
        "n_partial_core_lt80pct": int(
            np.asarray(
                ds[
                    "partial_core_lt80pct"
                ].values,
                dtype=int,
            ).sum()
        ),
        "thickness_min_m": (
            float(
                np.nanmin(
                    hi
                )
            )
            if finite_hi.any()
            else None
        ),
        "thickness_max_m": (
            float(
                np.nanmax(
                    hi
                )
            )
            if finite_hi.any()
            else None
        ),
        "thickness_mean_m": (
            float(
                np.nanmean(
                    hi
                )
            )
            if finite_hi.any()
            else None
        ),
        "time_start": (
            str(
                time[
                    good_time
                ].min()
            )
            if good_time.any()
            else None
        ),
        "time_end": (
            str(
                time[
                    good_time
                ].max()
            )
            if good_time.any()
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Process BEPSII-TS sea-ice cores into shuga "
            "point-observation Zarr stores."
        )
    )

    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    ap.add_argument(
        "--hemisphere",
        choices=(
            "SH",
            "NH",
            "both",
        ),
        default="SH",
    )

    ap.add_argument(
        "--keep-missing-thickness",
        action="store_true",
    )

    ap.add_argument(
        "--no-deduplicate",
        action="store_true",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    ap.add_argument(
        "--log-level",
        default="INFO",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level.upper(),
        ),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    LOGGER.info(
        "Opening BEPSII source: %s",
        args.input,
    )

    source = xr.open_dataset(
        args.input,
        decode_cf=True,
        mask_and_scale=True,
    )

    LOGGER.info(
        "Source records=%d variables=%d",
        source.sizes["n_cores"],
        len(
            source.data_vars
        ),
    )

    if "i_type" in source:
        vals = np.asarray(
            source["i_type"].values,
            dtype=float,
        )

        finite = np.isfinite(
            vals
        )

        values, counts = np.unique(
            vals[
                finite
            ],
            return_counts=True,
        )

        LOGGER.info(
            "Raw numeric i_type values: %s",
            dict(
                zip(
                    values.tolist(),
                    counts.tolist(),
                )
            ),
        )

    hemispheres = (
        ("SH", "NH")
        if args.hemisphere
        == "both"
        else (
            args.hemisphere,
        )
    )

    summaries: dict[str, dict] = {}

    for hem in hemispheres:
        points = _build_point_dataset(
            source,
            hemisphere=hem,
            valid_thickness_only=(
                not args.keep_missing_thickness
            ),
            deduplicate=(
                not args.no_deduplicate
            ),
        )

        all_summary = _summary(
            points
        )

        summaries[
            f"{hem}_all"
        ] = all_summary

        LOGGER.info(
            "%s all: n=%d fast=%d pack=%d unknown=%d "
            "inconsistent=%d time=%s -> %s",
            hem,
            all_summary[
                "n_observations"
            ],
            all_summary[
                "n_fast_ice"
            ],
            all_summary[
                "n_pack_ice"
            ],
            all_summary[
                "n_unknown_type"
            ],
            all_summary[
                "n_type_campaign_inconsistent"
            ],
            all_summary[
                "time_start"
            ],
            all_summary[
                "time_end"
            ],
        )

        fast = _subset_by_type(
            points,
            "is_fast_ice",
        )

        pack = _subset_by_type(
            points,
            "is_pack_ice",
        )

        unknown = _subset_by_type(
            points,
            "is_unknown_ice_type",
        )

        fast_summary = _summary(
            fast
        )

        pack_summary = _summary(
            pack
        )

        unknown_summary = _summary(
            unknown
        )

        summaries[
            f"{hem}_fast"
        ] = fast_summary

        summaries[
            f"{hem}_pack"
        ] = pack_summary

        summaries[
            f"{hem}_unknown"
        ] = unknown_summary

        LOGGER.info(
            "%s fast: n=%d mean_hi=%.3f max_hi=%.3f >3m=%d >5m=%d",
            hem,
            fast_summary[
                "n_observations"
            ],
            fast_summary[
                "thickness_mean_m"
            ],
            fast_summary[
                "thickness_max_m"
            ],
            fast_summary[
                "n_thickness_gt_3m"
            ],
            fast_summary[
                "n_thickness_gt_5m"
            ],
        )

        LOGGER.info(
            "%s pack: n=%d mean_hi=%.3f max_hi=%.3f",
            hem,
            pack_summary[
                "n_observations"
            ],
            pack_summary[
                "thickness_mean_m"
            ],
            pack_summary[
                "thickness_max_m"
            ],
        )

        if not args.dry_run:
            base = (
                args.output_root
                / hem
            )

            _write_zarr(
                points,
                base
                / "all"
                / "BEPSII_TS_points.zarr",
                overwrite=args.overwrite,
            )

            _write_zarr(
                fast,
                base
                / "fast"
                / "BEPSII_TS_fast_points.zarr",
                overwrite=args.overwrite,
            )

            _write_zarr(
                pack,
                base
                / "pack"
                / "BEPSII_TS_pack_points.zarr",
                overwrite=args.overwrite,
            )

            if (
                unknown.sizes[
                    "obs"
                ]
                > 0
            ):
                _write_zarr(
                    unknown,
                    base
                    / "unknown"
                    / "BEPSII_TS_unknown_points.zarr",
                    overwrite=args.overwrite,
                )

    if args.dry_run:
        print(
            json.dumps(
                summaries,
                indent=2,
            )
        )
        return

    summary_path = (
        args.output_root
        / "BEPSII_TS_processing_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summaries,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Summary -> %s",
        summary_path,
    )


if __name__ == "__main__":
    main()
