#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shuga.observations.insitu_thickness import (
    AADC_RECORDS,
    DEFAULT_ROOT,
    build_session,
    download_aadc_record,
    download_afiac,
    download_bepsii,
    write_manifest,
)


SOURCE_ALIASES = {
    "afiac",
    "mawson",
    "davis",
    "casey",
    "stations",
    "soe",
    "asac2500",
    "davis2015",
    "bepsii",
    "all",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Discover and download raw Antarctic in-situ fast-ice "
            "thickness datasets for shuga."
        )
    )

    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    ap.add_argument(
        "--sources",
        default="afiac,stations,asac2500,davis2015,bepsii",
        help=(
            "Comma-separated aliases: afiac, mawson, davis, casey, stations, "
            "soe, asac2500, davis2015, bepsii, all."
        ),
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve current source/download URLs without transferring data.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    ap.add_argument(
        "--include-bepsii-logsheets",
        action="store_true",
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--log-level",
        default="INFO",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    requested = {
        source.strip().lower()
        for source in args.sources.split(",")
        if source.strip()
    }

    unknown = requested - SOURCE_ALIASES
    if unknown:
        raise ValueError(
            f"Unknown source aliases: {sorted(unknown)}"
        )

    if "all" in requested:
        requested = {
            "afiac",
            "stations",
            "soe",
            "asac2500",
            "davis2015",
            "bepsii",
        }

    if "stations" in requested:
        requested.update(
            {"mawson", "davis", "casey"}
        )

    session = build_session(args.retries)
    records = []

    if "afiac" in requested:
        records += download_afiac(
            session,
            root=args.root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    mapping = {
        "mawson": "mawson_station",
        "davis": "davis_station",
        "casey": "casey_station",
        "soe": "soe_fast_ice",
        "asac2500": "asac2500",
        "davis2015": "davis2015",
    }

    for alias, key in mapping.items():
        if alias not in requested:
            continue

        info = AADC_RECORDS[key]

        records += download_aadc_record(
            session,
            key=key,
            entry_id=info["entry_id"],
            title=info["title"],
            root=args.root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if "bepsii" in requested:
        records += download_bepsii(
            session,
            root=args.root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            include_logsheets=args.include_bepsii_logsheets,
        )

    manifest_name = (
        "discovery_manifest.json"
        if args.dry_run
        else "download_manifest.json"
    )

    manifest = args.root / manifest_name

    write_manifest(
        records,
        manifest,
    )

    for record in records:
        logging.info(
            "%-14s %-32s %s",
            record.status,
            record.source,
            record.local_path or record.url,
        )

    failed = [
        record
        for record in records
        if record.status in {"failed", "not-found"}
    ]

    manual = [
        record
        for record in records
        if record.status in {"manual", "metadata-only"}
    ]

    logging.info(
        "Summary records=%d failed=%d manual_or_metadata=%d manifest=%s",
        len(records),
        len(failed),
        len(manual),
        manifest,
    )

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
