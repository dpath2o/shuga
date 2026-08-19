#!/usr/bin/env python3
"""
Download ESA CCI sea-ice thickness from the current CEDA THREDDS catalogue.

The current CCI THREDDS server exposes sea-ice-thickness products through
top-level dataset collections such as:

  esacci.SEAICE.mon.L3C.SITHICK.RA-2.Envisat.SH50KMEASE2.4-0.r1
  esacci.SEAICE.mon.L3C.SITHICK.SIRAL.CryoSat-2.SH50KMEASE2.4-0.r1
  esacci.SEAICE.mon.L3C.SITHICK.SRAL.Sentinel-3A.SH50KMEASE2.4-0.r1
  esacci.SEAICE.mon.L3C.SITHICK.SRAL.Sentinel-3B.SH50KMEASE2.4-0.r1

This downloader:
  * reads the THREDDS root catalog.xml;
  * selects matching SEAICE/SITHICK catalogue references;
  * by default keeps the newest product version for each sensor/hemisphere;
  * follows each nested catalog recursively;
  * downloads the original NetCDF files using each dataset's urlPath;
  * skips existing files when Content-Length matches;
  * resumes partial downloads through .part files and HTTP Range requests.

No assumption is made about legacy CEDA directory layout.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests

LOGGER = logging.getLogger("download_ESA_SIT")

THREDDS_ROOT = "https://data.cci.ceda.ac.uk/thredds"
ROOT_CATALOG = f"{THREDDS_ROOT}/catalog.xml"
DEFAULT_DEST = Path("/g/data/gv90/da1339/SeaIce/ESA/CCI")
XLINK        = "{http://www.w3.org/1999/xlink}href"
SENSOR_CANON = {"envisat"    : "envisat",
                "cryosat-2"  : "cryosat2",
                "cryosat2"   : "cryosat2",
                "sentinel-3a": "sentinel3a",
                "sentinel3a" : "sentinel3a",
                "sentinel-3b": "sentinel3b",
                "sentinel3b" : "sentinel3b",
                "ers-2"      : "ers2",
                "ers2"       : "ers2"}

@dataclass(frozen=True)
class Collection:
    name: str
    href: str
    sensor: str
    hemisphere: str
    version: tuple[int, ...]


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def fetch_xml(session: requests.Session, url: str) -> ET.Element:
    LOGGER.debug("GET catalog %s", url)
    r = session.get(url, timeout=90)
    r.raise_for_status()
    return ET.fromstring(r.content)


def product_version(name: str) -> tuple[int, ...]:
    # Supports ".4-0.r1", ".3-0.r1", ".2-0.r1", etc.
    m = re.search(r"\.(\d+(?:-\d+)*)\.r\d+(?:/)?$", name)
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("-"))


def infer_sensor(name: str) -> str | None:
    low = name.lower()
    for token, canon in SENSOR_CANON.items():
        if token in low:
            return canon
    return None


def infer_hemisphere(name: str) -> str | None:
    # Current collection names include NH25KMEASE2 or SH50KMEASE2.
    if ".nh" in name.lower() or "nh25kmease2" in name.lower():
        return "nh"
    if ".sh" in name.lower() or "sh50kmease2" in name.lower():
        return "sh"
    return None


def discover_collections(
    session: requests.Session,
    *,
    root_catalog: str,
    sensors: set[str],
    hemispheres: set[str],
    latest_only: bool,
) -> list[Collection]:
    root = fetch_xml(session, root_catalog)

    candidates: list[Collection] = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "catalogRef":
            continue
        name = elem.attrib.get("name") or elem.attrib.get(
            "{http://www.w3.org/1999/xlink}title", ""
        )
        href = elem.attrib.get(XLINK)
        if not name or not href:
            continue

        upper = name.upper()
        if "ESACCI.SEAICE." not in upper:
            continue
        if ".SITHICK." not in upper:
            continue
        # Focus on gridded collated SIT products suitable for comparison.
        if ".L3C." not in upper:
            continue

        sensor = infer_sensor(name)
        hem = infer_hemisphere(name)
        if sensor is None or hem is None:
            continue
        if sensor not in sensors or hem not in hemispheres:
            continue

        candidates.append(
            Collection(
                name=name.rstrip("/"),
                href=urljoin(root_catalog, href),
                sensor=sensor,
                hemisphere=hem,
                version=product_version(name.rstrip("/")),
            )
        )

    if not latest_only:
        return sorted(candidates, key=lambda c: (c.hemisphere, c.sensor, c.version))

    best: dict[tuple[str, str], Collection] = {}
    for c in candidates:
        key = (c.sensor, c.hemisphere)
        if key not in best or c.version > best[key].version:
            best[key] = c

    return sorted(best.values(), key=lambda c: (c.hemisphere, c.sensor))


def recursive_datasets(
    session: requests.Session,
    catalog_url: str,
    *,
    seen: set[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Return [(dataset_name, urlPath), ...] from a THREDDS catalog tree.

    Nested catalogRefs are followed recursively. Dataset urlPath values are the
    canonical service-relative paths and are later combined with /fileServer/.
    """
    if seen is None:
        seen = set()
    if catalog_url in seen:
        return []
    seen.add(catalog_url)

    root = fetch_xml(session, catalog_url)
    out: list[tuple[str, str]] = []

    for elem in root.iter():
        tag = _local_tag(elem.tag)
        if tag == "dataset":
            url_path = elem.attrib.get("urlPath")
            name = elem.attrib.get("name", "")
            if url_path and name.lower().endswith(".nc"):
                out.append((name, url_path))

    for elem in root.iter():
        if _local_tag(elem.tag) != "catalogRef":
            continue
        href = elem.attrib.get(XLINK)
        if not href:
            continue
        nested = urljoin(catalog_url, href)
        out.extend(recursive_datasets(session, nested, seen=seen))

    # stable de-duplication
    dedup = {}
    for name, url_path in out:
        dedup[url_path] = name
    return [(name, path) for path, name in sorted(dedup.items())]


def remote_size(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        if not r.ok:
            return None
        value = r.headers.get("Content-Length")
        return int(value) if value else None
    except requests.RequestException:
        return None


def download_file(
    session: requests.Session,
    *,
    url: str,
    dest: Path,
    retries: int,
) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(session, url)

    if dest.exists():
        if expected is None or dest.stat().st_size == expected:
            LOGGER.info("SKIP existing: %s", dest)
            return "skip"
        LOGGER.warning(
            "Existing file size differs from remote; re-downloading: %s", dest
        )

    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            ) as r:
                r.raise_for_status()

                # If server ignored Range, restart rather than append duplicate bytes.
                append = offset > 0 and r.status_code == 206
                mode = "ab" if append else "wb"

                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)

            if expected is not None and part.stat().st_size != expected:
                raise IOError(
                    f"size mismatch after download: "
                    f"{part.stat().st_size} != {expected}"
                )

            part.replace(dest)
            LOGGER.info("DOWNLOADED: %s", dest)
            return "download"

        except Exception as exc:
            LOGGER.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                retries,
                url,
                exc,
            )
            time.sleep(min(30, 2 * attempt))

    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ESA CCI SIT through current CEDA THREDDS catalogs."
    )
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--catalog", default=ROOT_CATALOG)
    ap.add_argument(
        "--sensors",
        default="envisat,cryosat2,sentinel3a,sentinel3b",
        help="Comma-separated canonical sensor names.",
    )
    ap.add_argument("--hemispheres", default="nh,sh")
    ap.add_argument(
        "--all-versions",
        action="store_true",
        help="Download all catalogued product versions instead of only newest.",
    )
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sensors = {x.strip().lower() for x in args.sensors.split(",") if x.strip()}
    hemispheres = {
        x.strip().lower() for x in args.hemispheres.split(",") if x.strip()
    }

    unknown = sensors.difference(set(SENSOR_CANON.values()))
    if unknown:
        raise ValueError(f"Unsupported sensor names: {sorted(unknown)}")

    ses = requests.Session()
    ses.headers.update({"User-Agent": "shuga-ESA-SIT-downloader/0.2"})

    collections = discover_collections(
        ses,
        root_catalog=args.catalog,
        sensors=sensors,
        hemispheres=hemispheres,
        latest_only=not args.all_versions,
    )

    if not collections:
        raise RuntimeError(
            "No matching ESA CCI SIT collections discovered from "
            f"{args.catalog}"
        )

    LOGGER.info("Discovered %d selected SIT collections", len(collections))
    for c in collections:
        LOGGER.info(
            "COLLECTION sensor=%s hemisphere=%s version=%s name=%s",
            c.sensor,
            c.hemisphere.upper(),
            ".".join(map(str, c.version)),
            c.name,
        )

    n_down = n_skip = n_fail = n_filtered = 0

    for c in collections:
        files = recursive_datasets(ses, c.href)
        LOGGER.info("%s: catalogued NetCDF files=%d", c.name, len(files))

        for name, url_path in files:
            # Filter by year from filename when requested.
            years = re.findall(r"(?:19|20)\d{2}", name)
            year = int(years[-1]) if years else None

            if year is not None:
                if args.year_min is not None and year < args.year_min:
                    n_filtered += 1
                    continue
                if args.year_max is not None and year > args.year_max:
                    n_filtered += 1
                    continue

            file_url = f"{THREDDS_ROOT}/fileServer/{url_path.lstrip('/')}"

            # Preserve shuga-friendly institution/product hierarchy.
            version_str = "-".join(map(str, c.version))
            local = (
                args.dest
                / "L3C"
                / c.sensor
                / c.hemisphere
                / f"v{version_str}"
                / name
            )

            if args.dry_run:
                LOGGER.info("[DRY] %s -> %s", file_url, local)
                continue

            result = download_file(
                ses,
                url=file_url,
                dest=local,
                retries=args.retries,
            )
            n_down += result == "download"
            n_skip += result == "skip"
            n_fail += result == "fail"

    LOGGER.info(
        "Summary downloaded=%d skipped=%d failed=%d filtered=%d",
        n_down,
        n_skip,
        n_fail,
        n_filtered,
    )

    if n_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
