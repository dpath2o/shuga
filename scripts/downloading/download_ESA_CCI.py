#!/usr/bin/env python3
"""
Declarative ESA CCI sea-ice mirror for shuga.

This downloader intentionally avoids traversing old/unwanted ESA CCI product
versions. The exact catalog roots required by the project are listed in
DOWNLOAD_ROOTS below.

Selected products
-----------------
1. Sea-ice thickness:
   - v4.0 ONLY
   - L2P and L3C
   - Envisat, CryoSat-2, Sentinel-3A, Sentinel-3B
   - NH and SH

2. Sea-ice concentration:
   - L4 / ssmi_ssmis / 12.5km / v3.0
   - NH and SH

3. Drift-aware thickness:
   - NH only
   - recurse beneath the product root, rejecting any SH paths

Local structure
---------------
/g/data/gv90/da1339/SeaIce/ESA/CCI/
    thickness/
    concentration/
    thickness_drift_aware/

The hierarchy beneath each selected remote root is preserved.

Network behaviour
-----------------
- THREDDS catalogs are retried with exponential backoff.
- requests Session uses urllib3 retry handling for transient HTTP errors.
- complete local files are skipped when Content-Length matches.
- .part files are resumed through HTTP Range requests where supported.
"""
from __future__ import annotations

import argparse
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("download_ESA_CCI")

THREDDS_ROOT = "https://data.cci.ceda.ac.uk/thredds"
CATALOG_PREFIX = f"{THREDDS_ROOT}/catalog/"
FILESERVER_PREFIX = f"{THREDDS_ROOT}/fileServer/"
DEFAULT_DEST = Path("/g/data/gv90/da1339/SeaIce/ESA/CCI")

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


@dataclass(frozen=True)
class DownloadRoot:
    product: str
    remote_root: str
    local_prefix: str
    hemisphere: str | None = None
    sensor: str | None = None
    level: str | None = None


# ---------------------------------------------------------------------
# Exact, auditable product roots.
# ---------------------------------------------------------------------

SIT_BASE = "esacci/sea_ice/data/sea_ice_thickness"

DOWNLOAD_ROOTS: tuple[DownloadRoot, ...] = (
    # ------------------------
    # Sea-ice thickness v4.0
    # ------------------------
    DownloadRoot("thickness", f"{SIT_BASE}/L2P/envisat/v4.0/NH",
                 "thickness/L2P/envisat/v4.0/NH", "NH", "envisat", "L2P"),
    DownloadRoot("thickness", f"{SIT_BASE}/L2P/envisat/v4.0/SH",
                 "thickness/L2P/envisat/v4.0/SH", "SH", "envisat", "L2P"),

    DownloadRoot("thickness", f"{SIT_BASE}/L2P/cryosat2/v4.0/NH",
                 "thickness/L2P/cryosat2/v4.0/NH", "NH", "cryosat2", "L2P"),
    DownloadRoot("thickness", f"{SIT_BASE}/L2P/cryosat2/v4.0/SH",
                 "thickness/L2P/cryosat2/v4.0/SH", "SH", "cryosat2", "L2P"),

    DownloadRoot("thickness", f"{SIT_BASE}/L2P/sentinel3a/v4.0/NH",
                 "thickness/L2P/sentinel3a/v4.0/NH", "NH", "sentinel3a", "L2P"),
    DownloadRoot("thickness", f"{SIT_BASE}/L2P/sentinel3a/v4.0/SH",
                 "thickness/L2P/sentinel3a/v4.0/SH", "SH", "sentinel3a", "L2P"),

    DownloadRoot("thickness", f"{SIT_BASE}/L2P/sentinel3b/v4.0/NH",
                 "thickness/L2P/sentinel3b/v4.0/NH", "NH", "sentinel3b", "L2P"),
    DownloadRoot("thickness", f"{SIT_BASE}/L2P/sentinel3b/v4.0/SH",
                 "thickness/L2P/sentinel3b/v4.0/SH", "SH", "sentinel3b", "L2P"),

    DownloadRoot("thickness", f"{SIT_BASE}/L3C/envisat/v4.0/NH",
                 "thickness/L3C/envisat/v4.0/NH", "NH", "envisat", "L3C"),
    DownloadRoot("thickness", f"{SIT_BASE}/L3C/envisat/v4.0/SH",
                 "thickness/L3C/envisat/v4.0/SH", "SH", "envisat", "L3C"),

    DownloadRoot("thickness", f"{SIT_BASE}/L3C/cryosat2/v4.0/NH",
                 "thickness/L3C/cryosat2/v4.0/NH", "NH", "cryosat2", "L3C"),
    DownloadRoot("thickness", f"{SIT_BASE}/L3C/cryosat2/v4.0/SH",
                 "thickness/L3C/cryosat2/v4.0/SH", "SH", "cryosat2", "L3C"),

    DownloadRoot("thickness", f"{SIT_BASE}/L3C/sentinel3a/v4.0/NH",
                 "thickness/L3C/sentinel3a/v4.0/NH", "NH", "sentinel3a", "L3C"),
    DownloadRoot("thickness", f"{SIT_BASE}/L3C/sentinel3a/v4.0/SH",
                 "thickness/L3C/sentinel3a/v4.0/SH", "SH", "sentinel3a", "L3C"),

    DownloadRoot("thickness", f"{SIT_BASE}/L3C/sentinel3b/v4.0/NH",
                 "thickness/L3C/sentinel3b/v4.0/NH", "NH", "sentinel3b", "L3C"),
    DownloadRoot("thickness", f"{SIT_BASE}/L3C/sentinel3b/v4.0/SH",
                 "thickness/L3C/sentinel3b/v4.0/SH", "SH", "sentinel3b", "L3C"),

    # -----------------------------------------------
    # Sea-ice concentration: exact required product
    # -----------------------------------------------
    DownloadRoot(
        "concentration",
        "esacci/sea_ice/data/sea_ice_concentration/L4/ssmi_ssmis/12.5km/v3.0/NH",
        "concentration/L4/ssmi_ssmis/12.5km/v3.0/NH",
        "NH",
        "ssmi_ssmis",
        "L4",
    ),
    DownloadRoot(
        "concentration",
        "esacci/sea_ice/data/sea_ice_concentration/L4/ssmi_ssmis/12.5km/v3.0/SH",
        "concentration/L4/ssmi_ssmis/12.5km/v3.0/SH",
        "SH",
        "ssmi_ssmis",
        "L4",
    ),

    # ------------------------------------------------------
    # Drift-aware SIT: no SH product; recurse from base root
    # ------------------------------------------------------
    DownloadRoot(
        "thickness_drift_aware",
        "esacci/sea_ice/data/thickness_drift_aware",
        "thickness_drift_aware",
        "NH",
        None,
        None,
    ),
)


def _tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build_session(total_retries: int = 5) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "shuga-ESA-CCI-downloader/0.5"}
    )
    return session


def catalog_xml(remote_root: str) -> str:
    return f"{CATALOG_PREFIX}{remote_root.strip('/')}/catalog.xml"


def fetch_xml(
    session: requests.Session,
    url: str,
    *,
    attempts: int = 5,
) -> ET.Element:
    """
    Fetch THREDDS XML with an outer retry loop as protection against
    RemoteDisconnected / transient server-side connection closes.
    """
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            LOGGER.debug("GET catalog %s", url)
            response = session.get(
                url,
                timeout=(30, 120),
            )
            response.raise_for_status()
            return ET.fromstring(response.content)

        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break

            delay = min(30, 2 ** (attempt - 1))
            LOGGER.warning(
                "Catalog request failed %d/%d: %s; retrying in %ds",
                attempt,
                attempts,
                url,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def recursive_catalog(
    session: requests.Session,
    root_catalog: str,
    *,
    seen: set[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Return unique (dataset_name, urlPath) pairs below one selected root.
    """
    if seen is None:
        seen = set()

    if root_catalog in seen:
        return []
    seen.add(root_catalog)

    root = fetch_xml(session, root_catalog)
    files: dict[str, str] = {}

    for elem in root.iter():
        if _tag(elem.tag) != "dataset":
            continue

        url_path = elem.attrib.get("urlPath")
        if not url_path or not url_path.lower().endswith(".nc"):
            continue

        files[url_path] = elem.attrib.get(
            "name",
            PurePosixPath(url_path).name,
        )

    child_catalogs: list[str] = []
    for elem in root.iter():
        if _tag(elem.tag) != "catalogRef":
            continue

        href = elem.attrib.get(XLINK_HREF)
        if href:
            child_catalogs.append(urljoin(root_catalog, href))

    for child in child_catalogs:
        try:
            for name, url_path in recursive_catalog(
                session,
                child,
                seen=seen,
            ):
                files[url_path] = name
        except Exception as exc:
            LOGGER.warning(
                "Cannot traverse %s after retries: %s",
                child,
                exc,
            )

    return [
        (name, path)
        for path, name in sorted(files.items())
    ]


def remote_relative_path(
    selected_root: DownloadRoot,
    url_path: str,
) -> PurePosixPath:
    remote = PurePosixPath(url_path.lstrip("/"))
    base = PurePosixPath(selected_root.remote_root)

    try:
        return remote.relative_to(base)
    except ValueError:
        # THREDDS can prepend internal path elements. Search for the exact
        # selected root as a subsequence.
        parts = remote.parts
        base_parts = base.parts

        for i in range(
            len(parts) - len(base_parts) + 1
        ):
            if tuple(parts[i:i + len(base_parts)]) == base_parts:
                return PurePosixPath(
                    *parts[i + len(base_parts):]
                )

        return PurePosixPath("_unresolved") / remote.name


def file_url(url_path: str) -> str:
    return FILESERVER_PREFIX + url_path.lstrip("/")


def remote_size(
    session: requests.Session,
    url: str,
) -> int | None:
    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=(30, 60),
        )
        if not response.ok:
            return None

        value = response.headers.get("Content-Length")
        return int(value) if value else None

    except requests.RequestException:
        return None


def download_one(
    session: requests.Session,
    *,
    url: str,
    dest: Path,
    attempts: int = 5,
) -> str:
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = remote_size(
        session,
        url,
    )

    if dest.exists():
        if expected is None or dest.stat().st_size == expected:
            LOGGER.info("SKIP complete: %s", dest)
            return "skip"

        LOGGER.warning(
            "Existing file size mismatch: local=%d remote=%s %s",
            dest.stat().st_size,
            expected,
            dest,
        )

    part = dest.with_suffix(
        dest.suffix + ".part"
    )

    for attempt in range(1, attempts + 1):
        offset = (
            part.stat().st_size
            if part.exists()
            else 0
        )

        headers = (
            {"Range": f"bytes={offset}-"}
            if offset
            else {}
        )

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 240),
            ) as response:
                response.raise_for_status()

                append = (
                    offset > 0
                    and response.status_code == 206
                )

                mode = "ab" if append else "wb"

                with open(part, mode) as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            handle.write(chunk)

            if (
                expected is not None
                and part.stat().st_size != expected
            ):
                raise IOError(
                    "Downloaded size mismatch: "
                    f"{part.stat().st_size} != {expected}"
                )

            part.replace(dest)
            LOGGER.info("DOWNLOADED: %s", dest)
            return "download"

        except Exception as exc:
            LOGGER.warning(
                "Download attempt %d/%d failed: %s: %s",
                attempt,
                attempts,
                url,
                exc,
            )

            if attempt < attempts:
                time.sleep(
                    min(30, 2 ** (attempt - 1))
                )

    LOGGER.error("FAILED: %s", url)
    return "fail"


def drift_aware_is_nh(url_path: str) -> bool:
    """
    Defensive NH-only filter for the drift-aware product.
    """
    upper_parts = {
        p.upper()
        for p in PurePosixPath(url_path).parts
    }

    if "SH" in upper_parts:
        return False

    if "NH" in upper_parts:
        return True

    upper = url_path.upper()

    if "/SH/" in upper:
        return False

    if "/NH/" in upper:
        return True

    # If the archive has no explicit NH component in the path, keep it.
    # The authoritative product itself is NH-only.
    return True


def selected_roots(
    *,
    products: set[str],
    hemispheres: set[str] | None,
    sensors: set[str] | None,
    levels: set[str] | None,
) -> list[DownloadRoot]:
    roots: list[DownloadRoot] = []

    for root in DOWNLOAD_ROOTS:
        if root.product not in products:
            continue

        if (
            hemispheres
            and root.hemisphere
            and root.hemisphere not in hemispheres
        ):
            continue

        if (
            sensors
            and root.sensor
            and root.sensor not in sensors
        ):
            continue

        if (
            levels
            and root.level
            and root.level not in levels
        ):
            continue

        roots.append(root)

    return roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Declarative ESA CCI sea-ice mirror."
    )

    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
    )

    parser.add_argument(
        "--products",
        default=(
            "thickness,"
            "concentration,"
            "thickness_drift_aware"
        ),
    )

    parser.add_argument(
        "--hemispheres",
        default="NH,SH",
        help=(
            "Optional filter. Drift-aware remains NH-only "
            "regardless of this argument."
        ),
    )

    parser.add_argument(
        "--sensors",
        default="",
        help=(
            "Optional SIT sensor filter, e.g. "
            "envisat,cryosat2"
        ),
    )

    parser.add_argument(
        "--levels",
        default="",
        help="Optional SIT level filter, e.g. L2P,L3C.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    products = {
        x.strip()
        for x in args.products.split(",")
        if x.strip()
    }

    valid_products = {
        "thickness",
        "concentration",
        "thickness_drift_aware",
    }

    unknown = products - valid_products
    if unknown:
        raise ValueError(
            f"Unknown products={sorted(unknown)}"
        )

    hemispheres = {
        x.strip().upper()
        for x in args.hemispheres.split(",")
        if x.strip()
    } or None

    sensors = {
        x.strip().lower()
        for x in args.sensors.split(",")
        if x.strip()
    } or None

    levels = {
        x.strip().upper()
        for x in args.levels.split(",")
        if x.strip()
    } or None

    roots = selected_roots(
        products=products,
        hemispheres=hemispheres,
        sensors=sensors,
        levels=levels,
    )

    if not roots:
        raise RuntimeError(
            "No download roots remain after filtering."
        )

    args.dest.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = build_session(
        total_retries=args.retries,
    )

    LOGGER.info(
        "Selected declarative roots=%d",
        len(roots),
    )

    for root in roots:
        LOGGER.info(
            "ROOT product=%s level=%s sensor=%s hemisphere=%s remote=%s",
            root.product,
            root.level or "-",
            root.sensor or "-",
            root.hemisphere or "-",
            root.remote_root,
        )

    total_catalogued = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for i, root in enumerate(roots, 1):
        LOGGER.info("=" * 88)
        LOGGER.info(
            "[%d/%d] Scanning %s",
            i,
            len(roots),
            root.remote_root,
        )

        try:
            files = recursive_catalog(
                session,
                catalog_xml(root.remote_root),
            )
        except Exception as exc:
            LOGGER.error(
                "ROOT FAILED after retries: %s: %s",
                root.remote_root,
                exc,
            )
            total_failed += 1
            continue

        if root.product == "thickness_drift_aware":
            files = [
                (name, path)
                for name, path in files
                if drift_aware_is_nh(path)
            ]

        total_catalogued += len(files)

        LOGGER.info(
            "ROOT catalogued NetCDF files=%d",
            len(files),
        )

        for _, url_path in files:
            rel = remote_relative_path(
                root,
                url_path,
            )

            dest = (
                args.dest
                / Path(root.local_prefix)
                / Path(str(rel))
            )

            url = file_url(url_path)

            if args.dry_run:
                LOGGER.info(
                    "[DRY] %s -> %s",
                    url,
                    dest,
                )
                continue

            result = download_one(
                session,
                url=url,
                dest=dest,
                attempts=args.retries,
            )

            total_downloaded += result == "download"
            total_skipped += result == "skip"
            total_failed += result == "fail"

    LOGGER.info("=" * 88)
    LOGGER.info(
        "FINAL SUMMARY roots=%d catalogued=%d "
        "downloaded=%d skipped=%d failed=%d",
        len(roots),
        total_catalogued,
        total_downloaded,
        total_skipped,
        total_failed,
    )

    if total_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
