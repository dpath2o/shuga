from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/g/data/gv90/da1339/SeaIce/InSitu/raw")

AADC_BASE = "https://data.aad.gov.au"
CKAN_API = "https://www.data.gov.au/data/api/3/action"
ZENODO_API = "https://zenodo.org/api/records"

# Direct AADC EDS endpoint documented for the Meiners/AFIAC dataset.
AFIAC_DIRECT_URL = "https://data.aad.gov.au/eds/4836/download"

# Metadata records. Runtime discovery follows each current AADC "Download the
# dataset" link rather than relying on historical storage URLs.
AADC_RECORDS = {
    "casey_station": {
        "entry_id": "AADC-00107",
        "title": "Casey Station Antarctica Ice Thickness Data",
    },
    "mawson_station": {
        "entry_id": None,
        "title": "Mawson Station Antarctica Ice Thickness Data",
    },
    "davis_station": {
        "entry_id": None,
        "title": "Davis Station Antarctica Ice Thickness Data",
    },
    "soe_fast_ice": {
        "entry_id": "SOE_fast_ice_thickness",
        "title": "Fast ice thickness at Davis, Mawson and Casey",
    },
    "asac2500": {
        "entry_id": "ASAC_2500",
        "title": "Variability of the coastal Antarctic climate derived from fast-ice observations.",
    },
    "davis2015": {
        "entry_id": "AAS_4298_Davis_Ice_Transects",
        "title": "Ice-physics transects collected in fast ice areas at Davis Station in November-December 2015",
    },
}

ZENODO_RECORDS = {
    "bepsii": {
        "record_id": 19203653,
        "title": "BEPSII-TS sea-ice core compilation",
    },
}

DOWNLOAD_EXTENSIONS = {
    ".zip", ".csv", ".txt", ".xlsx", ".xls", ".nc", ".nc4", ".dat",
    ".json", ".doc", ".docx", ".pdf", ".rtf", ".tsv",
}

NONDATA_FORMATS = {
    "PNG", "JPG", "JPEG", "GIF", "SVG",
}


@dataclass
class DownloadRecord:
    source: str
    title: str
    url: str
    local_path: str
    status: str
    size_bytes: int | None = None
    sha256: str | None = None
    note: str | None = None


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def build_session(retries: int = 5) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
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
        {"User-Agent": "shuga-insitu-thickness-downloader/0.2"}
    )
    return session


def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w.\-]+", "_", value.strip())
    return value.strip("_") or "download"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _filename_from_response(
    response: requests.Response,
    fallback: str,
) -> str:
    cd = response.headers.get("Content-Disposition", "")
    match = re.search(
        r'filename\*?=(?:UTF-8\'\')?"?([^";]+)',
        cd,
        flags=re.I,
    )
    if match:
        return Path(match.group(1).strip()).name

    final_name = Path(urlparse(response.url).path).name
    if final_name and "." in final_name:
        return final_name

    return fallback


def _html_links(base_url: str, html: str) -> list[tuple[str, str]]:
    parser = _LinkParser()
    parser.feed(html)

    out: list[tuple[str, str]] = []
    for href, label in parser.links:
        if not href:
            continue
        out.append((urljoin(base_url, href), label))
    return out


def _download_candidates(
    base_url: str,
    html: str,
) -> list[str]:
    candidates = []

    for url, label in _html_links(base_url, html):
        suffix = Path(urlparse(url).path).suffix.lower()
        text = label.lower()

        if (
            suffix in DOWNLOAD_EXTENSIONS
            or "/eds/" in url
            or "/dataset/" in url
            or "download the dataset" in text
            or "download dataset" in text
            or text == "download"
            or "get data" in text
        ):
            if url != base_url:
                candidates.append(url)

    def rank(url: str) -> tuple[int, int, str]:
        suffix = Path(urlparse(url).path).suffix.lower()
        return (
            0 if "/eds/" in url else 1,
            0 if suffix in DOWNLOAD_EXTENSIONS else 1,
            url,
        )

    return sorted(set(candidates), key=rank)


def inspect_url(
    session: requests.Session,
    url: str,
) -> dict:
    r = session.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(30, 120),
    )
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].lower()

    result = {
        "requested_url": url,
        "final_url": r.url,
        "content_type": ctype,
        "content_length": r.headers.get("Content-Length"),
        "content_disposition": r.headers.get("Content-Disposition"),
    }

    if ctype == "text/html":
        result["download_candidates"] = _download_candidates(
            r.url,
            r.text,
        )
    else:
        result["filename"] = _filename_from_response(
            r,
            "download",
        )

    r.close()
    return result


def download_url(
    session: requests.Session,
    *,
    url: str,
    outdir: Path,
    fallback_name: str,
    overwrite: bool = False,
    dry_run: bool = False,
    recursion_depth: int = 0,
) -> DownloadRecord:
    if recursion_depth > 8:
        return DownloadRecord(
            source="http",
            title=fallback_name,
            url=url,
            local_path="",
            status="manual",
            note="Exceeded HTML/download-link recursion depth.",
        )

    outdir.mkdir(parents=True, exist_ok=True)

    with session.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(30, 240),
    ) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").split(";")[0].lower()

        if ctype == "text/html":
            candidates = _download_candidates(r.url, r.text)

            if not candidates:
                return DownloadRecord(
                    source="html",
                    title=fallback_name,
                    url=r.url,
                    local_path="",
                    status="manual",
                    note="No machine-detectable data/download child link.",
                )

            if dry_run:
                return DownloadRecord(
                    source="html",
                    title=fallback_name,
                    url=candidates[0],
                    local_path=str(outdir),
                    status="discovered",
                    note=f"Resolved HTML landing page to {len(candidates)} candidate link(s).",
                )

            return download_url(
                session,
                url=candidates[0],
                outdir=outdir,
                fallback_name=fallback_name,
                overwrite=overwrite,
                dry_run=False,
                recursion_depth=recursion_depth + 1,
            )

        filename = _filename_from_response(r, fallback_name)
        target = outdir / _safe_name(filename)
        expected = (
            int(r.headers["Content-Length"])
            if r.headers.get("Content-Length", "").isdigit()
            else None
        )

        if dry_run:
            return DownloadRecord(
                source="http",
                title=fallback_name,
                url=r.url,
                local_path=str(target),
                status="discovered",
                size_bytes=expected,
            )

        if target.exists() and not overwrite:
            if expected is None or target.stat().st_size == expected:
                return DownloadRecord(
                    source="http",
                    title=fallback_name,
                    url=r.url,
                    local_path=str(target),
                    status="skip",
                    size_bytes=target.stat().st_size,
                    sha256=_sha256(target),
                )

        part = target.with_suffix(target.suffix + ".part")
        offset = (
            part.stat().st_size
            if part.exists() and not overwrite
            else 0
        )

    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with session.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 240),
    ) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").split(";")[0].lower()

        if ctype == "text/html":
            candidates = _download_candidates(r.url, r.text)
            if not candidates:
                return DownloadRecord(
                    source="html",
                    title=fallback_name,
                    url=r.url,
                    local_path="",
                    status="manual",
                )

            return download_url(
                session,
                url=candidates[0],
                outdir=outdir,
                fallback_name=fallback_name,
                overwrite=overwrite,
                dry_run=False,
                recursion_depth=recursion_depth + 1,
            )

        filename = _filename_from_response(r, fallback_name)
        target = outdir / _safe_name(filename)
        part = target.with_suffix(target.suffix + ".part")

        append = offset > 0 and r.status_code == 206
        mode = "ab" if append else "wb"

        with part.open(mode) as fh:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)

    part.replace(target)

    return DownloadRecord(
        source="http",
        title=fallback_name,
        url=url,
        local_path=str(target),
        status="download",
        size_bytes=target.stat().st_size,
        sha256=_sha256(target),
    )


def aadc_metadata_url(entry_id: str) -> str:
    return f"{AADC_BASE}/metadata/records/{entry_id}"


def discover_aadc_download_url(
    session: requests.Session,
    entry_id: str,
) -> str | None:
    url = aadc_metadata_url(entry_id)
    r = session.get(
        url,
        timeout=(30, 120),
    )
    r.raise_for_status()

    links = _html_links(r.url, r.text)

    preferred = []
    for child, label in links:
        label_low = label.lower()
        if (
            "download the dataset" in label_low
            or "download dataset" in label_low
            or "/eds/" in child
        ):
            preferred.append(child)

    if preferred:
        preferred.sort(
            key=lambda x: 0 if "/eds/" in x else 1
        )
        return preferred[0]

    return None


def _ckan_search_exact(
    session: requests.Session,
    title: str,
) -> dict | None:
    r = session.get(
        f"{CKAN_API}/package_search",
        params={"q": f'title:"{title}"', "rows": 20},
        timeout=(30, 120),
    )
    r.raise_for_status()

    payload = r.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN search failed for {title!r}")

    results = payload["result"]["results"]

    exact = [
        package
        for package in results
        if package.get("title", "").strip().lower()
        == title.strip().lower()
    ]
    if exact:
        return exact[0]

    return results[0] if results else None


def _resource_is_candidate(resource: dict) -> bool:
    url = str(resource.get("url", ""))
    name = str(resource.get("name", "")).lower()
    fmt = str(resource.get("format", "")).upper()
    suffix = Path(urlparse(url).path).suffix.lower()

    if fmt in NONDATA_FORMATS:
        return False

    return (
        suffix in DOWNLOAD_EXTENSIONS
        or "/eds/" in url
        or "download" in name
        or "get data" in name
        or fmt
        in {
            "CSV",
            "ZIP",
            "XLS",
            "XLSX",
            "NETCDF",
            "TXT",
            "DOC",
            "DOCX",
            "TSV",
        }
    )


def download_afiac(
    session: requests.Session,
    *,
    root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[DownloadRecord]:
    rec = download_url(
        session,
        url=AFIAC_DIRECT_URL,
        outdir=root / "AADC" / "AFIAC",
        fallback_name="AFIAC_AAS_4298",
        overwrite=overwrite,
        dry_run=dry_run,
    )
    rec.source = "AADC:AFIAC"
    rec.title = "Antarctic fast-ice ice-core compilation (Meiners et al.)"
    return [rec]


def download_aadc_record(
    session: requests.Session,
    *,
    key: str,
    entry_id: str | None,
    title: str,
    root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[DownloadRecord]:
    outdir = root / "AADC" / key
    outdir.mkdir(parents=True, exist_ok=True)

    records: list[DownloadRecord] = []

    # Preferred route: current AADC metadata page -> current download URL.
    if entry_id:
        metadata_url = aadc_metadata_url(entry_id)

        try:
            download_url_current = discover_aadc_download_url(
                session,
                entry_id,
            )

            metadata = {
                "entry_id": entry_id,
                "title": title,
                "metadata_url": metadata_url,
                "download_url": download_url_current,
            }
            (outdir / "aadc_metadata_resolution.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            if download_url_current:
                rec = download_url(
                    session,
                    url=download_url_current,
                    outdir=outdir,
                    fallback_name=key,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                rec.source = f"AADC:{entry_id}"
                rec.title = title
                records.append(rec)
                return records

        except Exception as exc:
            LOGGER.warning(
                "AADC metadata-route discovery failed for %s: %s",
                key,
                exc,
            )

    # Fallback route: discover current public resources via data.gov.au CKAN.
    package = _ckan_search_exact(session, title)

    if package is None:
        records.append(
            DownloadRecord(
                source=f"AADC:{entry_id or key}",
                title=title,
                url="",
                local_path="",
                status="not-found",
                note="Neither AADC metadata nor data.gov.au discovery returned a source.",
            )
        )
        return records

    (outdir / "data_gov_au_package.json").write_text(
        json.dumps(package, indent=2),
        encoding="utf-8",
    )

    resources = [
        resource
        for resource in package.get("resources", [])
        if _resource_is_candidate(resource)
    ]

    if not resources:
        records.append(
            DownloadRecord(
                source=f"data.gov.au:{key}",
                title=title,
                url=package.get("url", ""),
                local_path=str(
                    outdir / "data_gov_au_package.json"
                ),
                status="metadata-only",
                note="No machine-downloadable resource detected in CKAN package.",
            )
        )
        return records

    for i, resource in enumerate(resources, 1):
        url = resource.get("url")
        if not url:
            continue

        fallback = (
            resource.get("name")
            or f"{key}_{i}"
        )

        try:
            rec = download_url(
                session,
                url=url,
                outdir=outdir,
                fallback_name=_safe_name(str(fallback)),
                overwrite=overwrite,
                dry_run=dry_run,
            )
            rec.source = f"data.gov.au:{key}"
            rec.title = title
            records.append(rec)

        except Exception as exc:
            records.append(
                DownloadRecord(
                    source=f"data.gov.au:{key}",
                    title=title,
                    url=url,
                    local_path="",
                    status="failed",
                    note=str(exc),
                )
            )

    return records


def download_bepsii(
    session: requests.Session,
    *,
    root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    include_logsheets: bool = False,
) -> list[DownloadRecord]:
    info = ZENODO_RECORDS["bepsii"]
    record_id = info["record_id"]

    r = session.get(
        f"{ZENODO_API}/{record_id}",
        timeout=(30, 120),
    )
    r.raise_for_status()
    meta = r.json()

    outdir = root / "Zenodo" / "BEPSII_TS"
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "zenodo_record.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    records: list[DownloadRecord] = []

    for file_info in meta.get("files", []):
        name = (
            file_info.get("key")
            or file_info.get("filename")
            or "zenodo_file"
        )
        name_low = name.lower()

        keep = (
            "data" in name_low
            or "readme" in name_low
            or "documentation" in name_low
            or "doc" in name_low
            or name_low.endswith(".xlsx")
            or name_low.endswith(".csv")
            or name_low.endswith(".zip")
        )

        if "logsheet" in name_low and not include_logsheets:
            keep = False

        if not keep:
            continue

        links = file_info.get("links", {})
        url = links.get("self") or links.get("content")

        if not url:
            continue

        try:
            rec = download_url(
                session,
                url=url,
                outdir=outdir,
                fallback_name=name,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            rec.source = f"Zenodo:{record_id}"
            rec.title = meta.get(
                "metadata",
                {},
            ).get(
                "title",
                info["title"],
            )
            records.append(rec)

        except Exception as exc:
            records.append(
                DownloadRecord(
                    source=f"Zenodo:{record_id}",
                    title=info["title"],
                    url=url,
                    local_path="",
                    status="failed",
                    note=str(exc),
                )
            )

    return records


def write_manifest(
    records: list[DownloadRecord],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            [asdict(record) for record in records],
            indent=2,
        ),
        encoding="utf-8",
    )
