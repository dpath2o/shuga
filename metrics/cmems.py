from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterable, Sequence

import xarray as xr

from shuga.classify.cmems import CMEMSClassifier
from shuga.core.naming import normalize_method
from shuga.io.zarr_writing import sanitise_for_zarr_write
from shuga.metrics.calculations import (
    compute_area_series,
    compute_volume_series,
    compute_thickness_series,
    compute_persistence_mask,
    compute_temporal_mean,
)
from shuga.observations.CMEMS import DEFAULT_ROOT, open_cmems, static_store_path


CMEMS_METRIC_GROUPS: dict[str, list[str]] = {
    "cmems_fi_core": ["FIA", "FIV", "FIT", "FIP", "FIHI"],
    "cmems_pi_core": ["PIA", "PIV", "PIT", "PIP", "PIHI"],
    "cmems_si_core": ["SIA", "SIV", "SIT", "SIP", "SIHI"],
}
CMEMS_METRIC_GROUPS["cmems_core"] = (
    CMEMS_METRIC_GROUPS["cmems_fi_core"]
    + CMEMS_METRIC_GROUPS["cmems_pi_core"]
    + CMEMS_METRIC_GROUPS["cmems_si_core"]
)

_SUPPORTED = set(CMEMS_METRIC_GROUPS["cmems_core"])


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def expand_cmems_metric_names(
    *,
    metric_names: str | Iterable[str] | None = None,
    metric_groups: str | Iterable[str] | None = None,
) -> list[str]:
    names = _as_list(metric_names)
    groups = _as_list(metric_groups)
    # Default to the complete CMEMS set only when neither explicit names
    # nor groups were supplied. Thus metric_names="SIA" remains SIA-only.
    if not names and not groups:
        groups = ["cmems_core"]

    out: list[str] = []
    for group in groups:
        if group not in CMEMS_METRIC_GROUPS:
            raise ValueError(
                f"Unknown CMEMS metric group {group!r}; "
                f"available={sorted(CMEMS_METRIC_GROUPS)}"
            )
        out.extend(CMEMS_METRIC_GROUPS[group])
    out.extend(names)

    result: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in _SUPPORTED:
            raise ValueError(
                f"CMEMS metric {name!r} is not supported by the current source fields. "
                f"Supported={sorted(_SUPPORTED)}"
            )
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def metric_domain(name: str) -> str:
    if name.startswith("FI"):
        return "FI"
    if name.startswith("PI"):
        return "PI"
    if name.startswith("SI"):
        return "SI"
    raise ValueError(f"Cannot infer CMEMS metric domain from {name!r}")


class CMEMSMetrics:
    """
    Metrics runner for CMEMS native-grid FI/PI/SI products.

    This intentionally reuses shuga.metrics.calculations instead of duplicating
    the area/volume/thickness/persistence formulae.
    """

    def __init__(
        self,
        *,
        root: str | Path = DEFAULT_ROOT,
        start_date: str,
        end_date: str,
        hemisphere: str = "SH",
        ispd_thresh: float = 5.0e-4,
        aice_thresh: float = 0.15,
        bin_window: int = 11,
        bin_min_days: int = 9,
        roll_window: int = 15,
        chunks: dict[str, int] | None = None,
        area_scale: float = 1.0e9,
        volume_scale: float = 1.0e12,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.hemisphere = str(hemisphere).upper()
        self.ispd_thresh = float(ispd_thresh)
        self.aice_thresh = float(aice_thresh)
        self.bin_window = int(bin_window)
        self.bin_min_days = int(bin_min_days)
        self.roll_window = int(roll_window)
        self.chunks = chunks or {"time": 31, "latitude": 256, "longitude": 540}
        self.area_scale = float(area_scale)
        self.volume_scale = float(volume_scale)
        self.logger = logger or logging.getLogger("shuga.metrics.cmems")

        self.classifier = CMEMSClassifier(
            root=self.root,
            start_date=self.start_date,
            end_date=self.end_date,
            hemisphere=self.hemisphere,
            ispd_thresh=self.ispd_thresh,
            aice_thresh=self.aice_thresh,
            bin_window=self.bin_window,
            bin_min_days=self.bin_min_days,
            roll_window=self.roll_window,
            chunks=self.chunks,
            logger=self.logger,
        )

        self._data_cache: xr.Dataset | None = None
        self._static_cache: xr.Dataset | None = None
        self._mask_cache: dict[tuple[str, str], xr.DataArray] = {}

    def _data(self) -> xr.Dataset:
        if self._data_cache is None:
            self._data_cache = open_cmems(
                root=self.root,
                start_date=self.start_date,
                end_date=self.end_date,
                hemisphere=self.hemisphere,
                variables=("siconc", "sithick"),
                padding_days=0,
                chunks=self.chunks,
            )
        return self._data_cache

    def _static(self) -> xr.Dataset:
        if self._static_cache is None:
            store = static_store_path(self.root, self.hemisphere)
            if not store.exists():
                raise FileNotFoundError(
                    f"CMEMS static store does not exist: {store}. "
                    "Run classification first (or ensure_static_store())."
                )
            self._static_cache = xr.open_zarr(store, consolidated=True)
        return self._static_cache

    def _mask(self, domain: str, method: str) -> xr.DataArray:
        domain = domain.upper()
        method = normalize_method(method)
        key = (domain, method)
        if key in self._mask_cache:
            return self._mask_cache[key]

        store = self.classifier.classification_store(domain, method)
        if not store.exists():
            raise FileNotFoundError(
                f"Required {domain} classification does not exist: {store}"
            )
        ds = xr.open_zarr(store, consolidated=True, chunks=self.chunks)
        var = f"{domain}_mask"
        if var not in ds:
            raise KeyError(f"{store} does not contain {var!r}")
        mask = ds[var].sel(time=slice(self.start_date, self.end_date))
        self._mask_cache[key] = mask
        return mask

    def metrics_store(self, domain: str, method: str) -> Path:
        domain = domain.upper()
        if domain == "SI":
            return self.classifier.domain_root("SI") / "mets.zarr"
        return (
            self.classifier.classification_store(domain, method).parent
            / "mets.zarr"
        )

    def _compute(self, name: str, method: str) -> xr.DataArray:
        ds = self._data()
        static = self._static()
        aice = ds["aice"]
        hi = ds["hi"]
        area = static["tarea"]

        domain = metric_domain(name)
        if domain == "SI":
            mask = self._mask("SI", method)
        else:
            mask = self._mask(domain, method)

        aice, hi, mask = xr.align(aice, hi, mask, join="inner")

        if name.endswith("A"):
            long_names = {
                "FIA": "Fast Ice Area",
                "PIA": "Pack Ice Area",
                "SIA": "Sea Ice Area",
            }
            return compute_area_series(
                aice,
                area,
                mask,
                name=name,
                long_name=long_names[name],
                scale=self.area_scale,
            )

        if name.endswith("V"):
            long_names = {
                "FIV": "Fast Ice Volume",
                "PIV": "Pack Ice Volume",
                "SIV": "Sea Ice Volume",
            }
            # Mirror the current shuga convention: FI/PI are mask restricted;
            # SI volume is the full concentration-weighted field.
            use_mask = None if name == "SIV" else mask
            return compute_volume_series(
                aice,
                hi,
                area,
                use_mask,
                name=name,
                long_name=long_names[name],
                scale=self.volume_scale,
            )

        if name.endswith("T") and name in {"FIT", "PIT", "SIT"}:
            long_names = {
                "FIT": "Fast Ice Thickness",
                "PIT": "Pack Ice Thickness",
                "SIT": "Sea Ice Thickness",
            }
            use_mask = None if name == "SIT" else mask
            return compute_thickness_series(
                aice,
                hi,
                area,
                use_mask,
                name=name,
                long_name=long_names[name],
            )

        if name.endswith("P"):
            long_names = {
                "FIP": "Fast Ice Persistence",
                "PIP": "Pack Ice Persistence",
                "SIP": "Sea Ice Persistence",
            }
            return compute_persistence_mask(
                mask,
                name=name,
                long_name=long_names[name],
            )

        if name.endswith("IHI"):
            # This branch is retained for readability but current names are
            # FIHI/PIHI/SIHI, so test them explicitly below.
            pass

        if name in {"FIHI", "PIHI", "SIHI"}:
            long_names = {
                "FIHI": "Fast Ice Mean Thickness",
                "PIHI": "Pack Ice Mean Thickness",
                "SIHI": "Sea Ice Mean Thickness",
            }
            return compute_temporal_mean(
                hi.where(mask),
                name=name,
                long_name=long_names[name],
            )

        raise ValueError(f"Unsupported CMEMS metric {name!r}")


    def _prepare_for_zarr(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Rechunk computed CMEMS metrics onto a uniform Zarr-safe layout.

        Alignment of annual CMEMS input chunks with classification-mask chunks
        can create highly fragmented/irregular Dask chunks along time. Zarr
        requires uniform chunks except for the final chunk, so normalise the
        output dataset immediately before writing.
        """
        chunk_map: dict[str, int] = {}

        if "time" in ds.dims:
            chunk_map["time"] = min(
                int(self.chunks.get("time", 31)),
                int(ds.sizes["time"]),
            )

        if "latitude" in ds.dims:
            chunk_map["latitude"] = min(
                int(self.chunks.get("latitude", 256)),
                int(ds.sizes["latitude"]),
            )

        if "longitude" in ds.dims:
            chunk_map["longitude"] = min(
                int(self.chunks.get("longitude", 540)),
                int(ds.sizes["longitude"]),
            )

        for dim, size in ds.sizes.items():
            if dim not in chunk_map:
                chunk_map[dim] = int(size)

        out = sanitise_for_zarr_write(ds)

        if chunk_map:
            self.logger.info(
                "Rechunking CMEMS metrics for Zarr write: %s",
                ", ".join(f"{dim}={size}" for dim, size in chunk_map.items()),
            )
            out = out.chunk(chunk_map)

        return out

    def compute_metrics(
        self,
        method: str,
        *,
        metric_names: str | Iterable[str] | None = None,
        metric_groups: str | Iterable[str] | None = None,
        overwrite: bool = False,
        update_missing_only: bool = True,
    ) -> dict[str, Path]:
        method = normalize_method(method)
        requested = expand_cmems_metric_names(
            metric_names=metric_names,
            metric_groups=metric_groups,
        )

        by_domain: dict[str, list[str]] = {"FI": [], "PI": [], "SI": []}
        for name in requested:
            by_domain[metric_domain(name)].append(name)

        outputs: dict[str, Path] = {}
        for domain, names in by_domain.items():
            if not names:
                continue

            store = self.metrics_store(domain, method)
            existing_names: set[str] = set()
            if store.exists() and not overwrite:
                existing = xr.open_zarr(store, consolidated=True)
                existing_names = set(existing.data_vars)
                existing.close()

            todo = names
            if update_missing_only and not overwrite:
                todo = [name for name in names if name not in existing_names]

            if not todo:
                self.logger.info(
                    "%s metrics already present in %s; nothing to do",
                    domain,
                    store,
                )
                outputs[domain] = store
                continue

            self.logger.info(
                "Computing CMEMS %s metrics for %s: %s",
                domain,
                method,
                ", ".join(todo),
            )
            out = xr.Dataset({name: self._compute(name, method) for name in todo})
            out.attrs.update(
                {
                    "source": "CMEMS 0.083-degree daily",
                    "hemisphere": self.hemisphere,
                    "grid_type": "native",
                    "classification_method": method if domain != "SI" else "concentration",
                    "ispd_thresh_m_s": self.ispd_thresh,
                    "aice_thresh": self.aice_thresh,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                }
            )

            # xr.align() between annual CMEMS data and classified masks can
            # produce irregular time chunks (e.g. 24,7,24,7,...). Normalise
            # chunks here so xarray/Zarr receives a legal chunk structure.
            out = self._prepare_for_zarr(out)

            store.parent.mkdir(parents=True, exist_ok=True)

            if overwrite or not store.exists():
                out.to_zarr(store, mode="w", consolidated=True)
            else:
                # Only missing variables are written, avoiding a read/rewrite
                # of the existing metrics store.
                out.to_zarr(store, mode="a", consolidated=True)

            outputs[domain] = store
            self.logger.info("Wrote CMEMS %s metrics: %s", domain, store)

        return outputs
