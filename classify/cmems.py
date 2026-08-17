from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from shuga.core.naming import method_dirname, normalize_method, threshold_tag_dir
from shuga.observations.CMEMS import (
    DEFAULT_ROOT,
    ensure_static_store,
    open_cmems,
)


class CMEMSClassifier:
    """
    Native-grid FI/PI/SI classifier for the CMEMS 0.083-degree daily product.

    CMEMS usi/vsi are already collocated eastward/northward sea-ice velocity
    components, so native speed is simply hypot(usi, vsi). No CICE grid
    reconstruction is performed.
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
        self.logger = logger or logging.getLogger("shuga.classify.cmems")
        self._ds_cache: xr.Dataset | None = None

    def _required_padding_days(self, methods: Sequence[str]) -> int:
        pads = [0]
        methods = [normalize_method(m) for m in methods]
        if "binary-days" in methods:
            pads.append(self.bin_window // 2)
        if "rolling-mean" in methods:
            pads.append(self.roll_window // 2)
        return max(pads)

    def _load(self, methods: Sequence[str]) -> xr.Dataset:
        if self._ds_cache is None:
            padding = self._required_padding_days(methods)
            self.logger.info(
                "Opening CMEMS %s..%s with %d padding days",
                self.start_date,
                self.end_date,
                padding,
            )
            self._ds_cache = open_cmems(
                root=self.root,
                start_date=self.start_date,
                end_date=self.end_date,
                hemisphere=self.hemisphere,
                variables=("siconc", "usi", "vsi"),
                padding_days=padding,
                chunks=self.chunks,
            )
        return self._ds_cache

    def _crop(self, da: xr.DataArray) -> xr.DataArray:
        return da.sel(time=slice(self.start_date, self.end_date))

    def compute_speed(self, ds: xr.Dataset) -> xr.DataArray:
        speed = np.hypot(ds["uice"], ds["vice"]).astype(np.float32)
        speed.name = "ice_speed"
        speed.attrs.update(
            {
                "long_name": "CMEMS native-grid sea-ice speed magnitude",
                "units": "m s-1",
                "grid_type": "native",
            }
        )
        return speed

    def sea_ice_mask(self, aice: xr.DataArray) -> xr.DataArray:
        mask = (aice >= self.aice_thresh).astype(bool)
        mask.name = "SI_mask"
        mask.attrs.update(
            {
                "long_name": "Sea-ice mask from CMEMS concentration",
                "definition": "SI_mask = aice >= aice_thresh",
                "aice_thresh": self.aice_thresh,
            }
        )
        return mask

    def compute_raw_fi_mask(self, ds: xr.Dataset) -> xr.DataArray:
        speed = self.compute_speed(ds)
        aice = ds["aice"]
        mask = (
            (aice > self.aice_thresh)
            & np.isfinite(speed)
            & (speed > 0.0)
            & (speed <= self.ispd_thresh)
        )
        mask.name = "FI_mask"
        mask.attrs.update(
            {
                "long_name": "CMEMS fast-ice raw daily mask",
                "ispd_thresh_m_s": self.ispd_thresh,
                "aice_thresh": self.aice_thresh,
                "classification_method": "raw",
                "grid_type": "native",
            }
        )
        return mask.astype(bool)

    def classify_fi(self, method: str, ds: xr.Dataset | None = None) -> xr.DataArray:
        method = normalize_method(method)
        ds = ds if ds is not None else self._load((method,))

        if method == "raw":
            mask = self.compute_raw_fi_mask(ds)

        elif method == "binary-days":
            raw = self.compute_raw_fi_mask(ds).astype(np.int16)
            mask = (
                raw.rolling(
                    time=self.bin_window,
                    center=True,
                    min_periods=self.bin_min_days,
                ).sum()
                >= self.bin_min_days
            )
            mask.name = "FI_mask"
            mask.attrs.update(
                {
                    "long_name": "CMEMS fast-ice binary-days mask",
                    "classification_method": "binary-days",
                    "bin_window": self.bin_window,
                    "bin_min_days": self.bin_min_days,
                    "grid_type": "native",
                }
            )

        elif method == "rolling-mean":
            speed = self.compute_speed(ds)
            aice = ds["aice"]
            roll_speed = speed.rolling(
                time=self.roll_window,
                center=True,
                min_periods=self.roll_window,
            ).mean()
            mask = (
                (aice > self.aice_thresh)
                & np.isfinite(roll_speed)
                & (roll_speed > 0.0)
                & (roll_speed <= self.ispd_thresh)
            )
            mask.name = "FI_mask"
            mask.attrs.update(
                {
                    "long_name": "CMEMS fast-ice rolling-mean-speed mask",
                    "classification_method": "rolling-mean",
                    "roll_window": self.roll_window,
                    "grid_type": "native",
                }
            )
        else:  # normalize_method already protects this.
            raise ValueError(method)

        return self._crop(mask.astype(bool))

    def classify_pi(self, fi_mask: xr.DataArray, aice: xr.DataArray) -> xr.DataArray:
        aice = self._crop(aice)
        fi_mask, aice = xr.align(fi_mask, aice, join="exact")
        pi = self.sea_ice_mask(aice) & (~fi_mask.astype(bool))
        pi.name = "PI_mask"
        pi.attrs.update(
            {
                "long_name": "CMEMS pack-ice mask derived as sea ice excluding fast ice",
                "definition": "PI_mask = SI_mask & ~FI_mask",
                "aice_thresh": self.aice_thresh,
            }
        )
        return pi.astype(bool)

    def domain_root(self, domain: str) -> Path:
        domain = str(domain).strip().upper()
        if domain == "SI":
            return self.root / self.hemisphere / "SI"
        if domain not in {"FI", "PI"}:
            raise ValueError(f"Unsupported ice domain {domain!r}")
        return (
            self.root
            / self.hemisphere
            / f"ispd_thresh_{threshold_tag_dir(self.ispd_thresh)}"
            / domain
            / "native"
        )

    def classification_store(self, domain: str, method: str = "raw") -> Path:
        domain = str(domain).strip().upper()
        if domain == "SI":
            return self.domain_root("SI") / "data.zarr"
        return (
            self.domain_root(domain)
            / method_dirname(
                method,
                bin_window=self.bin_window,
                bin_min_days=self.bin_min_days,
                roll_window=self.roll_window,
            )
            / "data.zarr"
        )

    def _write_mask(
        self,
        mask: xr.DataArray,
        store: Path,
        *,
        overwrite: bool,
    ) -> Path:
        if store.exists() and not overwrite:
            self.logger.info("Classification store exists; skipping: %s", store)
            return store

        store.parent.mkdir(parents=True, exist_ok=True)
        ds_out = mask.to_dataset()
        ds_out.attrs.update(
            {
                "source": "CMEMS 0.083-degree daily",
                "hemisphere": self.hemisphere,
                "grid_type": "native",
                "start_date": self.start_date,
                "end_date": self.end_date,
            }
        )
        ds_out = ds_out.chunk(
            {
                "time": min(31, max(1, int(ds_out.sizes.get("time", 1)))),
                "latitude": min(256, int(ds_out.sizes["latitude"])),
                "longitude": min(540, int(ds_out.sizes["longitude"])),
            }
        )
        self.logger.info("Writing classification: %s", store)
        ds_out.to_zarr(store, mode="w", consolidated=True)
        return store

    def run_methods(
        self,
        *,
        methods: Sequence[str] = ("raw", "binary-days", "rolling-mean"),
        overwrite: bool = False,
        overwrite_static: bool = False,
    ) -> dict[str, Path]:
        methods = tuple(normalize_method(m) for m in methods)
        ds = self._load(methods)

        static = ensure_static_store(
            root=self.root,
            hemisphere=self.hemisphere,
            overwrite=overwrite_static,
        )
        self.logger.info("CMEMS static grid store: %s", static)

        # SI is method-independent.
        si = self._crop(self.sea_ice_mask(ds["aice"]))
        outputs: dict[str, Path] = {}
        outputs["SI"] = self._write_mask(
            si,
            self.classification_store("SI"),
            overwrite=overwrite,
        )

        for method in methods:
            fi = self.classify_fi(method, ds=ds)
            pi = self.classify_pi(fi, ds["aice"])

            fi_store = self.classification_store("FI", method)
            pi_store = self.classification_store("PI", method)

            outputs[f"FI:{method}"] = self._write_mask(
                fi, fi_store, overwrite=overwrite
            )
            outputs[f"PI:{method}"] = self._write_mask(
                pi, pi_store, overwrite=overwrite
            )

        return outputs
