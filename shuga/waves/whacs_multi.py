from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NCFile

from shuga.grid.cice import CICEGridwork
from shuga.waves.whacs import WHACSRegridder


WHACS_SPECTRAL_SETS = ("GRID", "GLOB", "BUOYS", "NIWA", "SCHISM")


class WHACSMultiSourceRegridder(WHACSRegridder):
    """Regrid the union of all five WHACS full-spectral point archives.

    WHACS stores hourly directional spectra in five parallel monthly files:
    ``GRID``, ``GLOB``, ``BUOYS``, ``NIWA`` and ``SCHISM``. The base
    :class:`WHACSRegridder` was originally written around the ``GRID`` file
    alone. This subclass preserves the tested direction integration,
    conservative 28->25 frequency remap, station-to-CICE IDW interpolation and
    NetCDF writer, but replaces the monthly source loader with the union of all
    five point sets.

    Fixed station coordinates duplicated between source sets are removed before
    interpolation. Source-set priority is the order in ``WHACS_SPECTRAL_SETS``;
    thus a GRID spectrum is retained when an identical location occurs in a
    later source set.
    """

    spectral_sets: tuple[str, ...] = WHACS_SPECTRAL_SETS
    duplicate_coord_decimals: int = 5

    @staticmethod
    def _set_path(anchor_grid_path: Path, source_set: str) -> Path:
        token = "_GRID_"
        if token not in anchor_grid_path.name:
            raise ValueError(
                f"Expected GRID anchor filename containing {token!r}; got {anchor_grid_path.name}"
            )
        return anchor_grid_path.with_name(
            anchor_grid_path.name.replace(token, f"_{source_set}_", 1)
        )

    @staticmethod
    def _assert_same_axis(
        reference: xr.Dataset,
        candidate: xr.Dataset,
        name: str,
        source_set: str,
    ) -> None:
        if name not in reference or name not in candidate:
            raise KeyError(f"Missing required WHACS axis {name!r} in {source_set}")
        a = np.asarray(reference[name].values)
        b = np.asarray(candidate[name].values)
        if a.shape != b.shape:
            raise ValueError(
                f"WHACS {source_set} {name} shape {b.shape} != reference {a.shape}"
            )
        if np.issubdtype(a.dtype, np.datetime64) or np.issubdtype(b.dtype, np.datetime64):
            same = np.array_equal(a, b)
        else:
            same = np.allclose(a, b, rtol=1.0e-7, atol=1.0e-10, equal_nan=True)
        if not same:
            raise ValueError(f"WHACS {source_set} {name} axis differs from GRID reference")

    def _deduplicate_station_indices(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> np.ndarray:
        """Return first-occurrence indices for fixed lon/lat pairs."""
        lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
        lat = np.asarray(lat, dtype=np.float64)
        if lon.shape != lat.shape:
            raise ValueError("WHACS station longitude/latitude shapes differ")
        valid = np.isfinite(lon) & np.isfinite(lat)
        scale = 10 ** int(self.duplicate_coord_decimals)
        lon_key = np.rint(lon * scale).astype(np.int64)
        lat_key = np.rint(lat * scale).astype(np.int64)
        keep: list[int] = []
        seen: set[tuple[int, int]] = set()
        for i in range(lon.size):
            if not valid[i]:
                continue
            key = (int(lon_key[i]), int(lat_key[i]))
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return np.asarray(keep, dtype=np.int64)

    def open_whacs_month(self, path: str | Path, *, time_chunk: int = 1) -> xr.Dataset:
        anchor = Path(path)
        source_paths = [self._set_path(anchor, source_set) for source_set in self.spectral_sets]
        missing = [p for p in source_paths if not p.exists()]
        if missing:
            missing_text = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Missing WHACS spectral source files:\n  {missing_text}")

        parts: list[xr.Dataset] = []
        reference: xr.Dataset | None = None
        station_offset = 0
        counts: dict[str, int] = {}

        for source_set, source_path in zip(self.spectral_sets, source_paths):
            self._log(f"Opening WHACS {source_set}: {source_path}")
            ds = super().open_whacs_month(source_path, time_chunk=time_chunk)

            if reference is None:
                reference = ds
            else:
                for axis in ("time", "frequency", "direction", "frequency_lo", "frequency_hi"):
                    self._assert_same_axis(reference, ds, axis, source_set)

            keep_vars = ["efth"]
            for name in ("frequency_lo", "frequency_hi"):
                if name in ds:
                    keep_vars.append(name)
            part = ds[keep_vars].copy()
            part = part.assign_coords(
                station_lon=ds["station_lon"],
                station_lat=ds["station_lat"],
            )
            nstation = int(part.sizes["station"])
            part = part.assign_coords(
                station=np.arange(station_offset, station_offset + nstation, dtype=np.int32),
                source_set=("station", np.full(nstation, source_set, dtype="U8")),
            )
            station_offset += nstation
            counts[source_set] = nstation
            parts.append(part)

        combined = xr.concat(
            parts,
            dim="station",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
        )

        before = int(combined.sizes["station"])
        keep = self._deduplicate_station_indices(
            combined["station_lon"].values,
            combined["station_lat"].values,
        )
        combined = combined.isel(station=keep)
        combined = combined.assign_coords(
            station=np.arange(combined.sizes["station"], dtype=np.int32)
        )
        after = int(combined.sizes["station"])

        self._current_whacs_path = ";".join(str(p) for p in source_paths)
        combined.attrs.update(
            whacs_source_sets=",".join(self.spectral_sets),
            whacs_source_station_counts=",".join(
                f"{name}:{counts[name]}" for name in self.spectral_sets
            ),
            whacs_station_count_before_dedup=before,
            whacs_station_count_after_dedup=after,
            whacs_duplicate_coordinate_decimals=int(self.duplicate_coord_decimals),
        )
        self._log(
            "Combined WHACS source stations: "
            + ", ".join(f"{name}={counts[name]}" for name in self.spectral_sets)
            + f"; total={before}, unique={after}, duplicates_removed={before-after}"
        )
        return combined

    def get_target_grid(self, paths) -> xr.Dataset:
        """Load target grid using the current CICEGridwork ``pth_cfg`` API."""
        if self._target_grid is not None:
            return self._target_grid
        gridwork = CICEGridwork(pth_cfg=paths, logger=self.logger)
        bundle = gridwork.load_cice_grid(build_faces=False)
        lon = bundle.tgrid["TLON"]
        lat = bundle.tgrid["TLAT"]
        if self.config.target_lon_type:
            lon = xr.DataArray(
                gridwork.normalise_longitudes(lon.values, to=self.config.target_lon_type),
                dims=lon.dims,
                coords=lon.coords,
                attrs=lon.attrs,
            )
        if bundle.mask is not None:
            ocean_mask = bundle.mask.astype(np.int8)
        else:
            ocean_mask = xr.where(np.isfinite(lon) & np.isfinite(lat), 1, 0).astype(np.int8)
        hemisphere_mask = self._build_hemisphere_mask(lat)
        active_mask = (
            ocean_mask.astype(bool)
            & hemisphere_mask
            & np.isfinite(lon)
            & np.isfinite(lat)
        ).astype(np.int8)
        self._target_grid = xr.Dataset(
            data_vars={
                "TLON": lon.astype(np.float32),
                "TLAT": lat.astype(np.float32),
                "ocean_mask": ocean_mask.astype(np.int8),
                "target_active_mask": active_mask.astype(np.int8),
            }
        )
        return self._target_grid

    def _is_complete_forcing_file(self, path: Path, expected_nt: int | None = None) -> bool:
        """Require a completed forcing file built from the current five-source geometry."""
        if not super()._is_complete_forcing_file(path, expected_nt=expected_nt):
            return False
        try:
            with NCFile(path, mode="r") as nc:
                actual = str(getattr(nc, "source_sets", ""))
                expected = ",".join(self.spectral_sets)
                return actual == expected
        except Exception:
            return False

    def _initialise_forcing_file(self, **kwargs) -> None:
        super()._initialise_forcing_file(**kwargs)
        out = Path(kwargs["out"])
        with NCFile(out, mode="a") as nc:
            nc.source_sets = ",".join(self.spectral_sets)
            nc.source_geometry = "union of fixed WHACS GRID,GLOB,BUOYS,NIWA,SCHISM spectral points"
            nc.duplicate_station_policy = (
                f"first source-set occurrence retained after lon/lat rounding to "
                f"{self.duplicate_coord_decimals} decimal degrees"
            )
