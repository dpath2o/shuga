from __future__ import annotations

import calendar
import fcntl
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as NCFile

from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec
from shuga.observations import SeaIceObservations
from shuga.waves.cawcr import CAWCRRegridConfig, CAWCRRegridder, _time_to_netcdf_numeric


WHACS_SOURCE_ROOT = Path(
    "/g/data/ia39/WP3/release/ACS_hindcast/spec/release/WP3/WHACS/"
    "BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/spectra/1hr/efth"
)

# Noah Day's nfreq=25 WaveWatch-style frequency grid in Icepack.
CICE25_WAVEFREQ = np.asarray(
    [
        0.04118, 0.045298, 0.0498278, 0.05481058, 0.06029164,
        0.06632081, 0.07295289, 0.08024818, 0.08827299, 0.09710029,
        0.10681032, 0.11749136, 0.1292405, 0.14216454, 0.15638101,
        0.17201911, 0.18922101, 0.20814312, 0.22895744, 0.25185317,
        0.27703848, 0.30474234, 0.33521661, 0.36873826, 0.40561208,
    ],
    dtype=np.float64,
)
CICE25_FREQ_RATIO = 1.1


def cice25_frequency_metadata() -> xr.Dataset:
    """Return the exact 25-bin frequency metadata expected by Noah's Icepack branch."""
    root_ratio = np.sqrt(CICE25_FREQ_RATIO)
    lo = CICE25_WAVEFREQ / root_ratio
    hi = CICE25_WAVEFREQ * root_ratio
    width = hi - lo
    nfreq = np.arange(CICE25_WAVEFREQ.size, dtype=np.int32)
    return xr.Dataset(
        data_vars={
            "wavefreq": ("nfreq", CICE25_WAVEFREQ.astype(np.float32)),
            "wavefreq_lo": ("nfreq", lo.astype(np.float32)),
            "wavefreq_hi": ("nfreq", hi.astype(np.float32)),
            "dwavefreq": ("nfreq", width.astype(np.float32)),
        },
        coords={"nfreq": nfreq},
        attrs={
            "frequency_grid": "NoahDay CICE/Icepack nfreq=25 geometric WaveWatch grid",
            "frequency_ratio": CICE25_FREQ_RATIO,
        },
    )


class WHACSRegridder(CAWCRRegridder):
    """
    Build monthly hourly WHACS wave spectra for standalone CICE6.

    Processing sequence
    -------------------
    1. Integrate WHACS ``efth(f, theta)`` over direction to obtain ``E(f)``.
    2. Conservatively remap spectral variance from the native WHACS frequency
       bins onto the 25-bin frequency grid expected by Noah Day's Icepack code.
    3. Regrid station spectra to the native CICE T grid using static sparse
       inverse-distance weights.
    4. Zero incident wave energy where daily NSIDC SIC >= ``sic_threshold``;
       Noah's CICE wave propagation then carries incident wave energy into ice.
    5. Write a lean hourly monthly NetCDF containing the CICE forcing field
       ``efreq(time,nfreq,nj,ni)`` plus frequency and grid coordinates.

    The output filename intentionally retains the historical
    ``CAWCR_efreq_for_CICE6_YYYYMM.nc`` convention so existing downstream CICE
    namelist/code paths can be revised without another forcing rename.
    """

    def __init__(
        self,
        config: CAWCRRegridConfig,
        *,
        source_root: str | Path = WHACS_SOURCE_ROOT,
        logger=None,
    ) -> None:
        super().__init__(config=config, logger=logger)
        self.source_root = Path(source_root)
        self._current_whacs_path: Path | None = None

    # ------------------------------------------------------------------
    # WHACS paths / source loading
    # ------------------------------------------------------------------
    def whacs_file(self, year: int, month: int) -> Path:
        last_day = calendar.monthrange(year, month)[1]
        name = (
            f"efth_WHACS_hindcast_spec_GRID_1hr_"
            f"{year:04d}{month:02d}010000-{year:04d}{month:02d}{last_day:02d}2300.nc"
        )
        return self.source_root / name

    def open_whacs_month(self, path: str | Path, *, time_chunk: int = 1) -> xr.Dataset:
        # Dask-backed reads are important here: native efth has dimensions
        # time x station x frequency x direction and is much larger than memory
        # needed for a single regridding chunk.
        chunks = {
            "time": max(1, int(time_chunk)),
            "station": 512,
            "frequency": -1,
            "direction": -1,
        }
        ds = xr.open_dataset(path, chunks=chunks)
        return self.normalise_cawcr_names(ds)

    # ------------------------------------------------------------------
    # spectral remapping
    # ------------------------------------------------------------------
    @staticmethod
    def _source_frequency_edges(ds_raw: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        if "frequency_lo" in ds_raw and "frequency_hi" in ds_raw:
            lo = np.asarray(ds_raw["frequency_lo"].values, dtype=np.float64)
            hi = np.asarray(ds_raw["frequency_hi"].values, dtype=np.float64)
        else:
            f = np.asarray(ds_raw["frequency"].values, dtype=np.float64)
            edges = np.empty(f.size + 1, dtype=np.float64)
            edges[1:-1] = 0.5 * (f[:-1] + f[1:])
            edges[0] = f[0] - 0.5 * (f[1] - f[0])
            edges[-1] = f[-1] + 0.5 * (f[-1] - f[-2])
            lo, hi = edges[:-1], edges[1:]
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
            raise ValueError("Invalid WHACS frequency-bin bounds.")
        return lo, hi

    @staticmethod
    def _spectral_overlap_matrix(
        src_lo: np.ndarray,
        src_hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        freq_ds = cice25_frequency_metadata()
        tgt_lo = freq_ds["wavefreq_lo"].values.astype(np.float64)
        tgt_hi = freq_ds["wavefreq_hi"].values.astype(np.float64)
        tgt_width = tgt_hi - tgt_lo

        overlap = np.maximum(
            0.0,
            np.minimum(src_hi[:, None], tgt_hi[None, :])
            - np.maximum(src_lo[:, None], tgt_lo[None, :]),
        )
        # Mapping converts source density E_j [m2 s] to target density E_k:
        # E_k = sum_j(E_j * overlap_jk) / delta_f_k.
        mapping = overlap / tgt_width[None, :]
        return mapping, tgt_lo, tgt_hi

    def remap_to_cice25(
        self,
        efreq_station_native: xr.DataArray,
        ds_raw: xr.Dataset,
    ) -> xr.DataArray:
        """Conservatively remap E(f) from WHACS bins to the CICE25 bins."""
        src_lo, src_hi = self._source_frequency_edges(ds_raw)
        mapping, _, _ = self._spectral_overlap_matrix(src_lo, src_hi)

        mapper = xr.DataArray(
            mapping.astype(np.float32),
            dims=("frequency", "cice_frequency"),
            coords={
                "frequency": efreq_station_native["frequency"],
                "cice_frequency": CICE25_WAVEFREQ.astype(np.float32),
            },
            name="frequency_overlap_density_mapping",
        )
        out = xr.dot(efreq_station_native, mapper, dim="frequency")
        out = out.rename({"cice_frequency": "frequency"})
        out = out.transpose("time", "station", "frequency")
        out = out.assign_coords(frequency=CICE25_WAVEFREQ.astype(np.float32))
        out.name = "efreq_station"
        out.attrs.update(
            {
                "long_name": "direction-integrated WHACS spectrum conservatively remapped to CICE25",
                "units": efreq_station_native.attrs.get("units", "m2 s"),
                "spectral_remap": "bin-overlap variance-conserving density remap",
            }
        )
        return out

    def _log_spectral_qc(
        self,
        native: xr.DataArray,
        remapped: xr.DataArray,
        ds_raw: xr.Dataset,
        max_stations: int = 128,
    ) -> None:
        """Log sampled retained m0/Hs after the 28->25 frequency remap."""
        ns = int(native.sizes["station"])
        stride = max(1, ns // max_stations)
        native_sample = native.isel(time=0, station=slice(0, None, stride))
        remap_sample = remapped.isel(time=0, station=slice(0, None, stride))

        src_lo, src_hi = self._source_frequency_edges(ds_raw)
        src_width = xr.DataArray(
            (src_hi - src_lo).astype(np.float32),
            dims=("frequency",),
            coords={"frequency": native["frequency"]},
        )
        tgt = cice25_frequency_metadata()
        tgt_width = xr.DataArray(
            tgt["dwavefreq"].values.astype(np.float32),
            dims=("frequency",),
            coords={"frequency": remapped["frequency"]},
        )

        src_m0 = (native_sample * src_width).sum("frequency").compute().values
        tgt_m0 = (remap_sample * tgt_width).sum("frequency").compute().values
        good = np.isfinite(src_m0) & np.isfinite(tgt_m0) & (src_m0 > 1.0e-12)
        if not np.any(good):
            self._log("Spectral QC: no positive sampled m0 values available.")
            return

        ratio = tgt_m0[good] / src_m0[good]
        self._log(
            "Spectral QC retained m0 (CICE25/native WHACS): "
            f"median={np.nanmedian(ratio):.5f}, "
            f"p05={np.nanpercentile(ratio, 5):.5f}, "
            f"p95={np.nanpercentile(ratio, 95):.5f}, "
            f"n={ratio.size}"
        )

    @staticmethod
    def _validate_hourly_month(time_values: np.ndarray, year: int, month: int) -> None:
        start = pd.Timestamp(year=year, month=month, day=1, hour=0)
        stop = start + pd.offsets.MonthBegin(1)
        expected = pd.date_range(start, stop, freq="1h", inclusive="left")
        actual = pd.DatetimeIndex(pd.to_datetime(time_values))
        if len(actual) != len(expected):
            raise ValueError(
                f"WHACS hourly record count mismatch for {year:04d}-{month:02d}: "
                f"got {len(actual)}, expected {len(expected)}."
            )
        if not np.array_equal(actual.values, expected.values):
            missing = expected.difference(actual)
            extra = actual.difference(expected)
            raise ValueError(
                f"WHACS timestamps are not a complete exact hourly month for {year:04d}-{month:02d}; "
                f"missing={list(missing[:5])}, extra={list(extra[:5])}."
            )

    # ------------------------------------------------------------------
    # shared static weight files with inter-process locking
    # ------------------------------------------------------------------
    @staticmethod
    def _weight_lock_path(path: Path) -> Path:
        return path.with_name(path.name + ".lock")

    def build_or_load_station_weights(
        self,
        station_lon: np.ndarray,
        station_lat: np.ndarray,
        paths: ShugaPaths,
        overwrite: bool = False,
    ):
        if self._station_weights is not None and self._station_diag is not None and not overwrite:
            return self._station_weights, self._station_diag
        path = self.config.weights_path
        if path is None:
            return super().build_or_load_station_weights(
                station_lon, station_lat, paths=paths, overwrite=overwrite
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._weight_lock_path(path)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # Another PBS array member may have built the file while this process waited.
            if path.exists() and not overwrite:
                self._station_weights = None
                self._station_diag = None
            return super().build_or_load_station_weights(
                station_lon, station_lat, paths=paths, overwrite=overwrite
            )

    def build_or_load_sic_weights(
        self,
        src_lon: np.ndarray,
        src_lat: np.ndarray,
        *,
        paths: ShugaPaths,
        overwrite: bool = False,
    ):
        if self._sic_weights is not None and not overwrite:
            return self._sic_weights
        path = self.config.sic_weights_path
        if path is None:
            return super().build_or_load_sic_weights(
                src_lon, src_lat, paths=paths, overwrite=overwrite
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._weight_lock_path(path)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists() and not overwrite:
                self._sic_weights = None
            return super().build_or_load_sic_weights(
                src_lon, src_lat, paths=paths, overwrite=overwrite
            )

    @staticmethod
    def _is_complete_forcing_file(path: Path, expected_nt: int | None = None) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with NCFile(path, mode="r") as nc:
                if str(getattr(nc, "completed", "false")).lower() != "true":
                    return False
                if "efreq" not in nc.variables:
                    return False
                if len(nc.dimensions.get("nfreq", [])) != 25:
                    return False
                if expected_nt is not None and len(nc.dimensions.get("time", [])) != expected_nt:
                    return False
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # lean production NetCDF writer
    # ------------------------------------------------------------------
    def _initialise_forcing_file(
        self,
        *,
        out: Path,
        ds_grid: xr.Dataset,
        time_values: np.ndarray,
        overwrite: bool,
        complevel: int = 3,
    ) -> None:
        if out.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists: {out}")
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)

        freq_ds = cice25_frequency_metadata()
        nt = int(time_values.size)
        nf = int(freq_ds.sizes["nfreq"])
        nj = int(ds_grid.sizes["nj"])
        ni = int(ds_grid.sizes["ni"])

        with NCFile(out, mode="w", format="NETCDF4") as nc:
            nc.createDimension("time", nt)
            nc.createDimension("nfreq", nf)
            nc.createDimension("nj", nj)
            nc.createDimension("ni", ni)

            time_num, time_units, time_calendar = _time_to_netcdf_numeric(time_values)
            vtime = nc.createVariable("time", "f8", ("time",))
            vtime[:] = time_num
            vtime.units = time_units
            vtime.calendar = time_calendar
            vtime.standard_name = "time"

            vn = nc.createVariable("nfreq", "i4", ("nfreq",))
            vn[:] = freq_ds["nfreq"].values

            vj = nc.createVariable("nj", "i4", ("nj",))
            vi = nc.createVariable("ni", "i4", ("ni",))
            vj[:] = np.arange(nj, dtype=np.int32)
            vi[:] = np.arange(ni, dtype=np.int32)

            for name in ("wavefreq", "wavefreq_lo", "wavefreq_hi", "dwavefreq"):
                var = nc.createVariable(name, "f4", ("nfreq",))
                var[:] = freq_ds[name].values.astype(np.float32)
                var.units = "s-1"

            vlon = nc.createVariable(
                "TLON", "f4", ("nj", "ni"), zlib=True, complevel=1, shuffle=True
            )
            vlat = nc.createVariable(
                "TLAT", "f4", ("nj", "ni"), zlib=True, complevel=1, shuffle=True
            )
            vlon[:] = ds_grid["TLON"].values.astype(np.float32)
            vlat[:] = ds_grid["TLAT"].values.astype(np.float32)
            vlon.units = "degrees_east"
            vlat.units = "degrees_north"

            # Chunk one hour at a time and retain all frequencies together. This
            # matches the intended CICE access pattern while keeping chunk sizes modest.
            nc.createVariable(
                "efreq",
                "f4",
                ("time", "nfreq", "nj", "ni"),
                zlib=True,
                complevel=int(complevel),
                shuffle=True,
                chunksizes=(1, nf, min(128, nj), min(128, ni)),
            )

            nc.title = "Hourly WHACS incident-wave spectra regridded for standalone CICE6"
            nc.source_product = "WHACS BoM-CSIRO hindcast, WWIII-v6.07, ERA5 forced"
            nc.source_file = str(self._current_whacs_path)
            nc.output_field = "efreq(time,nfreq,nj,ni)"
            nc.spectrum_units = "m2 s"
            nc.cice_nfreq = 25
            nc.frequency_grid = "Noah Day CICE/Icepack nfreq=25 geometric WaveWatch grid"
            nc.spectral_remap = "direction integration then conservative frequency-bin overlap"
            nc.sic_mask = f"NSIDC daily SIC < {self.config.sic_threshold:.3f} supplies incident waves; otherwise efreq=0"
            nc.station_regrid = (
                f"IDW k={self.config.k_nearest}, power={self.config.idw_power}, "
                f"radius_km={self.config.radius_km}"
            )
            nc.station_weights = str(self.config.weights_path)
            nc.target_lat_max = float(self.config.target_lat_max)

    def _append_forcing_chunk(
        self,
        *,
        out: Path,
        i0: int,
        i1: int,
        efreq_masked_chunk: xr.DataArray,
    ) -> None:
        arr = (
            efreq_masked_chunk
            .transpose("time", "frequency", "nj", "ni")
            .values.astype(np.float32, copy=False)
        )
        # Incident wave forcing must be finite and non-negative for CICE.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(arr, 0.0, out=arr)
        with NCFile(out, mode="a") as nc:
            nc["efreq"][i0:i1, :, :, :] = arr
            nc.sync()

    # ------------------------------------------------------------------
    # end-to-end monthly workflow
    # ------------------------------------------------------------------
    def prepare_month(
        self,
        year: int,
        month: int,
        *,
        paths: ShugaPaths,
        obs_class: Optional[SeaIceObservations] = None,
        overwrite_weights: bool = False,
        overwrite_sic_weights: bool = False,
        overwrite_output: bool = False,
        time_chunk: int = 1,
        complevel: int = 3,
    ) -> Path:
        if not (1 <= month <= 12):
            raise ValueError(f"month must be 1..12; got {month}")

        dt0 = pd.Timestamp(year=year, month=month, day=1)
        dtN = dt0 + pd.offsets.MonthEnd(0)
        source = self.whacs_file(year, month)
        self._current_whacs_path = source
        if not source.exists():
            raise FileNotFoundError(f"WHACS monthly source file not found: {source}")

        if self.config.output_path is None:
            self.config.output_path = paths.cawcr_regridded_file(year, month)
        if self.config.weights_path is None:
            self.config.weights_path = (
                paths.wave_weights_root_path
                / f"map_WHACSstations_to_ACCESS-OM3-025_idw_k{self.config.k_nearest}.npz"
            )
        if self.config.sic_weights_path is None:
            self.config.sic_weights_path = paths.nsidc2cice_weight_file

        out = Path(self.config.output_path)
        expected_nt = 24 * calendar.monthrange(year, month)[1]
        if not overwrite_output and self._is_complete_forcing_file(out, expected_nt=expected_nt):
            self._log(f"Completed output exists; skipping: {out}")
            return out

        # Write to a per-process partial file, then atomically replace the final
        # path only after all hourly records have been written successfully.
        work_out = out.with_name(out.name + f".partial.{os.getpid()}")
        if work_out.exists():
            work_out.unlink()

        self._log(f"WHACS source: {source}")
        self._log(f"CICE output : {out}")
        self._log(f"Weights     : {self.config.weights_path}")

        ds_raw = self.open_whacs_month(source, time_chunk=time_chunk)
        ds_raw = self._subset_time_window(ds_raw, dt0, dtN)
        self._validate_hourly_month(ds_raw["time"].values, year, month)

        self._log("Integrating WHACS directional spectrum over theta")
        native = self.collapse_directional_spectrum(ds_raw)
        self._log("Conservatively remapping WHACS frequency bins -> CICE25")
        efreq_station = self.remap_to_cice25(native, ds_raw)
        self._log_spectral_qc(native, efreq_station, ds_raw)

        obs_cfg = paths.obs_cfg or ObservationSpec()
        if obs_class is None:
            run_cfg = paths.run_cfg
            if run_cfg is None:
                raise ValueError("WHACS prepare_month requires ShugaPaths(run_cfg=RunSpec(...)).")
            obs_class = SeaIceObservations(
                run_cfg=run_cfg,
                obs_cfg=obs_cfg,
                pth_cfg=paths,
                logger=self.logger,
            )

        self._log("Loading daily NSIDC SIC and regridding it to the CICE T grid")
        sic_daily = obs_class.load_nsidc_daily(
            start_date=dt0.strftime("%Y-%m-%d"),
            end_date=dtN.strftime("%Y-%m-%d"),
            hemisphere=self.config.hemisphere,
        )
        sic_daily_cice = self._regrid_daily_sic_to_cice_all(
            sic_daily,
            paths=paths,
            sic_var=obs_cfg.nsidc_sic_var,
            overwrite_weights=overwrite_sic_weights,
        )

        ds_grid = self.get_target_grid(paths)

        # Build/load static station weights once before opening the output.
        first = efreq_station.isel(time=slice(0, 1))
        self._log("Building/loading static WHACS-station -> CICE weights")
        self._regrid_station_spectra_chunk_to_cice(
            first,
            paths=paths,
            overwrite_weights=overwrite_weights,
        )

        self._initialise_forcing_file(
            out=work_out,
            ds_grid=ds_grid,
            time_values=efreq_station["time"].values,
            overwrite=overwrite_output,
            complevel=complevel,
        )

        nt = int(efreq_station.sizes["time"])
        for i0 in range(0, nt, max(1, int(time_chunk))):
            i1 = min(i0 + max(1, int(time_chunk)), nt)
            self._log(f"Processing hourly chunk {i0}:{i1} / {nt}")
            station_chunk = efreq_station.isel(time=slice(i0, i1))
            efreq_unmasked, _ = self._regrid_station_spectra_chunk_to_cice(
                station_chunk,
                paths=paths,
                overwrite_weights=False,
            )
            sic_hourly = self.expand_daily_sic_to_hourly(
                sic_daily_cice,
                efreq_unmasked["time"],
            )
            efreq_masked, _ = self.apply_ice_edge_mask(efreq_unmasked, sic_hourly)
            self._append_forcing_chunk(
                out=work_out,
                i0=i0,
                i1=i1,
                efreq_masked_chunk=efreq_masked,
            )
            del station_chunk, efreq_unmasked, sic_hourly, efreq_masked

        with NCFile(work_out, mode="a") as nc:
            nc.completed = "true"
            nc.completed_utc = pd.Timestamp.utcnow().isoformat()

        os.replace(work_out, out)
        self._log(f"Completed WHACS monthly forcing: {out}")
        ds_raw.close()
        return out
