from __future__         import annotations
from dataclasses        import dataclass
from pathlib            import Path
from typing             import Optional
import numpy            as np
import pandas           as pd
import xarray           as xr
from netCDF4            import Dataset as NCFile
from scipy              import sparse
from scipy.spatial      import cKDTree
from shuga.core.paths   import ShugaPaths
from shuga.observations import SeaIceObservations
from shuga.core.types   import ObservationSpec
from shuga.grid.cice    import CICEGridwork

EARTH_RADIUS_KM = 6371.0

@dataclass(slots=True)
class CAWCRRegridConfig:
    """Configuration for monthly CAWCR -> CICE spectral forcing generation."""
    output_path      : str | Path | None = None
    source_var       : str = "Efth"
    station_lon_name : str = "longitude"
    station_lat_name : str = "latitude"
    time_dim         : str = "time"
    station_dim      : str = "station"
    frequency_dim    : str = "frequency"
    direction_dim    : str = "direction"
    frequency_lo_name: str = "frequency1"
    frequency_hi_name: str = "frequency2"
    k_nearest        : int = 5
    idw_power        : float = 2.5
    radius_km        : float = 1000.0
    sic_threshold    : float = 0.15
    hemisphere       : str = "SH"
    target_lat_max   : float = -35.0
    target_lat_min   : float = 35.0
    fill_value       : float = 0.0
    weights_path     : str | Path | None = None
    sic_weights_path : str | Path | None = None
    target_lon_type  : str = "-180-180"

    def __post_init__(self) -> None:
        if self.output_path is not None:
            self.output_path = Path(self.output_path)
        if self.weights_path is not None:
            self.weights_path = Path(self.weights_path)
        if self.sic_weights_path is not None:
            self.sic_weights_path = Path(self.sic_weights_path)

def _time_to_netcdf_numeric(time_values: np.ndarray) -> tuple[np.ndarray, str, str]:
    if np.issubdtype(time_values.dtype, np.datetime64):
        epoch = np.datetime64("1970-01-01T00:00:00")
        out = ((time_values - epoch) / np.timedelta64(1, "s")).astype(np.float64)
        return out, "seconds since 1970-01-01 00:00:00", "standard"
    return np.asarray(time_values, dtype=np.float64), "", "standard"

class CAWCRRegridder:
    """
    Regrid CAWCR station spectra to the native CICE T grid and mask by NSIDC SIC.

    The resulting dataset writes the *masked* spectrum to ``efreq`` because the
    CICE wave forcing code is hard-wired to read that field name.
    """

    def __init__(self, config: CAWCRRegridConfig, logger=None) -> None:
        self.config = config
        self.logger = logger
        self._target_grid     : xr.Dataset | None            = None
        self._station_weights : sparse.csr_matrix | None     = None
        self._station_diag    : dict[str, np.ndarray] | None = None
        self._sic_weights     : sparse.csr_matrix | None     = None

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    @staticmethod
    def _parse_month_window(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        dt0 = pd.Timestamp(start_date).normalize()
        dtN = pd.Timestamp(end_date).normalize()
        if dtN < dt0:
            raise ValueError(f"end_date ({end_date}) is earlier than start_date ({start_date}).")
        if (dt0.year != dtN.year) or (dt0.month != dtN.month):
            raise ValueError("prepare_month() only supports a single calendar month. "
                             f"Got start_date={start_date} and end_date={end_date}.")
        return dt0, dtN

    @staticmethod
    def _subset_time_window(ds: xr.Dataset, dt0: pd.Timestamp, dtN: pd.Timestamp) -> xr.Dataset:
        # inclusive of the final day
        t1 = dtN + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        ds_sub = ds.sel(time=slice(dt0, t1))
        if ds_sub.sizes.get("time", 0) == 0:
            raise ValueError(f"No CAWCR timesteps found in requested window "
                             f"{dt0.strftime('%Y-%m-%d')} to {dtN.strftime('%Y-%m-%d')}.")
        return ds_sub

    # ------------------------------------------------------------------
    # source loading / normalisation
    # ------------------------------------------------------------------
    def open_cawcr_month(self, path: str | Path, chunks: Optional[dict] = None) -> xr.Dataset:
        ds = xr.open_dataset(path, chunks=chunks)
        return self.normalise_cawcr_names(ds)

    def normalise_cawcr_names(self, ds: xr.Dataset) -> xr.Dataset:
        rename_map: dict[str, str] = {}
        rename_map |= self._rename_if_needed(ds, self.config.time_dim, "time")
        rename_map |= self._rename_if_needed(ds, self.config.station_dim, "station")
        rename_map |= self._rename_if_needed(ds, self.config.frequency_dim, "frequency")
        rename_map |= self._rename_if_needed(ds, self.config.direction_dim, "direction")
        rename_map |= self._rename_if_needed(ds, self.config.station_lon_name, "station_lon")
        rename_map |= self._rename_if_needed(ds, self.config.station_lat_name, "station_lat")
        rename_map |= self._rename_if_needed(ds, self.config.source_var, "efth")
        if self.config.frequency_lo_name in ds and self.config.frequency_lo_name != "frequency_lo":
            rename_map[self.config.frequency_lo_name] = "frequency_lo"
        if self.config.frequency_hi_name in ds and self.config.frequency_hi_name != "frequency_hi":
            rename_map[self.config.frequency_hi_name] = "frequency_hi"
        ds = ds.rename(rename_map)
        for name in ("station_lon", "station_lat"):
            if name in ds and name not in ds.coords:
                ds = ds.set_coords(name)
        return ds

    @staticmethod
    def _rename_if_needed(ds: xr.Dataset, old: str, new: str) -> dict[str, str]:
        if old == new:
            return {}
        if old in ds.variables or old in ds.coords or old in ds.dims:
            return {old: new}
        return {}

    def collapse_directional_spectrum(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Integrate Efth(f, theta) over direction to obtain E(f).

        Uses periodic directional bin widths derived from the directional centres
        along their native cyclic ordering, so wrapped grids like
        82.5, 67.5, ..., 7.5, 352.5, ... are handled correctly.
        """
        theta           = ds["direction"].astype(float).values
        # Convert to radians if the direction coordinate is in degrees
        theta_rad       = np.deg2rad(theta) if np.nanmax(np.abs(theta)) > (2 * np.pi + 1e-6) else theta.astype(np.float64)
        # Unwrap along the *existing* cyclic order so the sequence becomes monotonic
        theta_unwrapped = np.unwrap(theta_rad)
        # Build bin edges from centre midpoints
        edges           = np.empty(theta_unwrapped.size + 1, dtype=np.float64)
        edges[1:-1]     = 0.5 * (theta_unwrapped[:-1] + theta_unwrapped[1:])
        edges[0]        = theta_unwrapped[0] - 0.5 * (theta_unwrapped[1] - theta_unwrapped[0])
        edges[-1]       = theta_unwrapped[-1] + 0.5 * (theta_unwrapped[-1] - theta_unwrapped[-2])
        dtheta          = np.abs(np.diff(edges))
        # Sanity checks
        if not np.all(np.isfinite(dtheta)):
            raise ValueError("Directional bin widths contain non-finite values.")
        if np.any(dtheta <= 0):
            raise ValueError(f"Directional bin widths must be positive; got min={dtheta.min()}")
        if not np.isclose(dtheta.sum(), 2 * np.pi, rtol=1e-3, atol=1e-3):
            raise ValueError(f"Directional bin widths should sum to 2π; got {dtheta.sum()} rad "
                             f"({np.rad2deg(dtheta.sum())} deg)")
        dtheta_da = xr.DataArray(dtheta.astype(np.float32),
                                 dims   = ("direction",),
                                 coords = {"direction": ds["direction"]},
                                 name   = "dtheta",
                                 attrs  = {"long_name": "direction bin width", "units": "radian"})
        # Source spectrum should be non-negative; clip tiny negatives defensively
        efth               = xr.where(ds["efth"] < 0, 0.0, ds["efth"])
        efreq_station      = (efth * dtheta_da).sum("direction", skipna=True)
        efreq_station      = efreq_station.transpose("time", "station", "frequency")
        efreq_station.name = "efreq_station"
        efreq_station      = efreq_station.assign_coords(station_lon=self._as_station_coord(ds["station_lon"]),
                                                         station_lat=self._as_station_coord(ds["station_lat"]))
        efreq_station.attrs.update({"long_name": "direction-integrated wave energy spectrum",
                                    "units"    : ds["efth"].attrs.get("units", "m2 s")})
        return efreq_station

    @staticmethod
    def _as_station_coord(coord: xr.DataArray) -> xr.DataArray:
        if coord.dims == ("station",):
            return coord
        if "time" in coord.dims:
            coord = coord.isel(time=0, drop=True)
        squeeze_dims = [d for d in coord.dims if d != "station" and coord.sizes[d] == 1]
        if squeeze_dims:
            coord = coord.squeeze(squeeze_dims, drop=True)
        other_dims = [d for d in coord.dims if d != "station"]
        if other_dims:
            raise ValueError(f"Station coordinate must reduce to ('station',); got {coord.dims}")
        return coord

    # ------------------------------------------------------------------
    # frequency helpers / Hs
    # ------------------------------------------------------------------
    def get_frequency_metadata(self, ds_raw: xr.Dataset) -> xr.Dataset:
        wavefreq = ds_raw["frequency"].astype(np.float32)
        if "frequency_lo" in ds_raw and "frequency_hi" in ds_raw:
            wavefreq_lo = ds_raw["frequency_lo"].astype(np.float32)
            wavefreq_hi = ds_raw["frequency_hi"].astype(np.float32)
            dwavefreq   = (wavefreq_hi - wavefreq_lo).astype(np.float32)
        else:
            dwavefreq = xr.DataArray(np.gradient(wavefreq.values.astype(np.float64)).astype(np.float32),
                                     dims   = ("frequency",),
                                     coords = {"frequency": wavefreq},
                                     name   = "dwavefreq")
            wavefreq_lo = xr.DataArray((wavefreq - 0.5 * dwavefreq).astype(np.float32), dims=("frequency",), coords={"frequency": wavefreq})
            wavefreq_hi = xr.DataArray((wavefreq + 0.5 * dwavefreq).astype(np.float32), dims=("frequency",), coords={"frequency": wavefreq})
        return xr.Dataset(data_vars = {"wavefreq"    : ("nfreq", wavefreq.values.astype(np.float32)),
                                       "wavefreq_lo" : ("nfreq", wavefreq_lo.values.astype(np.float32)),
                                       "wavefreq_hi" : ("nfreq", wavefreq_hi.values.astype(np.float32)),
                                       "dwavefreq"   : ("nfreq", dwavefreq.values.astype(np.float32))},
                          coords    = {"nfreq": np.arange(wavefreq.size, dtype=np.int32)})

    @staticmethod
    def compute_hs(efreq: xr.DataArray, dwavefreq: xr.DataArray) -> xr.DataArray:
        m0      = (efreq * dwavefreq).sum("nfreq", skipna=True)
        hs      = 4.0 * np.sqrt(xr.where(m0 > 0, m0, 0.0))
        hs      = hs.astype(np.float32)
        hs.name = "hs"
        hs.attrs.update({"long_name": "significant wave height", "units": "m"})
        return hs

    # ------------------------------------------------------------------
    # target grid / weights
    # -----------------------------------------------------------------
    def get_target_grid(self, paths: ShugaPaths) -> xr.Dataset:
        if self._target_grid is not None:
            return self._target_grid
        gridwork = CICEGridwork(pth_cfg=paths, logger=self.logger)
        bundle   = gridwork.load_cice_grid(build_faces=False)
        lon      = bundle.tgrid["TLON"]
        lat      = bundle.tgrid["TLAT"]
        if self.config.target_lon_type:
            lon = xr.DataArray(gridwork.normalise_longitudes(lon.values, to=self.config.target_lon_type),
                               dims   = lon.dims,
                               coords = lon.coords,
                               attrs  = lon.attrs)
        if bundle.mask is not None:
            ocean_mask = bundle.mask.astype(np.int8)
        else:
            ocean_mask = xr.where(np.isfinite(lon) & np.isfinite(lat), 1, 0).astype(np.int8)
        hemisphere_mask   = self._build_hemisphere_mask(lat)
        active_mask       = (ocean_mask.astype(bool) & hemisphere_mask & np.isfinite(lon) & np.isfinite(lat)).astype(np.int8)
        self._target_grid = xr.Dataset(data_vars={"TLON"              : lon.astype(np.float32),
                                                  "TLAT"              : lat.astype(np.float32),
                                                  "ocean_mask"        : ocean_mask.astype(np.int8),
                                                  "target_active_mask": active_mask.astype(np.int8)})
        return self._target_grid

    def _build_hemisphere_mask(self, lat: xr.DataArray) -> xr.DataArray:
        hemi = self.config.hemisphere.upper()
        if hemi.startswith("S"):
            return lat <= float(self.config.target_lat_max)
        return lat >= float(self.config.target_lat_min)

    def build_or_load_station_weights(self, station_lon: np.ndarray, station_lat: np.ndarray, paths: ShugaPaths, 
                                      overwrite: bool = False) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
        if self._station_weights is not None and self._station_diag is not None and not overwrite:
            return self._station_weights, self._station_diag
        if self.config.weights_path is not None and self.config.weights_path.exists() and not overwrite:
            self._log(f"Loading station weights from {self.config.weights_path}")
            self._station_weights, self._station_diag = self._load_sparse_weights(self.config.weights_path)
            return self._station_weights, self._station_diag
        ds_grid       = self.get_target_grid(paths)
        active        = ds_grid["target_active_mask"].values.astype(bool)
        tgt_lon       = ds_grid["TLON"].values[active]
        tgt_lat       = ds_grid["TLAT"].values[active]
        weights, diag = self._build_idw_weights(src_lon   = station_lon,
                                                src_lat   = station_lat,
                                                tgt_lon   = tgt_lon,
                                                tgt_lat   = tgt_lat,
                                                k         = self.config.k_nearest,
                                                power     = self.config.idw_power,
                                                radius_km = self.config.radius_km)
        self._station_weights = weights
        self._station_diag    = diag
        if self.config.weights_path is not None:
            self.config.weights_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_sparse_weights(self.config.weights_path, weights, diag)
        return weights, diag

    # --- in CAWCRRegridder.build_or_load_sic_weights ---
    def build_or_load_sic_weights(self, src_lon: np.ndarray, src_lat: np.ndarray, *, paths: ShugaPaths,
                                  overwrite: bool = False) -> sparse.csr_matrix:
        if self._sic_weights is not None and not overwrite:
            return self._sic_weights
        if self.config.sic_weights_path is not None and self.config.sic_weights_path.exists() and not overwrite:
            self._log(f"Loading SIC weights from {self.config.sic_weights_path}")
            matrix, _         = self._load_sparse_weights(self.config.sic_weights_path)
            self._sic_weights = matrix
            return matrix
        ds_grid           = self.get_target_grid(paths)
        active            = ds_grid["target_active_mask"].values.astype(bool)
        tgt_lon           = ds_grid["TLON"].values[active]
        tgt_lat           = ds_grid["TLAT"].values[active]
        matrix            = self._build_nearest_weights(src_lon, src_lat, tgt_lon, tgt_lat)
        self._sic_weights = matrix
        if self.config.sic_weights_path is not None:
            self.config.sic_weights_path.parent.mkdir(parents=True, exist_ok=True)
            diag = {"distance_km"   : np.full(matrix.shape[0], np.nan, dtype=np.float32),
                    "n_source_used" : np.ones(matrix.shape[0], dtype=np.int16),
                    "weight_sum"    : np.ones(matrix.shape[0], dtype=np.float32),
                    "valid_target"  : np.ones(matrix.shape[0], dtype=np.int8)}
            self._save_sparse_weights(self.config.sic_weights_path, matrix, diag)
        return matrix

    @staticmethod
    def _lonlat_to_unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        return np.column_stack([np.cos(lat) * np.cos(lon),
                                np.cos(lat) * np.sin(lon),
                                np.sin(lat)])

    def _build_idw_weights(self,
                           src_lon   : np.ndarray,
                           src_lat   : np.ndarray,
                           tgt_lon   : np.ndarray,
                           tgt_lat   : np.ndarray,
                           k         : int,
                           power     : float,
                           radius_km : float) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
        src_xyz      = self._lonlat_to_unit_xyz(src_lon, src_lat)
        tgt_xyz      = self._lonlat_to_unit_xyz(tgt_lon, tgt_lat)
        tree         = cKDTree(src_xyz)
        k_eff        = min(k, src_xyz.shape[0])
        d_chord, idx = tree.query(tgt_xyz, k=k_eff)
        if k_eff == 1:
            d_chord = d_chord[:, None]
            idx     = idx[:, None]
        d_chord       = np.clip(d_chord, 0.0, 2.0)
        central_angle = 2.0 * np.arcsin(0.5 * d_chord)
        distance_km   = EARTH_RADIUS_KM * central_angle
        within_radius = distance_km <= radius_km
        valid_target  = within_radius.any(axis=1)
        safe_distance = np.where(distance_km == 0.0, 1.0e-12, distance_km)
        raw_weight    = np.where(within_radius, safe_distance ** (-power), 0.0)
        zero_hit      = (distance_km == 0.0) & within_radius
        zero_any      = zero_hit.any(axis=1)
        if zero_any.any():
            raw_weight[zero_any, :] = zero_hit[zero_any, :].astype(np.float64)
        weight_sum  = raw_weight.sum(axis=1)
        norm_weight = np.divide(raw_weight, weight_sum[:, None], out=np.zeros_like(raw_weight), where=weight_sum[:, None] > 0)
        row         = np.repeat(np.arange(tgt_lon.size), k_eff)
        col         = idx.reshape(-1)
        data        = norm_weight.reshape(-1)
        keep        = data > 0
        matrix      = sparse.csr_matrix((data[keep], (row[keep], col[keep])), shape=(tgt_lon.size, src_lon.size))
        diag        = { "distance_km"   : np.where(valid_target, np.nanmin(np.where(within_radius, distance_km, np.nan), axis=1), np.nan).astype(np.float32),
                        "n_source_used" : within_radius.sum(axis=1).astype(np.int16),
                        "weight_sum"    : weight_sum.astype(np.float32),
                        "valid_target"  : valid_target.astype(np.int8)}
        return matrix, diag

    def _build_nearest_weights(self,
                               src_lon: np.ndarray,
                               src_lat: np.ndarray,
                               tgt_lon: np.ndarray,
                               tgt_lat: np.ndarray) -> sparse.csr_matrix:
        src_xyz = self._lonlat_to_unit_xyz(src_lon, src_lat)
        tgt_xyz = self._lonlat_to_unit_xyz(tgt_lon, tgt_lat)
        tree    = cKDTree(src_xyz)
        _, idx  = tree.query(tgt_xyz, k=1)
        row     = np.arange(tgt_lon.size)
        col     = idx.astype(np.int64)
        data    = np.ones(tgt_lon.size, dtype=np.float32)
        return sparse.csr_matrix((data, (row, col)), shape=(tgt_lon.size, src_lon.size))

    @staticmethod
    def _save_sparse_weights(path: Path, matrix: sparse.csr_matrix, diag: dict[str, np.ndarray]) -> None:
        np.savez_compressed(path,
                            data    = matrix.data,
                            indices = matrix.indices,
                            indptr  = matrix.indptr,
                            shape   = np.asarray(matrix.shape, dtype=np.int64),
                            **diag)

    @staticmethod
    def _load_sparse_weights(path: Path) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as npz:
            matrix = sparse.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))
            diag   = {k: npz[k] for k in npz.files if k not in {"data", "indices", "indptr", "shape"}}
        return matrix, diag

    # ------------------------------------------------------------------
    # regridding
    # ------------------------------------------------------------------
    def regrid_station_spectra_to_cice(self, efreq_station: xr.DataArray, paths: ShugaPaths,
                                       overwrite_weights: bool = False) -> tuple[xr.DataArray, xr.Dataset]:
        if tuple(efreq_station.dims) != ("time", "station", "frequency"):
            efreq_station = efreq_station.transpose("time", "station", "frequency")
        station_lon  = efreq_station["station_lon"].values.astype(np.float64)
        station_lat  = efreq_station["station_lat"].values.astype(np.float64)
        matrix, diag = self.build_or_load_station_weights(station_lon, station_lat, paths=paths, overwrite=overwrite_weights)
        src          = efreq_station.fillna(0.0).values.astype(np.float32)
        nt, ns, nf   = src.shape
        src2d        = np.transpose(src, (0, 2, 1)).reshape(nt * nf, ns)
        out2d        = src2d @ matrix.T
        out_active   = np.asarray(out2d, dtype=np.float32).reshape(nt, nf, matrix.shape[0])
        out_active   = np.transpose(out_active, (0, 2, 1))
        ds_grid      = self.get_target_grid(paths)
        active       = ds_grid["target_active_mask"].values.astype(bool)
        full         = np.full((nt, ds_grid.dims["nj"], ds_grid.dims["ni"], nf), self.config.fill_value, dtype=np.float32)
        full[:, active, :] = out_active
        efreq_grid   = xr.DataArray(full,
                                    dims   = ("time", "nj", "ni", "frequency"),
                                    coords = {"time"     : efreq_station["time"].values,
                                              "nj"       : np.arange(ds_grid.dims["nj"], dtype=np.int32),
                                              "ni"       : np.arange(ds_grid.dims["ni"], dtype=np.int32),
                                              "frequency": efreq_station["frequency"].values.astype(np.float32),
                                              "TLON"     : (("nj", "ni"), ds_grid["TLON"].values),
                                              "TLAT"     : (("nj", "ni"), ds_grid["TLAT"].values)},
                                    name   = "efreq_unmasked",
                                    attrs  = {"long_name": "CAWCR spectra regridded to CICE grid before SIC masking",
                                              "units"    : efreq_station.attrs.get("units", "m2 s")})
        diag_ds      = xr.Dataset(data_vars = {"distance_to_nearest_station_km": (("nj", "ni"), self._scatter_active_to_grid(diag["distance_km"], active, np.nan).astype(np.float32)),
                                               "n_station_neighbours": (("nj", "ni"), self._scatter_active_to_grid(diag["n_source_used"], active, 0).astype(np.int16)),
                                               "station_weight_sum": (("nj", "ni"), self._scatter_active_to_grid(diag["weight_sum"], active, 0.0).astype(np.float32)),
                                               "station_interp_valid": (("nj", "ni"), self._scatter_active_to_grid(diag["valid_target"], active, 0).astype(np.int8))},
                                  coords    = {"nj": efreq_grid["nj"], "ni": efreq_grid["ni"]})
        return efreq_grid, diag_ds

    def regrid_daily_sic_to_cice(self, sic_daily: xr.DataArray | xr.Dataset, *, paths: ShugaPaths,
                                 sic_var           : str  = "cdr_seaice_conc",
                                 overwrite_weights : bool = False) -> xr.DataArray:
        sic                  = self._normalise_sic(sic_daily, sic_var=sic_var)
        src_lon2d, src_lat2d = self._extract_lonlat_2d(sic)
        valid_src            = np.isfinite(src_lon2d) & np.isfinite(src_lat2d)
        src_lon              = src_lon2d[valid_src]
        src_lat              = src_lat2d[valid_src]
        matrix               = self.build_or_load_sic_weights(src_lon, src_lat, paths=paths, overwrite=overwrite_weights)
        src                  = sic.values.astype(np.float32)
        nt                   = src.shape[0]
        src2d                = src.reshape(nt, -1)[:, valid_src.ravel()]
        out_active           = np.asarray(src2d @ matrix.T, dtype=np.float32)
        ds_grid              = self.get_target_grid(paths)
        active               = ds_grid["target_active_mask"].values.astype(bool)
        full                 = np.full((nt, ds_grid.sizes["nj"], ds_grid.sizes["ni"]), np.nan, dtype=np.float32)
        full[:, active]      = out_active
        out                  = xr.DataArray(full,
                                            dims   = ("time", "nj", "ni"),
                                            coords = {"time": sic["time"].values,
                                                      "nj": np.arange(ds_grid.sizes["nj"], dtype=np.int32),
                                                      "ni": np.arange(ds_grid.sizes["ni"], dtype=np.int32),
                                                      "TLON": (("nj", "ni"), ds_grid["TLON"].values),
                                                      "TLAT": (("nj", "ni"), ds_grid["TLAT"].values)},
                                            name  = "sic",
                                            attrs = {"long_name": "NSIDC daily sea ice concentration on CICE grid",
                                                     "units": sic.attrs.get("units", "1")})
        return out

    def expand_daily_sic_to_hourly(self, sic_daily_on_cice: xr.DataArray, hourly_time: xr.DataArray) -> xr.DataArray:
        time_daily      = pd.to_datetime(sic_daily_on_cice["time"].values).normalize()
        sic_norm        = sic_daily_on_cice.copy().assign_coords(time=time_daily)
        hourly_days     = pd.to_datetime(hourly_time.values).normalize()
        sic_hourly      = sic_norm.reindex(time=hourly_days)
        sic_hourly      = sic_hourly.assign_coords(time=hourly_time.values)
        sic_hourly.name = "sic"
        return sic_hourly.astype(np.float32)

    def apply_ice_edge_mask(self, efreq_unmasked: xr.DataArray, sic_hourly: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        if tuple(efreq_unmasked.dims) != ("time", "nj", "ni", "frequency"):
            efreq_unmasked = efreq_unmasked.transpose("time", "nj", "ni", "frequency")
        sic3d             = sic_hourly.transpose("time", "nj", "ni")
        open_water_mask   = xr.where(np.isfinite(sic3d) & (sic3d < self.config.sic_threshold), 1, 0).astype(np.int8)
        efreq_masked      = xr.where(open_water_mask.astype(bool), efreq_unmasked, 0.0).astype(np.float32)
        efreq_masked.name = "efreq"
        efreq_masked.attrs.update({"long_name": f"CAWCR spectra masked where NSIDC SIC >= {self.config.sic_threshold:.2f}"})
        return efreq_masked, open_water_mask

    # ------------------------------------------------------------------
    # SIC helpers
    # ------------------------------------------------------------------
    def _normalise_sic(self, sic: xr.DataArray | xr.Dataset, *,
                       sic_var: str = "cdr_seaice_conc") -> xr.DataArray:
        lon_candidates = ("longitude", "lon", "LONGITUDE", "LON")
        lat_candidates = ("latitude", "lat", "LATITUDE", "LAT")
        if isinstance(sic, xr.Dataset):
            if sic_var in sic.data_vars:
                da = sic[sic_var]
            elif "cdr_seaice_conc" in sic.data_vars:
                da = sic["cdr_seaice_conc"]
            else:
                candidates = [name for name, var in sic.data_vars.items()
                              if "time" in var.dims and var.ndim >= 3]
                if not candidates:
                    raise ValueError("Could not identify SIC variable in Dataset.")
                da = sic[candidates[0]]
            coord_map = {}
            for name in lon_candidates:
                if name in sic.data_vars:
                    coord_map[name] = sic[name]
                    break
            for name in lat_candidates:
                if name in sic.data_vars:
                    coord_map[name] = sic[name]
                    break
            if coord_map:
                da = da.assign_coords(coord_map)
            sic = da
        if "time" not in sic.dims:
            raise ValueError("SIC data must contain a time dimension.")
        if sic.dtype.kind not in "fiu":
            sic = sic.astype(np.float32)
        vmax = float(sic.max(skipna=True).compute()) if hasattr(sic.data, "compute") else float(np.nanmax(sic.values))
        if vmax > 1.5:
            sic = sic / 100.0
        return sic.astype(np.float32)

    def _extract_lonlat_2d(self, da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        lon_candidates = ("longitude", "lon", "LONGITUDE", "LON")
        lat_candidates = ("latitude", "lat", "LATITUDE", "LAT")
        lon_name       = next((n for n in lon_candidates if n in da.coords), None)
        lat_name       = next((n for n in lat_candidates if n in da.coords), None)
        if lon_name is None or lat_name is None:
            raise KeyError(f"Could not find lon/lat coordinates on SIC data. "
                           f"Available coords: {list(da.coords)}")
        lon = da.coords[lon_name]
        lat = da.coords[lat_name]
        if lon.ndim != 2 or lat.ndim != 2:
            raise ValueError(f"Expected 2D lon/lat coordinates; got lon.ndim={lon.ndim}, lat.ndim={lat.ndim}")
        return lon.values.astype(np.float64), lat.values.astype(np.float64)

    # def build_output_dataset(
    #     self,
    #     ds_raw: xr.Dataset,
    #     efreq_unmasked: xr.DataArray,
    #     efreq_masked: xr.DataArray,
    #     sic_hourly: xr.DataArray,
    #     open_water_mask: xr.DataArray,
    #     diag_ds: xr.Dataset,
    #     *,
    #     paths: ShugaPaths,
    # ) -> xr.Dataset:
    #     freq_ds = self.get_frequency_metadata(ds_raw)
    #     nfreq_index = freq_ds["nfreq"].values.astype(np.int32)

    #     # Convert the spectrum dimension to an integer index axis.
    #     # Keep the physical frequencies in wavefreq / wavefreq_lo / wavefreq_hi / dwavefreq.
    #     efreq_unmasked_out = (
    #         efreq_unmasked
    #         .transpose("time", "frequency", "nj", "ni")
    #         .rename({"frequency": "nfreq"})
    #         .assign_coords(nfreq=nfreq_index)
    #         .astype(np.float32)
    #     )

    #     efreq_masked_out = (
    #         efreq_masked
    #         .transpose("time", "frequency", "nj", "ni")
    #         .rename({"frequency": "nfreq"})
    #         .assign_coords(nfreq=nfreq_index)
    #         .astype(np.float32)
    #     )

    #     dwavefreq = xr.DataArray(
    #         freq_ds["dwavefreq"].values.astype(np.float32),
    #         dims=("nfreq",),
    #         coords={"nfreq": nfreq_index},
    #         name="dwavefreq",
    #     )

    #     hs_unmasked = self.compute_hs(efreq_unmasked_out, dwavefreq).rename("hs_unmasked")
    #     hs_masked = self.compute_hs(efreq_masked_out, dwavefreq).rename("hs_masked")

    #     ds_grid = self.get_target_grid(paths)

    #     ds_out = xr.Dataset(
    #         data_vars={
    #             "efreq": efreq_masked_out,
    #             "efreq_unmasked": efreq_unmasked_out,
    #             "hs_masked": hs_masked,
    #             "hs_unmasked": hs_unmasked,
    #             "sic": sic_hourly.astype(np.float32),
    #             "open_water_mask": open_water_mask.astype(np.int8),
    #             "TLON": ds_grid["TLON"].astype(np.float32),
    #             "TLAT": ds_grid["TLAT"].astype(np.float32),
    #             "ocean_mask": ds_grid["ocean_mask"].astype(np.int8),
    #             "target_active_mask": ds_grid["target_active_mask"].astype(np.int8),
    #             **{name: da for name, da in diag_ds.data_vars.items()},
    #         },
    #         coords={
    #             "time": efreq_masked_out["time"].values,
    #             "nfreq": nfreq_index,
    #             "nj": efreq_masked_out["nj"].values,
    #             "ni": efreq_masked_out["ni"].values,
    #         },
    #         attrs={
    #             "title": "CAWCR spectra regridded to native CICE grid and masked by NSIDC SIC",
    #             "sic_threshold": float(self.config.sic_threshold),
    #             "hemisphere": self.config.hemisphere,
    #             "idw_k_nearest": int(self.config.k_nearest),
    #             "idw_power": float(self.config.idw_power),
    #             "idw_radius_km": float(self.config.radius_km),
    #         },
    #     )

    #     for name in freq_ds.data_vars:
    #         ds_out[name] = freq_ds[name]

    #     return ds_out

    def _scatter_active_chunk_to_grid(self,
                                      out_active: np.ndarray,  # (nt_chunk, n_active, nf)
                                      active_mask: np.ndarray, # (nj, ni) bool
                                      nj: int, ni: int, nf: int) -> np.ndarray:
        """
        Scatter active-cell chunk back to dense native grid.
        Returns shape (nt_chunk, nj, ni, nf), float32.
        """
        nt_chunk                = out_active.shape[0]
        full                    = np.full((nt_chunk, nj, ni, nf), self.config.fill_value, dtype=np.float32)
        full[:, active_mask, :] = out_active
        return full

    def _regrid_station_spectra_chunk_to_cice(self,
                                              efreq_station_chunk: xr.DataArray, *,   # (time, station, frequency)
                                              paths, overwrite_weights: bool = False) -> tuple[xr.DataArray, xr.Dataset]:
        """
        Chunked version of station-spectrum regridding.
        Returns efreq_unmasked_chunk(time, nj, ni, frequency) and static diag_ds.
        """
        if tuple(efreq_station_chunk.dims) != ("time", "station", "frequency"):
            efreq_station_chunk = efreq_station_chunk.transpose("time", "station", "frequency")
        station_lon  = efreq_station_chunk["station_lon"].values.astype(np.float64)
        station_lat  = efreq_station_chunk["station_lat"].values.astype(np.float64)
        matrix, diag = self.build_or_load_station_weights(station_lon, station_lat, paths=paths, overwrite=overwrite_weights)
        src          = efreq_station_chunk.fillna(0.0).values.astype(np.float32)            # (nt, ns, nf)
        nt, ns, nf   = src.shape
        # do dense multiply for this chunk only
        src2d        = np.transpose(src, (0, 2, 1)).reshape(nt * nf, ns)                    # (nt*nf, ns)
        out2d        = src2d @ matrix.T                                                     # (nt*nf, n_active)
        out_active   = np.asarray(out2d, dtype=np.float32).reshape(nt, nf, matrix.shape[0])
        out_active   = np.transpose(out_active, (0, 2, 1))                                  # (nt, n_active, nf)
        ds_grid      = self.get_target_grid(paths)
        active       = ds_grid["target_active_mask"].values.astype(bool)
        nj           = int(ds_grid.sizes["nj"])
        ni           = int(ds_grid.sizes["ni"])
        full         = self._scatter_active_chunk_to_grid(out_active, active, nj, ni, nf)
        efreq_grid   = xr.DataArray(full,
                                    dims   = ("time", "nj", "ni", "frequency"),
                                    coords = {"time"     : efreq_station_chunk["time"].values,
                                              "nj"       : np.arange(nj, dtype=np.int32),
                                              "ni"       : np.arange(ni, dtype=np.int32),
                                              "frequency": efreq_station_chunk["frequency"].values.astype(np.float32),
                                              "TLON"     : (("nj", "ni"), ds_grid["TLON"].values),
                                              "TLAT"     : (("nj", "ni"), ds_grid["TLAT"].values)},
                                    name   = "efreq_unmasked",
                                    attrs  = {"long_name": "CAWCR spectra regridded to CICE grid before SIC masking",
                                              "units"    : efreq_station_chunk.attrs.get("units", "m2 s")})
        # diagnostics are static, so only build once
        diag_ds = xr.Dataset(data_vars = {"distance_to_nearest_station_km": (("nj", "ni"),
                                                                             self._scatter_active_to_grid(diag["distance_km"], active, np.nan).astype(np.float32)),
                                          "n_station_neighbours"          : (("nj", "ni"),
                                                                             self._scatter_active_to_grid(diag["n_source_used"], active, 0).astype(np.int16)),
                                          "station_weight_sum"            : (("nj", "ni"),
                                                                             self._scatter_active_to_grid(diag["weight_sum"], active, 0.0).astype(np.float32)),
                                          "station_interp_valid"          : (("nj", "ni"),
                                                                             self._scatter_active_to_grid(diag["valid_target"], active, 0).astype(np.int8))},
                             coords    = {"nj": efreq_grid["nj"], "ni": efreq_grid["ni"]})
        return efreq_grid, diag_ds

    def _regrid_daily_sic_to_cice_all(self, sic_daily: xr.DataArray | xr.Dataset, *, paths,
                                      sic_var          : str  = "cdr_seaice_conc",
                                      overwrite_weights: bool = False) -> xr.DataArray:
        """
        Regrid daily SIC once for the whole requested date range.
        This object is small compared with efreq and is safe to keep in memory.
        """
        sic                  = self._normalise_sic(sic_daily, sic_var=sic_var)
        src_lon2d, src_lat2d = self._extract_lonlat_2d(sic)
        valid_src            = np.isfinite(src_lon2d) & np.isfinite(src_lat2d)
        src_lon              = src_lon2d[valid_src]
        src_lat              = src_lat2d[valid_src]
        matrix               = self.build_or_load_sic_weights(src_lon, src_lat, paths=paths, overwrite=overwrite_weights)
        src                  = sic.values.astype(np.float32)                        # (nt_daily, y, x)
        nt                   = src.shape[0]
        src2d                = src.reshape(nt, -1)[:, valid_src.ravel()]         # (nt_daily, n_valid_src)
        out_active           = np.asarray(src2d @ matrix.T, dtype=np.float32)
        ds_grid              = self.get_target_grid(paths)
        active               = ds_grid["target_active_mask"].values.astype(bool)
        nj                   = int(ds_grid.sizes["nj"])
        ni                   = int(ds_grid.sizes["ni"])
        full                 = np.full((nt, nj, ni), np.nan, dtype=np.float32)
        full[:, active]      = out_active
        return xr.DataArray(full,
                            dims   = ("time", "nj", "ni"),
                            coords = {"time": sic["time"].values,
                                      "nj"   : np.arange(nj, dtype=np.int32),
                                      "ni"   : np.arange(ni, dtype=np.int32),
                                      "TLON" : (("nj", "ni"), ds_grid["TLON"].values),
                                      "TLAT" : (("nj", "ni"), ds_grid["TLAT"].values)},
                            name   = "sic",
                            attrs  = {"long_name": "NSIDC daily sea ice concentration on CICE grid",
                                      "units"    : sic.attrs.get("units", "1")})

    def _initialise_month_file(self, *,
                               out         : Path,
                               ds_raw      : xr.Dataset,
                               ds_grid     : xr.Dataset,
                               diag_ds     : xr.Dataset,
                               time_values : np.ndarray,
                               freq_ds     : xr.Dataset,
                               overwrite   : bool,
                               time_chunk  : int,
                               zlib        : bool = True,
                               complevel   : int  = 3,
                               shuffle     : bool = True) -> None:
        if out.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists: {out}")
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        nt       = int(time_values.size)
        nf       = int(freq_ds.sizes["nfreq"])
        nj       = int(ds_grid.sizes["nj"])
        ni       = int(ds_grid.sizes["ni"])
        chunk4D  = (min(time_chunk, nt), nf, min(128, nj), min(128, ni))
        chunk3D  = (min(time_chunk, nt), min(128, nj), min(128, ni))
        fill_val = np.float32(np.nan)
        with NCFile(out, mode="w", format="NETCDF4") as nc:
            nc.createDimension("time", nt)
            nc.createDimension("nfreq", nf)
            nc.createDimension("nj", nj)
            nc.createDimension("ni", ni)
            time_num, time_units, time_calendar = _time_to_netcdf_numeric(time_values)
            vtime                               = nc.createVariable("time", "f8", ("time",))
            vtime[:]                            = time_num
            if time_units:
                vtime.units    = time_units
                vtime.calendar = time_calendar
            vfreq    = nc.createVariable("nfreq", "i4", ("nfreq",))
            vfreq[:] = freq_ds["nfreq"].values.astype(np.int32)
            vnj      = nc.createVariable("nj", "i4", ("nj",))
            vni      = nc.createVariable("ni", "i4", ("ni",))
            vnj[:]   = np.arange(nj, dtype=np.int32)
            vni[:]   = np.arange(ni, dtype=np.int32)
            # static grid/diag vars
            static_2d = {"TLON"                           : ds_grid["TLON"].values.astype(np.float32),
                         "TLAT"                           : ds_grid["TLAT"].values.astype(np.float32),
                         "ocean_mask"                     : ds_grid["ocean_mask"].values.astype(np.int8),
                         "target_active_mask"             : ds_grid["target_active_mask"].values.astype(np.int8),
                         "distance_to_nearest_station_km" : diag_ds["distance_to_nearest_station_km"].values.astype(np.float32),
                         "n_station_neighbours"           : diag_ds["n_station_neighbours"].values.astype(np.int16),
                         "station_weight_sum"             : diag_ds["station_weight_sum"].values.astype(np.float32),
                         "station_interp_valid"           : diag_ds["station_interp_valid"].values.astype(np.int8)}
            for name, arr in static_2d.items():
                dtype = "f4"
                if arr.dtype.kind in "iu":
                    dtype = "i2" if arr.dtype.itemsize <= 2 else "i4"
                if arr.dtype.kind == "b":
                    dtype = "i1"
                v = nc.createVariable(name, dtype, ("nj", "ni"))
                v[:] = arr
            # frequency vars
            for name in ("wavefreq", "wavefreq_lo", "wavefreq_hi", "dwavefreq"):
                v = nc.createVariable(name, "f4", ("nfreq",))
                v[:] = freq_ds[name].values.astype(np.float32)
            # time varying 4D
            nc.createVariable("efreq", "f4", ("time", "nfreq", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk4D,
                              fill_value = fill_val)
            nc.createVariable("efreq_unmasked", "f4", ("time", "nfreq", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk4D,
                              fill_value = fill_val)
            # time varying 3D
            nc.createVariable("hs_masked", "f4", ("time", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk3D,
                              fill_value = fill_val)
            nc.createVariable("hs_unmasked", "f4", ("time", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk3D,
                              fill_value = fill_val)
            nc.createVariable("sic", "f4", ("time", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk3D,
                              fill_value = fill_val)
            nc.createVariable("open_water_mask", "i1", ("time", "nj", "ni"),
                              zlib       = zlib,
                              complevel  = complevel,
                              shuffle    = shuffle,
                              chunksizes = chunk3D,
                              fill_value = np.int8(-1))
            nc.title                 = "CAWCR spectra regridded to native CICE grid and masked by NSIDC SIC"
            nc.sic_threshold         = float(self.config.sic_threshold)
            nc.hemisphere            = self.config.hemisphere
            nc.idw_k_nearest         = int(self.config.k_nearest)
            nc.idw_power             = float(self.config.idw_power)
            nc.idw_radius_km         = float(self.config.radius_km)
            nc.cawcr_source_file     = str(self._current_cawcr_path)
            nc.efreq_field_is_masked = "true"

    def _append_month_chunk(self, *,
                            out                   : Path,
                            i0                    : int,
                            i1                    : int,
                            efreq_unmasked_chunk  : xr.DataArray,          # time, nj, ni, frequency
                            efreq_masked_chunk    : xr.DataArray,          # time, nj, ni, frequency
                            sic_hourly_chunk      : xr.DataArray,          # time, nj, ni
                            open_water_mask_chunk : xr.DataArray,          # time, nj, ni
                            dwavefreq             : xr.DataArray) -> None:  # nfreq
        nfreq_index        = np.arange(efreq_unmasked_chunk.sizes["frequency"], dtype=np.int32)
        efreq_unmasked_out = (efreq_unmasked_chunk
                              .transpose("time", "frequency", "nj", "ni")
                              .rename({"frequency": "nfreq"})
                              .assign_coords(nfreq=nfreq_index)
                              .astype(np.float32))
        efreq_masked_out   = (efreq_masked_chunk
                              .transpose("time", "frequency", "nj", "ni")
                              .rename({"frequency": "nfreq"})
                              .assign_coords(nfreq=nfreq_index)
                              .astype(np.float32))
        dw                 = xr.DataArray(dwavefreq.values.astype(np.float32), dims=("nfreq",), coords={"nfreq": nfreq_index})
        hs_unmasked        = self.compute_hs(efreq_unmasked_out, dw).astype(np.float32)
        hs_masked          = self.compute_hs(efreq_masked_out, dw).astype(np.float32)
        with NCFile(out, mode="a") as nc:
            nc["efreq_unmasked"][i0:i1, :, :, :] = efreq_unmasked_out.values
            nc["efreq"][i0:i1, :, :, :]          = efreq_masked_out.values
            nc["hs_unmasked"][i0:i1, :, :]       = hs_unmasked.values
            nc["hs_masked"][i0:i1, :, :]         = hs_masked.values
            nc["sic"][i0:i1, :, :]               = sic_hourly_chunk.values.astype(np.float32)
            nc["open_water_mask"][i0:i1, :, :]   = open_water_mask_chunk.values.astype(np.int8)
            nc.sync()

    def prepare_month(self, start_date: str, end_date: str, *, paths: ShugaPaths,
                      obs_class             : Optional["SeaIceObservations"] = None,
                      overwrite_weights     : bool = False,
                      overwrite_sic_weights : bool = False,
                      write                 : bool = True,
                      overwrite_output      : bool = False,
                      time_chunk            : int  = 6,
                      return_dataset        : bool = False) -> xr.Dataset | Path:
        """
        Streaming monthly CAWCR->CICE builder.

        With write=True, this writes the monthly NetCDF incrementally and returns
        the output Path. With return_dataset=True and write=False, it can still
        return a full in-memory dataset for very small tests.
        """
        dt0, dtN = self._parse_month_window(start_date, end_date)
        year     = dt0.year
        month    = dt0.month
        if self.config.output_path is None:
            self.config.output_path = paths.cawcr_regridded_file(year, month)
        if self.config.weights_path is None:
            self.config.weights_path = paths.cawcr2cice_weight_file(year, month)
        if self.config.sic_weights_path is None:
            self.config.sic_weights_path = paths.nsidc2cice_weight_file
        cawcr_path = paths.cawcr_file(year, month)
        self._current_cawcr_path = cawcr_path
        self._log(f"Preparing CAWCR month for {dt0:%Y-%m-%d} to {dtN:%Y-%m-%d} from {cawcr_path}")
        if not Path(cawcr_path).exists():
            raise FileNotFoundError(f"CAWCR monthly source file not found: {cawcr_path}")
        obs_cfg = paths.observations or ObservationSpec()
        if obs_class is None:
            obs_class = SeaIceObservations(paths.run)
        sic_daily = obs_class.load_nsidc_daily(start_date = dt0.strftime("%Y-%m-%d"),
                                               end_date   = dtN.strftime("%Y-%m-%d"),
                                               hemisphere = self.config.hemisphere)
        if sic_daily is None:
            raise ValueError(f"SeaIceObservations.load_nsidc_daily() returned None for "
                             f"{dt0:%Y-%m-%d} to {dtN:%Y-%m-%d}.")
        self._log(f"opening {cawcr_path} and subsetting over time {dt0} to {dtN}")
        ds_raw = self.open_cawcr_month(cawcr_path)
        ds_raw = self._subset_time_window(ds_raw, dt0, dtN)
        self._log("collapsing directional spectrum")
        efreq_station = self.collapse_directional_spectrum(ds_raw)
        self._log("regridding NSIDC SIC to CICE")
        sic_daily_cice = self._regrid_daily_sic_to_cice_all(sic_daily,
                                                            paths             = paths,
                                                            sic_var           = obs_cfg.nsidc_sic_var,
                                                            overwrite_weights = overwrite_sic_weights)
        ds_grid = self.get_target_grid(paths)
        freq_ds = self.get_frequency_metadata(ds_raw)
        # build diagnostics once from a tiny first chunk
        first_chunk = efreq_station.isel(time=slice(0, min(time_chunk, efreq_station.sizes["time"])))
        self._log("building station weights / diagnostics")
        _, diag_ds = self._regrid_station_spectra_chunk_to_cice(first_chunk, paths=paths, overwrite_weights=overwrite_weights)
        out        = self.config.output_path
        if out is None:
            raise ValueError("config.output_path must be set before write=True prepare_month().")
        out = Path(out)
        if write:
            self._log("initialising monthly output file")
            self._initialise_month_file(out=out,
                                        ds_raw=ds_raw,
                                        ds_grid=ds_grid,
                                        diag_ds=diag_ds,
                                        time_values=efreq_station["time"].values,
                                        freq_ds=freq_ds,
                                        overwrite=overwrite_output,
                                        time_chunk=time_chunk)
        all_chunks = []
        time_values = efreq_station["time"].values
        nt = efreq_station.sizes["time"]
        dwavefreq = xr.DataArray(freq_ds["dwavefreq"].values.astype(np.float32),
                                 dims=("nfreq",),
                                 coords={"nfreq": freq_ds["nfreq"].values})
        for i0 in range(0, nt, time_chunk):
            i1 = min(i0 + time_chunk, nt)
            self._log(f"processing chunk {i0}:{i1} of {nt}")
            efreq_station_chunk = efreq_station.isel(time=slice(i0, i1))
            efreq_unmasked_chunk, _ = self._regrid_station_spectra_chunk_to_cice(efreq_station_chunk, paths=paths, overwrite_weights=False)
            sic_hourly_chunk = self.expand_daily_sic_to_hourly(sic_daily_cice, efreq_unmasked_chunk["time"])
            efreq_masked_chunk, open_water_mask_chunk = self.apply_ice_edge_mask(efreq_unmasked_chunk, sic_hourly_chunk)
            if write:
                self._append_month_chunk(out=out,
                                         i0=i0,
                                         i1=i1,
                                         efreq_unmasked_chunk=efreq_unmasked_chunk,
                                         efreq_masked_chunk=efreq_masked_chunk,
                                         sic_hourly_chunk=sic_hourly_chunk,
                                         open_water_mask_chunk=open_water_mask_chunk,
                                         dwavefreq=dwavefreq)
            if return_dataset:
                all_chunks.append(xr.Dataset(data_vars={"efreq": efreq_masked_chunk.rename({"frequency": "nfreq"}).transpose("time", "nfreq", "nj", "ni"),
                                                        "efreq_unmasked": efreq_unmasked_chunk.rename({"frequency": "nfreq"}).transpose("time", "nfreq", "nj", "ni"),
                                                        "sic": sic_hourly_chunk.astype(np.float32),
                                                        "open_water_mask": open_water_mask_chunk.astype(np.int8)}))
            del efreq_station_chunk, efreq_unmasked_chunk, sic_hourly_chunk, efreq_masked_chunk, open_water_mask_chunk
        if write and not return_dataset:
            self._log(f"Wrote {out}")
            return out
        if return_dataset:
            ds_joined = xr.concat(all_chunks, dim="time")
            ds_joined["TLON"] = ds_grid["TLON"].astype(np.float32)
            ds_joined["TLAT"] = ds_grid["TLAT"].astype(np.float32)
            ds_joined["ocean_mask"] = ds_grid["ocean_mask"].astype(np.int8)
            ds_joined["target_active_mask"] = ds_grid["target_active_mask"].astype(np.int8)
            for name in freq_ds.data_vars:
                ds_joined[name] = freq_ds[name]
            for name, da in diag_ds.data_vars.items():
                ds_joined[name] = da
            return ds_joined
        raise RuntimeError("Unexpected prepare_month() control path.")

    @staticmethod
    def _scatter_active_to_grid(values: np.ndarray, active_mask: np.ndarray, fill_value) -> np.ndarray:
        out              = np.full(active_mask.shape, fill_value, dtype=np.asarray(values).dtype)
        out[active_mask] = values
        return out

    def write_month(self, ds_out: xr.Dataset,
                    overwrite: bool = False,
                    time_chunk: int = 6) -> Path:
        """
        Write CAWCR->CICE monthly forcing dataset incrementally to NetCDF.

        Notes
        -----
        This avoids `ds_out.to_netcdf()` over the whole dataset in one shot.
        It writes time-varying variables in chunks along the time dimension.

        Important:
        This reduces write-time memory pressure, but it does NOT solve the
        main peak-memory issue if `ds_out` has already been fully materialised
        in memory upstream.
        """
        from netCDF4 import Dataset as NCFile
        out = self.config.output_path
        if out is None:
            raise ValueError("self.config.output_path must be set before calling write_month().")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists: {out}")
            out.unlink()
        self._log(f"Writing monthly CAWCR file to {out} with time_chunk={time_chunk}")
        # ----------------------------
        # Dimension sizes
        # ----------------------------
        nt = int(ds_out.sizes["time"])
        nj = int(ds_out.sizes["nj"])
        ni = int(ds_out.sizes["ni"])
        nf = int(ds_out.sizes["nfreq"])
        # ----------------------------
        # Variable groups
        # ----------------------------
        static_vars = ["TLON", "TLAT", "ocean_mask", "target_active_mask",
                       "distance_to_nearest_station_km", "n_station_neighbours", "station_weight_sum", "station_interp_valid",
                       "wavefreq", "wavefreq_lo", "wavefreq_hi", "dwavefreq"]
        time_vars_4d = ["efreq","efreq_unmasked"]
        time_vars_3d = ["hs_masked", "hs_unmasked", "sic", "open_water_mask"]
        # retain only vars that actually exist
        static_vars = [v for v in static_vars if v in ds_out]
        time_vars_4d = [v for v in time_vars_4d if v in ds_out]
        time_vars_3d = [v for v in time_vars_3d if v in ds_out]
        # ----------------------------
        # File creation
        # ----------------------------
        with NCFile(out, mode="w", format="NETCDF4") as nc:
            # dimensions
            nc.createDimension("time", nt)
            nc.createDimension("nfreq", nf)
            nc.createDimension("nj", nj)
            nc.createDimension("ni", ni)
            # global attrs
            for key, value in ds_out.attrs.items():
                try:
                    setattr(nc, key, value)
                except Exception:
                    setattr(nc, key, str(value))
            # ------------------------
            # coordinate variables
            # ------------------------
            v_time = nc.createVariable("time", "f8", ("time",))
            v_nfreq = nc.createVariable("nfreq", "i4", ("nfreq",))
            v_nj = nc.createVariable("nj", "i4", ("nj",))
            v_ni = nc.createVariable("ni", "i4", ("ni",))
            # preserve time units/calendar if possible
            time_da = ds_out["time"]
            time_vals = time_da.values
            # write time as raw numeric if already numeric, else datetime64 ns -> seconds since epoch
            if np.issubdtype(time_vals.dtype, np.datetime64):
                epoch = np.datetime64("1970-01-01T00:00:00")
                time_num = ((time_vals - epoch) / np.timedelta64(1, "s")).astype(np.float64)
                v_time[:] = time_num
                v_time.units = "seconds since 1970-01-01 00:00:00"
                v_time.calendar = "standard"
            else:
                v_time[:] = np.asarray(time_vals, dtype=np.float64)
            v_nfreq[:] = np.asarray(ds_out["nfreq"].values, dtype=np.int32)
            v_nj[:] = np.asarray(ds_out["nj"].values, dtype=np.int32)
            v_ni[:] = np.asarray(ds_out["ni"].values, dtype=np.int32)
            # ------------------------
            # helper for attrs
            # ------------------------
            def _copy_attrs(src_da, dst_var):
                for key, value in src_da.attrs.items():
                    try:
                        setattr(dst_var, key, value)
                    except Exception:
                        setattr(dst_var, key, str(value))
            # ------------------------
            # create static variables
            # ------------------------
            static_created = {}
            for name in static_vars:
                da = ds_out[name]
                dims = da.dims
                dtype = "f4"
                if da.dtype.kind in "iu":
                    dtype = "i4" if da.dtype.itemsize >= 4 else "i2"
                elif da.dtype.kind == "b":
                    dtype = "i1"
                zlib = False
                complevel = 0
                chunksizes = None
                if dims == ("nj", "ni"):
                    chunksizes = (min(256, nj), min(256, ni))
                elif dims == ("nfreq",):
                    chunksizes = (nf,)
                var = nc.createVariable(name, dtype, dims,
                                        zlib=zlib,
                                        complevel=complevel,
                                        shuffle=False,
                                        chunksizes=chunksizes)
                _copy_attrs(da, var)
                static_created[name] = var
            # ------------------------
            # create time-varying vars
            # ------------------------
            time_created = {}
            for name in time_vars_4d:
                da = ds_out[name]
                var = nc.createVariable(name, "f4", ("time", "nfreq", "nj", "ni"),
                                        zlib=True,
                                        complevel=3,
                                        shuffle=True,
                                        chunksizes=(min(time_chunk, nt), nf, min(128, nj), min(128, ni)),
                                        fill_value=np.float32(np.nan))
                _copy_attrs(da, var)
                time_created[name] = var
            for name in time_vars_3d:
                da = ds_out[name]
                if da.dtype.kind in "iu":
                    dtype = "i2" if da.dtype.itemsize <= 2 else "i4"
                    fill_value = 0
                else:
                    dtype = "f4"
                    fill_value = np.float32(np.nan) if name == "sic" else np.float32(0.0)
                var = nc.createVariable(name, dtype, ("time", "nj", "ni"),
                                        zlib=True,
                                        complevel=3,
                                        shuffle=True,
                                        chunksizes=(min(time_chunk, nt), min(128, nj), min(128, ni)),
                                        fill_value=fill_value)
                _copy_attrs(da, var)
                time_created[name] = var
            # ------------------------
            # write static variables once
            # ------------------------
            self._log("Writing static variables")
            for name, var in static_created.items():
                da = ds_out[name]
                arr = da.values
                if da.dtype.kind == "f":
                    arr = np.asarray(arr, dtype=np.float32)
                elif da.dtype.kind in "iu":
                    arr = np.asarray(arr)
                elif da.dtype.kind == "b":
                    arr = np.asarray(arr, dtype=np.int8)
                var[:] = arr
            # ------------------------
            # write time-varying variables chunk-by-chunk
            # ------------------------
            for i0 in range(0, nt, time_chunk):
                i1 = min(i0 + time_chunk, nt)
                self._log(f"Writing time chunk {i0}:{i1} / {nt}")
                for name in time_vars_4d:
                    da = ds_out[name].isel(time=slice(i0, i1))
                    arr = da.values
                    if hasattr(arr, "compute"):
                        arr = arr.compute()
                    arr = np.asarray(arr, dtype=np.float32)
                    time_created[name][i0:i1, :, :, :] = arr
                for name in time_vars_3d:
                    da = ds_out[name].isel(time=slice(i0, i1))
                    arr = da.values
                    if hasattr(arr, "compute"):
                        arr = arr.compute()
                    if da.dtype.kind == "f":
                        arr = np.asarray(arr, dtype=np.float32)
                    elif da.dtype.kind in "iu":
                        arr = np.asarray(arr)
                    elif da.dtype.kind == "b":
                        arr = np.asarray(arr, dtype=np.int8)
                    time_created[name][i0:i1, :, :] = arr
                nc.sync()
        self._log(f"Wrote {out}")
        return out


