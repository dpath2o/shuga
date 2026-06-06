from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import xarray as xr
from shuga.core.paths import ShugaPaths
from shuga.core.types import CICEGridSpec

@dataclass(slots=True)
class CICEGridBundle:
    tgrid      : xr.Dataset
    ugrid      : xr.Dataset | None = None
    egrid      : xr.Dataset | None = None
    ngrid      : xr.Dataset | None = None
    mask       : xr.DataArray | None = None
    mask_mod   : xr.DataArray | None = None
    bathymetry : xr.DataArray | None = None
    grid_kind  : str = "cice"
    source_path: str | None = None
    metadata   : dict[str, Any] | None = None

class CICEGridwork:
    """
    Grid loader and geometry builder for native CICE and MOM6 supergrid files.

    This is the shuga equivalent of the grid-loading half of AFIM's
    ``sea_ice_gridwork.py``. It standardises all public lon/lat outputs to
    degrees and publishes ``TLON`` / ``TLAT`` on the T-grid regardless of the
    source file naming convention.
    """
    def __init__(self, paths: ShugaPaths, grid_spec: CICEGridSpec | None = None, logger=None) -> None:
        self.paths     = paths
        self.grid_spec = grid_spec or paths.cice_grid or CICEGridSpec()
        self.logger    = logger
        self._grid_bundle: CICEGridBundle | None = None

    #------------------------------------------------------------------------------
    # helpers
    #------------------------------------------------------------------------------
    @staticmethod
    def normalise_longitudes(lon, to: str = "0-360", eps: float = 1e-12):
        lon_wrapped = ((lon % 360) + 360) % 360
        if to == "0-360":
            if isinstance(lon_wrapped, xr.DataArray):
                return xr.where(np.isclose(lon_wrapped, 360.0, atol=eps), 0.0, lon_wrapped)
            return np.where(np.isclose(lon_wrapped, 360.0, atol=eps), 0.0, lon_wrapped)
        if to == "-180-180":
            lon_180 = ((lon_wrapped + 180.0) % 360.0) - 180.0
            if isinstance(lon_180, xr.DataArray):
                return xr.where(np.isclose(lon_180, 180.0, atol=eps), -180.0, lon_180)
            return np.where(np.isclose(lon_180, 180.0, atol=eps), -180.0, lon_180)
        raise ValueError("to must be '0-360' or '-180-180'")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def _infer_deg_from_grid_units(self, values, name: str, units: str | None = None):
        arr = np.asarray(values, dtype="float64")
        unit = str(units or "").strip().lower()
        if "radian" in unit:
            return np.rad2deg(arr)
        if "degree" in unit:
            return arr
        finite = np.isfinite(arr)
        if finite.any() and np.nanmax(np.abs(arr[finite])) <= (2.0 * np.pi + 1e-6):
            self._log(f"Treating {name} as radians based on value range.")
            return np.rad2deg(arr)
        return arr

    def _to_meters(self, values, units: str | None, name: str = "metric"):
        arr = np.asarray(values, dtype="float64")
        unit = str(units or "").strip().lower()
        if unit in {"m", "meter", "meters", "metre", "metres"}:
            return arr
        if unit in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
            return arr / 100.0
        if unit in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            return arr * 1000.0
        if not unit:
            self._warn(f"No units for {name}; assuming meters.")
            return arr
        self._warn(f"Unknown units {units!r} for {name}; assuming meters.")
        return arr

    def _pick(self, ds: xr.Dataset, *names: str) -> str | None:
        for name in names:
            if name in ds.variables:
                return name
        return None

    def _open_ocean_mask_on_tgrid(self,
                                  P_mask : str | Path | None = None,
                                  P_topog: str | Path | None = None,
                                  nx     : int | None = None,
                                  ny     : int | None = None) -> xr.DataArray | None:
        def _shape_ok(da: xr.DataArray, source: str) -> bool:
            if da.ndim != 2:
                return False
            if ny is None or nx is None:
                return True
            if da.shape != (int(ny), int(nx)):
                self._warn(f"Skipping {source} mask because shape {da.shape} does not "
                           f"match target T-grid shape {(int(ny), int(nx))}.")
                return False
            return True
        if P_mask is None:
            P_mask = self.paths.cice_kmt_path
        if P_mask is not None and Path(P_mask).exists():
            ds = xr.open_dataset(P_mask, decode_times=False)
            var = self._pick(ds, "kmt", "KMT", "wet", "mask")
            if var is not None:
                da = ds[var]
                if da.ndim == 2:
                    if tuple(da.dims) != ("nj", "ni"):
                        da = da.rename({da.dims[-2]: "nj", da.dims[-1]: "ni"})
                    if not _shape_ok(da, str(P_mask)):
                        return None
                    return xr.where(np.isfinite(da) & (da > 0), True, False)
        if P_topog is None:
            P_topog = self.paths.cice_bathymetry_path
        if P_topog is not None and Path(P_topog).exists():
            ds = xr.open_dataset(P_topog, decode_times=False)
            var = self._pick(ds, "depth", "DEPTH", "bathymetry", "topog", "bathy")
            if var is not None:
                da = ds[var]
                if da.ndim == 2:
                    if tuple(da.dims) != ("nj", "ni"):
                        da = da.rename({da.dims[-2]: "nj", da.dims[-1]: "ni"})
                    if not _shape_ok(da, str(P_topog)):
                        return None
                    return xr.where(np.isfinite(da) & (da > 0), True, False)
        return None

    def _circular_mean_lon_deg(self, *lon_deg_arrays):
        lon  = np.stack(lon_deg_arrays, axis=0)
        lon  = self.normalise_longitudes(lon, to="0-360")
        lonr = np.deg2rad(lon)
        x    = np.cos(lonr).mean(axis=0)
        y    = np.sin(lonr).mean(axis=0)
        out  = np.rad2deg(np.arctan2(y, x))
        return self.normalise_longitudes(out, to="0-360")

    #------------------------------------------------------------------------------
    # APIs
    #------------------------------------------------------------------------------
    def build_grid_faces(self, lon, lat, source_in_radians: bool = False):
        if source_in_radians:
            lon = np.rad2deg(np.asarray(lon))
            lat = np.rad2deg(np.asarray(lat))
        lon               = self.normalise_longitudes(lon, to="0-360")
        lat               = np.asarray(lat, dtype="float64")
        nj, ni            = lat.shape
        lon_b             = np.full((nj + 1, ni + 1), np.nan, dtype=float)
        lat_b             = np.full((nj + 1, ni + 1), np.nan, dtype=float)
        lat_b[1:-1, 1:-1] = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:])
        lon_b[1:-1, 1:-1] = self._circular_mean_lon_deg(lon[:-1, :-1], lon[:-1, 1:], lon[1:, :-1], lon[1:, 1:])
        lat_b[0   , 1:-1] = 0.5  * (lat[0, :-1  ] + lat[0, 1: ])
        lon_b[0   , 1:-1] = self._circular_mean_lon_deg(lon[0, :-1  ], lon[0, 1: ])
        lat_b[-1  , 1:-1] = 0.5  * (lat[-1, :-1 ] + lat[-1, 1:])
        lon_b[-1  , 1:-1] = self._circular_mean_lon_deg(lon[-1, :-1 ], lon[-1, 1:])
        lat_b[1:-1, 0   ] = 0.5  * (lat[:-1, 0  ] + lat[1:, 0 ])
        lon_b[1:-1, 0   ] = self._circular_mean_lon_deg(lon[:-1, 0  ], lon[1:, 0 ])
        lat_b[1:-1, -1  ] = 0.5  * (lat[:-1, -1 ] + lat[1:, -1])
        lon_b[1:-1, -1  ] = self._circular_mean_lon_deg(lon[:-1, -1 ], lon[1:, -1])
        lat_b[0   , 0   ] = lat[0, 0]
        lat_b[0   , -1  ] = lat[0, -1]
        lat_b[-1  , 0   ] = lat[-1, 0]
        lat_b[-1  , -1  ] = lat[-1, -1]
        lon_b[0   , 0   ] = lon[0, 0]
        lon_b[0   , -1  ] = lon[0, -1]
        lon_b[-1  , 0   ] = lon[-1, 0]
        lon_b[-1  , -1  ] = lon[-1, -1]
        lon_b             = self.normalise_longitudes(lon_b, to="0-360")
        lat_e             = 0.5 * (lat_b[:-1, :] + lat_b[1:, :])
        lon_e             = self._circular_mean_lon_deg(lon_b[:-1, :], lon_b[1:, :])
        lat_n             = 0.5 * (lat_b[:, :-1] + lat_b[:, 1:])
        lon_n             = self._circular_mean_lon_deg(lon_b[:, :-1], lon_b[:, 1:])
        return lon_b, lat_b, lon_e, lat_e, lon_n, lat_n

    def load_super_grid(self,
                        P_grid  : str | Path | None = None,
                        lon_type: str | None = None,
                        nx      : int | None = None,
                        ny      : int | None = None,
                        nx_in   : int | None = None,
                        ny_in   : int | None = None):
        if nx is None:
            nx = nx_in
        if ny is None:
            ny = ny_in
        if P_grid is None:
            P_grid = self.paths.cice_grid_path
        if lon_type is None:
            lon_type = self.grid_spec.lon_type
        self._log(f"Opening grid geometry: {P_grid}")
        ds       = xr.open_dataset(P_grid, decode_times=False)
        is_super = ("nxp" in ds.sizes) and ("nyp" in ds.sizes) and ("x" in ds.variables) and ("y" in ds.variables)
        if is_super:
            grid_kind = "mom6_supergrid"
            x_deg     = self._infer_deg_from_grid_units(ds["x"].values, "x", ds["x"].attrs.get("units"))
            y_deg     = self._infer_deg_from_grid_units(ds["y"].values, "y", ds["y"].attrs.get("units"))
            nxp       = int(ds.sizes["nxp"])
            nyp       = int(ds.sizes["nyp"])
            nx_d      = (nxp - 1) // 2
            ny_d      = (nyp - 1) // 2
            if (2 * nx_d + 1) != nxp or (2 * ny_d + 1) != nyp:
                raise RuntimeError(f"Unexpected supergrid dims: nxp={nxp}, nyp={nyp}")
            if nx is not None and int(nx) != nx_d:
                raise RuntimeError(f"Supergrid-derived nx={nx_d} != requested nx={nx}")
            if ny is not None and int(ny) != ny_d:
                raise RuntimeError(f"Supergrid-derived ny={ny_d} != requested ny={ny}")
            tlon = self.normalise_longitudes(x_deg[1::2, 1::2], to=lon_type)
            tlat = y_deg[1::2, 1::2]
            if "angle_dx" in ds.variables:
                ang_deg    = self._infer_deg_from_grid_units(ds["angle_dx"].values, "angle_dx", ds["angle_dx"].attrs.get("units"))
                anglet_rad = np.deg2rad(ang_deg[1::2, 1::2])
            else:
                self._warn("angle_dx missing; estimating angle from x/y finite differences.")
                dxlon      = x_deg[1::2, 2::2] - x_deg[1::2, 0::2]
                dylat      = y_deg[1::2, 2::2] - y_deg[1::2, 0::2]
                anglet_rad = np.arctan2(dylat, dxlon)[:, :nx_d]
            dx_m = self._to_meters(ds["dx"].values, ds["dx"].attrs.get("units"), name="dx")
            dy_m = self._to_meters(ds["dy"].values, ds["dy"].attrs.get("units"), name="dy")
            dx_m = dx_m[1::2, 0::2] + dx_m[1::2, 1::2]
            dy_m = dy_m[0::2, 1::2] + dy_m[1::2, 1::2]
            if "area" in ds.variables:
                a = ds["area"].astype("float64").values
                if a.shape == (2 * ny_d, 2 * nx_d):
                    area_m2 = a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]
                else:
                    self._warn(f"Unexpected area shape {a.shape}; expected {(2 * ny_d, 2 * nx_d)}")
                    area_m2 = np.full((ny_d, nx_d), np.nan, dtype="float64")
            else:
                area_m2 = np.full((ny_d, nx_d), np.nan, dtype="float64")
            return tlon, tlat, anglet_rad, dx_m, dy_m, area_m2, grid_kind, ds
        grid_kind = "cice"
        v_tlon    = self._pick(ds, "tlon", "TLON", "t_lon", "lon_t")
        v_tlat    = self._pick(ds, "tlat", "TLAT", "t_lat", "lat_t")
        v_angt    = self._pick(ds, "anglet", "angleT", "ANGLET", "angle_t")
        v_hte     = self._pick(ds, "hte", "HTE", "dxT", "dxt")
        v_htn     = self._pick(ds, "htn", "HTN", "dyT", "dyt")
        for v, nm in [(v_tlon, "tlon"), (v_tlat, "tlat"), (v_angt, "anglet"), (v_hte, "hte"), (v_htn, "htn")]:
            if v is None:
                raise KeyError(f"Could not find required grid variable '{nm}' in {P_grid}")
        tlon   = self._infer_deg_from_grid_units(ds[v_tlon].values, "tlon", ds[v_tlon].attrs.get("units"))
        tlat   = self._infer_deg_from_grid_units(ds[v_tlat].values, "tlat", ds[v_tlat].attrs.get("units"))
        tlon   = self.normalise_longitudes(tlon, to=lon_type)
        angt   = np.asarray(ds[v_angt].values, dtype="float64")
        finite = np.isfinite(angt)
        if finite.any() and np.nanmax(np.abs(angt[finite])) <= (2.0 * np.pi + 1e-6):
            anglet_rad = angt
        else:
            anglet_rad = np.deg2rad(angt)
        dx_m    = self._to_meters(ds[v_hte].values, ds[v_hte].attrs.get("units"), name=v_hte)
        dy_m    = self._to_meters(ds[v_htn].values, ds[v_htn].attrs.get("units"), name=v_htn)
        area_m2 = None
        for v in ("tarea", "TAREA", "area"):
            if v in ds.variables and ds[v].ndim == 2:
                area_m2 = np.asarray(ds[v].values, dtype="float64")
                break
        if area_m2 is None:
            area_m2 = np.asarray(dx_m, dtype="float64") * np.asarray(dy_m, dtype="float64")
        if (nx is not None and tlon.shape[1] != int(nx)) or (ny is not None and tlon.shape[0] != int(ny)):
            self._warn(f"CICE tlon shape {tlon.shape} does not match requested (ny,nx)=({ny},{nx})")
        return tlon, tlat, anglet_rad, dx_m, dy_m, area_m2, grid_kind, ds

    def load_cice_grid(self,
                       P_grid     : str | Path | None = None,
                       P_mask_org : str | Path | None = None,
                       P_mask_mod : str | Path | None = None,
                       slice_hem  : bool = False,
                       build_faces: bool = True,
                       nx         : int | None = None,
                       ny         : int | None = None) -> CICEGridBundle:
        if self._grid_bundle is not None and P_grid is None and P_mask_org is None and P_mask_mod is None and not slice_hem and build_faces:
            return self._grid_bundle
        tlon, tlat, anglet_rad, dx_m, dy_m, area_m2, grid_kind, ds_grid = self.load_super_grid(P_grid   = P_grid,
                                                                                               lon_type = self.grid_spec.lon_type,
                                                                                               nx       = nx,
                                                                                               ny       = ny)
        if tuple(np.asarray(tlon).shape) != tuple(np.asarray(tlat).shape):
            raise ValueError("TLON and TLAT shapes do not match.")
        nj, ni = tlat.shape
        tgrid = xr.Dataset(data_vars = {"TLON"  : (("nj", "ni"), np.asarray(tlon, dtype="float64")),
                                        "TLAT"  : (("nj", "ni"), np.asarray(tlat, dtype="float64")),
                                        "ANGLET": (("nj", "ni"), np.asarray(anglet_rad, dtype="float64")),
                                        "HTE"   : (("nj", "ni"), np.asarray(dx_m, dtype="float64")),
                                        "HTN"   : (("nj", "ni"), np.asarray(dy_m, dtype="float64")),
                                        "TAREA" : (("nj", "ni"), np.asarray(area_m2, dtype="float64"))},
                           coords    = {"nj": np.arange(nj, dtype=np.int32),
                                        "ni": np.arange(ni, dtype=np.int32)},
                           attrs     = {"grid_kind"  : grid_kind,
                                        "source_path": str(P_grid or self.paths.cice_grid_path)})
        if slice_hem:
            hemi = self.paths.hemisphere
            if hemi == "SH":
                selector = tgrid["TLAT"] <= 0.0
            else:
                selector = tgrid["TLAT"] >= 0.0
            tgrid = tgrid.where(selector)
        mask_org   = self._open_ocean_mask_on_tgrid(P_mask=P_mask_org, nx=ni, ny=nj)
        mask_mod   = self._open_ocean_mask_on_tgrid(P_mask=P_mask_mod, nx=ni, ny=nj) if P_mask_mod is not None else None
        bathy      = None
        bathy_path = self.paths.cice_bathymetry_path
        if bathy_path is not None and Path(bathy_path).exists():
            ds_b = xr.open_dataset(bathy_path, decode_times=False)
            v = self._pick(ds_b, "depth", "DEPTH", "bathymetry", "topog", "bathy")
            if v is not None and ds_b[v].ndim == 2:
                bathy = ds_b[v]
                if tuple(bathy.dims) != ("nj", "ni"):
                    bathy = bathy.rename({bathy.dims[-2]: "nj", bathy.dims[-1]: "ni"})
        ugrid = None
        egrid = None
        ngrid = None
        if build_faces:
            lon_b, lat_b, lon_e, lat_e, lon_n, lat_n = self.build_grid_faces(tgrid["TLON"].values, tgrid["TLAT"].values)
            ugrid = xr.Dataset(data_vars = {"ULON": (("nj_b", "ni_b"), lon_b),
                                            "ULAT": (("nj_b", "ni_b"), lat_b)},
                               coords    = {"nj_b": np.arange(lon_b.shape[0], dtype=np.int32),
                                            "ni_b": np.arange(lon_b.shape[1], dtype=np.int32)})
            egrid = xr.Dataset(data_vars = {"ELON": (("nj", "ni_b"), lon_e),
                                            "ELAT": (("nj", "ni_b"), lat_e)},
                               coords    = {"nj"  : np.arange(lon_e.shape[0], dtype=np.int32),
                                            "ni_b": np.arange(lon_e.shape[1], dtype=np.int32)})
            ngrid = xr.Dataset(data_vars = {"NLON": (("nj_b", "ni"), lon_n),
                                            "NLAT": (("nj_b", "ni"), lat_n)},
                               coords    = {"nj_b": np.arange(lon_n.shape[0], dtype=np.int32),
                                            "ni"  : np.arange(lon_n.shape[1], dtype=np.int32)})
        assets = self.paths.resolve_cice_grid_assets()
        bundle = CICEGridBundle(tgrid       = tgrid,
                                ugrid       = ugrid,
                                egrid       = egrid,
                                ngrid       = ngrid,
                                mask        = mask_org,
                                mask_mod    = mask_mod,
                                bathymetry  = bathy,
                                grid_kind   = grid_kind,
                                source_path = str(P_grid or self.paths.cice_grid_path),
                                metadata    = {"ice_in_file"    : str(assets.get("ice_in_file")) if assets.get("ice_in_file") is not None else None,
                                               "ice_diag_file"  : str(assets.get("ice_diag_file")) if assets.get("ice_diag_file") is not None else None,
                                               "grid_file"      : str(assets.get("grid_file")) if assets.get("grid_file") is not None else None,
                                               "kmt_file"       : str(assets.get("kmt_file")) if assets.get("kmt_file") is not None else None,
                                               "bathymetry_file": str(assets.get("bathymetry_file")) if assets.get("bathymetry_file") is not None else None,
                                               "f2_file"        : str(assets.get("f2_file")) if assets.get("f2_file") is not None else None})
        if P_grid is None and P_mask_org is None and P_mask_mod is None and not slice_hem and build_faces:
            self._grid_bundle = bundle
        return bundle

    def _open_loose_static_zarr_arrays(self, P_: Path, *, chunks: dict | None = None) -> xr.Dataset:
        """
        Open a directory containing loose per-variable zarr arrays but no root zarr group.

        This handles stores that look like:

            CICE_0p25_Cgrid_coords.zarr/
                TLON/
                TLAT/
                tarea/
                ...

        rather than a proper xarray Dataset zarr store with root .zgroup/zarr.json.
        """
        import dask.array as da
        var_dirs = sorted([p for p in P_.iterdir() if p.is_dir()])
        if not var_dirs:
            raise FileNotFoundError(f"No zarr array directories found in {P_}")
        data_vars: dict[str, xr.DataArray] = {}
        for d in var_dirs:
            name = d.name
            try:
                arr = da.from_zarr(str(d))
            except Exception as exc:
                self._warn(f"Skipping {d}: could not open as zarr array: {exc}")
                continue
            if arr.ndim == 2:
                dims = ("nj", "ni")
            elif arr.ndim == 1:
                dims = (f"{name}_dim",)
            elif arr.ndim == 0:
                dims = ()
            else:
                dims = tuple(f"{name}_dim_{i}" for i in range(arr.ndim))
            data_vars[name] = xr.DataArray(arr, dims=dims, name=name)
        if not data_vars:
            raise RuntimeError(f"No readable zarr arrays found in {P_}")
        ds = xr.Dataset(data_vars)
        if chunks is not None:
            ds = ds.chunk(chunks)
        ds.attrs.update(source_path=str(P_),
                        grid_kind="cice_static_loose_zarr_arrays",
                        loader="CICEGridwork._open_loose_static_zarr_arrays",
                        warning="Opened from loose zarr arrays because no root zarr group metadata was present.")
        return ds

    def load_cice_static(self, P_cice_static_store: str | Path | None = None, *,
                         variables    : Iterable[str] | None = None,
                         require      : Iterable[str] = ("TLON", "TLAT"),
                         hemisphere   : str | None = None,
                         south_lat_max: float | None = None,
                         lon_type     : str | None = None,
                         chunks       : dict | None = None,
                         consolidated : bool = False,
                         add_aliases  : bool = True) -> xr.Dataset:
        """
        Load the persistent CICE static-coordinate zarr store.

        Default store:
            ~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr

        Parameters
        ----------
        P_cice_static_store
            Optional explicit zarr path. If omitted, use
            ``self.paths.resolve_static_store()``.
        variables
            Optional variable list to return. If omitted, all variables in the
            static store are returned.
        require
            Variables that must be present before optional subsetting. By default
            TLON and TLAT are required.
        hemisphere
            Optional hemisphere subset. Use "SH" or "NH". This is a latitude-sign
            subset and is intentionally conservative.
        south_lat_max
            Optional southern-latitude row subset, e.g. -45.0 for Antarctic
            comparison workflows. If supplied, rows are kept where any TLAT in the
            row is <= south_lat_max.
        lon_type
            Optional longitude convention for longitude variables:
            "0-360" or "-180-180". If None, leave stored values unchanged.
        chunks
            Optional xarray chunks passed to ``xr.open_zarr``.
        consolidated
            Whether to use consolidated zarr metadata.
        add_aliases
            If True, add uppercase aliases such as TAREA from tarea when useful.
            Existing variables are not overwritten.

        Returns
        -------
        xr.Dataset
            Static CICE coordinate/metric/mask dataset.
        """
        if P_cice_static_store is None:
            P_ = self.paths.resolve_static_store()
            if P_ is None:
                raise FileNotFoundError("Could not find CICE static-coordinate zarr store. "
                                        f"Default expected at {self.paths.default_cice_static_store_path}")
        else:
            P_ = Path(P_cice_static_store).expanduser()
        P_ = Path(P_).expanduser()
        if not P_.exists():
            raise FileNotFoundError(P_)
        self._log(f"Opening CICE static-coordinate store: {P_}")
        try:
            ds = xr.open_zarr(P_, consolidated=consolidated, chunks=chunks)
        except Exception as exc:
            msg = str(exc)
            if "No group found" in msg or "GroupNotFoundError" in exc.__class__.__name__:
                self._warn(f"{P_} is not a valid xarray zarr group; attempting loose-array fallback.")
                ds = self._open_loose_static_zarr_arrays(P_, chunks=chunks)
            else:
                raise
        missing_required = [v for v in require if v not in ds]
        if missing_required:
            raise KeyError(f"Required CICE static variable(s) missing from {P_}: "
                           f"{missing_required}. Available variables: {list(ds.data_vars)}")
        if add_aliases:
            alias_pairs = {"tarea": "TAREA",
                           "uarea": "UAREA",
                           "earea": "EAREA",
                           "narea": "NAREA",
                           "tmask": "TMASK",
                           "umask": "UMASK",
                           "emask": "EMASK",
                           "nmask": "NMASK",
                           "dxt": "DXT",
                           "dyt": "DYT",
                           "dxu": "DXU",
                           "dyu": "DYU",
                           "dxe": "DXE",
                           "dye": "DYE",
                           "dxn": "DXN",
                           "dyn": "DYN"}
            for src, dst in alias_pairs.items():
                if src in ds and dst not in ds:
                    ds[dst] = ds[src]
        if lon_type is not None:
            if lon_type not in {"0-360", "-180-180"}:
                raise ValueError("lon_type must be None, '0-360', or '-180-180'.")
            for lon_name in ("TLON", "ULON", "ELON", "NLON"):
                if lon_name in ds:
                    ds[lon_name] = self.normalise_longitudes(ds[lon_name], to=lon_type)
        # Optional latitude/hemisphere subsetting. This is row-based so it remains
        # cheap and preserves the full circumpolar x direction.
        if hemisphere is not None or south_lat_max is not None:
            if "TLAT" not in ds:
                raise KeyError("TLAT is required for hemisphere/south_lat_max subsetting.")
            tlat = ds["TLAT"]
            if tlat.ndim != 2:
                raise ValueError(f"Expected TLAT to be 2-D, got dims={tlat.dims}")
            ydim, xdim = tlat.dims
            if south_lat_max is not None:
                row_mask = (tlat <= float(south_lat_max)).any(dim=xdim).compute()
            else:
                hemi = self.paths.canonical_hemisphere(hemisphere or self.paths.hemisphere)
                if hemi == "SH":
                    row_mask = (tlat <= 0.0).any(dim=xdim).compute()
                else:
                    row_mask = (tlat >= 0.0).any(dim=xdim).compute()
            rows = np.where(row_mask.values)[0]
            if rows.size == 0:
                raise ValueError(f"No CICE static rows matched hemisphere={hemisphere!r}, south_lat_max={south_lat_max!r}.")
            ds = ds.isel({ydim: slice(int(rows.min()), int(rows.max()) + 1)})
        if variables is not None:
            requested = list(dict.fromkeys([*require, *variables]))
            present = [v for v in requested if v in ds]
            missing = [v for v in requested if v not in ds]
            if missing:
                self._warn(f"Requested static variable(s) missing from {P_}: {missing}")
            ds = ds[present]
        ds.attrs.update(source_path = str(P_), grid_kind = "cice_static_zarr", loader = "CICEGridwork.load_cice_static")
        return ds

    def open_cice_static(self, *args, **kwargs) -> xr.Dataset:
        """Backward-compatible alias for load_cice_static()."""
        return self.load_cice_static(*args, **kwargs)
