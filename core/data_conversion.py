from __future__ import annotations
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import xarray as xr
from .paths import ShugaPaths

__all__         = ["NC2Zarr", "NC2ZarrResult"]
_MONTH_RE       = re.compile(r"^\d{4}-\d{2}$")
_DATE_RE        = re.compile(r"(\d{4}-\d{2}-\d{2})\.nc$")
_HOURLY_FILE_RE = re.compile(r"^iceh(?:_inst)?\.(\d{4}-\d{2}-\d{2})-(\d{5})\.nc$")
_DAY_GROUP_RE   = re.compile(r"^\d{4}_\d{2}_\d{2}$")

@dataclass(slots=True)
class NC2ZarrResult:
    cice_store      : Path
    static_store    : Path | None
    months_scanned  : int = 0
    months_written  : int = 0
    months_skipped  : int = 0
    months_rewritten: int = 0
    daily_files_seen: int = 0
    daily_files_used: int = 0

class NC2Zarr:
    """
    Convert daily CICE ``iceh.YYYY-MM-DD.nc`` history files into grouped monthly
    Zarr and maintain a companion ``iceh_static.zarr`` store.

    Dynamic fields are written to ``iceh_daily.zarr/YYYY-MM``. Static grid,
    mask, and geometry fields are written once to ``iceh_static.zarr`` and are
    intended to be merged back in at read time by ``shuga.io.zarr_loading``.
    """

    def __init__(self, paths: ShugaPaths, *,
                 logger       : logging.Logger | None = None,
                 chunks       : dict           | None = None,
                 netcdf_engine:                   str = "scipy") -> None:
        self.paths  = paths
        self.logger = logger or logging.getLogger(__name__)
        self.chunks = chunks
        self.netcdf_engine = netcdf_engine

    # ------------------------------------------------------------------
    # helpers: TO MOVE TO shuga/grid/math.py **NEW FILE/MODULE**
    # ------------------------------------------------------------------
    def _latlon_to_degrees(self, da: xr.DataArray, target: str) -> xr.DataArray:
        values = np.asarray(da.values, dtype="float64")
        units = str(da.attrs.get("units", "")).strip().lower()
        finite = np.isfinite(values)
        if "radian" in units or (finite.any() and np.nanmax(np.abs(values[finite])) <= 2.0 * np.pi + 1e-6):
            values = np.rad2deg(values)
        if target.endswith("LON"):
            values = self._normalise_lon_values(values)
        attrs = dict(da.attrs)
        attrs["units"] = "degrees"
        attrs.setdefault("standard_name", "longitude" if target.endswith("LON") else "latitude")
        return xr.DataArray(values, dims=da.dims, coords=self._dim_coords(da), attrs=attrs, name=target)

    def _normalise_lon_values(self, lon: np.ndarray) -> np.ndarray:
        lon_wrapped = ((lon % 360.0) + 360.0) % 360.0
        spec = self.paths.cice_grid
        lon_type = getattr(spec, "lon_type", "-180-180") if spec is not None else "-180-180"

        if lon_type == "-180-180":
            out = ((lon_wrapped + 180.0) % 360.0) - 180.0
            return np.where(np.isclose(out, 180.0), -180.0, out)
        return np.where(np.isclose(lon_wrapped, 360.0), 0.0, lon_wrapped)

    def _angle_to_radians(self, da: xr.DataArray, target: str) -> xr.DataArray:
        values = np.asarray(da.values, dtype="float64")
        units = str(da.attrs.get("units", "")).strip().lower()
        finite = np.isfinite(values)
        if "degree" in units or (finite.any() and np.nanmax(np.abs(values[finite])) > 2.0 * np.pi + 1e-6):
            values = np.deg2rad(values)
        attrs = dict(da.attrs)
        attrs["units"] = "radians"
        return xr.DataArray(values, dims=da.dims, coords=self._dim_coords(da), attrs=attrs, name=target)

    def _metric_to_meters(self, da: xr.DataArray, target: str) -> xr.DataArray:
        values = np.asarray(da.values, dtype="float64")
        units = str(da.attrs.get("units", "")).strip().lower()
        if unit_is_cm := units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
            values = values / 100.0
        elif units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            values = values * 1000.0
        attrs = dict(da.attrs)
        attrs["units"] = "m"
        return xr.DataArray(values, dims=da.dims, coords=self._dim_coords(da), attrs=attrs, name=target)

    def _area_to_m2(self, da: xr.DataArray, target: str) -> xr.DataArray:
        values = np.asarray(da.values, dtype="float64")
        units = str(da.attrs.get("units", "")).strip().lower().replace(" ", "")
        if units in {"cm^2", "cm2"}:
            values = values / 10_000.0
        elif units in {"km^2", "km2"}:
            values = values * 1_000_000.0
        attrs = dict(da.attrs)
        attrs["units"] = "m^2"
        return xr.DataArray(values, dims=da.dims, coords=self._dim_coords(da), attrs=attrs, name=target)

    def _dim_coords(self, da: xr.DataArray) -> dict[str, xr.DataArray]:
        return {dim: da.coords[dim] for dim in da.dims if dim in da.coords}

    # ------------------------------------------------------------------
    # helpers: discovery / update checks
    # ------------------------------------------------------------------
    def _select_hourly_files_between_dates(self, hourly_dir: Path,
                                           dt0_str: str | None,
                                           dtN_str: str | None) -> list[Path]:
        if not hourly_dir.exists():
            raise FileNotFoundError(f"Hourly CICE NetCDF directory not found: {hourly_dir}")
        dt0 = pd.to_datetime(dt0_str) if dt0_str is not None else None
        dtN = pd.to_datetime(dtN_str) if dtN_str is not None else None
        files: list[tuple[pd.Timestamp, Path]] = []
        for path in sorted(hourly_dir.glob("iceh*.nc")):
            match = _HOURLY_FILE_RE.match(path.name)
            if not match:
                continue
            date_part = match.group(1)
            sec_part  = int(match.group(2))
            ts = pd.to_datetime(date_part) + pd.to_timedelta(sec_part, unit="s")
            if dt0 is not None and ts < dt0:
                continue
            # Date-only dtN means keep the whole day.
            if dtN is not None:
                dtN_eff = dtN
                if str(dtN_str).strip() == dtN.strftime("%Y-%m-%d"):
                    dtN_eff = dtN + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                if ts > dtN_eff:
                    continue
            files.append((ts, path))
        return [p for _, p in files]

    def _group_hourly_files_by_day(self, files: Iterable[Path]) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = {}
        for path in files:
            match = _HOURLY_FILE_RE.match(path.name)
            if not match:
                self.logger.warning("Skipping unrecognised hourly filename: %s", path.name)
                continue
            day = pd.to_datetime(match.group(1)).strftime("%Y_%m_%d")
            grouped.setdefault(day, []).append(path)
        return grouped

    def _select_daily_files_between_dates(self, daily_dir: Path,
                                          dt0_str: str | None,
                                          dtN_str: str | None) -> list[Path]:
        if not daily_dir.exists():
            raise FileNotFoundError(f"Daily CICE NetCDF directory not found: {daily_dir}")
        dt0 = pd.to_datetime(dt0_str) if dt0_str is not None else None
        dtN = pd.to_datetime(dtN_str) if dtN_str is not None else None
        files: list[tuple[pd.Timestamp, Path]] = []
        for path in sorted(daily_dir.glob("iceh.*.nc")):
            match = _DATE_RE.search(path.name)
            if not match:
                continue
            ts = pd.to_datetime(match.group(1))
            if dt0 is not None and ts < dt0:
                continue
            if dtN is not None and ts > dtN:
                continue
            files.append((ts, path))
        return [p for _, p in files]

    def _group_files_by_month(self, files: Iterable[Path]) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = {}
        for path in files:
            match = _DATE_RE.search(path.name)
            if not match:
                self.logger.warning("Skipping unrecognised filename: %s", path.name)
                continue
            month = pd.to_datetime(match.group(1)).strftime("%Y-%m")
            grouped.setdefault(month, []).append(path)
        return grouped

    def _month_needs_rewrite(self, *, cice_store: Path, month: str,
                             source_files: list[Path],
                             overwrite: bool) -> tuple[bool, str]:
        if overwrite:
            return True, "overwrite requested"
        group_path = cice_store / month
        if not group_path.exists():
            return True, "group missing"
        try:
            ds = xr.open_zarr(cice_store, group=month, consolidated=False)
        except Exception as exc:
            self.logger.warning("[%s] reopening existing group failed; will rewrite (%s)", month, exc)
            return True, f"failed to reopen existing group: {exc}"
        try:
            attrs = ds.attrs or {}
            old_sig = attrs.get("_nc2zarr_source_signature")
            new_sig = self._source_signature(source_files)
            if old_sig is not None:
                return (old_sig != new_sig), ("signature changed" if old_sig != new_sig else "signature match")
            expected_count = len(source_files)
            actual_count = int(ds.sizes.get("time", 0))
            if expected_count != actual_count:
                return True, f"time count changed ({actual_count} -> {expected_count})"
            return False, "existing group count matches source"
        finally:
            ds.close()

    def _source_signature(self, files: list[Path]) -> str:
        parts = [f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in sorted(files)]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def _attach_source_metadata(self, ds: xr.Dataset, files: list[Path]) -> xr.Dataset:
        meta = {"_nc2zarr_source_signature": self._source_signature(files),
                "_nc2zarr_file_count": len(files),
                "_nc2zarr_first_file": files[0].name,
                "_nc2zarr_last_file": files[-1].name,
                "_nc2zarr_source_filenames": json.dumps([p.name for p in files])}
        ds = ds.copy()
        ds.attrs = dict(ds.attrs)
        ds.attrs.update(meta)
        return ds

    def _strip_unsafe_zarr_encoding(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy()
        for name in list(ds.variables):
            enc = dict(ds[name].encoding)
            for key in ["source", "original_shape", "coordinates", "chunksizes", "preferred_chunks"]:
                enc.pop(key, None)
            ds[name].encoding = enc
        return ds

    def _write_grouped_month(self, cice_store: Path, month: str, ds: xr.Dataset) -> None:
        group_path = cice_store / month
        if group_path.exists():
            shutil.rmtree(group_path)
        ds.to_zarr(cice_store, group=month, mode="a", consolidated=False)

    def _delete_source_files(self, files: Iterable[Path]) -> None:
        for path in files:
            try:
                path.unlink()
            except Exception as exc:
                self.logger.warning("Could not delete %s: %s", path, exc)

    # ------------------------------------------------------------------
    # helpers: static extraction / comparison
    # ------------------------------------------------------------------
    def _build_iceh_static_dataset_from_grid_assets(self, *,
                                                    assets: dict[str, Path | None],
                                                    metadata_file: Path) -> xr.Dataset:
        """
        Construct a history-compatible static dataset from CICEGridwork plus any
        native variables available in the grid file.

        The main recoverable fields are:

        - TLON/TLAT, ULON/ULAT, NLON/NLAT, ELON/ELAT when present in the grid file;
        - ANGLE/ANGLET;
        - HTE/HTN and derived dxt/dyt/dxu/dyu/dxe/dye/dxn/dyn fallbacks;
        - tarea/uarea/earea/narea;
        - tmask plus derived umask/nmask/emask/blkmask;
        - bathymetry if a bathymetry file is available;
        - NCAT if it can be parsed from ice_in or ice_diag.d.

        VGRDa/VGRDb/VGRDi/VGRDs are not invented here because they are not
        recoverable from a horizontal grid file.
        """
        from shuga.grid.cice import CICEGridwork
        grid_file       = Path(assets["grid_file"]).expanduser()
        kmt_file        = Path(assets["kmt_file"]).expanduser() if assets.get("kmt_file") is not None else None
        bathymetry_file = (Path(assets["bathymetry_file"]).expanduser() if assets.get("bathymetry_file") is not None else None)
        gridwork        = CICEGridwork(self.paths, logger=self.logger)
        bundle          = gridwork.load_cice_grid(P_grid      = grid_file,
                                                  P_mask_org  = kmt_file,
                                                  build_faces = False)
        ds_static = xr.Dataset()
        # T-grid fields from CICEGridwork are already normalised consistently.
        for name in ("TLON", "TLAT"):
            if name in bundle.tgrid:
                ds_static = ds_static.assign_coords({name: self._prepare_iceh_static_da(bundle.tgrid[name], name=name, context="grid-assets")})
        tgrid_var_map = {"ANGLET": "ANGLET",
                         "HTE"   : "HTE",
                         "HTN"   : "HTN",
                         "tarea" : "TAREA"}
        for target, source in tgrid_var_map.items():
            if source in bundle.tgrid:
                da = self._prepare_iceh_static_da(bundle.tgrid[source], name=target, context="grid-assets")
                da.name = target
                ds_static[target] = da
        # Pull native CICE-grid fields that CICEGridwork intentionally does not
        # fully expose, especially U/N/E lon/lat from ACCESS-OM3-025_Cgrid.nc.
        ds_grid = xr.open_dataset(grid_file, decode_times=False)
        try:
            coord_map = {"TLON": ("TLON", "tlon", "t_lon", "lon_t"),
                         "TLAT": ("TLAT", "tlat", "t_lat", "lat_t"),
                         "ULON": ("ULON", "ulon", "u_lon", "lon_u"),
                         "ULAT": ("ULAT", "ulat", "u_lat", "lat_u"),
                         "NLON": ("NLON", "nlon", "n_lon", "lon_n"),
                         "NLAT": ("NLAT", "nlat", "n_lat", "lat_n"),
                         "ELON": ("ELON", "elon", "e_lon", "lon_e"),
                         "ELAT": ("ELAT", "elat", "e_lat", "lat_e")}
            for target, candidates in coord_map.items():
                if target in ds_static.coords:
                    continue
                da = self._native_grid_da(ds_grid, target, candidates, kind="lonlat")
                if da is not None:
                    ds_static = ds_static.assign_coords({target: da})
            var_map = {"ANGLE"  : ("ANGLE", "angle", "angle_u"),
                       "ANGLET" : ("ANGLET", "anglet", "angleT", "angle_t"),
                       "HTE"    : ("HTE", "hte"),
                       "HTN"    : ("HTN", "htn"),
                       "dxt"    : ("dxt", "DXT"),
                       "dyt"    : ("dyt", "DYT"),
                       "dxu"    : ("dxu", "DXU"),
                       "dyu"    : ("dyu", "DYU"),
                       "dxe"    : ("dxe", "DXE"),
                       "dye"    : ("dye", "DYE"),
                       "dxn"    : ("dxn", "DXN"),
                       "dyn"    : ("dyn", "DYN"),
                       "tarea"  : ("tarea", "TAREA", "area_t", "area"),
                       "uarea"  : ("uarea", "UAREA", "area_u"),
                       "earea"  : ("earea", "EAREA", "area_e"),
                       "narea"  : ("narea", "NAREA", "area_n"),
                       "tmask"  : ("tmask", "TMASK"),
                       "umask"  : ("umask", "UMASK"),
                       "emask"  : ("emask", "EMASK"),
                       "nmask"  : ("nmask", "NMASK"),
                       "blkmask": ("blkmask", "BLKMASK")}
            for target, candidates in var_map.items():
                if target in ds_static:
                    continue
                if target in {"ANGLE", "ANGLET"}:
                    kind = "angle"
                elif target in {"HTE", "HTN", "dxt", "dyt", "dxu", "dyu", "dxe", "dye", "dxn", "dyn"}:
                    kind = "metric"
                elif target in {"tarea", "uarea", "earea", "narea"}:
                    kind = "area"
                elif target.endswith("mask") or target == "blkmask":
                    kind = "mask"
                else:
                    kind = None
                da = self._native_grid_da(ds_grid, target, candidates, kind=kind)
                if da is not None:
                    ds_static[target] = da
        finally:
            ds_grid.close()
        # Mask from KMT/bathymetry if not present in the grid file.
        if "tmask" not in ds_static and bundle.mask is not None:
            tmask = self._prepare_iceh_static_da(bundle.mask, name="tmask", context="grid-assets-mask")
            ds_static["tmask"] = tmask.astype(np.int8)
        # Preserve bathymetry as useful provenance/QC if available.
        if bathymetry_file is not None and bathymetry_file.exists() and bundle.bathymetry is not None:
            bathy = self._prepare_iceh_static_da(bundle.bathymetry, name="bathymetry", context="grid-assets-bathymetry")
            bathy.name = "bathymetry"
            ds_static["bathymetry"] = bathy
        ds_static = self._fill_missing_static_geometry_from_tgrid(ds_static)
        ds_static = self._fill_missing_masks_from_tmask(ds_static)
        ncat      = self._parse_ncat_from_metadata(metadata_file)
        if ncat is not None and "NCAT" not in ds_static:
            ds_static["NCAT"] = xr.DataArray(np.int32(ncat), attrs={"long_name": "number of ice categories"})
        ds_static.attrs.update({"static_source": "grid_assets",
                                "run_metadata_file": str(metadata_file),
                                "grid_file": str(grid_file),
                                "kmt_file": str(kmt_file) if kmt_file is not None else "",
                                "bathymetry_file": str(bathymetry_file) if bathymetry_file is not None else ""})
        return ds_static

    def _build_iceh_static_zarr_from_grid_assets(self, *,
                                                 static_store: str | Path | None = None,
                                                 overwrite: bool = False) -> Path | None:
        """
        Build a minimal iceh_static.zarr from resolved CICE grid assets.

        This is the fallback used when CICE history files were written with
        static/grid output disabled.
        """
        static_store = Path(static_store).expanduser() if static_store is not None else self.paths.resolve_static_store_target()
        if static_store.exists() and not overwrite:
            return static_store
        if static_store.exists() and overwrite:
            shutil.rmtree(static_store)
        try:
            from shuga.grid.cice import CICEGridwork
            gridwork = CICEGridwork(paths=self.paths, logger=self.logger)
            bundle = gridwork.load_cice_grid(build_faces=True)
            ds_static = xr.Dataset()
            tgrid = bundle.tgrid
            if "TLON" in tgrid:
                ds_static = ds_static.assign_coords({"TLON": tgrid["TLON"]})
            if "TLAT" in tgrid:
                ds_static = ds_static.assign_coords({"TLAT": tgrid["TLAT"]})
            if bundle.ugrid is not None:
                if "ULON" in bundle.ugrid:
                    ds_static = ds_static.assign_coords({"ULON": bundle.ugrid["ULON"]})
                if "ULAT" in bundle.ugrid:
                    ds_static = ds_static.assign_coords({"ULAT": bundle.ugrid["ULAT"]})
            if bundle.egrid is not None:
                if "ELON" in bundle.egrid:
                    ds_static = ds_static.assign_coords({"ELON": bundle.egrid["ELON"]})
                if "ELAT" in bundle.egrid:
                    ds_static = ds_static.assign_coords({"ELAT": bundle.egrid["ELAT"]})
            if bundle.ngrid is not None:
                if "NLON" in bundle.ngrid:
                    ds_static = ds_static.assign_coords({"NLON": bundle.ngrid["NLON"]})
                if "NLAT" in bundle.ngrid:
                    ds_static = ds_static.assign_coords({"NLAT": bundle.ngrid["NLAT"]})
            if "ANGLET" in tgrid:
                ds_static["ANGLET"] = tgrid["ANGLET"]
                ds_static["ANGLE"] = tgrid["ANGLET"]
            if "HTE" in tgrid:
                ds_static["HTE"] = tgrid["HTE"]
                ds_static["dxt"] = tgrid["HTE"]
            if "HTN" in tgrid:
                ds_static["HTN"] = tgrid["HTN"]
                ds_static["dyt"] = tgrid["HTN"]
            if "TAREA" in tgrid:
                ds_static["tarea"] = tgrid["TAREA"]
            elif "HTE" in tgrid and "HTN" in tgrid:
                ds_static["tarea"] = tgrid["HTE"] * tgrid["HTN"]
            if bundle.mask is not None:
                mask = bundle.mask
                if tuple(mask.dims) != ("nj", "ni") and mask.ndim == 2:
                    mask = mask.rename({mask.dims[-2]: "nj", mask.dims[-1]: "ni"})
                ds_static["tmask"] = mask.astype("int8")
            ds_static.attrs.update({
                "source": "resolved CICE grid assets",
                "note": "Built because iceh history did not contain static grid fields.",
            })
            if not ds_static.data_vars and not ds_static.coords:
                self.logger.warning("Resolved grid assets produced an empty static dataset.")
                return None
            ds_static = self._prepare_iceh_static_for_write(ds_static)
            static_store.parent.mkdir(parents=True, exist_ok=True)
            ds_static.to_zarr(static_store, mode="w", consolidated=False)
            self.logger.info("Wrote static store from resolved grid assets: %s", static_store)
            return static_store
        except Exception as exc:
            self.logger.warning("Could not build static store from resolved grid assets: %s", exc)
            return None

    # def build_iceh_static_zarr_from_grid_assets(self, *,
    #                                             static_store: str | Path | None = None,
    #                                             overwrite: bool = False) -> Path | None:
    #     """
    #     Build ``iceh_static.zarr`` from external CICE grid/mask/bathymetry assets.

    #     This is intended for modern space-saving CICE history output where
    #     ``f_tlon``, ``f_tlat``, ``f_tmask``, ``f_tarea`` etc. are all disabled.

    #     A run metadata file is required. At least one of these must exist:

    #         <output_root>/ice_in
    #         <output_root>/ice_diag.d

    #     Without either file, this method returns ``None`` rather than silently
    #     guessing the grid provenance.
    #     """
    #     static_store = (Path(static_store).expanduser() if static_store is not None else self.paths.resolve_static_store_target())
    #     if static_store.exists() and not overwrite:
    #         self.logger.info("Static store already exists, skipping: %s", static_store)
    #         return static_store
    #     if static_store.exists() and overwrite:
    #         self.logger.info("Overwriting existing static store: %s", static_store)
    #         shutil.rmtree(static_store)
    #     metadata_file = self._resolve_run_metadata_file_for_grid_static()
    #     if metadata_file is None:
    #         self.logger.warning("Cannot build %s from grid assets because neither ice_in nor ice_diag.d "
    #                             "was found under the simulation archive/output roots.", static_store)
    #         return None
    #     assets    = self.paths.resolve_cice_grid_assets()
    #     grid_file = assets.get("grid_file")
    #     if grid_file is None or not Path(grid_file).exists():
    #         self.logger.warning("Cannot build %s from grid assets: resolved grid_file is missing or does not exist: %s", static_store, grid_file)
    #         return None
    #     ds_static = self._build_iceh_static_dataset_from_grid_assets(assets = assets, metadata_file = metadata_file)
    #     if not ds_static.data_vars and not ds_static.coords:
    #         self.logger.warning("Grid-asset fallback produced an empty static dataset; not writing %s", static_store)
    #         return None
    #     ds_static = self._prepare_iceh_static_for_write(ds_static)
    #     static_store.parent.mkdir(parents=True, exist_ok=True)
    #     ds_static.to_zarr(static_store, mode="w", consolidated=False)
    #     self.logger.info("Wrote static store from grid assets: %s", static_store)
    #     return static_store

    def _resolve_run_metadata_file_for_grid_static(self) -> Path | None:
        """
        Prefer ice_in over ice_diag.d, but accept either as proof of run/grid provenance.
        """
        try:
            ice_in = self.paths.resolve_ice_in_file()
        except Exception as exc:
            self.logger.debug("Could not resolve ice_in file: %s", exc)
            ice_in = None
        if ice_in is not None and ice_in.exists():
            return ice_in
        resolver = getattr(self.paths, "resolve_ice_diag_file", None)
        if callable(resolver):
            try:
                ice_diag = resolver()
            except Exception as exc:
                self.logger.debug("Could not resolve ice_diag.d file: %s", exc)
                ice_diag = None
            if ice_diag is not None and ice_diag.exists():
                return ice_diag
        # Defensive fallback for older ShugaPaths during transition.
        for candidate in (self.paths.output_root / "ice_diag.d",
                          self.paths.output_root / "run" / "ice_diag.d",
                          self.paths.output_root / "config" / "ice_diag.d",
                          self.paths.output_root / "history" / "ice_diag.d"):
            if candidate.exists():
                return candidate
        return None

    def _pick_native_grid_name(self, ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if name in ds.variables:
                return name
        return None

    def _native_grid_da(self,
                        ds: xr.Dataset,
                        target: str,
                        candidates: tuple[str, ...],
                        *,
                        kind: str | None = None) -> xr.DataArray | None:
        source = self._pick_native_grid_name(ds, candidates)
        if source is None:
            return None
        da = ds[source]
        da = self._prepare_iceh_static_da(da, name=target, context="native-grid")
        if da.ndim == 2:
            rename: dict[str, str] = {}
            if da.dims[-2] != "nj":
                rename[da.dims[-2]] = "nj"
            if da.dims[-1] != "ni":
                rename[da.dims[-1]] = "ni"
            if rename:
                da = da.rename(rename)
        if kind == "lonlat":
            da = self._latlon_to_degrees(da, target)
        elif kind == "angle":
            da = self._angle_to_radians(da, target)
        elif kind == "metric":
            da = self._metric_to_meters(da, target)
        elif kind == "area":
            da = self._area_to_m2(da, target)
        elif kind == "mask":
            da = da.astype(np.int8)
        da.name = target
        return da

    def _fill_missing_static_geometry_from_tgrid(self, ds_static: xr.Dataset) -> xr.Dataset:
        ds = ds_static.copy()
        # Conservative fallbacks: enough for loaders/plotters/metrics that need
        # dimensions and approximate geometry, without pretending to reconstruct
        # unavailable CICE internals exactly.
        if "tarea" in ds:
            for name in ("uarea", "earea", "narea"):
                if name not in ds:
                    ds[name] = ds["tarea"].copy()
                    ds[name].attrs.update({"long_name": f"{name} fallback copied from tarea",
                                           "units"    : ds["tarea"].attrs.get("units", "m^2")})
        metric_fallbacks = {"dxt": "HTE",
                            "dxu": "HTE",
                            "dxe": "HTE",
                            "dxn": "HTE",
                            "dyt": "HTN",
                            "dyu": "HTN",
                            "dye": "HTN",
                            "dyn": "HTN"}

        for target, source in metric_fallbacks.items():
            if target not in ds and source in ds:
                ds[target] = ds[source].copy()
                ds[target].attrs.update({"long_name": f"{target} fallback copied from {source}",
                                         "units"    : ds[source].attrs.get("units", "m")})
        return ds

    def _fill_missing_masks_from_tmask(self, ds_static: xr.Dataset) -> xr.Dataset:
        ds = ds_static.copy()
        if "tmask" not in ds:
            return ds
        ocean = ds["tmask"].astype(bool)
        # CICE-style conservative masks from neighbouring T cells.
        east      = ocean.roll(ni=-1, roll_coords=False)
        north     = ocean.shift(nj=-1, fill_value=False)
        northeast = east.shift(nj=-1, fill_value=False)
        derived   = {"emask"  : ocean & east,
                     "nmask"  : ocean & north,
                     "umask"  : ocean & east & north & northeast,
                     "blkmask": ocean}
        for name, mask in derived.items():
            if name not in ds:
                ds[name] = mask.astype(np.int8)
                ds[name].attrs.update({"long_name": f"{name} derived from tmask"})
        ds["tmask"] = ds["tmask"].astype(np.int8)
        return ds

    def _parse_ncat_from_metadata(self, metadata_file: Path) -> int | None:
        try:
            text = metadata_file.read_text(errors="ignore")
        except Exception:
            return None
        parsed = ShugaPaths._parse_ice_in_scalar_lines(text)
        for key in ("ncat", "ncat_in", "ncat_ice"):
            value = parsed.get(key)
            if value is None:
                continue
            try:
                return int(float(str(value).strip()))
            except ValueError:
                continue
        return None

    def _get_iceh_static_field_names(self, extra_coords=None, extra_vars=None) -> dict[str, list[str]]:
        default_coords = ["ELAT", "ELON",
                          "NLAT", "NLON",
                          "TLAT", "TLON",
                          "ULAT", "ULON"]
        default_vars = ["ANGLE", "ANGLET", "HTE", "HTN", "NCAT",
                        "VGRDa", "VGRDb", "VGRDi", "VGRDs",
                        "blkmask",
                        "dxe", "dxn", "dxt", "dxu",
                        "dye", "dyn", "dyt", "dyu",
                        "earea", "emask",
                        "narea", "nmask",
                        "tarea", "tmask",
                        "uarea", "umask"]
        coords = list(dict.fromkeys(default_coords + list(extra_coords or [])))
        vars_ = list(dict.fromkeys(default_vars + list(extra_vars or [])))
        return {"coords": coords, "vars": vars_, "all": coords + vars_}

    def _prepare_iceh_static_da(self, da: xr.DataArray, *, name: str | None = None, context: str = "static") -> xr.DataArray:
        da = da.copy()
        if "time" in da.dims:
            self.logger.warning("%s: %r unexpectedly has a time dimension; using the first time slice for static handling.", context, name or da.name)
            da = da.isel(time=0, drop=True)
        drop_now = [c for c in da.coords if (c == "time" or c == "time_bounds" or c not in da.dims)]
        if drop_now:
            da = da.drop_vars(drop_now, errors="ignore")
        return da

    def _build_iceh_static_dataset(self, ds: xr.Dataset, *,
                                   static_coords=None,
                                   static_vars=None,
                                   log_missing: bool = True) -> xr.Dataset:
        names = self._get_iceh_static_field_names(extra_coords=static_coords, extra_vars=static_vars)
        ds_out = xr.Dataset()
        missing_coords: list[str] = []
        missing_vars: list[str] = []
        for name in names["coords"]:
            if name in ds.coords:
                da = ds.coords[name]
            elif name in ds.variables:
                da = ds[name]
            else:
                missing_coords.append(name)
                continue
            ds_out = ds_out.assign_coords({name: self._prepare_iceh_static_da(da, name=name, context="extract-coord")})
        for name in names["vars"]:
            if name in ds.variables or name in ds.coords:
                da = ds[name]
            else:
                missing_vars.append(name)
                continue
            ds_out[name] = self._prepare_iceh_static_da(da, name=name, context="extract-var")
        ds_out = ds_out.drop_vars(["time", "time_bounds"], errors="ignore")
        if log_missing:
            if missing_coords:
                self.logger.warning("Missing expected iceh static coords: %s", missing_coords)
            if missing_vars:
                self.logger.warning("Missing expected iceh static vars: %s", missing_vars)
        return ds_out

    def _prepare_iceh_static_for_write(self, ds_static: xr.Dataset, *, cast_float32: bool = True) -> xr.Dataset:
        ds = ds_static.copy().drop_vars(["time", "time_bounds"], errors="ignore")
        if cast_float32:
            for name in list(ds.coords):
                if np.issubdtype(ds[name].dtype, np.floating) and ds[name].dtype != np.float32:
                    ds = ds.assign_coords({name: ds[name].astype(np.float32)})
            for name in list(ds.data_vars):
                if np.issubdtype(ds[name].dtype, np.floating) and ds[name].dtype != np.float32:
                    ds[name] = ds[name].astype(np.float32)
        return self._strip_unsafe_zarr_encoding(ds)

    def _warn_if_iceh_static_differs(self, ds_new: xr.Dataset, ds_ref: xr.Dataset, *, context: str = "") -> None:
        names = self._get_iceh_static_field_names()["all"]
        prefix = f"[{context}] " if context else ""
        for name in names:
            in_new = (name in ds_new.coords) or (name in ds_new.data_vars)
            in_ref = (name in ds_ref.coords) or (name in ds_ref.data_vars)
            if in_new and not in_ref:
                self.logger.warning("%sstatic name %r present in new data but absent from iceh_static.zarr", prefix, name)
                continue
            if in_ref and not in_new:
                self.logger.warning("%sstatic name %r present in iceh_static.zarr but absent from new data", prefix, name)
                continue
            if not in_new and not in_ref:
                continue
            where_new = "coord" if name in ds_new.coords else "data_var"
            where_ref = "coord" if name in ds_ref.coords else "data_var"
            if where_new != where_ref:
                self.logger.warning("%sstatic name %r changed role: new=%s, on_disk=%s", prefix, name, where_new, where_ref)
            da_new = self._prepare_iceh_static_da(ds_new[name], name=name, context="compare-new")
            da_ref = self._prepare_iceh_static_da(ds_ref[name], name=name, context="compare-ref")
            if da_new.dims != da_ref.dims:
                self.logger.warning("%sstatic name %r dims differ: new=%s, on_disk=%s", prefix, name, da_new.dims, da_ref.dims)
                continue
            if tuple(da_new.shape) != tuple(da_ref.shape):
                self.logger.warning("%sstatic name %r shape differs: new=%s, on_disk=%s", prefix, name, tuple(da_new.shape), tuple(da_ref.shape))
                continue
            if da_new.dtype != da_ref.dtype:
                self.logger.warning("%sstatic name %r dtype differs: new=%s, on_disk=%s", prefix, name, da_new.dtype, da_ref.dtype)
            a = np.asarray(da_new.values)
            b = np.asarray(da_ref.values)
            try:
                same = np.array_equal(a, b, equal_nan=True)
            except TypeError:
                same = np.array_equal(a, b)
            if not same:
                self.logger.warning("%sstatic name %r values differ from iceh_static.zarr", prefix, name)

    def _drop_iceh_static_from_monthly(self, ds: xr.Dataset, *, static_coords=None, static_vars=None) -> xr.Dataset:
        names = self._get_iceh_static_field_names(extra_coords=static_coords, extra_vars=static_vars)
        drop_now = [n for n in names["all"] if n in ds.variables or n in ds.coords]
        ds_out = ds.drop_vars(drop_now, errors="ignore") if drop_now else ds
        ds_out = ds_out.drop_vars("time_bounds", errors="ignore")
        if "time" in ds_out.coords:
            ds_out["time"].attrs.pop("bounds", None)
        return self._strip_unsafe_zarr_encoding(ds_out)

    def _maybe_update_static_store(self, ds_static_new: xr.Dataset, *, static_store: Path, overwrite: bool, context: str) -> None:
        if not ds_static_new.data_vars and not ds_static_new.coords:
            self.logger.warning("[%s] extracted empty static dataset; skipping static-store update", context)
            return
        static_store.parent.mkdir(parents=True, exist_ok=True)
        if static_store.exists():
            ds_static_ref = xr.open_zarr(static_store, consolidated=False)
            try:
                self._warn_if_iceh_static_differs(ds_static_new, ds_static_ref, context=context)
            finally:
                ds_static_ref.close()
            if not overwrite:
                return
            shutil.rmtree(static_store)
        ds_to_write = self._prepare_iceh_static_for_write(ds_static_new)
        ds_to_write.to_zarr(static_store, mode="w", consolidated=False)
        self.logger.info("[%s] wrote static store %s", context, static_store)

    # ------------------------------------------------------------------
    # helpers: zarr hygiene
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # helpers: 
    # ------------------------------------------------------------------

    def _iceh_frequency(self) -> str:
        freq = str(getattr(self.paths, "iceh_frequency", getattr(self.paths.run, "iceh_frequency", "daily"))).lower()
        if freq not in {"daily", "hourly"}:
            raise ValueError(f"Unsupported iceh_frequency={freq!r}; expected 'daily' or 'hourly'.")
        return freq

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def ensure_iceh_stores(self, *,
                           dt0_str         : str | None = None,
                           dtN_str         : str | None = None,
                           daily_root      : str | Path | None = None,
                           hourly_root     : str | Path | None = None,
                           overwrite       : bool = False,
                           overwrite_static: bool = False,
                           delete_original : bool = False,
                           netcdf_engine   : str | None = None) -> NC2ZarrResult:
        freq = self._iceh_frequency()
        if freq == "hourly":
            result = self.hourly_iceh_to_daily_zarr(
                dt0_str          = dt0_str,
                dtN_str          = dtN_str,
                hourly_root      = hourly_root,
                overwrite        = overwrite,
                delete_original  = delete_original,
                netcdf_engine    = netcdf_engine,
                overwrite_static = overwrite_static,
            )
        else:
            result = self.daily_iceh_to_monthly_zarr(
                dt0_str          = dt0_str,
                dtN_str          = dtN_str,
                daily_root       = daily_root,
                overwrite        = overwrite,
                delete_original  = delete_original,
                netcdf_engine    = netcdf_engine,
                overwrite_static = overwrite_static,
            )
        if result.static_store is None or not result.static_store.exists():
            static_store = self.ensure_iceh_static_store(
                dt0_str     = dt0_str,
                dtN_str     = dtN_str,
                daily_root  = daily_root,
                hourly_root = hourly_root,
                overwrite   = overwrite_static,
            )
            result.static_store = static_store
        return result

    # def ensure_iceh_stores(self, *,
    #                        dt0_str         : str | None = None,
    #                        dtN_str         : str | None = None,
    #                        daily_root      : str | Path | None = None,
    #                        hourly_root     : str | Path | None = None,
    #                        overwrite       : bool = False,
    #                        overwrite_static: bool = False,
    #                        delete_original : bool = False,
    #                        netcdf_engine   : str | None = None) -> NC2ZarrResult:
    #     """
    #     Ensure the dynamic ice-history Zarr store and companion iceh_static.zarr
    #     exist for either daily or hourly CICE history output.
    #     """
    #     freq = str(getattr(self.paths.run, "iceh_frequency", "daily")).strip().lower()
    #     if freq == "hourly":
    #         result = self.hourly_iceh_to_daily_zarr(dt0_str          = dt0_str,
    #                                                 dtN_str          = dtN_str,
    #                                                 hourly_root      = hourly_root,
    #                                                 overwrite        = overwrite,
    #                                                 delete_original  = delete_original,
    #                                                 netcdf_engine    = netcdf_engine,
    #                                                 overwrite_static = overwrite_static)
    #     elif freq == "daily":
    #         result = self.daily_iceh_to_monthly_zarr(dt0_str          = dt0_str,
    #                                                  dtN_str          = dtN_str,
    #                                                  daily_root       = daily_root,
    #                                                  overwrite        = overwrite,
    #                                                  delete_original  = delete_original,
    #                                                  netcdf_engine    = netcdf_engine,
    #                                                  overwrite_static = overwrite_static)
    #     else:
    #         raise ValueError(f"Unsupported iceh_frequency={freq!r}. Use 'daily' or 'hourly'.")
    #     if result.static_store is None or not result.static_store.exists():
    #         static_store = self.ensure_iceh_static_store(dt0_str                 = dt0_str,
    #                                                      dtN_str                 = dtN_str,
    #                                                      daily_root              = daily_root,
    #                                                      hourly_root             = hourly_root,
    #                                                      overwrite               = overwrite_static,
    #                                                      allow_gridwork_fallback = True)
    #         result.static_store = static_store
    #     return result

    def hourly_iceh_to_daily_zarr(self, *,
                                  dt0_str          : str | None = None,
                                  dtN_str          : str | None = None,
                                  hourly_root      : str | Path | None = None,
                                  overwrite        : bool = False,
                                  delete_original  : bool = False,
                                  netcdf_engine    : str | None = None,
                                  overwrite_static : bool = False) -> NC2ZarrResult:
        """
        Convert instantaneous hourly CICE files like

            iceh_inst.YYYY-MM-DD-SSSSS.nc

        into grouped daily Zarr:

            iceh_hourly.zarr/YYYY_MM_DD

        Unlike daily averaged CICE output, instantaneous hourly output is not
        shifted back by one day.
        """
        engine       = netcdf_engine or self.netcdf_engine
        hourly_dir   = self.paths.resolve_hourly_iceh_root(hourly_root)
        cice_store   = self.paths.resolve_cice_store_target()
        static_store = self.paths.resolve_static_store_target()
        result       = NC2ZarrResult(cice_store=cice_store, static_store=static_store)
        cice_store.mkdir(parents=True, exist_ok=True)
        source_files = self._select_hourly_files_between_dates(hourly_dir, dt0_str, dtN_str)
        result.daily_files_seen = len(source_files)
        if not source_files:
            self.logger.info("No CICE hourly NetCDF files found in %s for [%s, %s]", hourly_dir, dt0_str, dtN_str)
            return result
        day_groups = self._group_hourly_files_by_day(source_files)
        result.months_scanned = len(day_groups)
        result.daily_files_used = sum(len(v) for v in day_groups.values())
        for day, files in sorted(day_groups.items()):
            needs_rewrite, reason = self._month_needs_rewrite(cice_store   = cice_store,
                                                              month        = day,
                                                              source_files = files,
                                                              overwrite    = overwrite)
            if not needs_rewrite:
                result.months_skipped += 1
                self.logger.info("[%s] skipping grouped hourly day (%s)", day, reason)
                if delete_original:
                    self._delete_source_files(files)
                continue
            if reason == "overwrite requested" or (cice_store / day).exists():
                result.months_rewritten += int((cice_store / day).exists())
            else:
                result.months_written += 1
            self.logger.info("[%s] opening %d hourly NetCDF files with engine=%s", day, len(files), engine)
            ds_all = xr.open_mfdataset(files,
                                       engine     = engine,
                                       parallel   = True,
                                       combine    = "nested",
                                       concat_dim ="time",
                                       coords     = "minimal",
                                       data_vars  = "minimal",
                                       compat     = "override",
                                       join       = "override",
                                       cache      = False,
                                       chunks     = self.chunks)
            if self.chunks:
                ds_all = ds_all.chunk(self.chunks)
            if "time" not in ds_all.coords:
                raise KeyError(f"[{day}] opened hourly dataset does not contain a time coordinate")
            # Important: hourly files are instantaneous, so no daily-output
            # timestamp correction is applied here.
            ds_static_new = self._build_iceh_static_dataset(ds_all, log_missing=True)
            self._maybe_update_static_store(ds_static_new,
                                            static_store = static_store,
                                            overwrite    = overwrite_static,
                                            context      = f"hourly day {day}")
            ds_dynamic = self._drop_iceh_static_from_monthly(ds_all)
            ds_dynamic = self._attach_source_metadata(ds_dynamic, files)
            self._write_grouped_month(cice_store, day, ds_dynamic)
            self.logger.info("[%s] wrote grouped hourly Zarr to %s", day, cice_store / day)
            ds_all.close()
            if delete_original:
                self._delete_source_files(files)
        return result

    def daily_iceh_to_monthly_zarr(self, *,
                                   dt0_str          : str | None = None,
                                   dtN_str          : str | None = None,
                                   daily_root       : str | Path | None = None,
                                   overwrite        : bool = False,
                                   delete_original  : bool = False,
                                   netcdf_engine    : str | None = None,
                                   overwrite_static : bool = False) -> NC2ZarrResult:
        """
        Convert daily CICE ``iceh.YYYY-MM-DD.nc`` files into the grouped monthly
        ``iceh_daily.zarr`` store.

        Behaviour mirrors the AFIM implementation:
        - NetCDF files are grouped by filename month.
        - ``time`` is shifted back by one day before write.
        - Static fields are extracted once into ``iceh_static.zarr``.
        - Existing monthly groups are skipped unless a rewrite is required or
          ``overwrite=True``.
        """
        engine       = netcdf_engine or self.netcdf_engine
        daily_dir    = self.paths.resolve_daily_iceh_root(daily_root)
        cice_store   = self.paths.resolve_cice_store_target()
        static_store = self.paths.resolve_static_store_target()
        result       = NC2ZarrResult(cice_store=cice_store, static_store=static_store)
        cice_store.mkdir(parents=True, exist_ok=True)
        source_files = self._select_daily_files_between_dates(daily_dir, dt0_str, dtN_str)
        result.daily_files_seen = len(source_files)
        if not source_files:
            self.logger.info("No CICE daily NetCDF files found in %s for [%s, %s]", daily_dir, dt0_str, dtN_str)
            return result
        month_groups = self._group_files_by_month(source_files)
        result.months_scanned = len(month_groups)
        result.daily_files_used = sum(len(v) for v in month_groups.values())
        for month, files in sorted(month_groups.items()):
            needs_rewrite, reason = self._month_needs_rewrite(cice_store   = cice_store,
                                                              month        = month,
                                                              source_files = files,
                                                              overwrite    = overwrite)
            if not needs_rewrite:
                result.months_skipped += 1
                self.logger.info("[%s] skipping grouped month (%s)", month, reason)
                if delete_original:
                    self._delete_source_files(files)
                continue
            if reason == "overwrite requested" or (cice_store / month).exists():
                result.months_rewritten += int((cice_store / month).exists())
            else:
                result.months_written += 1
            self.logger.info("[%s] opening %d NetCDF files with engine=%s", month, len(files), engine)
            ds_all = xr.open_mfdataset(files,
                                       engine     = engine,
                                       parallel   = True,
                                       combine    = "nested",
                                       concat_dim = "time",
                                       coords     = "minimal",
                                       data_vars  = "minimal",
                                       compat     = "override",
                                       join       = "override",
                                       cache      = False,
                                       chunks     = self.chunks)
            if self.chunks:
                ds_all = ds_all.chunk(self.chunks)
            if "time" not in ds_all.coords:
                raise KeyError(f"[{month}] opened dataset does not contain a time coordinate")
            self.logger.info("[%s] shifting time coordinate back by one day", month)
            ds_all = ds_all.assign_coords(time=ds_all["time"] - np.timedelta64(1, "D"))
            ds_static_new = self._build_iceh_static_dataset(ds_all, log_missing=True)
            self._maybe_update_static_store(ds_static_new,
                                            static_store = static_store,
                                            overwrite    = overwrite_static,
                                            context      = f"month {month}")
            ds_dynamic = self._drop_iceh_static_from_monthly(ds_all)
            ds_dynamic = self._attach_source_metadata(ds_dynamic, files)
            self._write_grouped_month(cice_store, month, ds_dynamic)
            self.logger.info("[%s] wrote grouped dynamic Zarr to %s", month, cice_store / month)
            ds_all.close()
            if delete_original:
                self._delete_source_files(files)
        return result

    def ensure_iceh_static_store(self, *,
                                 dt0_str          : str | None = None,
                                 dtN_str          : str | None = None,
                                 daily_root       : str | Path | None = None,
                                 hourly_root      : str | Path | None = None,
                                 overwrite        : bool = False,
                                 verify_all_groups: bool = True) -> Path | None:
        """
        Ensure iceh_static.zarr exists.

        Preference order:
        1. build from the active grouped iceh Zarr store if static fields are present;
        2. build from the first matching NetCDF source file;
        3. build a minimal static store from resolved CICE grid assets.
        """
        static_store = self.paths.resolve_static_store_target()
        freq = self._iceh_frequency()
        if static_store.exists() and not overwrite:
            self.logger.info("Static store already exists, skipping: %s", static_store)
            return static_store
        if static_store.exists() and overwrite:
            self.logger.info("Overwriting static store: %s", static_store)
            shutil.rmtree(static_store)
        cice_store = self.paths.resolve_cice_store_target()
        if cice_store.exists():
            built = self.build_iceh_static_zarr_from_grouped_iceh(
                cice_store           = cice_store,
                static_store         = static_store,
                frequency            = freq,
                overwrite            = False,
                verify_all_groups    = verify_all_groups,
                allow_empty_fallback = True,
            )
            if built is not None and built.exists():
                return built
        if freq == "hourly":
            try:
                source_dir = self.paths.resolve_hourly_iceh_root(hourly_root)
                source_files = self._select_hourly_files_between_dates(source_dir, dt0_str, dtN_str)
            except FileNotFoundError as exc:
                self.logger.warning("Could not inspect hourly NetCDF files for static fields: %s", exc)
                source_files = []
        else:
            try:
                source_dir = self.paths.resolve_daily_iceh_root(daily_root)
                source_files = self._select_daily_files_between_dates(source_dir, dt0_str, dtN_str)
            except FileNotFoundError as exc:
                self.logger.warning("Could not inspect daily NetCDF files for static fields: %s", exc)
                source_files = []
        if source_files:
            self.logger.info("Building %s from first available %s NetCDF file %s",
                             static_store, freq, source_files[0])
            ds = xr.open_dataset(source_files[0], engine=self.netcdf_engine)
            try:
                ds_static = self._build_iceh_static_dataset(ds, log_missing=True)
                if ds_static.data_vars or ds_static.coords:
                    ds_static = self._prepare_iceh_static_for_write(ds_static)
                    static_store.parent.mkdir(parents=True, exist_ok=True)
                    ds_static.to_zarr(static_store, mode="w", consolidated=False)
                    return static_store
                self.logger.warning("No static variables could be extracted from %s; trying grid assets.",
                                    source_files[0])
            finally:
                ds.close()
        else:
            self.logger.warning("No %s NetCDF files available to build %s; trying grid assets.",
                                freq, static_store)
        return self._build_iceh_static_zarr_from_grid_assets(static_store=static_store)

    # def ensure_iceh_static_store(self, *,
    #                              dt0_str          : str | None = None,
    #                              dtN_str          : str | None = None,
    #                              daily_root       : str | Path | None = None,
    #                              overwrite        : bool = False,
    #                              verify_all_groups: bool = True) -> Path | None:
    #     """
    #     Ensure ``iceh_static.zarr`` exists.

    #     Preference order:
    #     1. build from grouped ``iceh_daily.zarr`` if those groups still contain
    #        static content;
    #     2. build from the first available daily NetCDF file if it still contains
    #        static content;
    #     3. build from grid assets resolved from ``ice_in`` or ``ice_diag.d``.

    #     If neither ``ice_in`` nor ``ice_diag.d`` can be found, fallback (3) is
    #     deliberately disabled because the grid/mask provenance is then ambiguous.
    #     """
    #     static_store = self.paths.resolve_static_store_target()
    #     if static_store.exists() and not overwrite:
    #         self.logger.info("Static store already exists, skipping: %s", static_store)
    #         return static_store
    #     if static_store.exists() and overwrite:
    #         self.logger.info("Overwriting static store: %s", static_store)
    #         shutil.rmtree(static_store)
    #     cice_store = self.paths.resolve_cice_store_target()
    #     if cice_store.exists():
    #         built = self.build_iceh_static_zarr_from_grouped_daily(cice_store           = cice_store,
    #                                                                static_store         = static_store,
    #                                                                overwrite            = False,
    #                                                                verify_all_groups    = verify_all_groups,
    #                                                                allow_empty_fallback = True)
    #         if built is not None and built.exists():
    #             return built
    #     source_files: list[Path] = []
    #     try:
    #         daily_dir = self.paths.resolve_daily_iceh_root(daily_root)
    #         source_files = self._select_daily_files_between_dates(daily_dir, dt0_str, dtN_str)
    #     except FileNotFoundError as exc:
    #         self.logger.info("No daily NetCDF root available for static extraction (%s); trying grid-asset fallback.", exc)
    #     if source_files:
    #         self.logger.info("Building %s from first available daily file %s", static_store, source_files[0])
    #         ds = xr.open_dataset(source_files[0], engine=self.netcdf_engine)
    #         try:
    #             ds_static = self._build_iceh_static_dataset(ds, log_missing=True)
    #             if ds_static.data_vars or ds_static.coords:
    #                 ds_static = self._prepare_iceh_static_for_write(ds_static)
    #                 static_store.parent.mkdir(parents=True, exist_ok=True)
    #                 ds_static.to_zarr(static_store, mode="w", consolidated=False)
    #                 self.logger.info("Wrote static store from daily NetCDF: %s", static_store)
    #                 return static_store
    #             self.logger.warning("Daily file %s contains no usable static fields; trying grid-asset fallback.", source_files[0])
    #         finally:
    #             ds.close()
    #     else:
    #         self.logger.info("No daily NetCDF files available to build %s; trying grid-asset fallback.", static_store)
    #     return self.build_iceh_static_zarr_from_grid_assets(static_store = static_store, overwrite = False)

    def build_iceh_static_zarr_from_grouped_iceh(self, *,
                                                 cice_store: str | Path | None = None,
                                                 static_store: str | Path | None = None,
                                                 frequency: str | None = None,
                                                 overwrite: bool = False,
                                                 verify_all_groups: bool = True,
                                                 allow_empty_fallback: bool = False) -> Path | None:
        """
        Build iceh_static.zarr from an existing grouped iceh store.

        Daily stores use YYYY-MM groups.
        Hourly stores use YYYY_MM_DD groups.
        """
        freq = str(frequency or self._iceh_frequency()).lower()
        if freq not in {"daily", "hourly"}:
            raise ValueError(f"Unsupported frequency={freq!r}")

        group_re = _DAY_GROUP_RE if freq == "hourly" else _MONTH_RE

        cice_store = Path(cice_store).expanduser() if cice_store is not None else self.paths.resolve_cice_store_target()
        static_store = Path(static_store).expanduser() if static_store is not None else self.paths.resolve_static_store_target()

        if not cice_store.exists():
            raise FileNotFoundError(f"Grouped CICE store not found: {cice_store}")

        groups = sorted(p.name for p in cice_store.iterdir() if p.is_dir() and group_re.fullmatch(p.name))
        if not groups:
            raise FileNotFoundError(f"No grouped {freq} iceh groups found under {cice_store}")

        if static_store.exists() and not overwrite:
            self.logger.info("Static store already exists, skipping: %s", static_store)
            return static_store

        if static_store.exists() and overwrite:
            self.logger.info("Overwriting existing static store: %s", static_store)
            shutil.rmtree(static_store)

        ref_group = groups[0]
        self.logger.info("Building static store from grouped %s group %s", freq, ref_group)

        ds_ref = xr.open_zarr(cice_store, group=ref_group, consolidated=False)
        try:
            ds_static = self._build_iceh_static_dataset(ds_ref, log_missing=True)

            if not ds_static.data_vars and not ds_static.coords:
                if allow_empty_fallback:
                    self.logger.info(
                        "Grouped %s group %s contains no static content; falling back to NetCDF/grid assets",
                        freq,
                        ref_group,
                    )
                    return None
                raise ValueError(f"No static content found in grouped {freq} group {ref_group}")

            if verify_all_groups and len(groups) > 1:
                self.logger.info("Checking static consistency across %d grouped %s groups", len(groups), freq)
                for g in groups[1:]:
                    ds_g = xr.open_zarr(cice_store, group=g, consolidated=False)
                    try:
                        ds_g_static = self._build_iceh_static_dataset(ds_g, log_missing=False)
                        if ds_g_static.data_vars or ds_g_static.coords:
                            self._warn_if_iceh_static_differs(
                                ds_g_static,
                                ds_static,
                                context=f"group {g} vs {ref_group}",
                            )
                    finally:
                        ds_g.close()

            ds_static = self._prepare_iceh_static_for_write(ds_static)
            static_store.parent.mkdir(parents=True, exist_ok=True)
            ds_static.to_zarr(static_store, mode="w", consolidated=False)
            self.logger.info("Wrote static store: %s", static_store)
            return static_store

        finally:
            ds_ref.close()


    def build_iceh_static_zarr_from_grouped_daily(self, *,
                                                  cice_store: str | Path | None = None,
                                                  static_store: str | Path | None = None,
                                                  overwrite: bool = False,
                                                  verify_all_groups: bool = True,
                                                  allow_empty_fallback: bool = False) -> Path | None:
        return self.build_iceh_static_zarr_from_grouped_iceh(
            cice_store           = cice_store,
            static_store         = static_store,
            frequency            = "daily",
            overwrite            = overwrite,
            verify_all_groups    = verify_all_groups,
            allow_empty_fallback = allow_empty_fallback,
        )

    # def build_iceh_static_zarr_from_grouped_daily(self, *,
    #                                               cice_store: str | Path | None = None,
    #                                               static_store: str | Path | None = None,
    #                                               overwrite: bool = False,
    #                                               verify_all_groups: bool = True,
    #                                               allow_empty_fallback: bool = False) -> Path | None:
    #     """
    #     Build ``iceh_static.zarr`` from an existing grouped ``iceh_daily.zarr``
    #     root if those monthly groups still contain static fields.
    #     """
    #     cice_store = Path(cice_store).expanduser() if cice_store is not None else self.paths.resolve_cice_store_target()
    #     static_store = Path(static_store).expanduser() if static_store is not None else self.paths.resolve_static_store_target()
    #     if not cice_store.exists():
    #         raise FileNotFoundError(f"Grouped CICE store not found: {cice_store}")
    #     groups = sorted(p.name for p in cice_store.iterdir() if p.is_dir() and _MONTH_RE.fullmatch(p.name))
    #     if not groups:
    #         raise FileNotFoundError(f"No YYYY-MM groups found under {cice_store}")
    #     if static_store.exists() and not overwrite:
    #         self.logger.info("Static store already exists, skipping: %s", static_store)
    #         return static_store
    #     if static_store.exists() and overwrite:
    #         self.logger.info("Overwriting existing static store: %s", static_store)
    #         shutil.rmtree(static_store)
    #     ref_group = groups[0]
    #     self.logger.info("Building static store from grouped month %s", ref_group)
    #     ds_ref = xr.open_zarr(cice_store, group=ref_group, consolidated=False)
    #     try:
    #         ds_static = self._build_iceh_static_dataset(ds_ref, log_missing=True)
    #         if not ds_static.data_vars and not ds_static.coords:
    #             if allow_empty_fallback:
    #                 self.logger.info("Grouped month %s contains no static content; falling back to daily NetCDF source", ref_group)
    #                 return None
    #             raise ValueError(f"No static content found in grouped month {ref_group}")
    #         if verify_all_groups and len(groups) > 1:
    #             self.logger.info("Checking static consistency across %d grouped months", len(groups))
    #             for g in groups[1:]:
    #                 ds_g = xr.open_zarr(cice_store, group=g, consolidated=False)
    #                 try:
    #                     ds_g_static = self._build_iceh_static_dataset(ds_g, log_missing=False)
    #                     if ds_g_static.data_vars or ds_g_static.coords:
    #                         self._warn_if_iceh_static_differs(ds_g_static, ds_static, context=f"group {g} vs {ref_group}")
    #                 finally:
    #                     ds_g.close()
    #         ds_static = self._prepare_iceh_static_for_write(ds_static)
    #         static_store.parent.mkdir(parents=True, exist_ok=True)
    #         ds_static.to_zarr(static_store, mode="w", consolidated=False)
    #         self.logger.info("Wrote static store: %s", static_store)
    #         return static_store
    #     finally:
    #         ds_ref.close()

