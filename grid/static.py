from __future__ import annotations
import logging, re, shutil
from pathlib import Path
import numpy as np
import xarray as xr
from shuga.core.paths    import ShugaPaths
from shuga.grid.cice     import CICEGridwork
from shuga.grid.geometry import (angle_to_radians,
                                 area_to_m2,
                                 coerce_2d_dims_to_nj_ni,
                                 dim_coords,
                                 latlon_to_degrees,
                                 metric_to_meters,
                                 pick_variable)

class CICEStaticBuilder:
    """
    Build history-compatible iceh_static.zarr datasets from CICE grid assets.

    This owns the logic formerly embedded in NC2Zarr:
    - recover TLON/TLAT/ULON/ULAT/NLON/NLAT/ELON/ELAT where available;
    - recover ANGLE/ANGLET, HTE/HTN, areas, masks, bathymetry;
    - derive conservative fallback geometry and masks when native fields are absent;
    - parse NCAT from ice_in or ice_diag.d when available.
    """

    def __init__(self, pth_cfg: ShugaPaths, *, logger: logging.Logger | None = None) -> None:
        self.pth_cfg = pth_cfg
        self.logger = logger or logging.getLogger(__name__)

    @property
    def lon_type(self) -> str:
        spec = self.pth_cfg.G_cice_cfg
        return getattr(spec, "lon_type", "-180-180") if spec is not None else "-180-180"

    @staticmethod
    def _tgrid_shape(ds: xr.Dataset) -> tuple[int, int] | None:
        """
        Return the canonical T-grid shape from an existing static dataset.

        The static builder treats any variable/coordinate on dims ('nj', 'ni')
        as the T-grid template. Fields on other dimensions, e.g. corner/edge
        grids, are allowed to coexist.
        """
        for name in ("TLAT", "TLON", "tarea", "HTE", "HTN"):
            if name in ds:
                da = ds[name]
            elif name in ds.coords:
                da = ds.coords[name]
            else:
                continue
            if da.ndim == 2 and tuple(da.dims) == ("nj", "ni"):
                return int(da.sizes["nj"]), int(da.sizes["ni"])
        if "nj" in ds.sizes and "ni" in ds.sizes:
            return int(ds.sizes["nj"]), int(ds.sizes["ni"])
        return None

    def _compatible_with_tgrid(self, ds: xr.Dataset, da: xr.DataArray, *, name: str, source: str) -> bool:
        """
        Validate fields that claim to live on the T grid.

        Only dims exactly ('nj', 'ni') are checked here. Non-T-grid fields
        such as ('nj_b', 'ni_b') are allowed.
        """
        if da.ndim != 2 or tuple(da.dims) != ("nj", "ni"):
            return True
        expected = self._tgrid_shape(ds)
        if expected is None:
            return True
        actual = (int(da.sizes["nj"]), int(da.sizes["ni"]))
        if actual == expected:
            return True
        self.logger.warning("Skipping static field %s from %s because shape %s does not match existing T-grid shape %s.", name, source, actual, expected)
        return False

    def _assign_data_var_if_compatible(self, ds: xr.Dataset,name: str, da: xr.DataArray, *, source: str) -> xr.Dataset:
        if self._compatible_with_tgrid(ds, da, name=name, source=source):
            ds[name] = da
        return ds

    def _assign_coord_if_compatible(self, ds: xr.Dataset, name: str, da: xr.DataArray, *, source: str) -> xr.Dataset:
        if self._compatible_with_tgrid(ds, da, name=name, source=source):
            ds = ds.assign_coords({name: da})
        return ds

    def resolve_run_metadata_file(self) -> Path | None:
        """
        Prefer ice_in over ice_diag.d, but accept either as run/grid provenance.
        """
        try:
            ice_in = self.pth_cfg.resolve_ice_in_file()
        except Exception as exc:
            self.logger.debug("Could not resolve ice_in file: %s", exc)
            ice_in = None
        if ice_in is not None and ice_in.exists():
            return ice_in
        try:
            ice_diag = self.pth_cfg.resolve_ice_diag_file()
        except Exception as exc:
            self.logger.debug("Could not resolve ice_diag.d file: %s", exc)
            ice_diag = None
        if ice_diag is not None and ice_diag.exists():
            return ice_diag
        for candidate in (self.pth_cfg.output_root / "ice_diag.d",
                          self.pth_cfg.output_root / "run" / "ice_diag.d",
                          self.pth_cfg.output_root / "config" / "ice_diag.d",
                          self.pth_cfg.output_root / "history" / "ice_diag.d"):
            if candidate.exists():
                return candidate
        return None

    def build_dataset_from_resolved_assets(self, *, require_metadata: bool = True) -> xr.Dataset | None:
        metadata_file = self.resolve_run_metadata_file()
        if metadata_file is None and require_metadata:
            self.logger.warning("Cannot build static CICE dataset because neither ice_in nor ice_diag.d was found.")
            return None
        assets    = self.pth_cfg.resolve_cice_grid_assets()
        grid_file = assets.get("grid_file")
        if grid_file is None or not Path(grid_file).expanduser().exists():
            self.logger.warning("Cannot build static CICE dataset: resolved grid_file is missing: %s", grid_file)
            return None
        return self.build_dataset_from_grid_assets(assets = assets, metadata_file = metadata_file)

    def write_zarr_from_resolved_assets(self, *,
                                        static_store: str | Path | None = None,
                                        overwrite: bool = False,
                                        require_metadata: bool = True) -> Path | None:
        target = (Path(static_store).expanduser() if static_store is not None else self.pth_cfg.resolve_static_store_target())
        if target.exists() and not overwrite:
            self.logger.info("Static store already exists, skipping: %s", target)
            return target
        if target.exists() and overwrite:
            self.logger.info("Overwriting existing static store: %s", target)
            shutil.rmtree(target)
        ds_static = self.build_dataset_from_resolved_assets(require_metadata = require_metadata)
        if ds_static is None:
            return None
        if not ds_static.data_vars and not ds_static.coords:
            self.logger.warning("Static builder produced an empty dataset; not writing %s", target)
            return None
        ds_static = self.prepare_for_write(ds_static)
        target.parent.mkdir(parents = True, exist_ok = True)
        ds_static.to_zarr(target, mode = "w", consolidated = False)
        self.logger.info("Wrote static store from CICE grid assets: %s", target)
        return target

    def build_dataset_from_grid_assets(self, *, assets: dict[str, Path | None], metadata_file: Path | None = None) -> xr.Dataset:
        grid_file_raw = assets.get("grid_file")
        if grid_file_raw is None:
            raise FileNotFoundError("Cannot build static dataset: assets['grid_file'] is None.")
        grid_file       = Path(grid_file_raw).expanduser()
        kmt_file        = Path(assets["kmt_file"]).expanduser() if assets.get("kmt_file") is not None else None
        bathymetry_file = (Path(assets["bathymetry_file"]).expanduser() if assets.get("bathymetry_file") is not None else None)
        gridwork        = CICEGridwork(self.pth_cfg, logger = self.logger)
        bundle          = gridwork.load_cice_grid(P_grid = grid_file, P_mask_org = kmt_file, build_faces = True)
        ds_static       = xr.Dataset()
        # Core T-grid fields from CICEGridwork.
        for name in ("TLON", "TLAT"):
            if name in bundle.tgrid:
                da        = self.prepare_da(bundle.tgrid[name], name = name, context = "grid-assets")
                ds_static = self._assign_coord_if_compatible(ds_static, name, da, source = "CICEGridwork.tgrid")
        tgrid_var_map = {"ANGLET": "ANGLET",
                         "HTE"   : "HTE",
                         "HTN"   : "HTN",
                         "tarea" : "TAREA"}
        for target, source in tgrid_var_map.items():
            if source in bundle.tgrid:
                da       = self.prepare_da(bundle.tgrid[source], name = target, context = "grid-assets")
                ds_static = self._assign_data_var_if_compatible(ds_static, target, da, source = "CICEGridwork.tgrid")
        # Constructed face/edge coordinates. Native grid-file values override below
        # only if these were not already present.
        if bundle.ugrid is not None:
            for target in ("ULON", "ULAT"):
                if target in bundle.ugrid and target not in ds_static:
                    ds_static = ds_static.assign_coords({target: self.prepare_da(bundle.ugrid[target], name = target, context = "grid-faces")})
        if bundle.egrid is not None:
            for target in ("ELON", "ELAT"):
                if target in bundle.egrid and target not in ds_static:
                    ds_static = ds_static.assign_coords({target: self.prepare_da(bundle.egrid[target], name = target, context = "grid-faces")})
        if bundle.ngrid is not None:
            for target in ("NLON", "NLAT"):
                if target in bundle.ngrid and target not in ds_static:
                    ds_static = ds_static.assign_coords({target: self.prepare_da(bundle.ngrid[target], name = target, context = "grid-faces")})
        # Native grid-file fields, if available.
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
                da = self.native_grid_da(ds_grid, target, candidates, kind="lonlat")
                if da is not None:
                    ds_static = self._assign_coord_if_compatible(ds_static, target, da, source=str(grid_file))
            var_map = {"ANGLE": ("ANGLE", "angle", "angle_u"),
                       "ANGLET": ("ANGLET", "anglet", "angleT", "angle_t"),
                       "HTE": ("HTE", "hte"),
                       "HTN": ("HTN", "htn"),
                       "dxt": ("dxt", "DXT"),
                       "dyt": ("dyt", "DYT"),
                       "dxu": ("dxu", "DXU"),
                       "dyu": ("dyu", "DYU"),
                       "dxe": ("dxe", "DXE"),
                       "dye": ("dye", "DYE"),
                       "dxn": ("dxn", "DXN"),
                       "dyn": ("dyn", "DYN"),
                       "tarea": ("tarea", "TAREA", "area_t", "area"),
                       "uarea": ("uarea", "UAREA", "area_u"),
                       "earea": ("earea", "EAREA", "area_e"),
                       "narea": ("narea", "NAREA", "area_n"),
                       "tmask": ("tmask", "TMASK"),
                       "umask": ("umask", "UMASK"),
                       "emask": ("emask", "EMASK"),
                       "nmask": ("nmask", "NMASK"),
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
                da = self.native_grid_da(ds_grid, target, candidates, kind=kind)
                if da is not None:
                    ds_static = self._assign_data_var_if_compatible(ds_static, target, da, source = str(grid_file))
        finally:
            ds_grid.close()
        if "tmask" not in ds_static and bundle.mask is not None:
            tmask     = self.prepare_da(bundle.mask, name = "tmask", context = "grid-assets-mask")
            tmask     = coerce_2d_dims_to_nj_ni(tmask).astype(np.int8)
            ds_static = self._assign_data_var_if_compatible(ds_static, "tmask", tmask, source = str(kmt_file) if kmt_file is not None else "CICEGridwork.mask")
        if bathymetry_file is not None and bathymetry_file.exists() and bundle.bathymetry is not None:
            bathy     = self.prepare_da(bundle.bathymetry, name = "bathymetry", context = "grid-assets-bathymetry")
            bathy     = coerce_2d_dims_to_nj_ni(bathy)
            ds_static = self._assign_data_var_if_compaible(ds_static, "bathymetry", bathy, source = str(bathymetry_file))
        ds_static = self.fill_missing_static_geometry_from_tgrid(ds_static)
        ds_static = self.fill_missing_masks_from_tmask(ds_static)
        ncat      = self.parse_ncat_from_metadata(metadata_file)
        if ncat is not None and "NCAT" not in ds_static:
            ds_static["NCAT"] = xr.DataArray(np.int32(ncat), attrs = {"long_name": "number of ice categories"})
        ds_static.attrs.update({"static_source"    : "grid_assets",
                                "run_metadata_file": str(metadata_file) if metadata_file is not None else "",
                                "grid_file"        : str(grid_file),
                                "kmt_file"         : str(kmt_file) if kmt_file is not None else "",
                                "bathymetry_file"  : str(bathymetry_file) if bathymetry_file is not None else ""})
        return ds_static

    def prepare_da(self, da: xr.DataArray, *, name: str, context: str) -> xr.DataArray:
        out = da
        if "time" in out.dims and out.sizes.get("time", 0) == 1:
            out = out.isel(time=0, drop=True)
        out = xr.DataArray(out.data, dims = out.dims, coords = dim_coords(out), attrs = dict(out.attrs), name = name)
        out.attrs.setdefault("static_builder_context", context)
        return out

    def native_grid_da(self, ds: xr.Dataset, target: str, candidates: tuple[str, ...], *, kind: str | None = None) -> xr.DataArray | None:
        source = pick_variable(ds, candidates)
        if source is None:
            return None
        da = self.prepare_da(ds[source], name = target, context = "native-grid")
        if da.ndim == 2:
            da = coerce_2d_dims_to_nj_ni(da)
        if kind == "lonlat":
            da = latlon_to_degrees(da, target=target, lon_type=self.lon_type)
        elif kind == "angle":
            da = angle_to_radians(da, target=target)
        elif kind == "metric":
            da = metric_to_meters(da, target=target)
        elif kind == "area":
            da = area_to_m2(da, target=target)
        elif kind == "mask":
            da = da.astype(np.int8)
        da.name = target
        return da

    def fill_missing_static_geometry_from_tgrid(self, ds_static: xr.Dataset) -> xr.Dataset:
        ds = ds_static.copy()
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
        if "ANGLE" not in ds and "ANGLET" in ds:
            ds["ANGLE"] = ds["ANGLET"].copy()
            ds["ANGLE"].attrs.update({"long_name": "ANGLE fallback copied from ANGLET"})
        return ds

    def fill_missing_masks_from_tmask(self, ds_static: xr.Dataset) -> xr.Dataset:
        ds = ds_static.copy()
        if "tmask" not in ds:
            return ds
        ocean     = ds["tmask"].astype(bool)
        east      = ocean.roll(ni=-1, roll_coords=False)
        north     = ocean.shift(nj=-1, fill_value=False)
        northeast = east.shift(nj=-1, fill_value=False)
        derived   = {"emask": ocean & east,
                     "nmask": ocean & north,
                     "umask": ocean & east & north & northeast,
                     "blkmask": ocean}
        for name, mask in derived.items():
            if name in ds:
                continue
            ds[name] = mask.astype(np.int8)
            ds[name].attrs.update({"long_name": f"{name} derived conservatively from tmask",
                                   "units"    : "1"})
        return ds

    @staticmethod
    def parse_ncat_from_metadata(metadata_file: Path | None) -> int | None:
        if metadata_file is None or not metadata_file.exists():
            return None
        text     = metadata_file.read_text(errors="ignore")
        patterns = (r"^\s*ncat\s*=\s*(\d+)",
                    r"^\s*NCAT\s*=\s*(\d+)",
                    r"number\s+of\s+ice\s+categories\D+(\d+)")
        for pattern in patterns:
            match = re.search(pattern, text, flags = re.IGNORECASE | re.MULTILINE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def prepare_for_write(ds: xr.Dataset) -> xr.Dataset:
        out = ds.copy()
        for name in list(out.variables):
            enc = dict(out[name].encoding)
            for key in ("source", "original_shape", "coordinates", "chunksizes", "preferred_chunks"):
                enc.pop(key, None)
            out[name].encoding = enc
        out.encoding = {}
        return out
