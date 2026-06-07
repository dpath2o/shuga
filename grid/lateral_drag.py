from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.paths import ShugaPaths
from shuga.core.types import CICEGridSpec, LateralDragSpec
from shuga.grid.cice import CICEGridwork

class FormFactors:
    """
    Build simplified coastal-drag form-factor fields from coastline and
    grounded-iceberg datasets.

    This is a structured shuga port of the lateral-drag half of AFIM's
    ``sea_ice_gridwork.py``. The implementations here are intentionally
    simplified but produce usable F2x/F2y NetCDF products and a combined field.
    """

    def __init__(self,
                 pth_cfg   : ShugaPaths,
                 G_cice_cfg: CICEGridSpec    | None = None,
                 LD_cfg    : LateralDragSpec | None = None,
                 logger = None) -> None:
        self.pth_cfg    = pth_cfg
        self.G_cice_cfg = G_cice_cfg or pth_cfg.G_cice_cfg or CICEGridSpec()
        self.spec       = LD_cfg or pth_cfg.LD_cfg or LateralDragSpec()
        self.logger     = logger
        self.gridwork   = CICEGridwork(pth_cfg = pth_cfg, G_cice_cfg = self.G_cice_cfg, logger = logger)

    #----------------------------------------------------------------------
    # helpers
    #----------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    @staticmethod
    def _transformer_to_proj(proj_crs: str):
        from pyproj import Transformer
        return Transformer.from_crs(4326, proj_crs, always_xy=True)

    @staticmethod
    def _lonlat_to_xy(lon_deg, lat_deg, proj_crs: str):
        T = FormFactors._transformer_to_proj(proj_crs)
        return T.transform(lon_deg, lat_deg)

    @staticmethod
    def _extract_xy_line_segments(gdf, stride: int = 1):
        seg_mid_xy: list[tuple[float, float]] = []
        seg_vec_xy: list[tuple[float, float]] = []
        def add_linestring(line):
            coords = np.asarray(line.coords)
            if coords.shape[0] < 2:
                return
            if stride > 1:
                coords = coords[::stride]
                if coords.shape[0] < 2:
                    return
            p0   = coords[:-1]
            p1   = coords[1:]
            mid  = 0.5 * (p0 + p1)
            vec  = p1 - p0
            keep = np.isfinite(mid).all(axis=1) & np.isfinite(vec).all(axis=1)
            for m, v in zip(mid[keep], vec[keep]):
                seg_mid_xy.append((float(m[0]), float(m[1])))
                seg_vec_xy.append((float(v[0]), float(v[1])))
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gt = geom.geom_type
            if gt == "LineString":
                add_linestring(geom)
            elif gt == "MultiLineString":
                for part in geom.geoms:
                    add_linestring(part)
            elif gt == "Polygon":
                add_linestring(geom.exterior)
                for ring in geom.interiors:
                    add_linestring(ring)
            elif gt == "MultiPolygon":
                for poly in geom.geoms:
                    add_linestring(poly.exterior)
                    for ring in poly.interiors:
                        add_linestring(ring)
        if not seg_mid_xy:
            raise RuntimeError("No coastline segments extracted from vector file.")
        return np.asarray(seg_mid_xy, dtype="float64"), np.asarray(seg_vec_xy, dtype="float64")

    def _load_grid_inputs(self):
        grid = self.gridwork.load_cice_grid(build_faces=False)
        tlon = grid.tgrid["TLON"].values
        tlat = grid.tgrid["TLAT"].values
        if grid.mask is not None:
            ocean = grid.mask.values.astype(bool)
        else:
            ocean = np.isfinite(tlon) & np.isfinite(tlat)
        mask = ocean & np.isfinite(tlon) & np.isfinite(tlat) & (tlat <= float(self.spec.lat_subset_max))
        return grid, tlon, tlat, mask

    @staticmethod
    def _empty_f2_dataset(tlon, tlat, f2x, f2y, attrs: dict[str, object]) -> xr.Dataset:
        ds = xr.Dataset(data_vars = {"F2x": (("nj", "ni"), f2x.astype("float32")),
                                     "F2y": (("nj", "ni"), f2y.astype("float32")),
                                     "lon": (("nj", "ni"), tlon.astype("float32")),
                                     "lat": (("nj", "ni"), tlat.astype("float32"))},
                        coords    = {"nj": np.arange(tlon.shape[0], dtype=np.int32),
                                     "ni": np.arange(tlon.shape[1], dtype=np.int32)},
                        attrs     = attrs)
        ds["F2x"].attrs.update({"long_name": "x-direction lateral drag form factor", "units": "1"})
        ds["F2y"].attrs.update({"long_name": "y-direction lateral drag form factor", "units": "1"})
        ds["lon"].attrs.update({"units": "degrees_east"})
        ds["lat"].attrs.update({"units": "degrees_north"})
        return ds

    def _read_grounded_iceberg_table(self, grounded_iceberg_file: str | Path | None = None) -> pd.DataFrame:
        import geopandas as gpd
        if grounded_iceberg_file is None:
            grounded_iceberg_file = self.pth_cfg.grounded_iceberg_file_path
        path = Path(grounded_iceberg_file)
        suffix = path.suffix.lower()
        if suffix in {".gpkg", ".shp", ".geojson"}:
            gdf = gpd.read_file(path)
            if "geometry" in gdf.columns:
                if gdf.crs is None:
                    gdf = gdf.set_crs(4326)
                gdf_ll = gdf.to_crs(4326)
                lon = gdf_ll.geometry.centroid.x.values
                lat = gdf_ll.geometry.centroid.y.values
                df = pd.DataFrame({"lon": lon, "lat": lat})
                for candidate in ("area", "Area", "AREA", "perimeter", "Perimeter", "PERIMETER"):
                    if candidate in gdf.columns:
                        df[candidate] = gdf[candidate].values
                return df
        df = pd.read_csv(path)
        rename = {}
        for cand in ("Longitude", "longitude", "LON"):
            if cand in df.columns:
                rename[cand] = "lon"
                break
        for cand in ("Latitude", "latitude", "LAT"):
            if cand in df.columns:
                rename[cand] = "lat"
                break
        df = df.rename(columns=rename)
        if not {"lon", "lat"}.issubset(df.columns):
            raise KeyError(f"Could not infer lon/lat columns from {path}")
        return df

    #----------------------------------------------------------------------
    # APIs
    #----------------------------------------------------------------------
    def build_F2_from_high_res_coastline(self,
                                         high_res_coast_file: str | Path | None = None,
                                         output_path        : str | Path | None = None,
                                         overwrite          : bool = False,
                                         stride             : int = 4) -> xr.Dataset:
        import geopandas as gpd
        from scipy.spatial import cKDTree
        if high_res_coast_file is None:
            high_res_coast_file = self.pth_cfg.high_res_coast_file_path
        if output_path is None:
            output_path = self.pth_cfg.coast_form_factors_path
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return xr.open_dataset(output_path)
        grid, tlon, tlat, mask = self._load_grid_inputs()
        self._log(f"Reading coastline vector: {high_res_coast_file}")
        gdf = gpd.read_file(high_res_coast_file)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf                    = gdf.to_crs(self.spec.proj_crs)
        seg_mid_xy, seg_vec_xy = self._extract_xy_line_segments(gdf, stride=max(1, int(stride)))
        tree                   = cKDTree(seg_mid_xy)
        xg, yg                 = self._lonlat_to_xy(tlon[mask], tlat[mask], self.spec.proj_crs)
        dist_m, idx            = tree.query(np.column_stack([xg, yg]), k=1)
        dist_km                = dist_m / 1000.0
        seg_vec                = seg_vec_xy[idx]
        seg_norm               = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
        seg_norm               = np.where(seg_norm > 0, seg_norm, 1.0)
        # Use coastline tangent to distribute drag between x/y components.
        ux        = np.abs(seg_vec[:, 0] / seg_norm)
        uy        = np.abs(seg_vec[:, 1] / seg_norm)
        magnitude = np.clip(1.0 - dist_km / float(self.spec.max_assign_km), 0.0, 1.0)
        f2x       = np.zeros_like(tlon, dtype="float64")
        f2y       = np.zeros_like(tlat, dtype="float64")
        f2x[mask] = magnitude * ux
        f2y[mask] = magnitude * uy
        ds_out    = self._empty_f2_dataset(tlon, tlat, f2x, f2y, attrs={"title"        : "Coastline-derived lateral drag form factors",
                                                                        "source_vector": str(high_res_coast_file),
                                                                        "grid_source"  : str(grid.source_path),
                                                                        "proj_crs"     : str(self.spec.proj_crs),
                                                                        "max_assign_km": float(self.spec.max_assign_km),
                                                                        "method"       : "nearest coastline segment, linear distance taper"})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enc = {"F2x": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "F2y": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lon": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lat": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"}}
        ds_out.to_netcdf(output_path, mode="w", encoding=enc)
        self._log(f"Wrote coastline form factors: {output_path}")
        return ds_out

    def build_F2_from_grounded_iceberg_dataframe(self,
                                                 grounded_iceberg_file: str | Path | None = None,
                                                 output_path          : str | Path | None = None,
                                                 overwrite            : bool = False) -> xr.Dataset:
        from scipy.spatial import cKDTree
        if grounded_iceberg_file is None:
            grounded_iceberg_file = self.pth_cfg.grounded_iceberg_file_path
        if output_path is None:
            output_path = self.pth_cfg.grounded_iceberg_form_factors_path
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return xr.open_dataset(output_path)
        grid, tlon, tlat, mask = self._load_grid_inputs()
        df                     = self._read_grounded_iceberg_table(grounded_iceberg_file)
        lon                    = df["lon"].to_numpy(dtype="float64")
        lat                    = df["lat"].to_numpy(dtype="float64")
        valid                  = np.isfinite(lon) & np.isfinite(lat)
        lon                    = lon[valid]
        lat                    = lat[valid]
        if lon.size == 0:
            raise RuntimeError("No valid grounded iceberg points after filtering.")
        xi, yi      = self._lonlat_to_xy(lon, lat, self.spec.proj_crs)
        tree        = cKDTree(np.column_stack([xi, yi]))
        xg, yg      = self._lonlat_to_xy(tlon[mask], tlat[mask], self.spec.proj_crs)
        dist_m, idx = tree.query(np.column_stack([xg, yg]), k=1)
        dist_km     = dist_m / 1000.0
        magnitude   = np.clip(1.0 - dist_km / float(self.spec.max_assign_km), 0.0, 1.0)
        # For GI-derived drag use an isotropic assignment by default.
        f2x       = np.zeros_like(tlon, dtype="float64")
        f2y       = np.zeros_like(tlat, dtype="float64")
        f2x[mask] = magnitude
        f2y[mask] = magnitude
        ds_out    = self._empty_f2_dataset(tlon, tlat, f2x, f2y, attrs={"title"        : "Grounded-iceberg-derived lateral drag form factors",
                                                                        "source_vector": str(grounded_iceberg_file),
                                                                        "grid_source"  : str(grid.source_path),
                                                                        "proj_crs"     : str(self.spec.proj_crs),
                                                                        "max_assign_km": float(self.spec.max_assign_km),
                                                                        "method"       : "nearest grounded iceberg point, linear distance taper"})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enc = {"F2x": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "F2y": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lon": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lat": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"}}
        ds_out.to_netcdf(output_path, mode="w", encoding=enc)
        self._log(f"Wrote grounded-iceberg form factors: {output_path}")
        return ds_out

    def build_F2_combined(self,
                          coast_path           : str | Path | None = None,
                          grounded_iceberg_path: str | Path | None = None,
                          output_path          : str | Path | None = None,
                          combine              : str | None = None,
                          overwrite            : bool = False) -> xr.Dataset:
        if coast_path is None:
            coast_path = self.pth_cfg.coast_form_factors_path
        if grounded_iceberg_path is None:
            grounded_iceberg_path = self.pth_cfg.grounded_iceberg_form_factors_path
        if output_path is None:
            output_path = self.pth_cfg.combined_form_factors_path
        if combine is None:
            combine = self.spec.f2_map_method
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return xr.open_dataset(output_path)
        ds_c = xr.open_dataset(coast_path)
        ds_g = xr.open_dataset(grounded_iceberg_path)
        if combine == "sum":
            f2x = np.clip(ds_c["F2x"].values + ds_g["F2x"].values, 0.0, 1.0)
            f2y = np.clip(ds_c["F2y"].values + ds_g["F2y"].values, 0.0, 1.0)
        elif combine == "mean":
            f2x = 0.5 * (ds_c["F2x"].values + ds_g["F2x"].values)
            f2y = 0.5 * (ds_c["F2y"].values + ds_g["F2y"].values)
        else:
            f2x = np.maximum(ds_c["F2x"].values, ds_g["F2x"].values)
            f2y = np.maximum(ds_c["F2y"].values, ds_g["F2y"].values)
        ds_out = self._empty_f2_dataset(ds_c["lon"].values, ds_c["lat"].values, f2x, f2y,
                                        attrs={"title"                  : "Combined coastline + grounded iceberg lateral drag form factors",
                                               "coast_source"           : str(coast_path),
                                               "grounded_iceberg_source": str(grounded_iceberg_path),
                                               "combine"                : str(combine)})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enc = {"F2x": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "F2y": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lon": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"},
               "lat": {"zlib": True, "complevel": int(self.spec.netcdf_compression), "dtype": "float32"}}
        ds_out.to_netcdf(output_path, mode="w", encoding=enc)
        self._log(f"Wrote combined form factors: {output_path}")
        return ds_out
