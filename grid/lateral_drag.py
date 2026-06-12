from __future__ import annotations
"""
Utilities for constructing CICE lateral-drag form-factor fields.

This module builds T-cell source fields used by the CICE lateral-drag
parameterisation. The main products are directional geometric form factors:

    FFx(nj, ni)
    FFy(nj, ni)

where `FFx` represents the local model-i / x-direction source contribution and
`FFy` represents the local model-j / y-direction source contribution. These
fields are written in the existing shuga / xarray / NetCDF convention:

    FFx(nj, ni)
    FFy(nj, ni)
    lon(nj, ni)
    lat(nj, ni)

The fields are intended to be supplied to CICE through the F2 input file
machinery, for example:

    F2_file       = "<path-to-form-factor-file>"
    F2x_varname   = "FFx"
    F2y_varname   = "FFy"

The fields written by this module are T-cell source fields. They are not
pre-mapped to CICE velocity faces. CICE reads the T-cell fields and maps them
internally to the appropriate E- and N-face fields used by the lateral-drag
stress calculation.

The primary construction methods are:

    build_FF_from_Hres_coast_Liu()
        Build coastline / ice-front form factors using within-cell projected
        high-resolution source-line length density.

    build_FF_from_GIB_perimeter()
        Build grounded-iceberg form factors using within-cell projected
        grounded-iceberg perimeter length density, with optional iceberg
        area-fraction diagnostics.

    build_FF_combined_CICE()
        Combine coastline and grounded-iceberg form-factor products using
        component-wise max, mean, or sum operations.

The current implementation is designed primarily for Antarctic landfast sea-ice
experiments, but the source-domain filtering machinery also supports an Arctic
configuration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import build_file_logger
from shuga.core.paths   import ShugaPaths
from shuga.core.types   import CICEGridSpec, LateralDragSpec
from shuga.grid.cice    import CICEGridwork

class FormFactors:
    """
    Build CICE lateral-drag form-factor fields from coastline and
    grounded-iceberg source geometry.

    `FormFactors` provides the shuga workflow for generating geometric
    lateral-drag source fields on the CICE T grid. These fields are used to
    represent unresolved lateral resistance associated with high-resolution
    coastline / ice-front geometry and grounded icebergs.

    The class supports three main product types:

    1. Coastline-only form factors
       High-resolution coastline or ice-front geometry is clipped to CICE
       T-cell polygons. The clipped source-line length is projected onto local
       model-grid directions and normalised by the local CICE grid metrics:

           FFx = sum(abs(projected source length along model-i)) / HTE
           FFy = sum(abs(projected source length along model-j)) / HTN

    2. Grounded-iceberg-only form factors
       Grounded-iceberg polygon perimeters are treated as source linework and
       accumulated using the same projected length-density calculation. Optional
       grounded-iceberg area-fraction diagnostics can also be generated.

    3. Combined coastline + grounded-iceberg form factors
       Existing coastline and grounded-iceberg products can be combined using
       component-wise `max`, `mean`, or `sum` operations.

    Output convention
    -----------------
    All production NetCDF files produced by this class follow the shuga / CICE
    file convention:

        FFx(nj, ni)
        FFy(nj, ni)
        lon(nj, ni)
        lat(nj, ni)

    where Python-side arrays are shaped `(nj, ni)`. This is the convention that
    matches the existing CICE NetCDF read pathway used by the lateral-drag
    implementation. The fields are T-cell source products; CICE subsequently
    maps them to velocity-face fields internally.

    Efficiency strategy
    -------------------
    The Liu-style source builders avoid brute-force global grid searches. Source
    vector geometry is first filtered to the requested polar domain, candidate
    CICE cells are restricted to the source envelope and model ocean mask, and
    exact projected intersections are then evaluated only for candidate cells
    near the supplied source geometry.

    This source-driven design ensures that form factors are generated from the
    high-resolution source files themselves, rather than from unrelated coarse
    model landmask coastlines.

    Parameters are supplied through shuga configuration objects:

    pth_cfg
        Path configuration, including CICE grid paths, source vector paths,
        output paths, and logging paths.

    G_cice_cfg
        CICE grid specification. If omitted, the value attached to `pth_cfg` is
        used; if that is also missing, a default `CICEGridSpec` is created.

    LD_cfg
        Lateral-drag form-factor specification. If omitted, the value attached
        to `pth_cfg` is used; if that is also missing, a default
        `LateralDragSpec` is created.

    logger
        Optional logger. If omitted, a file logger named
        `"shuga.lateral-drag"` is created.

    Notes
    -----
    This class supersedes the earlier simplified nearest-source distance-taper
    form-factor workflow for publication-quality Liu-style geometry products.
    Distance tapering, if required for sensitivity testing, should be applied
    as an explicit separate operation so that source-strength construction and
    spatial spreading remain scientifically separable.
    """

    def __init__(self,
                 pth_cfg   : ShugaPaths,
                 G_cice_cfg: CICEGridSpec    | None = None,
                 LD_cfg    : LateralDragSpec | None = None,
                 logger = None) -> None:
        """
        Initialise a form-factor builder.

        Parameters
        ----------
        pth_cfg : ShugaPaths
            shuga path-configuration object. This object supplies source-data
            paths, output paths, CICE grid configuration, lateral-drag
            configuration, and the default metrics/logging path.

        G_cice_cfg : CICEGridSpec or None, optional
            CICE grid specification used to load the model grid, grid metrics,
            and grid mask. If None, the constructor first attempts to use
            `pth_cfg.G_cice_cfg`. If that is also None, a default
            `CICEGridSpec()` is created.

        LD_cfg : LateralDragSpec or None, optional
            Lateral-drag specification controlling form-factor construction
            and combination settings. If None, the constructor first attempts
            to use `pth_cfg.LD_cfg`. If that is also None, a default
            `LateralDragSpec()` is created.

        logger : logging.Logger or None, optional
            Logger used by the form-factor workflow. If None, a file logger
            named `"shuga.lateral-drag"` is created using
            `pth_cfg.metrics_log_path()`.

        Attributes
        ----------
        pth_cfg : ShugaPaths
            Stored path configuration.

        G_cice_cfg : CICEGridSpec
            Resolved CICE grid specification.

        spec : LateralDragSpec
            Resolved lateral-drag specification.

        logger : logging.Logger
            Logger used for progress messages, diagnostics, and writeout
            status.

        gridwork : CICEGridwork
            Helper object used to load and process the CICE grid, including
            T-grid longitude/latitude, grid metrics, grid masks, and optional
            grid-cell face geometry.

        Notes
        -----
        The constructor does not build any form-factor products by itself. It
        only resolves configuration, logging, and gridwork dependencies. Source
        files and CICE grid files are read later by the build methods.
        """
        self.pth_cfg    = pth_cfg
        self.G_cice_cfg = G_cice_cfg or pth_cfg.G_cice_cfg or CICEGridSpec()
        self.spec       = LD_cfg or pth_cfg.LD_cfg or LateralDragSpec()
        self.logger     = logger or build_file_logger("shuga.lateral-drag", self.pth_cfg.metrics_log_path())
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
    def _empty_FF_dataset(tlon, tlat, FFx, FFy, attrs: dict[str, object]) -> xr.Dataset:
        ds = xr.Dataset(data_vars = {"FFx": (("nj", "ni"), FFx.astype("float32")),
                                     "FFy": (("nj", "ni"), FFy.astype("float32")),
                                     "lon": (("nj", "ni"), tlon.astype("float32")),
                                     "lat": (("nj", "ni"), tlat.astype("float32"))},
                        coords    = {"nj": np.arange(tlon.shape[0], dtype=np.int32),
                                     "ni": np.arange(tlon.shape[1], dtype=np.int32)},
                        attrs     = attrs)
        ds["FFx"].attrs.update({"long_name": "x-direction lateral drag form factor", "units": "1"})
        ds["FFy"].attrs.update({"long_name": "y-direction lateral drag form factor", "units": "1"})
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
    # CICE-compatible F2 output
    #
    # CICE expects a NetCDF file containing T-cell form-factor fields, usually:
    #
    #     FFx(i,j) -> read into F2x_in(nx_global, ny_global)
    #     FFy(i,j) -> read into F2y_in(nx_global, ny_global)
    #
    # CICE then maps these T-cell fields to velocity faces internally:
    #
    #     FFx T cells -> E faces -> F2E
    #     FFy T cells -> N faces -> F2N
    #
    # Therefore:
    #   * Python should NOT pre-map to E/N faces.
    #   * Python should write T-cell fields only.
    #   * FFx and FFy should be non-negative and finite.
    #   * Values do not need to be clipped to 1 for Liu-style length-density
    #     form factors, unless a bounded experiment is explicitly desired.
    #----------------------------------------------------------------------
    @staticmethod
    def _as_nonnegative_finite(arr,
                               fill_value: float = 0.0,
                               clip_max  : float | None = None,
                               dtype     = "float32"):
        """
        Return a finite, non-negative array suitable for CICE F2 input.

        CICE already protects against NaNs and negative values when mapping to
        faces, but doing this on write makes the NetCDF product self-consistent
        and easier to inspect.

        Parameters
        ----------
        arr : array-like
            Input form-factor field.
        fill_value : float
            Replacement for NaN/Inf.
        clip_max : float or None
            Optional upper cap. For Liu-style length-density form factors, this
            should normally be None, because the form factor may legitimately
            exceed 1 in convoluted coastal or grounded-iceberg cells.
        dtype : str
            Output dtype.
        """
        out = np.asarray(arr, dtype="float64").copy()
        out[~np.isfinite(out)] = float(fill_value)
        out = np.maximum(out, 0.0)
        if clip_max is not None:
            out = np.minimum(out, float(clip_max))
        return out.astype(dtype)

    @staticmethod
    def _empty_FF_dataset(tlon, tlat, FFx, FFy, attrs: dict[str, object]) -> xr.Dataset:
        """
        Create a CICE-readable T-cell F2 form-factor dataset.

        Important dimension-order note
        ------------------------------
        xarray/NCO/ncks show this file in CDL/C order as:

            FFx(nj, ni)
            FFy(nj, ni)

        where:
            nj = ny_global
            ni = nx_global

        This is intentional.

        CICE reads these variables through the NetCDF Fortran interface into
        arrays allocated as:

            F2x_in(nx_global, ny_global)
            F2y_in(nx_global, ny_global)

        The NetCDF Fortran interface handles the Fortran/C dimension-order
        convention. Therefore, for Python-side arrays shaped (nj, ni), we should
        write the variables as ("nj", "ni"), not transpose them to ("ni", "nj").

        These are T-cell source fields. CICE then maps:
            FFx -> F2E on E faces
            FFy -> F2N on N faces

        inside load_F2_form_factors().
        """
        FFx_out = np.asarray(FFx, dtype="float64").copy()
        FFy_out = np.asarray(FFy, dtype="float64").copy()
        FFx_out[~np.isfinite(FFx_out)] = 0.0
        FFy_out[~np.isfinite(FFy_out)] = 0.0
        FFx_out = np.maximum(FFx_out, 0.0)
        FFy_out = np.maximum(FFy_out, 0.0)
        ds = xr.Dataset(data_vars = {"FFx": (("nj", "ni"), FFx_out.astype("float32")),
                                     "FFy": (("nj", "ni"), FFy_out.astype("float32")),
                                     "lon": (("nj", "ni"), np.asarray(tlon).astype("float32")),
                                     "lat": (("nj", "ni"), np.asarray(tlat).astype("float32"))},
                        coords    = {"nj": np.arange(tlon.shape[0], dtype=np.int32),
                                     "ni": np.arange(tlon.shape[1], dtype=np.int32)},
                        attrs     = {**attrs,
                                     "CICE_read_note": ("Variables are written as (nj,ni) in CDL/xarray order. "
                                                        "CICE reads them through the NetCDF Fortran interface into "
                                                        "F2x_in(nx_global,ny_global) and F2y_in(nx_global,ny_global)."),
                                     "CICE_mapping_note": ("These are T-cell source fields. CICE maps FFx to F2E and FFy to F2N internally.")})
        ds["FFx"].attrs.update({"long_name"  : "x-direction T-cell lateral drag form factor",
                                "units"      : "1",
                                "CICE_target": "F2x_in, then mapped internally to F2E"})
        ds["FFy"].attrs.update({"long_name"  : "y-direction T-cell lateral drag form factor",
                                "units"      : "1",
                                "CICE_target": "F2y_in, then mapped internally to F2N"})
        ds["lon"].attrs.update({"units": "degrees_east"})
        ds["lat"].attrs.update({"units": "degrees_north"})
        return ds


    def _add_tcell_diagnostic_to_cice_ds(self, ds: xr.Dataset, name: str, data, tlon, long_name: str, units: str = "1",
                                         cice_dim_order: str | None = None):
        """
        Add a diagnostic T-cell variable to a CICE-compatible FF dataset.

        This handles the same possible dimension order as `_empty_FF_dataset_cice`.
        Use for diagnostics such as:
          * coastline projected length in metres;
          * GIB perimeter projected length in metres;
          * GIB area fraction;
          * number of GIBs intersecting a cell.
        """
        if cice_dim_order is None:
            cice_dim_order = ds.attrs.get("cice_dim_order", "ni_nj")
        data = np.asarray(data)
        if data.shape != np.asarray(tlon).shape:
            raise ValueError(f"Diagnostic '{name}' shape {data.shape} does not match T grid shape {np.asarray(tlon).shape}.")
        if cice_dim_order == "ni_nj":
            ds[name] = (("ni", "nj"), data.T)
        elif cice_dim_order == "nj_ni":
            ds[name] = (("nj", "ni"), data)
        else:
            raise ValueError("cice_dim_order must be either 'ni_nj' or 'nj_ni'.")
        ds[name].attrs.update({"long_name": long_name, "units": units, "location": "T cell"})
        return ds

    def assert_CICE_F2_file_compatibility(self, P_F2: str | Path,
                                          F2x_varname: str = "FFx",
                                          F2y_varname: str = "FFy",
                                          nx_global: int | None = None,
                                          ny_global: int | None = None) -> None:
        """
        Lightweight sanity check for a CICE F2 input file.

        This does not prove that the Fortran wrapper will read the file correctly,
        but it catches the common mistakes:
          * missing FFx/FFy variables;
          * NaN/Inf values;
          * negative form factors;
          * unexpected dimensions.
        """
        P_F2 = Path(P_F2)
        if not P_F2.exists():
            raise FileNotFoundError(P_F2)
        ds = xr.open_dataset(P_F2)
        try:
            for v in (F2x_varname, F2y_varname):
                if v not in ds:
                    raise KeyError(f"{P_F2} does not contain required variable '{v}'.")
                arr = ds[v].values
                if arr.ndim != 2:
                    raise ValueError(f"{v} must be 2-D, got shape {arr.shape}.")
                if not np.isfinite(arr).all():
                    raise ValueError(f"{v} contains NaN or Inf values.")
                if np.nanmin(arr) < 0:
                    raise ValueError(f"{v} contains negative values.")
            if nx_global is not None and ny_global is not None:
                shape = ds[F2x_varname].shape
                allowed = {(int(nx_global), int(ny_global)),
                           (int(ny_global), int(nx_global))}
                if shape not in allowed:
                    raise ValueError(f"{F2x_varname} shape {shape} does not match either "
                                     f"(nx_global, ny_global)=({nx_global}, {ny_global}) or "
                                     f"(ny_global, nx_global)=({ny_global}, {nx_global}).")
            print(f"CICE F2 compatibility check passed: {P_F2}")
        finally:
            ds.close()

    #----------------------------------------------------------------------
    # Liu-style and GIB-perimeter form-factor methods
    #
    # They compute geometric length-density form factors:
    #
    #     F2x(i,j) = sum(|line projection onto model-i direction|) / dx(i,j)
    #     F2y(i,j) = sum(|line projection onto model-j direction|) / dy(i,j)
    #
    # This is the Liu et al.-style construction for high-resolution coastline
    # geometry, and an analogous perimeter/contact-length construction for
    # grounded icebergs.
    #----------------------------------------------------------------------
    @staticmethod
    def _iter_line_parts(geom):
        """
        Yield LineString parts from an arbitrary Shapely geometry.

        This helper is deliberately permissive because intersections between
        grid-cell polygons and source geometries can return LineString,
        MultiLineString, GeometryCollection, or empty objects.

        Polygon boundaries are NOT extracted here. This function assumes the
        input is already linework or the result of intersecting linework with
        a polygon.
        """
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "LineString":
            yield geom
        elif gt == "MultiLineString":
            for part in geom.geoms:
                if part is not None and not part.is_empty:
                    yield part
        elif gt == "GeometryCollection":
            for part in geom.geoms:
                yield from FormFactors._iter_line_parts(part)

    @staticmethod
    def _iter_polygon_parts(geom):
        """
        Yield Polygon parts from Polygon/MultiPolygon/GeometryCollection objects.

        Used for grounded-iceberg area-fraction diagnostics.
        """
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "Polygon":
            if geom.is_valid:
                yield geom
            else:
                fixed = geom.buffer(0)
                if fixed is not None and not fixed.is_empty:
                    yield from FormFactors._iter_polygon_parts(fixed)
        elif gt == "MultiPolygon":
            for part in geom.geoms:
                yield from FormFactors._iter_polygon_parts(part)
        elif gt == "GeometryCollection":
            for part in geom.geoms:
                yield from FormFactors._iter_polygon_parts(part)

    @staticmethod
    def _iter_boundary_linework(geom):
        """
        Convert source geometries into linework.

        Rules:
          * LineString/MultiLineString are used directly.
          * Polygon/MultiPolygon are converted to their boundaries.
          * GeometryCollections are recursively searched.
          * Points are ignored.

        This lets the same code handle:
          * Natural Earth / ADD coastline linework;
          * coastline polygons;
          * grounded-iceberg polygons, where the perimeter is the source.
        """
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt in {"LineString", "MultiLineString"}:
            yield from FormFactors._iter_line_parts(geom)
        elif gt == "Polygon":
            yield from FormFactors._iter_line_parts(geom.boundary)
        elif gt == "MultiPolygon":
            for poly in geom.geoms:
                yield from FormFactors._iter_line_parts(poly.boundary)
        elif gt == "GeometryCollection":
            for part in geom.geoms:
                yield from FormFactors._iter_boundary_linework(part)

    def _read_projected_linework(self, P_vector: str | Path):
        """
        Read a vector file and return projected source linework.

        For coastline data, this is the coastline/ice-front linework.
        For grounded-iceberg polygons, this is the grounded-iceberg perimeter.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per source line geometry, projected into self.spec.proj_crs.
        """
        import geopandas as gpd
        P_vector = Path(P_vector)
        self._log(f"Reading vector linework: {P_vector}")
        gdf = gpd.read_file(P_vector)
        if gdf.empty:
            raise RuntimeError(f"No features found in vector file: {P_vector}")
        if gdf.crs is None:
            # Most of the products you are using are lon/lat.
            # If a future source file is not EPSG:4326, set the CRS upstream.
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(self.spec.proj_crs)
        lines = []
        source_ids = []
        for source_id, geom in enumerate(gdf.geometry):
            for line in self._iter_boundary_linework(geom):
                if line is None or line.is_empty:
                    continue
                if line.length <= 0:
                    continue
                lines.append(line)
                source_ids.append(source_id)
        if not lines:
            raise RuntimeError(f"No usable linework could be extracted from vector file: {P_vector}")
        out = gpd.GeoDataFrame({"source_id": source_ids}, geometry = lines, crs = self.spec.proj_crs)
        return out

    def _read_projected_polygons(self, P_vector: str | Path):
        """
        Read a vector file and return projected polygon geometry.

        This is mainly used for grounded-iceberg area-fraction diagnostics.
        It is not used for the perimeter-length form factor except as a source
        from which boundaries are extracted elsewhere.
        """
        import geopandas as gpd
        P_vector = Path(P_vector)
        self._log(f"Reading vector polygons: {P_vector}")
        gdf = gpd.read_file(P_vector)
        if gdf.empty:
            raise RuntimeError(f"No features found in vector file: {P_vector}")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(self.spec.proj_crs)
        polygons = []
        source_ids = []
        for source_id, geom in enumerate(gdf.geometry):
            for poly in self._iter_polygon_parts(geom):
                if poly is None or poly.is_empty:
                    continue
                if poly.area <= 0:
                    continue
                polygons.append(poly)
                source_ids.append(source_id)
        if not polygons:
            raise RuntimeError(f"No usable polygons could be extracted from vector file: {P_vector}")
        out = gpd.GeoDataFrame({"source_id": source_ids}, geometry = polygons, crs = self.spec.proj_crs)
        return out

    @staticmethod
    def _local_grid_basis_from_centres(x: np.ndarray, y: np.ndarray):
        """
        Estimate local model-grid basis vectors from projected T-cell centres.

        The Liu et al. formulation projects sub-grid coastline length onto the
        model x- and y-directions. In CICE this is more naturally interpreted as
        the local logical i- and j-directions of the model grid, not necessarily
        true east/north in the projection.

        This helper estimates those directions from neighbouring projected
        T-cell centres.

        Returns
        -------
        eix, eiy : ndarray
            Unit vector for the local model-i direction.
        ejx, ejy : ndarray
            Unit vector for the local model-j direction.
        dx_m, dy_m : ndarray
            Local grid spacings in metres, estimated from centre-to-centre
            distances in the i and j directions.
        """
        x               = np.asarray(x, dtype="float64")
        y               = np.asarray(y, dtype="float64")
        nj, ni          = x.shape
        # Vector in logical i direction.
        dix             = np.full_like(x, np.nan, dtype="float64")
        diy             = np.full_like(y, np.nan, dtype="float64")
        # Interior: centred difference between neighbouring cell centres.
        dix[:   , 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
        diy[:   , 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
        # Edges: one-sided difference.
        dix[:   , 0   ] = x[:, 1 ] - x[:, 0 ]
        diy[:   , 0   ] = y[:, 1 ] - y[:, 0 ]
        dix[:   , -1  ] = x[:, -1] - x[:, -2]
        diy[:   , -1  ] = y[:, -1] - y[:, -2]
        # Vector in logical j direction.
        djx             = np.full_like(x, np.nan, dtype="float64")
        djy             = np.full_like(y, np.nan, dtype="float64")
        djx[1:-1, :   ] = 0.5 * (x[2:, :] - x[:-2, :])
        djy[1:-1, :   ] = 0.5 * (y[2:, :] - y[:-2, :])
        djx[0   , :   ] = x[1 , :] - x[0 , :]
        djy[0   , :   ] = y[1 , :] - y[0 , :]
        djx[-1  , :   ] = x[-1, :] - x[-2, :]
        djy[-1  , :   ] = y[-1, :] - y[-2, :]
        dx_m            = np.hypot(dix, diy)
        dy_m            = np.hypot(djx, djy)
        # Unit vectors. Invalid values are left as NaN for masking later.
        eix             = np.divide(dix, dx_m, out = np.full_like(dix, np.nan), where = (dx_m > 0))
        eiy             = np.divide(diy, dx_m, out = np.full_like(diy, np.nan), where = (dx_m > 0))
        ejx             = np.divide(djx, dy_m, out = np.full_like(djx, np.nan), where = (dy_m > 0))
        ejy             = np.divide(djy, dy_m, out = np.full_like(djy, np.nan), where = (dy_m > 0))
        return eix, eiy, ejx, ejy, dx_m, dy_m


    def _build_projected_tcell_gdf(self):
        """
        Build an approximate projected T-cell polygon GeoDataFrame.

        This helper constructs one rotated parallelogram per valid CICE T cell
        using:
          * projected T-cell centre;
          * local logical i-direction;
          * local logical j-direction;
          * local centre-to-centre spacings.

        This is an approximation to the true curvilinear CICE grid-cell polygon.
        It is suitable for practical length-density mapping when explicit grid
        corner variables are unavailable. If exact CICE cell vertices become
        available through CICEGridwork, this helper should be swapped for a true
        vertex-based implementation.

        Returns
        -------
        cell_gdf : geopandas.GeoDataFrame
            Projected T-cell polygons with one row per valid ocean/subset cell.
        grid : object
            CICE gridwork object returned by _load_grid_inputs().
        tlon, tlat : ndarray
            Longitude and latitude arrays.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon
        grid, tlon, tlat, mask         = self._load_grid_inputs()
        x, y                           = self._lonlat_to_xy(tlon, tlat, self.spec.proj_crs)
        x                              = np.asarray(x, dtype="float64")
        y                              = np.asarray(y, dtype="float64")
        eix, eiy, ejx, ejy, dx_m, dy_m = self._local_grid_basis_from_centres(x, y)
        valid                          = (mask &
                                          np.isfinite(x)    & np.isfinite(y)    &
                                          np.isfinite(eix)  & np.isfinite(eiy)  &
                                          np.isfinite(ejx)  & np.isfinite(ejy)  &
                                          np.isfinite(dx_m) & np.isfinite(dy_m) &
                                          (dx_m > 0)        & (dy_m > 0)        )
        jj, ii                         = np.where(valid)
        rows                           = []
        geoms                          = []
        for cell_id, (j, i) in enumerate(zip(jj, ii)):
            cx     = x[j, i]
            cy     = y[j, i]
            # Half-width vector in logical i direction.
            hx_x   = 0.5 * dx_m[j, i] * eix[j, i]
            hx_y   = 0.5 * dx_m[j, i] * eiy[j, i]
            # Half-width vector in logical j direction.
            hy_x   = 0.5 * dy_m[j, i] * ejx[j, i]
            hy_y   = 0.5 * dy_m[j, i] * ejy[j, i]
            # Rotated parallelogram vertices; polygon is centred on the
            # CICE T-cell centre and oriented along the local model-grid i/j basis.
            coords = [(cx - hx_x - hy_x, cy - hx_y - hy_y),
                      (cx + hx_x - hy_x, cy + hx_y - hy_y),
                      (cx + hx_x + hy_x, cy + hx_y + hy_y),
                      (cx - hx_x + hy_x, cy - hx_y + hy_y)]
            poly   = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                continue
            rows.append({"cell_id"   : cell_id,
                         "j"         : int(j),
                         "i"         : int(i),
                         "x"         : float(cx),
                         "y"         : float(cy),
                         "dx_m"      : float(dx_m[j, i]),
                         "dy_m"      : float(dy_m[j, i]),
                         "eix"       : float(eix[j, i]),
                         "eiy"       : float(eiy[j, i]),
                         "ejx"       : float(ejx[j, i]),
                         "ejy"       : float(ejy[j, i]),
                         "cell_area" : float(poly.area)})
            geoms.append(poly)
        cell_gdf = gpd.GeoDataFrame(rows, geometry = geoms, crs = self.spec.proj_crs)
        if cell_gdf.empty:
            raise RuntimeError("No valid projected CICE T-cell polygons were built.")
        return cell_gdf, grid, tlon, tlat

    def _accumulate_projected_line_density(self, cell_gdf, line_gdf, progress_every: int = 10000):
        """
        Accumulate projected line-length density within each CICE grid cell.

        For each model cell, all source linework intersecting that cell is clipped
        to the cell polygon. Each clipped line segment is decomposed into small
        straight segments, projected onto the local model-grid i and j directions,
        and accumulated.

        The resulting form factors are:

            FFx = sum(|segment dot e_i|) / dx_m
            FFy = sum(|segment dot e_j|) / dy_m

        where e_i and e_j are local model-grid unit vectors estimated from the
        projected CICE grid.

        This is the practical Liu-style operation.
        """
        ncell        = len(cell_gdf)
        sum_i_m      = np.zeros(ncell, dtype="float64")
        sum_j_m      = np.zeros(ncell, dtype="float64")
        n_hits       = np.zeros(ncell, dtype="int32")
        source_geoms = list(line_gdf.geometry)
        source_sidx  = line_gdf.sindex
        for pos, row in enumerate(cell_gdf.itertuples(index=False)):
            if progress_every and pos > 0 and pos % progress_every == 0:
                self._log(f"Processed {pos:,} / {ncell:,} grid cells for line density")
            cell_poly = row.geometry
            # Fast bounding-box query first.
            candidate_idx = list(source_sidx.intersection(cell_poly.bounds))
            if not candidate_idx:
                continue
            for src_idx in candidate_idx:
                src_geom = source_geoms[src_idx]
                if src_geom is None or src_geom.is_empty:
                    continue
                if not src_geom.intersects(cell_poly):
                    continue
                # Clip the source line to the cell polygon.
                clipped = src_geom.intersection(cell_poly)
                if clipped is None or clipped.is_empty:
                    continue
                any_hit = False
                for line in self._iter_line_parts(clipped):
                    coords = np.asarray(line.coords, dtype="float64")
                    if coords.shape[0] < 2:
                        continue
                    vec = coords[1:, :] - coords[:-1, :]
                    finite = np.isfinite(vec).all(axis=1)
                    if not finite.any():
                        continue
                    vec = vec[finite]
                    # Project each clipped line segment onto the local CICE
                    # logical i and j directions; implements the projected-length numerator in the Liu
                    # form-factor equations, but using the local model-grid basis rather than global x/y axes.
                    proj_i = vec[:, 0] * row.eix + vec[:, 1] * row.eiy
                    proj_j = vec[:, 0] * row.ejx + vec[:, 1] * row.ejy
                    sum_i_m[row.cell_id] += np.abs(proj_i).sum()
                    sum_j_m[row.cell_id] += np.abs(proj_j).sum()
                    any_hit = True
                if any_hit:
                    n_hits[row.cell_id] += 1
        dx  = cell_gdf["dx_m"].to_numpy(dtype="float64")
        dy  = cell_gdf["dy_m"].to_numpy(dtype="float64")
        FFx = np.divide(sum_i_m, dx, out = np.zeros_like(sum_i_m), where = dx > 0)
        FFy = np.divide(sum_j_m, dy, out = np.zeros_like(sum_j_m), where = dy > 0)
        return FFx, FFy, sum_i_m, sum_j_m, n_hits

    def _accumulate_polygon_area_fraction(self, cell_gdf, polygon_gdf,
                                          progress_every: int = 10000,
                                          clip_area_fraction: bool = True):
        """
        Accumulate grounded-iceberg area fraction within each CICE grid cell.

        This is optional for GIBs. It is not part of the Liu coastline form factor,
        but it is useful as:
          * a diagnostic of obstacle occupancy;
          * an optional isotropic roughness contribution;
          * a sanity check for cells containing many grounded icebergs.

        Returns
        -------
        area_frac : ndarray
            Sum of grounded-iceberg area intersecting the cell divided by cell area.
        area_m2 : ndarray
            Raw intersected grounded-iceberg area in each cell.
        n_poly_hits : ndarray
            Number of grounded-iceberg polygon features intersecting each cell.
        """
        ncell = len(cell_gdf)
        area_m2     = np.zeros(ncell, dtype="float64")
        n_poly_hits = np.zeros(ncell, dtype="int32")
        poly_geoms = list(polygon_gdf.geometry)
        poly_sidx  = polygon_gdf.sindex
        for pos, row in enumerate(cell_gdf.itertuples(index=False)):
            if progress_every and pos > 0 and pos % progress_every == 0:
                self._log(f"Processed {pos:,} / {ncell:,} grid cells for GIB area")
            cell_poly = row.geometry
            candidate_idx = list(poly_sidx.intersection(cell_poly.bounds))
            if not candidate_idx:
                continue
            for src_idx in candidate_idx:
                poly = poly_geoms[src_idx]
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(cell_poly):
                    continue
                clipped = poly.intersection(cell_poly)
                if clipped is None or clipped.is_empty:
                    continue
                a = clipped.area
                if np.isfinite(a) and a > 0:
                    area_m2[row.cell_id] += a
                    n_poly_hits[row.cell_id] += 1
        cell_area = cell_gdf["cell_area"].to_numpy(dtype="float64")
        area_frac = np.divide(area_m2, cell_area, out = np.zeros_like(area_m2), where = cell_area > 0)
        if clip_area_fraction:
            # Area fractions should generally be <= 1 unless source polygons
            # overlap. Clipping makes the diagnostic easier to interpret.
            area_frac = np.clip(area_frac, 0.0, 1.0)
        return area_frac, area_m2, n_poly_hits

    @staticmethod
    def _scatter_cell_values_to_grid(tlon, tlat, cell_gdf, values, dtype="float32"):
        """
        Scatter a 1-D cell vector back onto the CICE T-grid shape.
        """
        out         = np.zeros_like(tlon, dtype = dtype)
        jj          = cell_gdf["j"].to_numpy(dtype = "int64")
        ii          = cell_gdf["i"].to_numpy(dtype = "int64")
        out[jj, ii] = np.asarray(values, dtype = dtype)
        return out

    #----------------------------------------------------------------------
    # Liu-style form-factor grid subsetting and accumulation:
    # 1. Only intersect T-cell polygons in the requested polar domain.
    # 2. Use the vector source itself to define where form factors can occur.
    #    This prevents unrelated coarse-grid land/coastlines from creating FFs.
    # 3. Use the CICE ocean/kmt mask as a secondary filter, not the primary
    #    source of coastline geometry.
    # 4. Accumulate by looping over source linework and querying nearby grid
    #    cells, rather than looping over every grid cell.
    #----------------------------------------------------------------------
    def _normalise_hemisphere(self, hemisphere: str | None = None) -> str:
        """
        Normalise hemisphere selection.

        Default behaviour is Antarctic/Southern Hemisphere because the present
        lateral-drag workflow is Antarctic landfast-ice focused.
        """
        if hemisphere is None:
            hemisphere = getattr(self.pth_cfg, "hemisphere", None) or "SH"
        hemisphere = str(hemisphere).upper()
        if hemisphere in {"S", "SOUTH", "SOUTHERN", "ANT", "ANTARCTIC"}:
            return "SH"
        if hemisphere in {"N", "NORTH", "NORTHERN", "ARC", "ARCTIC"}:
            return "NH"
        if hemisphere not in {"SH", "NH"}:
            raise ValueError("hemisphere must be one of 'SH' or 'NH'.")
        return hemisphere

    def _polar_source_lat_limit(self, hemisphere: str,
                                source_lat_limit: float | None = None) -> float:
        """
        Latitude threshold used to filter the source vector file.

        For Antarctic coastline form factors, use <= -60 by default. This
        deliberately excludes non-Antarctic southern landmasses such as Australia,
        South Africa, and southern South America if a non-Antarctic coastline file
        is accidentally supplied.

        For Arctic coastline form factors, use >= +60 by default.
        """
        hemisphere = self._normalise_hemisphere(hemisphere)
        if source_lat_limit is not None:
            return float(source_lat_limit)
        if hemisphere == "SH":
            return -60.0
        return 60.0

    def _filter_vector_to_polar_domain(self, gdf,
                                       hemisphere: str = "SH",
                                       source_lat_limit: float | None = None,
                                       source_lat_buffer_deg: float = 0.0):
        """
        Filter vector features to the requested polar domain before projection.

        This is intentionally applied to the SOURCE geometry, not merely the CICE
        grid. This ensures that if a broad/global coastline dataset is accidentally
        passed in, only the Antarctic or Arctic part of that source can generate
        form factors.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input vector features in any CRS.
        hemisphere : {"SH", "NH"}
            Requested polar domain.
        source_lat_limit : float or None
            Antarctic default: -60.0, keep features touching lat <= -60.
            Arctic default    : +60.0, keep features touching lat >= +60.
        source_lat_buffer_deg : float
            Optional latitude buffer applied to the source filter.
            Usually keep this at 0.0 for Antarctic coastline to avoid accidentally
            admitting southern South America.

        Returns
        -------
        geopandas.GeoDataFrame
            Filtered source features, still in the original CRS.
        """
        import geopandas as gpd
        hemisphere = self._normalise_hemisphere(hemisphere)
        lat_limit  = self._polar_source_lat_limit(hemisphere, source_lat_limit)
        if gdf.empty:
            return gdf
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        # Work in geographic coordinates for the polar source filter.
        gdf_ll = gdf.to_crs(4326)
        bounds = gdf_ll.bounds
        if hemisphere == "SH":
            # Keep any feature whose southern edge reaches the Antarctic domain.
            # With the default -60 deg threshold this avoids South America,
            # Australia, South Africa, etc.
            keep = bounds["miny"].to_numpy() <= (lat_limit + source_lat_buffer_deg)
        else:
            keep = bounds["maxy"].to_numpy() >= (lat_limit - source_lat_buffer_deg)
        gdf_out = gdf.loc[keep].copy()
        if gdf_out.empty:
            raise RuntimeError(f"Polar source filtering removed all features. hemisphere={hemisphere}, source_lat_limit={lat_limit}")
        return gdf_out

    def _read_projected_linework(self, P_vector: str | Path,
                                 hemisphere: str = "SH",
                                 source_lat_limit: float | None = None,
                                 source_lat_buffer_deg: float = 0.0):
        """
        Read a vector file, filter to the requested polar source domain, and
        return projected linework.

        For coastline data:
            source linework is the coastline/ice-front linework.

        For grounded-iceberg polygons:
            source linework is the iceberg perimeter.

        Critical point
        --------------
        The polar filter is applied to the source vector file itself. This is what
        prevents unrelated coasts from generating FF values. The coarse CICE kmt
        mask is only used later to decide which model cells are valid ocean cells.
        """
        import geopandas as gpd
        P_vector = Path(P_vector)
        self._log(f"Reading vector linework: {P_vector}")
        gdf = gpd.read_file(P_vector)
        if gdf.empty:
            raise RuntimeError(f"No features found in vector file: {P_vector}")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = self._filter_vector_to_polar_domain(gdf                   = gdf,
                                                  hemisphere            = hemisphere,
                                                  source_lat_limit      = source_lat_limit,
                                                  source_lat_buffer_deg = source_lat_buffer_deg)
        gdf = gdf.to_crs(self.spec.proj_crs)
        lines = []
        source_ids = []
        for source_id, geom in enumerate(gdf.geometry):
            for line in self._iter_boundary_linework(geom):
                if line is None or line.is_empty:
                    continue
                if line.length <= 0:
                    continue
                lines.append(line)
                source_ids.append(source_id)
        if not lines:
            raise RuntimeError(f"No usable linework could be extracted from vector file: {P_vector}")
        out = gpd.GeoDataFrame({"source_id": source_ids}, geometry = lines, crs = self.spec.proj_crs)
        return out.reset_index(drop=True)

    def _read_projected_polygons(self, P_vector: str | Path,
                                 hemisphere: str = "SH",
                                 source_lat_limit: float | None = None,
                                 source_lat_buffer_deg: float = 0.0):
        """
        Read a vector file, filter to the requested polar source domain, and
        return projected polygons.

        This is mainly for grounded-iceberg area-fraction diagnostics.
        """
        import geopandas as gpd
        P_vector = Path(P_vector)
        self._log(f"Reading vector polygons: {P_vector}")
        gdf = gpd.read_file(P_vector)
        if gdf.empty:
            raise RuntimeError(f"No features found in vector file: {P_vector}")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = self._filter_vector_to_polar_domain(gdf                   = gdf,
                                                  hemisphere            = hemisphere,
                                                  source_lat_limit      = source_lat_limit,
                                                  source_lat_buffer_deg = source_lat_buffer_deg)
        gdf = gdf.to_crs(self.spec.proj_crs)
        polygons = []
        source_ids = []
        for source_id, geom in enumerate(gdf.geometry):
            for poly in self._iter_polygon_parts(geom):
                if poly is None or poly.is_empty:
                    continue
                if poly.area <= 0:
                    continue
                polygons.append(poly)
                source_ids.append(source_id)
        if not polygons:
            raise RuntimeError(f"No usable polygons could be extracted from vector file: {P_vector}")
        out = gpd.GeoDataFrame({"source_id": source_ids}, geometry = polygons, crs = self.spec.proj_crs)
        return out.reset_index(drop=True)

    def _source_lonlat_bounds(self, source_gdf_projected):
        """
        Return lon/lat bounds for already-projected source geometry.

        Used only as an early grid-candidate filter. Exact FF assignment still
        requires geometric intersection with the projected source linework.
        """
        src_ll = source_gdf_projected.to_crs(4326)
        return src_ll.total_bounds  # minlon, minlat, maxlon, maxlat


    @staticmethod
    def _lon_in_bounds_mask(lon_deg, lon_min, lon_max, pad_deg: float = 0.0):
        """
        Longitude bounds mask with antimeridian handling.

        If the source spans almost all longitudes, return all True.
        """
        lon = ((np.asarray(lon_deg, dtype="float64") + 180.0) % 360.0) - 180.0
        lon_min = float(lon_min) - float(pad_deg)
        lon_max = float(lon_max) + float(pad_deg)
        # If broad circum-Antarctic source geometry spans nearly all longitudes,
        # longitude is not useful as a prefilter.
        if (lon_max - lon_min) >= 340.0:
            return np.ones_like(lon, dtype=bool)
        lon_min = ((lon_min + 180.0) % 360.0) - 180.0
        lon_max = ((lon_max + 180.0) % 360.0) - 180.0
        if lon_min <= lon_max:
            return (lon >= lon_min) & (lon <= lon_max)
        return (lon >= lon_min) | (lon <= lon_max)

    def _candidate_mask_for_source(self, tlon, tlat, ocean_mask, source_gdf_projected,
                                   hemisphere: str = "SH",
                                   source_lat_limit: float | None = None,
                                   grid_lat_pad_deg: float = 3.0,
                                   grid_lon_pad_deg: float = 3.0,
                                   use_ocean_mask: bool = True,
                                   use_index_half_hint: bool = False,
                                   use_coastal_neighbour_filter: bool = False,
                                   coastal_neighbour_radius: int = 2):
        """
        Build an intelligent T-cell candidate mask before constructing polygons.

        The mask has three layers:

        1. Polar-domain filter:
           Antarctic default is lat <= -60 plus padding. This removes the
           irrelevant half of the grid immediately.

        2. Source-bounds filter:
           Use the actual vector source bounds, with padding. This prevents
           unrelated geographical areas from being considered.

        3. Optional CICE ocean/coastal-neighbour filter:
           Use kmt-derived ocean mask to keep only ocean cells. Optionally restrict
           further to ocean cells close to the coarse land mask. This is OFF by
           default because the high-resolution coastline may not coincide perfectly
           with the coarse model landmask.

        The exact geometric intersection is still performed later. This mask only
        reduces the number of cells for which polygons are built and indexed.
        """
        hemisphere = self._normalise_hemisphere(hemisphere)
        lat_limit  = self._polar_source_lat_limit(hemisphere, source_lat_limit)
        tlon = np.asarray(tlon, dtype="float64")
        tlat = np.asarray(tlat, dtype="float64")
        candidate = np.isfinite(tlon) & np.isfinite(tlat)
        if hemisphere == "SH":
            candidate &= tlat <= (lat_limit + abs(float(grid_lat_pad_deg)))
        else:
            candidate &= tlat >= (lat_limit - abs(float(grid_lat_pad_deg)))
        # Optional index hint for this specific global grid. This should never be
        # the only domain filter, because grid orientation can change. It is only
        # an additional cheap prefilter after lat filtering.
        if use_index_half_hint:
            nj, ni = tlat.shape
            jj = np.arange(nj)[:, None]
            if hemisphere == "SH":
                candidate &= jj <= (nj // 2)
            else:
                candidate &= jj >= (nj // 2)
        # Restrict to the lon/lat envelope of the source geometry.
        minlon, minlat, maxlon, maxlat = self._source_lonlat_bounds(source_gdf_projected)
        candidate &= tlat >= (minlat - abs(float(grid_lat_pad_deg)))
        candidate &= tlat <= (maxlat + abs(float(grid_lat_pad_deg)))
        candidate &= self._lon_in_bounds_mask(tlon, lon_min = minlon, lon_max = maxlon, pad_deg = grid_lon_pad_deg)
        if use_ocean_mask and ocean_mask is not None:
            ocean = np.asarray(ocean_mask, dtype=bool)
            candidate &= ocean
            if use_coastal_neighbour_filter:
                # This is deliberately optional. It can greatly reduce candidates,
                # but it can also remove legitimate high-resolution coastline cells
                # where the coarse kmt landmask is displaced or too smooth.
                from scipy.ndimage import binary_dilation
                land = (~ocean) & np.isfinite(tlat)
                near_land = binary_dilation(land,iterations = max(1, int(coastal_neighbour_radius)))
                candidate &= near_land
        return candidate

    def _build_projected_tcell_gdf_for_source(self, source_gdf_projected,
                                              hemisphere: str = "SH",
                                              source_lat_limit: float | None = None,
                                              grid_lat_pad_deg: float = 3.0,
                                              grid_lon_pad_deg: float = 3.0,
                                              use_ocean_mask: bool = True,
                                              use_index_half_hint: bool = False,
                                              use_coastal_neighbour_filter: bool = False,
                                              coastal_neighbour_radius: int = 2):
        """
        Build projected T-cell polygons only for cells that could plausibly
        intersect the supplied source geometry.

        This method uses CICE grid faces from CICEGridwork when available, rather
        than building polygons for the full global grid. The resulting GeoDataFrame
        is much smaller and is spatially indexed for source-driven accumulation.

        Returns
        -------
        cell_gdf : geopandas.GeoDataFrame
            Candidate T-cell polygons, projected into self.spec.proj_crs.
        grid : CICEGridBundle
            Loaded CICE grid bundle.
        tlon, tlat : ndarray
            Full global T-grid lon/lat arrays in Python order (nj, ni).
        """
        import geopandas as gpd
        from shapely.geometry import Polygon
        # build_faces=True gives approximate CICE cell corner coordinates through
        # ULON/ULAT. This is preferable to synthetic centre-based rectangles.
        grid = self.gridwork.load_cice_grid(build_faces=True)
        tlon = grid.tgrid["TLON"].values
        tlat = grid.tgrid["TLAT"].values
        if grid.mask is not None:
            ocean_mask = grid.mask.values.astype(bool)
        else:
            ocean_mask = np.isfinite(tlon) & np.isfinite(tlat)
        candidate = self._candidate_mask_for_source(tlon                         = tlon,
                                                    tlat                         = tlat,
                                                    ocean_mask                   = ocean_mask,
                                                    source_gdf_projected          = source_gdf_projected,
                                                    hemisphere                   = hemisphere,
                                                    source_lat_limit             = source_lat_limit,
                                                    grid_lat_pad_deg             = grid_lat_pad_deg,
                                                    grid_lon_pad_deg             = grid_lon_pad_deg,
                                                    use_ocean_mask               = use_ocean_mask,
                                                    use_index_half_hint          = use_index_half_hint,
                                                    use_coastal_neighbour_filter = use_coastal_neighbour_filter,
                                                    coastal_neighbour_radius     = coastal_neighbour_radius)
        if not candidate.any():
            raise RuntimeError("No candidate CICE T cells remain after source/grid filtering.")
        # Project cell centres.
        x_t, y_t = self._lonlat_to_xy(tlon, tlat, self.spec.proj_crs)
        x_t = np.asarray(x_t, dtype="float64")
        y_t = np.asarray(y_t, dtype="float64")
        # Project approximate cell corners from CICEGridwork.
        ulon = grid.ugrid["ULON"].values
        ulat = grid.ugrid["ULAT"].values
        x_b, y_b = self._lonlat_to_xy(ulon, ulat, self.spec.proj_crs)
        x_b = np.asarray(x_b, dtype="float64")
        y_b = np.asarray(y_b, dtype="float64")
        # Use CICE grid metrics where available.
        dx_m = grid.tgrid["HTE"].values.astype("float64")
        dy_m = grid.tgrid["HTN"].values.astype("float64")
        jj, ii = np.where(candidate)
        rows = []
        geoms = []
        for cell_id, (j, i) in enumerate(zip(jj, ii)):
            # Each T-cell polygon uses the four surrounding corner points.
            # The precise orientation is not important for polygon area/intersection,
            # provided the ring is non-self-intersecting.
            coords = [(x_b[j,     i    ], y_b[j,     i    ]),
                      (x_b[j,     i + 1], y_b[j,     i + 1]),
                      (x_b[j + 1, i + 1], y_b[j + 1, i + 1]),
                      (x_b[j + 1, i    ], y_b[j + 1, i    ])]
            if not np.isfinite(np.asarray(coords)).all():
                continue
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                continue
            # Local model-i direction: midpoint of west side to midpoint of east side.
            west_mid = np.array([0.5 * (x_b[j, i] + x_b[j + 1, i]),
                                 0.5 * (y_b[j, i] + y_b[j + 1, i])])
            east_mid = np.array([0.5 * (x_b[j, i + 1] + x_b[j + 1, i + 1]),
                                 0.5 * (y_b[j, i + 1] + y_b[j + 1, i + 1])])
            vi       = east_mid - west_mid
            vi_norm  = np.hypot(vi[0], vi[1])
            # Local model-j direction: midpoint of south side to midpoint of north side.
            south_mid = np.array([0.5 * (x_b[j, i] + x_b[j, i + 1]),
                                  0.5 * (y_b[j, i] + y_b[j, i + 1])])
            north_mid = np.array([0.5 * (x_b[j + 1, i] + x_b[j + 1, i + 1]),
                                  0.5 * (y_b[j + 1, i] + y_b[j + 1, i + 1])])
            vj        = north_mid - south_mid
            vj_norm   = np.hypot(vj[0], vj[1])
            if vi_norm <= 0 or vj_norm <= 0:
                continue
            if not np.isfinite(dx_m[j, i]) or not np.isfinite(dy_m[j, i]):
                continue
            if dx_m[j, i] <= 0 or dy_m[j, i] <= 0:
                continue
            rows.append({"cell_id"   : int(cell_id),
                         "j"         : int(j),
                         "i"         : int(i),
                         "x"         : float(x_t[j, i]),
                         "y"         : float(y_t[j, i]),
                         "dx_m"      : float(dx_m[j, i]),
                         "dy_m"      : float(dy_m[j, i]),
                         "eix"       : float(vi[0] / vi_norm),
                         "eiy"       : float(vi[1] / vi_norm),
                         "ejx"       : float(vj[0] / vj_norm),
                         "ejy"       : float(vj[1] / vj_norm),
                         "cell_area" : float(poly.area)})
            geoms.append(poly)
        cell_gdf = gpd.GeoDataFrame(rows, geometry = geoms, crs = self.spec.proj_crs).reset_index(drop=True)
        if cell_gdf.empty:
            raise RuntimeError("No valid projected candidate T-cell polygons were built.")
        self._log(f"Built {len(cell_gdf):,} candidate T-cell polygons "
                  f"from {int(candidate.sum()):,} source-filtered candidate cells "
                  f"on global grid shape {tlon.shape}.")
        return cell_gdf, grid, tlon, tlat

    def _accumulate_projected_line_density_by_source(self, cell_gdf, line_gdf, progress_every: int = 1000):
        """
        Accumulate Liu-style projected line-length density by looping over source
        linework, not over all grid cells.

        This is the key efficiency improvement.

        For each source line:
          1. query the candidate T-cell spatial index;
          2. clip the source line to each intersecting T-cell polygon;
          3. accumulate projected line length along model-i and model-j;
          4. normalise by local CICE HTE/HTN.

        Because the loop is source-driven, form factors can only be created where
        the actual high-resolution source linework intersects candidate CICE cells.
        """
        ncell = len(cell_gdf)
        sum_i_m = np.zeros(ncell, dtype="float64")
        sum_j_m = np.zeros(ncell, dtype="float64")
        n_hits  = np.zeros(ncell, dtype="int32")
        cell_geoms = list(cell_gdf.geometry)
        cell_sidx  = cell_gdf.sindex
        eix  = cell_gdf["eix"].to_numpy(dtype="float64")
        eiy  = cell_gdf["eiy"].to_numpy(dtype="float64")
        ejx  = cell_gdf["ejx"].to_numpy(dtype="float64")
        ejy  = cell_gdf["ejy"].to_numpy(dtype="float64")
        dx_m = cell_gdf["dx_m"].to_numpy(dtype="float64")
        dy_m = cell_gdf["dy_m"].to_numpy(dtype="float64")
        nsrc = len(line_gdf)
        for src_pos, src_geom in enumerate(line_gdf.geometry):
            if progress_every and src_pos > 0 and src_pos % progress_every == 0:
                self._log(f"Processed {src_pos:,} / {nsrc:,} source line geometries")
            if src_geom is None or src_geom.is_empty:
                continue
            # Candidate CICE cells whose polygon bounding boxes intersect this
            # source line's bounding box.
            cell_posns = list(cell_sidx.intersection(src_geom.bounds))
            if not cell_posns:
                continue
            for cell_pos in cell_posns:
                cell_poly = cell_geoms[cell_pos]
                if cell_poly is None or cell_poly.is_empty:
                    continue
                if not src_geom.intersects(cell_poly):
                    continue
                clipped = src_geom.intersection(cell_poly)
                if clipped is None or clipped.is_empty:
                    continue
                any_hit = False
                for line in self._iter_line_parts(clipped):
                    coords = np.asarray(line.coords, dtype="float64")
                    if coords.shape[0] < 2:
                        continue
                    vec = coords[1:, :] - coords[:-1, :]
                    finite = np.isfinite(vec).all(axis=1)
                    if not finite.any():
                        continue
                    vec = vec[finite]
                    # Project clipped source-line segment vectors onto local CICE
                    # logical-i and logical-j directions. Absolute values follow
                    # the Liu-style projected-length density.
                    proj_i = vec[:, 0] * eix[cell_pos] + vec[:, 1] * eiy[cell_pos]
                    proj_j = vec[:, 0] * ejx[cell_pos] + vec[:, 1] * ejy[cell_pos]
                    sum_i_m[cell_pos] += np.abs(proj_i).sum()
                    sum_j_m[cell_pos] += np.abs(proj_j).sum()
                    any_hit = True
                if any_hit:
                    n_hits[cell_pos] += 1
        FFx = np.divide(sum_i_m, dx_m, out=np.zeros_like(sum_i_m), where=dx_m > 0)
        FFy = np.divide(sum_j_m, dy_m, out=np.zeros_like(sum_j_m), where=dy_m > 0)
        self._log("source-driven line-density accumulation finished: "
                  f"source lines={nsrc:,}; "
                  f"candidate cells={ncell:,}; "
                  f"intersected cells={int(np.count_nonzero(n_hits > 0)):,}; "
                  f"total projected i-length={np.nansum(sum_i_m):.4e} m; "
                  f"total projected j-length={np.nansum(sum_j_m):.4e} m; "
                  f"FFx max={np.nanmax(FFx):.4e}; "
                  f"FFy max={np.nanmax(FFy):.4e}")
        return FFx, FFy, sum_i_m, sum_j_m, n_hits

    def _accumulate_polygon_area_fraction_by_source(self, cell_gdf, polygon_gdf,
                                                    progress_every: int = 1000,
                                                    clip_area_fraction: bool = True):
        """
        Accumulate polygon area fraction by looping over source polygons.

        This is useful for GIB diagnostics:
            area_fraction = area(GIB polygons intersecting T cell) / T-cell area

        If there are dozens of GIBs in a grid cell, both:
            GIB_n_polygon_hits
            GIB_area_frac
        will reflect that roughness/occupancy.
        """
        ncell = len(cell_gdf)
        area_m2     = np.zeros(ncell, dtype="float64")
        n_poly_hits = np.zeros(ncell, dtype="int32")
        cell_geoms = list(cell_gdf.geometry)
        cell_sidx  = cell_gdf.sindex
        nsrc = len(polygon_gdf)
        for src_pos, poly in enumerate(polygon_gdf.geometry):
            if progress_every and src_pos > 0 and src_pos % progress_every == 0:
                self._log(f"Processed {src_pos:,} / {nsrc:,} source polygon geometries")
            if poly is None or poly.is_empty:
                continue
            cell_posns = list(cell_sidx.intersection(poly.bounds))
            if not cell_posns:
                continue
            for cell_pos in cell_posns:
                cell_poly = cell_geoms[cell_pos]
                if not poly.intersects(cell_poly):
                    continue
                clipped = poly.intersection(cell_poly)
                if clipped is None or clipped.is_empty:
                    continue
                a = clipped.area
                if np.isfinite(a) and a > 0:
                    area_m2[cell_pos] += a
                    n_poly_hits[cell_pos] += 1
        cell_area = cell_gdf["cell_area"].to_numpy(dtype="float64")
        area_frac = np.divide(area_m2, cell_area, out = np.zeros_like(area_m2), where = cell_area > 0)
        if clip_area_fraction:
            area_frac = np.clip(area_frac, 0.0, 1.0)
        self._log("source-driven polygon area-fraction accumulation finished: "
                  f"source polygons={nsrc:,}; "
                  f"candidate cells={ncell:,}; "
                  f"area-hit cells={int(np.count_nonzero(n_poly_hits > 0)):,}; "
                  f"total intersected area={np.nansum(area_m2):.4e} m2; "
                  f"max intersected area={np.nanmax(area_m2):.4e} m2; "
                  f"area_frac max={np.nanmax(area_frac):.4e}; "
                  f"max polygon hits per cell={int(np.nanmax(n_poly_hits))}")
        return area_frac, area_m2, n_poly_hits

    #----------------------------------------------------------------------
    # NetCDF write safely helpers
    #----------------------------------------------------------------------
    @staticmethod
    def _netcdf_safe_attr_value(value):
        """
        Convert Python / NumPy metadata values into NetCDF-safe attribute values.

        netCDF4 cannot write several common Python metadata types as attributes,
        especially bool / np.bool_. For example:

            True -> TypeError: illegal data type ... got b1

        This helper keeps attributes readable and robust by converting unsupported
        types to strings or simple numeric scalars.

        Rules
        -----
        * bool / np.bool_ -> "true" or "false"
        * None            -> "None"
        * Path            -> string path
        * NumPy scalars   -> native Python scalars, then rechecked
        * lists/tuples    -> comma-separated string
        * dicts           -> JSON-like string
        * unsupported     -> string representation
        """
        import json
        from pathlib import Path
        # Explicitly catch Python and NumPy booleans before integer handling.
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if value is None:
            return "None"
        if isinstance(value, Path):
            return str(value)
        # Convert NumPy scalar values to native Python values.
        if isinstance(value, np.generic):
            return FormFactors._netcdf_safe_attr_value(value.item())
        # Strings and simple numeric types are safe.
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            return value
        # Lists/tuples can contain mixed types. Keep them readable.
        if isinstance(value, (list, tuple)):
            return ", ".join(str(FormFactors._netcdf_safe_attr_value(v)) for v in value)
        # Dictionaries are useful provenance, but not valid NetCDF attributes.
        if isinstance(value, dict):
            try:
                clean = {str(k): FormFactors._netcdf_safe_attr_value(v) for k, v in value.items()}
                return json.dumps(clean)
            except Exception:
                return str(value)
        # NumPy arrays as attributes are brittle unless purely numeric.
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return FormFactors._netcdf_safe_attr_value(value.item())
            return ", ".join(str(FormFactors._netcdf_safe_attr_value(v)) for v in value.ravel())
        return str(value)

    @classmethod
    def _netcdf_safe_attrs(cls, attrs: dict | None) -> dict:
        """
        Sanitize an attribute dictionary for NetCDF writing.
        """
        if attrs is None:
            return {}
        return {str(k): cls._netcdf_safe_attr_value(v) for k, v in attrs.items()}

    @classmethod
    def _sanitize_dataset_for_netcdf(cls, ds: xr.Dataset) -> xr.Dataset:
        """
        Return a dataset with NetCDF-safe global and variable attributes.

        This does not alter data values. It only cleans metadata.
        """
        ds = ds.copy()
        ds.attrs = cls._netcdf_safe_attrs(ds.attrs)
        for v in ds.variables:
            ds[v].attrs = cls._netcdf_safe_attrs(ds[v].attrs)
        return ds

    def _default_FF_encoding(self, ds: xr.Dataset) -> dict:
        """
        Build conservative NetCDF encoding for FF products.

        Compression is optional but useful for full 1440 x 1080 fields. Integer
        diagnostic fields are kept as int32; all floating fields are written as
        float32 unless already explicitly integer.

        `_FillValue` is disabled for coordinate variables ni/nj. For data variables,
        float fields get NaN fill values; integer fields get no fill value.
        """
        complevel = int(getattr(self.spec, "netcdf_compression", 1))
        enc = {}
        for name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.integer):
                enc[name] = {"zlib": True, "complevel": complevel, "dtype": "int32", "_FillValue": None}
            else:
                enc[name] = {"zlib": True, "complevel": complevel, "dtype": "float32", "_FillValue": np.float32(np.nan)}
        for name in ds.coords:
            enc[name] = {"_FillValue": None}
        return enc

    def _write_FF_dataset(self, ds: xr.Dataset,
                          P_out: str | Path,
                          mode: str = "w",
                          engine: str = "netcdf4",
                          encoding: dict | None = None) -> xr.Dataset:
        """
        Safely write a form-factor dataset to NetCDF.

        Use this for all coastline, GIB, and combined FF products instead of
        calling:

            ds.to_netcdf(P_out, mode="w")

        directly.

        This prevents write failures from unsupported NetCDF attribute types such
        as bool, pathlib.Path, None, dict, list, NumPy scalar bools, etc.
        """
        P_out = Path(P_out)
        P_out.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("sanitising Dataset attributes for NetCDF write")
        ds_safe = self._sanitize_dataset_for_netcdf(ds)
        if encoding is None:
            encoding = self._default_FF_encoding(ds_safe)
        self.logger.info(f"writing NetCDF safely: {P_out}")
        ds_safe.to_netcdf(P_out, mode = mode, engine = engine, encoding = encoding)
        self.logger.info(f"NetCDF write complete: {P_out}")
        return ds_safe

    def build_FF_from_Hres_coast_Liu(self,
                                     P_Hres_cst: str | Path | None = None,
                                     P_out     : str | Path | None = None,
                                     overwrite : bool = False,
                                     clip_max  : float | None = None,
                                     hemisphere: str = "SH",
                                     source_lat_limit: float | None = None,
                                     grid_lat_pad_deg: float = 3.0,
                                     grid_lon_pad_deg: float = 3.0,
                                     use_ocean_mask: bool = True,
                                     use_index_half_hint: bool = False,
                                     use_coastal_neighbour_filter: bool = False,
                                     coastal_neighbour_radius: int = 2,
                                     progress_every: int = 1000) -> xr.Dataset:
        """
        Build coastline-derived lateral-drag form factors (F2: Liu et. al 2022)

        This method constructs a CICE-readable, T-cell-centred geometric form-factor
        field from high-resolution coastline or ice-front linework. It is intended as
        the coastline analogue of the form-factor construction described by
        Liu et al. (2022), where sub-grid coastline geometry is converted into
        directional length-density fields on the model grid.

        For each CICE T cell, the method clips the high-resolution source linework to
        the cell polygon and accumulates the absolute projected source length along
        the local model-grid i and j directions:

            FFx = sum(abs(projected source length along model-i)) / HTE
            FFy = sum(abs(projected source length along model-j)) / HTN

        where HTE and HTN are the local CICE grid metrics. The resulting fields are
        dimensionless. Values may exceed 1 in cells containing complex or convoluted
        coastline/ice-front geometry, and this is expected for a true length-density
        form factor. For this reason, `clip_max=None` is the recommended default.

        The output fields are written as T-cell source fields, not velocity-face
        fields. CICE subsequently reads:

            FFx -> F2x_in -> mapped internally to F2E
            FFy -> F2y_in -> mapped internally to F2N

        inside `load_F2_form_factors()`. Therefore this method should not pre-map the
        form factors to E or N faces. The NetCDF output follows the existing shuga/CICE
        convention of variables written as:

            FFx(nj, ni)
            FFy(nj, ni)

        using Python-side arrays shaped `(nj, ni)`.

        Efficiency strategy
        -------------------
        The direct, brute-force implementation would loop over every valid CICE T cell
        and test for intersections with the high-resolution source geometry. On a
        global grid this is unnecessarily expensive, especially because Antarctic
        coastline form factors can only be produced near the Antarctic source
        geometry.

        This method therefore applies several filters before performing exact geometry
        intersections:

        1. Source-domain filtering
           The vector source file is first filtered to the requested polar domain.
           For Antarctic builds, the default source filter keeps only source features
           reaching south of 60 deg S. This is deliberately applied to the source
           geometry itself, not only to the model grid. If a broader coastline dataset
           is accidentally supplied, this prevents unrelated coastlines such as South
           America, Australia, South Africa, or other non-Antarctic landmasses from
           generating form factors.

        2. Source-envelope grid filtering
           Candidate CICE T cells are restricted to the longitude/latitude envelope of
           the filtered high-resolution source geometry, with a small configurable
           padding. This reduces the number of grid cells for which polygons are built
           while still allowing the exact projected intersection test to determine the
           final form-factor assignment.

        3. Optional CICE ocean-mask filtering
           If `use_ocean_mask=True`, only CICE ocean T cells are retained as candidate
           cells. This is usually appropriate for lateral-drag source fields, because
           CICE ultimately applies the mapped F2 fields at active velocity points and
           masks land velocity points after reading the file.

        4. Optional grid-index half-domain hint
           If `use_index_half_hint=True`, the method also applies a crude hemispheric
           grid-index prefilter. For the present 1440 x 1080 global grid, Antarctic
           cells are expected to occur in the southern half of the grid. This can save
           time, but it is deliberately optional because it is grid-layout dependent.
           Latitude filtering is safer and should remain the primary polar-domain
           filter.

        5. Optional coarse coastal-neighbour filter
           If `use_coastal_neighbour_filter=True`, the candidate set is further
           restricted to ocean cells close to the coarse CICE land mask. This can
           greatly reduce the number of candidate cells, but it is not recommended for
           the first production build. The high-resolution Antarctic coastline or
           ice-front geometry may not coincide exactly with the coarse CICE `kmt`
           land/ocean transition. Using the coarse coastal-neighbour filter too early
           could remove valid high-resolution coastline/ice-front source cells.

        Final assignment rule
        ---------------------
        Even after the candidate filters above, a T cell receives a non-zero form
        factor only if the projected high-resolution source linework actually
        intersects that candidate T-cell polygon. This exact intersection step is what
        ensures that form factors are generated from the supplied coastline source
        geometry, rather than from the coarse model landmask alone.

        Recommended Antarctic defaults
        ------------------------------
        For the current Antarctic landfast-ice application, the recommended defaults
        are:

            hemisphere                   = "SH"
            source_lat_limit             = None      # defaults to -60 deg
            grid_lat_pad_deg             = 3.0
            grid_lon_pad_deg             = 3.0
            use_ocean_mask               = True
            use_index_half_hint          = False
            use_coastal_neighbour_filter = False
            clip_max                     = None

        These settings prioritise correctness and traceability. They restrict the
        calculation to the Antarctic source geometry and CICE ocean cells, but do not
        allow the coarse CICE landmask to decide where high-resolution coastline or
        ice-front geometry is permitted to contribute.

        Parameters
        ----------
        P_Hres_cst : str or pathlib.Path, optional
            Path to the high-resolution coastline or ice-front vector file. If None,
            `self.pth_cfg.high_res_coast_file_path` is used. The file may contain
            line, multiline, polygon, or multipolygon geometry. Polygon sources are
            converted to boundary linework before length-density accumulation.

        P_out : str or pathlib.Path, optional
            Path to the output NetCDF form-factor file. If None,
            `self.pth_cfg.coast_form_factors_path` is used.

        overwrite : bool, default False
            If False and `P_out` already exists, the existing dataset is opened and
            returned without rebuilding. If True, the form-factor product is rebuilt
            and the output file is overwritten.

        clip_max : float or None, default None
            Optional upper bound applied to FFx and FFy before writing. For Liu-style
            length-density form factors, None is recommended because physically
            meaningful values can exceed 1 in cells with convoluted source geometry.
            Use a finite value only for explicitly bounded sensitivity experiments.

        hemisphere : {"SH", "NH"}, default "SH"
            Polar domain to build. "SH" is the Antarctic/Southern Hemisphere case;
            "NH" is the Arctic/Northern Hemisphere case. Several aliases may be
            normalised internally if `_normalise_hemisphere()` supports them.

        source_lat_limit : float or None, default None
            Latitude threshold used to filter the source vector file before projection.
            If None, Antarctic builds default to -60 deg and Arctic builds default to
            +60 deg. For Antarctic builds, only source features reaching south of this
            latitude are retained.

        grid_lat_pad_deg : float, default 3.0
            Latitude padding applied when selecting candidate CICE T cells around the
            filtered source-envelope. This is a prefilter only; final assignment still
            requires exact projected source/cell intersection.

        grid_lon_pad_deg : float, default 3.0
            Longitude padding applied when selecting candidate CICE T cells around the
            filtered source-envelope. This is mainly useful when the source geometry is
            regional rather than circum-Antarctic.

        use_ocean_mask : bool, default True
            If True, only CICE ocean T cells are retained as candidates. This uses the
            model grid mask loaded through `CICEGridwork`. This filter should usually
            remain enabled for production F2 files.

        use_index_half_hint : bool, default False
            If True, applies an additional grid-index-based hemispheric prefilter. This
            may be useful on known global grid layouts but is intentionally disabled by
            default because it is less robust than latitude/source-geometry filtering.

        use_coastal_neighbour_filter : bool, default False
            If True, restricts candidates to ocean cells near the coarse CICE landmask.
            This can improve speed but may remove valid high-resolution source geometry
            if the coarse landmask and high-resolution coastline/ice-front product are
            displaced relative to each other. Recommended default is False.

        coastal_neighbour_radius : int, default 2
            Number of binary-dilation iterations used by the optional coarse
            coastal-neighbour filter. Only used when
            `use_coastal_neighbour_filter=True`.

        progress_every : int, default 1000
            Interval for progress logging during source-driven accumulation. The count
            refers to source line geometries, not grid cells.

        Returns
        -------
        xr.Dataset
            Dataset containing at minimum:

                FFx(nj, ni)
                    T-cell x/i-direction coastline form factor.

                FFy(nj, ni)
                    T-cell y/j-direction coastline form factor.

                lon(nj, ni), lat(nj, ni)
                    T-cell longitude and latitude.

            The dataset may also include diagnostic fields such as:

                coast_line_i_m(nj, ni)
                    Summed clipped source length projected onto the local model-i
                    direction, in metres.

                coast_line_j_m(nj, ni)
                    Summed clipped source length projected onto the local model-j
                    direction, in metres.

                coast_n_source_hits(nj, ni)
                    Number of source line geometries intersecting each T cell.

        Notes
        -----
        This method should be interpreted as a source-driven, within-cell,
        projected-length-density calculation. It is distinct from the earlier
        nearest-source distance-taper method. No taper radius is used here. If spatial
        spreading of source influence is desired, it should be applied as a separate,
        explicit post-processing step so that the Liu-style source strength and the
        chosen spreading operator remain scientifically separable.
        """
        if P_Hres_cst is None:
            P_Hres_cst = self.pth_cfg.high_res_coast_file_path
        if P_out is None:
            P_out = self.pth_cfg.coast_form_factors_path
        P_out = Path(P_out)
        if P_out.exists() and not overwrite:
            return xr.open_dataset(P_out)
        hemisphere = self._normalise_hemisphere(hemisphere)
        self.logger.info(f"reading: {P_Hres_cst}")
        line_gdf = self._read_projected_linework(P_vector              = P_Hres_cst,
                                                 hemisphere            = hemisphere,
                                                 source_lat_limit      = source_lat_limit,
                                                 source_lat_buffer_deg = 0.0)
        self.logger.info("building source-filtered candidate T-cell polygons")
        cell_gdf, grid, tlon, tlat = self._build_projected_tcell_gdf_for_source(source_gdf_projected          = line_gdf,
                                                                                hemisphere                   = hemisphere,
                                                                                source_lat_limit             = source_lat_limit,
                                                                                grid_lat_pad_deg             = grid_lat_pad_deg,
                                                                                grid_lon_pad_deg             = grid_lon_pad_deg,
                                                                                use_ocean_mask               = use_ocean_mask,
                                                                                use_index_half_hint          = use_index_half_hint,
                                                                                use_coastal_neighbour_filter = use_coastal_neighbour_filter,
                                                                                coastal_neighbour_radius     = coastal_neighbour_radius)
        self.logger.info("accumulating projected line density by source geometry")
        FFx_vec, FFy_vec, line_i_m, line_j_m, n_hits = (self._accumulate_projected_line_density_by_source(cell_gdf = cell_gdf, line_gdf = line_gdf, progress_every = progress_every))
        self.logger.info("line-density accumulation complete: "
                         f"FFx_vec min/max/sum = "
                         f"{np.nanmin(FFx_vec):.4e} / {np.nanmax(FFx_vec):.4e} / {np.nansum(FFx_vec):.4e}; "
                         f"FFy_vec min/max/sum = "
                         f"{np.nanmin(FFy_vec):.4e} / {np.nanmax(FFy_vec):.4e} / {np.nansum(FFy_vec):.4e}; "
                         f"nonzero cells = {int(np.count_nonzero((FFx_vec > 0) | (FFy_vec > 0))):,}")
        self.logger.info("diagnostic source-length vectors: "
                         f"line_i_m sum/max = {np.nansum(line_i_m):.4e} / {np.nanmax(line_i_m):.4e}; "
                         f"line_j_m sum/max = {np.nansum(line_j_m):.4e} / {np.nanmax(line_j_m):.4e}; "
                         f"source-hit cells = {int(np.count_nonzero(n_hits > 0)):,}")
        if clip_max is not None:
            self.logger.info(f"clipping FFx/FFy vectors to [0, {float(clip_max):.4e}]")
            FFx_vec = np.clip(FFx_vec, 0.0, float(clip_max))
            FFy_vec = np.clip(FFy_vec, 0.0, float(clip_max))
        else:
            self.logger.info("clip_max=None; preserving Liu-style values greater than 1 where present")
        self.logger.info("scattering FF vectors back to full CICE T-grid arrays")
        FFx = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, FFx_vec)
        FFy = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, FFy_vec)
        self.logger.info("full-grid FF arrays created: "
                         f"FFx shape={FFx.shape}, min/max/sum="
                         f"{np.nanmin(FFx):.4e}/{np.nanmax(FFx):.4e}/{np.nansum(FFx):.4e}; "
                         f"FFy shape={FFy.shape}, min/max/sum="
                         f"{np.nanmin(FFy):.4e}/{np.nanmax(FFy):.4e}/{np.nansum(FFy):.4e}; "
                         f"nonzero grid cells={int(np.count_nonzero((FFx > 0) | (FFy > 0))):,}")
        self.logger.info("creating output xarray Dataset")
        ds_out = self._empty_FF_dataset(tlon, tlat, FFx, FFy,
                                        attrs = {"title"        : "Liu-style coastline-derived lateral drag form factors",
                                                 "source_vector": str(P_Hres_cst),
                                                 "grid_source"  : str(grid.source_path),
                                                 "proj_crs"     : str(self.spec.proj_crs),
                                                 "hemisphere"   : hemisphere,
                                                 "source_lat_limit": self._polar_source_lat_limit(hemisphere, source_lat_limit),
                                                 "method"       : ("source-filtered, source-driven within-cell projected coastline-length density"),
                                                 "normalisation": ("FFx=sum(abs(projected source length along model-i))/HTE; "
                                                                   "FFy=sum(abs(projected source length along model-j))/HTN"),
                                                 "use_ocean_mask": bool(use_ocean_mask),
                                                 "use_index_half_hint": bool(use_index_half_hint),
                                                 "use_coastal_neighbour_filter": bool(use_coastal_neighbour_filter),
                                                 "coastal_neighbour_radius": int(coastal_neighbour_radius),
                                                 "clip_max": "None" if clip_max is None else float(clip_max)})
        self.logger.info("adding coastline diagnostic fields to output Dataset")
        coast_line_i_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, line_i_m)
        coast_line_j_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, line_j_m)
        coast_hits_grid   = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, n_hits, dtype="int32")
        ds_out["coast_line_i_m"] = (("nj", "ni"), coast_line_i_grid)
        ds_out["coast_line_j_m"] = (("nj", "ni"), coast_line_j_grid)
        ds_out["coast_n_source_hits"] = (("nj", "ni"), coast_hits_grid)
        ds_out["coast_line_i_m"].attrs.update({"long_name": "summed coastline length projected onto local model-i direction", "units": "m"})
        ds_out["coast_line_j_m"].attrs.update({"long_name": "summed coastline length projected onto local model-j direction", "units": "m"})
        ds_out["coast_n_source_hits"].attrs.update({"long_name": "number of source coastline geometries intersecting each T cell", "units": "1"})
        self.logger.info("diagnostic grids added: "
                         f"coast_line_i_m sum/max={np.nansum(coast_line_i_grid):.4e}/{np.nanmax(coast_line_i_grid):.4e}; "
                         f"coast_line_j_m sum/max={np.nansum(coast_line_j_grid):.4e}/{np.nanmax(coast_line_j_grid):.4e}; "
                         f"coast_n_source_hits max={int(np.nanmax(coast_hits_grid))}")
        P_out.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"writing Liu-style coastline form factors to NetCDF: {P_out}")
        ds_out = self._write_FF_dataset(ds_out, P_out, mode = "w")
        self.logger.info(f"NetCDF write complete: {P_out}")
        self.logger.info("final dataset summary: "
                         f"dims={dict(ds_out.dims)}; "
                         f"data_vars={list(ds_out.data_vars)}")
        self._log(f"Wrote Liu-style coastline form factors: {P_out}")
        return ds_out

    def build_FF_from_GIB_perimeter(self,
                                    P_GIB    : str | Path | None = None,
                                    P_out    : str | Path | None = None,
                                    overwrite: bool = False,
                                    include_area_fraction: bool = True,
                                    area_component_mode : str = "diagnostic",
                                    area_weight         : float = 1.0,
                                    clip_max            : float | None = None,
                                    clip_area_fraction  : bool = True,
                                    hemisphere          : str = "SH",
                                    source_lat_limit    : float | None = None,
                                    grid_lat_pad_deg    : float = 3.0,
                                    grid_lon_pad_deg    : float = 3.0,
                                    use_ocean_mask      : bool = True,
                                    use_index_half_hint : bool = False,
                                    use_coastal_neighbour_filter: bool = False,
                                    coastal_neighbour_radius: int = 2,
                                    progress_every      : int = 1000) -> xr.Dataset:
        """
        Build grounded-iceberg-derived lateral-drag form factors using iceberg
        perimeters as the source geometry.

        This method constructs a CICE-readable, T-cell-centred geometric form-factor
        field from grounded-iceberg vector geometry. It is the grounded-iceberg
        analogue of the Liu-style coastline form-factor builder, but instead of using
        coastline or ice-front linework, it treats the perimeter of each grounded
        iceberg as the potential lateral-contact boundary for sea ice.

        For each candidate CICE T cell, the method clips the grounded-iceberg
        perimeter linework to the cell polygon and accumulates the absolute projected
        perimeter length along the local model-grid i and j directions:

            FFx = sum(abs(projected GIB perimeter length along model-i)) / HTE
            FFy = sum(abs(projected GIB perimeter length along model-j)) / HTN

        where HTE and HTN are the local CICE grid metrics. The resulting fields are
        dimensionless. Values may exceed 1 when one or more grounded-iceberg
        perimeters contribute substantial contact length within a single model grid
        cell. This is expected for a true perimeter-length-density form factor, and
        therefore `clip_max=None` is the recommended default.

        The output fields are written as T-cell source fields, not velocity-face
        fields. CICE subsequently reads:

            FFx -> F2x_in -> mapped internally to F2E
            FFy -> F2y_in -> mapped internally to F2N

        inside `load_F2_form_factors()`. Therefore this method should not pre-map the
        form factors to E or N faces. The NetCDF output follows the existing
        shuga/CICE convention of variables written as:

            FFx(nj, ni)
            FFy(nj, ni)

        using Python-side arrays shaped `(nj, ni)`.

        Scientific interpretation
        -------------------------
        The perimeter-derived fields represent sub-grid grounded-iceberg contact
        geometry. They are intended to describe how much unresolved grounded-iceberg
        boundary length is available to exert lateral resistance on sea ice within
        each model grid cell.

        This is distinct from the earlier nearest-grounded-iceberg distance-taper
        method. No taper radius is used here. A grid cell receives a non-zero
        perimeter-derived form factor only if the grounded-iceberg perimeter actually
        intersects the candidate T-cell polygon.

        Optional area-fraction diagnostic
        ---------------------------------
        The method can also calculate grounded-iceberg area fraction within each CICE
        T cell:

            GIB_area_frac = area(GIB polygons intersecting T cell) / area(T cell)

        This quantity has a different physical interpretation from perimeter length
        density:

            perimeter length density
                Measures potential contact-boundary roughness.

            area fraction
                Measures obstacle occupancy within the model grid cell.

        The area fraction is useful because there may be many grounded icebergs within
        a single CICE grid cell. A cell containing dozens of small grounded icebergs
        may have a large perimeter density even if its total grounded-iceberg area
        fraction is modest. Conversely, a cell containing one large grounded iceberg
        may have a large area fraction but comparatively less perimeter complexity.

        By default, `area_component_mode="diagnostic"` stores area fraction as an
        output diagnostic but does not add it to FFx or FFy. This is recommended for
        the first production-quality form-factor products, because perimeter density
        and area occupancy should be inspected separately before deciding whether the
        area term should enter the dynamical drag field.

        Area-component modes
        --------------------
        The optional `area_component_mode` controls whether GIB area fraction affects
        the final FFx/FFy fields:

            "diagnostic"
                Store GIB_area_frac, GIB_area_m2, and GIB_n_polygon_hits as diagnostic
                variables only. FFx and FFy are based only on perimeter length density.
                This is the recommended default.

            "add"
                Add an isotropic area-fraction contribution to both components:

                    FFx = FFx_perimeter + area_weight * GIB_area_frac
                    FFy = FFy_perimeter + area_weight * GIB_area_frac

                This is experimental and should be interpreted as perimeter roughness
                plus obstacle occupancy.

            "replace"
                Replace the perimeter-derived form factors with an isotropic
                area-fraction field:

                    FFx = area_weight * GIB_area_frac
                    FFy = area_weight * GIB_area_frac

                This is also experimental. It treats grounded-iceberg occupancy,
                rather than contact length, as the source of lateral resistance.

        Efficiency strategy
        -------------------
        The direct brute-force approach would loop over every valid CICE T cell and
        test for intersections with every grounded-iceberg feature. That is expensive
        on the global grid and unnecessary because GIB form factors can only occur
        where the supplied GIB source geometry exists.

        This method therefore uses a source-driven, polar-filtered workflow:

        1. Source-domain filtering
           The GIB vector file is filtered to the requested polar domain before
           projection. For Antarctic builds, the default source filter keeps only
           source features reaching south of 60 deg S. This prevents any accidental
           non-polar or unrelated vector features from entering the form-factor
           product.

        2. Source perimeter extraction
           Polygon and multipolygon GIB features are converted to boundary linework.
           The perimeter, not the centroid, is used as the primary lateral-contact
           source geometry.

        3. Source-envelope grid filtering
           Candidate CICE T cells are restricted to the longitude/latitude envelope of
           the filtered GIB source geometry, with configurable latitude and longitude
           padding. This reduces the number of T-cell polygons that need to be built.

        4. Optional CICE ocean-mask filtering
           If `use_ocean_mask=True`, only CICE ocean T cells are retained as candidate
           cells. This is usually appropriate for the F2 input product because CICE
           later masks inactive land velocity points after reading and mapping the
           fields.

        5. Optional grid-index half-domain hint
           If `use_index_half_hint=True`, an additional hemispheric grid-index
           prefilter is applied. This can be useful on known global grid layouts but
           is disabled by default because it is less robust than latitude/source
           filtering.

        6. Optional coarse coastal-neighbour filter
           If `use_coastal_neighbour_filter=True`, candidates are further restricted
           to ocean cells near the coarse CICE landmask. This can improve speed, but
           it is not recommended for the first production build because grounded
           icebergs may occur offshore and should not be filtered solely by proximity
           to the coarse continental landmask.

        7. Source-driven accumulation
           The final accumulation loops over grounded-iceberg source geometries and
           queries the candidate T-cell spatial index. Form factors are only assigned
           after exact projected intersection between the GIB perimeter and the
           candidate T-cell polygon.

        Final assignment rule
        ---------------------
        Even after all prefilters, a grid cell receives a non-zero perimeter-derived
        form factor only where the supplied grounded-iceberg perimeter linework
        actually intersects that cell. The CICE landmask and candidate filters are
        used for efficiency and validity, not as independent sources of form-factor
        geometry.

        Recommended Antarctic defaults
        ------------------------------
        For the current Antarctic grounded-iceberg lateral-drag application, the
        recommended defaults are:

            hemisphere                   = "SH"
            source_lat_limit             = None          # defaults to -60 deg
            grid_lat_pad_deg             = 3.0
            grid_lon_pad_deg             = 3.0
            use_ocean_mask               = True
            use_index_half_hint          = False
            use_coastal_neighbour_filter = False
            include_area_fraction        = True
            area_component_mode          = "diagnostic"
            area_weight                  = 1.0
            clip_max                     = None
            clip_area_fraction           = True

        These settings prioritise traceability. They build the primary FFx/FFy fields
        from GIB perimeter length density, retain GIB area fraction as an audit
        diagnostic, and avoid using the coarse CICE landmask to suppress legitimate
        offshore grounded-iceberg source geometry.

        Parameters
        ----------
        P_GIB : str or pathlib.Path, optional
            Path to the grounded-iceberg vector file. If None,
            `self.pth_cfg.grounded_iceberg_file_path` is used. Polygon and multipolygon
            features are expected for the preferred perimeter and area-fraction
            calculations. Linework can also be used for perimeter-only calculations.

        P_out : str or pathlib.Path, optional
            Path to the output NetCDF form-factor file. If None,
            `self.pth_cfg.grounded_iceberg_form_factors_path` is used.

        overwrite : bool, default False
            If False and `P_out` already exists, the existing dataset is opened and
            returned without rebuilding. If True, the product is rebuilt and the output
            file is overwritten.

        include_area_fraction : bool, default True
            If True, calculate grounded-iceberg area-fraction diagnostics in addition
            to perimeter-derived FFx/FFy. If False, only perimeter length-density
            fields are generated.

        area_component_mode : {"diagnostic", "add", "replace"}, default "diagnostic"
            Controls whether the area-fraction field affects FFx/FFy. The recommended
            default is "diagnostic".

        area_weight : float, default 1.0
            Scalar multiplier applied to GIB_area_frac when
            `area_component_mode="add"` or `area_component_mode="replace"`. Ignored
            when `area_component_mode="diagnostic"`.

        clip_max : float or None, default None
            Optional upper bound applied to FFx and FFy before writing. For
            perimeter-length-density form factors, None is recommended because values
            may legitimately exceed 1 where grounded-iceberg perimeter density is high.

        clip_area_fraction : bool, default True
            If True, clip GIB_area_frac to the range [0, 1]. This is usually desirable
            for an occupancy diagnostic, particularly if source polygons overlap or
            contain small topology artefacts.

        hemisphere : {"SH", "NH"}, default "SH"
            Polar domain to build. "SH" is the Antarctic/Southern Hemisphere case;
            "NH" is the Arctic/Northern Hemisphere case. Several aliases may be
            normalised internally if `_normalise_hemisphere()` supports them.

        source_lat_limit : float or None, default None
            Latitude threshold used to filter the GIB source file before projection.
            If None, Antarctic builds default to -60 deg and Arctic builds default to
            +60 deg.

        grid_lat_pad_deg : float, default 3.0
            Latitude padding applied when selecting candidate CICE T cells around the
            filtered GIB source envelope. This is a prefilter only; final assignment
            still requires exact projected intersection.

        grid_lon_pad_deg : float, default 3.0
            Longitude padding applied when selecting candidate CICE T cells around the
            filtered GIB source envelope. This is mainly useful for regional source
            products.

        use_ocean_mask : bool, default True
            If True, only CICE ocean T cells are retained as candidates. This should
            usually remain enabled for production F2 files.

        use_index_half_hint : bool, default False
            If True, applies an additional grid-index-based hemispheric prefilter. This
            may improve speed on a known grid but is disabled by default because it is
            grid-layout dependent.

        use_coastal_neighbour_filter : bool, default False
            If True, restricts candidate cells to ocean cells near the coarse CICE
            landmask. This is not recommended for the default GIB build because
            grounded icebergs can occur offshore and should not be removed simply
            because they are not adjacent to the coarse landmask.

        coastal_neighbour_radius : int, default 2
            Number of binary-dilation iterations used by the optional coarse
            coastal-neighbour filter. Only used when
            `use_coastal_neighbour_filter=True`.

        progress_every : int, default 1000
            Interval for progress logging during source-driven perimeter and polygon
            accumulation. The count refers to source geometries, not grid cells.

        Returns
        -------
        xr.Dataset
            Dataset containing at minimum:

                FFx(nj, ni)
                    T-cell x/i-direction grounded-iceberg form factor.

                FFy(nj, ni)
                    T-cell y/j-direction grounded-iceberg form factor.

                lon(nj, ni), lat(nj, ni)
                    T-cell longitude and latitude.

            The dataset may also include diagnostic fields such as:

                GIB_perimeter_i_m(nj, ni)
                    Summed clipped GIB perimeter length projected onto the local
                    model-i direction, in metres.

                GIB_perimeter_j_m(nj, ni)
                    Summed clipped GIB perimeter length projected onto the local
                    model-j direction, in metres.

                GIB_n_perimeter_hits(nj, ni)
                    Number of GIB perimeter source geometries intersecting each T cell.

                GIB_area_frac(nj, ni)
                    Grounded-iceberg area fraction within each T cell.

                GIB_area_m2(nj, ni)
                    Grounded-iceberg polygon area intersecting each T cell, in square
                    metres.

                GIB_n_polygon_hits(nj, ni)
                    Number of GIB polygon features intersecting each T cell.

        Notes
        -----
        This method should be interpreted as a source-driven, within-cell,
        projected-perimeter-density calculation. It does not use grounded-iceberg
        centroids, nearest-neighbour assignment, or a taper radius. If spatial
        spreading of grounded-iceberg influence is desired, it should be applied as a
        separate, explicit post-processing step so that the source-strength calculation
        and the chosen spreading operator remain scientifically separable.
        """
        if area_component_mode not in {"diagnostic", "add", "replace"}:
            raise ValueError("area_component_mode must be one of: 'diagnostic', 'add', 'replace'")
        if P_GIB is None:
            P_GIB = self.pth_cfg.grounded_iceberg_file_path
        if P_out is None:
            P_out = self.pth_cfg.grounded_iceberg_form_factors_path
        P_out = Path(P_out)
        if P_out.exists() and not overwrite:
            self.logger.info(f"GIB perimeter FF file exists and overwrite=False; opening: {P_out}")
            return xr.open_dataset(P_out)
        hemisphere = self._normalise_hemisphere(hemisphere)
        src_lat_lim = self._polar_source_lat_limit(hemisphere, source_lat_limit)
        self.logger.info("starting GIB perimeter form-factor build")
        self.logger.info(f"input GIB vector file: {P_GIB}")
        self.logger.info(f"output NetCDF file  : {P_out}")
        self.logger.info("configuration: "
                         f"hemisphere={hemisphere}; "
                         f"source_lat_limit={src_lat_lim}; "
                         f"grid_lat_pad_deg={grid_lat_pad_deg}; "
                         f"grid_lon_pad_deg={grid_lon_pad_deg}; "
                         f"use_ocean_mask={use_ocean_mask}; "
                         f"use_index_half_hint={use_index_half_hint}; "
                         f"use_coastal_neighbour_filter={use_coastal_neighbour_filter}; "
                         f"coastal_neighbour_radius={coastal_neighbour_radius}; "
                         f"include_area_fraction={include_area_fraction}; "
                         f"area_component_mode={area_component_mode}; "
                         f"area_weight={area_weight}; "
                         f"clip_max={clip_max}")
        # ------------------------------------------------------------------
        # 1. Read GIB perimeter linework.
        # ------------------------------------------------------------------
        self.logger.info("reading and polar-filtering GIB perimeter linework")
        line_gdf = self._read_projected_linework(P_vector              = P_GIB,
                                                 hemisphere            = hemisphere,
                                                 source_lat_limit      = source_lat_limit,
                                                 source_lat_buffer_deg = 0.0)
        self.logger.info("GIB perimeter linework ready: "
                         f"n_line_geometries={len(line_gdf):,}; "
                         f"projected CRS={line_gdf.crs}; "
                         f"projected bounds={tuple(float(x) for x in line_gdf.total_bounds)}")
        try:
            total_line_length_m = float(line_gdf.geometry.length.sum())
            self.logger.info(f"total projected GIB perimeter length = {total_line_length_m:.4e} m")
        except Exception as exc:
            self.logger.info(f"could not calculate total GIB perimeter length for logging: {exc}")
        # ------------------------------------------------------------------
        # 2. Build candidate CICE T-cell polygons, restricted to the source
        #    envelope and optional model-grid filters.
        # ------------------------------------------------------------------
        self.logger.info("building source-filtered candidate T-cell polygons for GIB perimeter")
        cell_gdf, grid, tlon, tlat = self._build_projected_tcell_gdf_for_source(source_gdf_projected          = line_gdf,
                                                                                hemisphere                   = hemisphere,
                                                                                source_lat_limit             = source_lat_limit,
                                                                                grid_lat_pad_deg             = grid_lat_pad_deg,
                                                                                grid_lon_pad_deg             = grid_lon_pad_deg,
                                                                                use_ocean_mask               = use_ocean_mask,
                                                                                use_index_half_hint          = use_index_half_hint,
                                                                                use_coastal_neighbour_filter = use_coastal_neighbour_filter,
                                                                                coastal_neighbour_radius     = coastal_neighbour_radius)
        self.logger.info("candidate T-cell polygon build complete: "
                         f"n_candidate_cells={len(cell_gdf):,}; "
                         f"global_grid_shape={tlon.shape}; "
                         f"j_range={int(cell_gdf['j'].min())}:{int(cell_gdf['j'].max())}; "
                         f"i_range={int(cell_gdf['i'].min())}:{int(cell_gdf['i'].max())}")
        self.logger.info("candidate cell metric summary: "
                         f"HTE/dx min/max={np.nanmin(cell_gdf['dx_m'].to_numpy()):.4e}/"
                         f"{np.nanmax(cell_gdf['dx_m'].to_numpy()):.4e} m; "
                         f"HTN/dy min/max={np.nanmin(cell_gdf['dy_m'].to_numpy()):.4e}/"
                         f"{np.nanmax(cell_gdf['dy_m'].to_numpy()):.4e} m; "
                         f"cell_area min/max={np.nanmin(cell_gdf['cell_area'].to_numpy()):.4e}/"
                         f"{np.nanmax(cell_gdf['cell_area'].to_numpy()):.4e} m2")
        # ------------------------------------------------------------------
        # 3. Accumulate projected GIB perimeter length density.
        # ------------------------------------------------------------------
        self.logger.info("accumulating projected GIB perimeter density by source geometry")
        FFx_vec, FFy_vec, perim_i_m, perim_j_m, n_line_hits = (self._accumulate_projected_line_density_by_source(cell_gdf = cell_gdf, line_gdf = line_gdf, progress_every = progress_every))
        self.logger.info("GIB perimeter-density accumulation complete: "
                         f"FFx_vec min/max/sum="
                         f"{np.nanmin(FFx_vec):.4e}/{np.nanmax(FFx_vec):.4e}/{np.nansum(FFx_vec):.4e}; "
                         f"FFy_vec min/max/sum="
                         f"{np.nanmin(FFy_vec):.4e}/{np.nanmax(FFy_vec):.4e}/{np.nansum(FFy_vec):.4e}; "
                         f"nonzero perimeter cells={int(np.count_nonzero((FFx_vec > 0) | (FFy_vec > 0))):,}")
        self.logger.info("GIB perimeter diagnostic vectors: "
                         f"perim_i_m sum/max={np.nansum(perim_i_m):.4e}/{np.nanmax(perim_i_m):.4e} m; "
                         f"perim_j_m sum/max={np.nansum(perim_j_m):.4e}/{np.nanmax(perim_j_m):.4e} m; "
                         f"source-hit cells={int(np.count_nonzero(n_line_hits > 0)):,}; "
                         f"max perimeter hits per cell={int(np.nanmax(n_line_hits))}")
        # ------------------------------------------------------------------
        # 4. Optional GIB area fraction.
        # ------------------------------------------------------------------
        area_frac_vec = None
        area_m2_vec   = None
        n_poly_hits   = None
        if include_area_fraction:
            self.logger.info("include_area_fraction=True; reading and polar-filtering GIB polygons")
            polygon_gdf = self._read_projected_polygons(P_vector              = P_GIB,
                                                        hemisphere            = hemisphere,
                                                        source_lat_limit      = source_lat_limit,
                                                        source_lat_buffer_deg = 0.0)
            self.logger.info("GIB polygon geometry ready: "
                             f"n_polygon_geometries={len(polygon_gdf):,}; "
                             f"projected CRS={polygon_gdf.crs}; "
                             f"projected bounds={tuple(float(x) for x in polygon_gdf.total_bounds)}")

            try:
                total_polygon_area_m2 = float(polygon_gdf.geometry.area.sum())
                self.logger.info(f"total projected GIB polygon area = {total_polygon_area_m2:.4e} m2")
            except Exception as exc:
                self.logger.info(f"could not calculate total GIB polygon area for logging: {exc}")
            self.logger.info("accumulating GIB area fraction by source polygon geometry")
            area_frac_vec, area_m2_vec, n_poly_hits = (self._accumulate_polygon_area_fraction_by_source(cell_gdf           = cell_gdf,
                                                                                                        polygon_gdf        = polygon_gdf,
                                                                                                        progress_every     = progress_every,
                                                                                                        clip_area_fraction = clip_area_fraction))
            self.logger.info("GIB area-fraction accumulation complete: "
                             f"area_frac min/max/sum="
                             f"{np.nanmin(area_frac_vec):.4e}/{np.nanmax(area_frac_vec):.4e}/{np.nansum(area_frac_vec):.4e}; "
                             f"area_m2 sum/max={np.nansum(area_m2_vec):.4e}/{np.nanmax(area_m2_vec):.4e} m2; "
                             f"area-hit cells={int(np.count_nonzero(n_poly_hits > 0)):,}; "
                             f"max polygon hits per cell={int(np.nanmax(n_poly_hits))}; "
                             f"clip_area_fraction={clip_area_fraction}")
            if area_component_mode == "diagnostic":
                self.logger.info("area_component_mode='diagnostic'; GIB area fraction will be stored but not added to FFx/FFy")
            elif area_component_mode == "add":
                self.logger.info(f"area_component_mode='add'; adding isotropic area contribution area_weight * GIB_area_frac with area_weight={float(area_weight):.4e}")
                FFx_vec = FFx_vec + float(area_weight) * area_frac_vec
                FFy_vec = FFy_vec + float(area_weight) * area_frac_vec
                self.logger.info("after adding area contribution: "
                                 f"FFx_vec min/max/sum="
                                 f"{np.nanmin(FFx_vec):.4e}/{np.nanmax(FFx_vec):.4e}/{np.nansum(FFx_vec):.4e}; "
                                 f"FFy_vec min/max/sum="
                                 f"{np.nanmin(FFy_vec):.4e}/{np.nanmax(FFy_vec):.4e}/{np.nansum(FFy_vec):.4e}")
            elif area_component_mode == "replace":
                self.logger.info("area_component_mode='replace'; replacing perimeter density with "
                                 f"isotropic area_weight * GIB_area_frac, area_weight={float(area_weight):.4e}")
                FFx_vec = float(area_weight) * area_frac_vec
                FFy_vec = float(area_weight) * area_frac_vec
                self.logger.info("after replacing with area contribution: "
                                 f"FFx_vec min/max/sum="
                                 f"{np.nanmin(FFx_vec):.4e}/{np.nanmax(FFx_vec):.4e}/{np.nansum(FFx_vec):.4e}; "
                                 f"FFy_vec min/max/sum="
                                 f"{np.nanmin(FFy_vec):.4e}/{np.nanmax(FFy_vec):.4e}/{np.nansum(FFy_vec):.4e}")
        else:
            self.logger.info("include_area_fraction=False; skipping GIB polygon area-fraction calculation")
        # ------------------------------------------------------------------
        # 5. Optional clipping.
        # ------------------------------------------------------------------
        if clip_max is not None:
            self.logger.info(f"clipping GIB FFx/FFy vectors to [0, {float(clip_max):.4e}]")
            FFx_vec = np.clip(FFx_vec, 0.0, float(clip_max))
            FFy_vec = np.clip(FFy_vec, 0.0, float(clip_max))
            self.logger.info("after clipping: "
                             f"FFx_vec min/max/sum="
                             f"{np.nanmin(FFx_vec):.4e}/{np.nanmax(FFx_vec):.4e}/{np.nansum(FFx_vec):.4e}; "
                             f"FFy_vec min/max/sum="
                             f"{np.nanmin(FFy_vec):.4e}/{np.nanmax(FFy_vec):.4e}/{np.nansum(FFy_vec):.4e}")
        else:
            self.logger.info("clip_max=None; preserving Liu-style GIB perimeter values greater than 1 where present")
        # ------------------------------------------------------------------
        # 6. Scatter candidate-cell vectors back to the full CICE T grid.
        # ------------------------------------------------------------------
        self.logger.info("scattering GIB FF vectors back to full CICE T-grid arrays")
        FFx = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, FFx_vec)
        FFy = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, FFy_vec)
        self.logger.info("full-grid GIB FF arrays created: "
                         f"FFx shape={FFx.shape}, min/max/sum="
                         f"{np.nanmin(FFx):.4e}/{np.nanmax(FFx):.4e}/{np.nansum(FFx):.4e}; "
                         f"FFy shape={FFy.shape}, min/max/sum="
                         f"{np.nanmin(FFy):.4e}/{np.nanmax(FFy):.4e}/{np.nansum(FFy):.4e}; "
                         f"nonzero grid cells={int(np.count_nonzero((FFx > 0) | (FFy > 0))):,}")
        # ------------------------------------------------------------------
        # 7. Create output dataset.
        # ------------------------------------------------------------------
        self.logger.info("creating output xarray Dataset for GIB perimeter form factors")
        ds_out = self._empty_FF_dataset(tlon, tlat, FFx, FFy,
                                        attrs = {"title"        : "Grounded-iceberg perimeter-derived lateral drag form factors",
                                                 "source_vector": str(P_GIB),
                                                 "grid_source"  : str(grid.source_path),
                                                 "proj_crs"     : str(self.spec.proj_crs),
                                                 "hemisphere"   : hemisphere,
                                                 "source_lat_limit": src_lat_lim,
                                                 "method"       : ("source-filtered, source-driven within-cell projected grounded-iceberg perimeter-length density"),
                                                 "normalisation": ("FFx=sum(abs(projected GIB perimeter along model-i))/HTE; "
                                                                   "FFy=sum(abs(projected GIB perimeter along model-j))/HTN"),
                                                 "include_area_fraction": bool(include_area_fraction),
                                                 "area_component_mode"  : str(area_component_mode),
                                                 "area_weight"          : float(area_weight),
                                                 "use_ocean_mask"       : bool(use_ocean_mask),
                                                 "use_index_half_hint"  : bool(use_index_half_hint),
                                                 "use_coastal_neighbour_filter": bool(use_coastal_neighbour_filter),
                                                 "coastal_neighbour_radius": int(coastal_neighbour_radius),
                                                 "clip_area_fraction": bool(clip_area_fraction),
                                                 "clip_max": "None" if clip_max is None else float(clip_max)})
        # ------------------------------------------------------------------
        # 8. Add perimeter diagnostics.
        # ------------------------------------------------------------------
        self.logger.info("adding GIB perimeter diagnostic fields to output Dataset")
        GIB_perimeter_i_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, perim_i_m)
        GIB_perimeter_j_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, perim_j_m)
        GIB_line_hits_grid   = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, n_line_hits, dtype="int32")
        ds_out["GIB_perimeter_i_m"] = (("nj", "ni"), GIB_perimeter_i_grid)
        ds_out["GIB_perimeter_j_m"] = (("nj", "ni"), GIB_perimeter_j_grid)
        ds_out["GIB_n_perimeter_hits"] = (("nj", "ni"), GIB_line_hits_grid)
        ds_out["GIB_perimeter_i_m"].attrs.update({"long_name": "summed GIB perimeter length projected onto local model-i direction", "units": "m"})
        ds_out["GIB_perimeter_j_m"].attrs.update({"long_name": "summed GIB perimeter length projected onto local model-j direction", "units": "m"})
        ds_out["GIB_n_perimeter_hits"].attrs.update({"long_name": "number of GIB perimeter source geometries intersecting each T cell", "units": "1"})
        self.logger.info("GIB perimeter diagnostic grids added: "
                         f"GIB_perimeter_i_m sum/max="
                         f"{np.nansum(GIB_perimeter_i_grid):.4e}/{np.nanmax(GIB_perimeter_i_grid):.4e}; "
                         f"GIB_perimeter_j_m sum/max="
                         f"{np.nansum(GIB_perimeter_j_grid):.4e}/{np.nanmax(GIB_perimeter_j_grid):.4e}; "
                         f"GIB_n_perimeter_hits max={int(np.nanmax(GIB_line_hits_grid))}")
        # ------------------------------------------------------------------
        # 9. Add optional area diagnostics.
        # ------------------------------------------------------------------
        if include_area_fraction and area_frac_vec is not None:
            self.logger.info("adding GIB area-fraction diagnostic fields to output Dataset")
            GIB_area_frac_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, area_frac_vec)
            GIB_area_m2_grid   = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, area_m2_vec)
            GIB_poly_hits_grid = self._scatter_cell_values_to_grid(tlon, tlat, cell_gdf, n_poly_hits, dtype="int32")
            ds_out["GIB_area_frac"] = (("nj", "ni"), GIB_area_frac_grid)
            ds_out["GIB_area_m2"] = (("nj", "ni"), GIB_area_m2_grid)
            ds_out["GIB_n_polygon_hits"] = (("nj", "ni"), GIB_poly_hits_grid)
            ds_out["GIB_area_frac"].attrs.update({"long_name": "grounded-iceberg area fraction within CICE T cell", "units": "1"})
            ds_out["GIB_area_m2"].attrs.update({"long_name": "grounded-iceberg polygon area intersecting CICE T cell", "units": "m2"})
            ds_out["GIB_n_polygon_hits"].attrs.update({"long_name": "number of GIB polygon features intersecting each T cell", "units": "1"})
            self.logger.info("GIB area diagnostic grids added: "
                             f"GIB_area_frac min/max/sum="
                             f"{np.nanmin(GIB_area_frac_grid):.4e}/"
                             f"{np.nanmax(GIB_area_frac_grid):.4e}/"
                             f"{np.nansum(GIB_area_frac_grid):.4e}; "
                             f"GIB_area_m2 sum/max="
                             f"{np.nansum(GIB_area_m2_grid):.4e}/"
                             f"{np.nanmax(GIB_area_m2_grid):.4e}; "
                             f"GIB_n_polygon_hits max={int(np.nanmax(GIB_poly_hits_grid))}")
        # ------------------------------------------------------------------
        # 10. Write output.
        # ------------------------------------------------------------------
        P_out.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"writing GIB perimeter form factors to NetCDF: {P_out}")
        ds_out = self._write_FF_dataset(ds_out, P_out, mode="w")
        self.logger.info(f"NetCDF write complete: {P_out}")
        self.logger.info("final GIB perimeter dataset summary: "
                         f"dims={dict(ds_out.dims)}; "
                         f"data_vars={list(ds_out.data_vars)}")
        self._log(f"Wrote GIB perimeter form factors: {P_out}")
        return ds_out

    def build_FF_combined_CICE(self,
                               P_FF_cst          : str | Path | None = None,
                               P_FF_GIB          : str | Path | None = None,
                               P_out             : str | Path | None = None,
                               FF_combine_method : str | None = None,
                               overwrite         : bool = False,
                               clip_max          : float | None = None,
                               F2x_varname       : str = "FFx",
                               F2y_varname       : str = "FFy",
                               cice_dim_order    : str | None = None) -> xr.Dataset:
        """
        Combine coastline and grounded-iceberg T-cell form-factor products for CICE.

        This method combines two already-built form-factor products:

            1. coastline-derived FF product, usually from build_FF_from_Hres_coast_Liu()
            2. grounded-iceberg-derived FF product, usually from build_FF_from_GIB_perimeter()

        Both input files are expected to contain T-cell fields in the existing
        shuga/xarray convention:

            FFx(nj, ni)
            FFy(nj, ni)
            lon(nj, ni)
            lat(nj, ni)

        This is the convention that has already been working with the current CICE
        NetCDF read pathway. Although CICE allocates the target arrays as
        F2x_in(nx_global, ny_global) and F2y_in(nx_global, ny_global), the NetCDF
        Fortran interface handles the file/API ordering. Therefore this method
        should not transpose the arrays and should not write variables as
        FFx(ni, nj).

        The output remains a T-cell source product. It is not pre-mapped to CICE
        velocity faces. CICE later maps:

            FFx -> F2x_in -> F2E
            FFy -> F2y_in -> F2N

        inside load_F2_form_factors().

        Combination methods
        -------------------
        max
            Component-wise maximum:

                FFx = max(FFx_coast, FFx_GIB)
                FFy = max(FFy_coast, FFy_GIB)

            This preserves the strongest local geometric source term without
            amplifying overlap.

        mean
            Component-wise arithmetic mean:

                FFx = 0.5 * (FFx_coast + FFx_GIB)
                FFy = 0.5 * (FFy_coast + FFy_GIB)

            This is conservative where only one source class contributes strongly.

        sum
            Component-wise sum:

                FFx = FFx_coast + FFx_GIB
                FFy = FFy_coast + FFy_GIB

            This enhances cells where coastline and grounded-iceberg source terms
            overlap. For Liu-style length-density fields, the sum is not clipped
            unless `clip_max` is explicitly provided.

        Parameters
        ----------
        P_FF_cst : str or pathlib.Path, optional
            Path to the coastline form-factor NetCDF file.

        P_FF_GIB : str or pathlib.Path, optional
            Path to the grounded-iceberg form-factor NetCDF file.

        P_out : str or pathlib.Path, optional
            Path to the combined output NetCDF file.

        FF_combine_method : {"max", "mean", "sum"}, optional
            Method used to combine coastline and GIB FFx/FFy fields. If None,
            `self.spec.FF_map_method` is used. Note that this is the Python-side
            source-field combine method, not CICE's T-cell-to-face map method.

        overwrite : bool, default False
            If False and `P_out` exists, open and return the existing file.

        clip_max : float or None, default None
            Optional cap applied after combination. For Liu-style length-density
            fields, None is recommended because values may legitimately exceed 1.

        F2x_varname, F2y_varname : str
            Variable names to read from the input products and write to the output.
            The standard names are "FFx" and "FFy".

        cice_dim_order : str or None
            Deprecated compatibility argument. It is ignored. Production shuga/CICE
            FF products should remain in xarray/CDL order ("nj", "ni").

        Returns
        -------
        xr.Dataset
            Combined CICE-readable T-cell form-factor dataset.
        """

        # ------------------------------------------------------------------
        # 0. Resolve paths and options.
        # ------------------------------------------------------------------
        if P_FF_cst is None:
            P_FF_cst = self.pth_cfg.coast_form_factors_path
        if P_FF_GIB is None:
            P_FF_GIB = self.pth_cfg.grounded_iceberg_form_factors_path
        if P_out is None:
            P_out = self.pth_cfg.combined_form_factors_path
        if FF_combine_method is None:
            FF_combine_method = self.spec.FF_map_method
        FF_combine_method = str(FF_combine_method).lower()
        if FF_combine_method not in {"max", "mean", "sum"}:
            raise ValueError("FF_combine_method must be one of: 'max', 'mean', or 'sum'.")
        P_FF_cst = Path(P_FF_cst)
        P_FF_GIB = Path(P_FF_GIB)
        P_out    = Path(P_out)
        if cice_dim_order is not None:
            self.logger.info("cice_dim_order argument is deprecated/ignored for this method; "
                             "using established shuga/CICE file order FFx(nj, ni), FFy(nj, ni)")
        self.logger.info("starting combined CICE F2 form-factor build")
        self.logger.info(f"coast FF file : {P_FF_cst}")
        self.logger.info(f"GIB FF file   : {P_FF_GIB}")
        self.logger.info(f"output file   : {P_out}")
        self.logger.info("configuration: "
                         f"FF_combine_method={FF_combine_method}; "
                         f"clip_max={clip_max}; "
                         f"F2x_varname={F2x_varname}; "
                         f"F2y_varname={F2y_varname}; "
                         "file_order=(nj,ni)")
        if P_out.exists() and not overwrite:
            self.logger.info(f"combined FF file exists and overwrite=False; opening existing file: {P_out}")
            return xr.open_dataset(P_out)
        # ------------------------------------------------------------------
        # 1. Open input datasets.
        # ------------------------------------------------------------------
        self.logger.info("opening coastline and GIB form-factor datasets")
        ds_c = xr.open_dataset(P_FF_cst)
        ds_g = xr.open_dataset(P_FF_GIB)
        try:
            # --------------------------------------------------------------
            # 2. Validate required variables and shapes.
            # --------------------------------------------------------------
            self.logger.info("validating required FF variables and dimensions")
            for ds, label, path in [(ds_c, "coast", P_FF_cst), (ds_g, "GIB", P_FF_GIB)]:
                if F2x_varname not in ds or F2y_varname not in ds:
                    raise KeyError(f"{label} form-factor file missing {F2x_varname}/{F2y_varname}: {path}")
                if ds[F2x_varname].ndim != 2 or ds[F2y_varname].ndim != 2:
                    raise ValueError(f"{label} FF variables must be 2-D. "
                                     f"{F2x_varname}.ndim={ds[F2x_varname].ndim}, "
                                     f"{F2y_varname}.ndim={ds[F2y_varname].ndim}")
                if ds[F2x_varname].shape != ds[F2y_varname].shape:
                    raise ValueError(f"{label} FFx/FFy shapes differ: "
                                     f"{F2x_varname}={ds[F2x_varname].shape}, "
                                     f"{F2y_varname}={ds[F2y_varname].shape}")
            if "lon" not in ds_c or "lat" not in ds_c:
                raise KeyError("Coast FF file must contain lon/lat variables.")
            c_x = ds_c[F2x_varname].values
            c_y = ds_c[F2y_varname].values
            g_x = ds_g[F2x_varname].values
            g_y = ds_g[F2y_varname].values
            if c_x.shape != g_x.shape or c_y.shape != g_y.shape:
                raise ValueError("Coast and GIB form-factor products have different shapes. "
                                 f"coast {F2x_varname}={c_x.shape}, GIB {F2x_varname}={g_x.shape}; "
                                 f"coast {F2y_varname}={c_y.shape}, GIB {F2y_varname}={g_y.shape}")
            tlon = ds_c["lon"].values
            tlat = ds_c["lat"].values
            if tlon.shape != c_x.shape or tlat.shape != c_x.shape:
                raise ValueError(f"Coast lon/lat shapes do not match FF fields. lon={tlon.shape}, lat={tlat.shape}, FF={c_x.shape}")
            self.logger.info("input validation complete: "
                             f"shape={c_x.shape}; "
                             f"coast dims={ds_c[F2x_varname].dims}; "
                             f"GIB dims={ds_g[F2x_varname].dims}")
            self.logger.info("coast input summary: "
                             f"{F2x_varname} min/max/sum="
                             f"{np.nanmin(c_x):.4e}/{np.nanmax(c_x):.4e}/{np.nansum(c_x):.4e}; "
                             f"{F2y_varname} min/max/sum="
                             f"{np.nanmin(c_y):.4e}/{np.nanmax(c_y):.4e}/{np.nansum(c_y):.4e}; "
                             f"nonzero cells={int(np.count_nonzero((c_x > 0) | (c_y > 0))):,}")
            self.logger.info("GIB input summary: "
                             f"{F2x_varname} min/max/sum="
                             f"{np.nanmin(g_x):.4e}/{np.nanmax(g_x):.4e}/{np.nansum(g_x):.4e}; "
                             f"{F2y_varname} min/max/sum="
                             f"{np.nanmin(g_y):.4e}/{np.nanmax(g_y):.4e}/{np.nansum(g_y):.4e}; "
                             f"nonzero cells={int(np.count_nonzero((g_x > 0) | (g_y > 0))):,}")
            # --------------------------------------------------------------
            # 3. Clean inputs before combination.
            #
            #    This mirrors CICE's defensive behaviour:
            #      * NaN/Inf -> 0
            #      * negative -> 0
            #
            #    Do not clip upper values here unless clip_max is requested
            #    after combination.
            # --------------------------------------------------------------
            self.logger.info("cleaning non-finite and negative input values before combination")
            c_x = self._as_nonnegative_finite(c_x, clip_max=None, dtype="float64")
            c_y = self._as_nonnegative_finite(c_y, clip_max=None, dtype="float64")
            g_x = self._as_nonnegative_finite(g_x, clip_max=None, dtype="float64")
            g_y = self._as_nonnegative_finite(g_y, clip_max=None, dtype="float64")
            # --------------------------------------------------------------
            # 4. Combine source fields.
            # --------------------------------------------------------------
            self.logger.info(f"combining coastline and GIB FF fields using method='{FF_combine_method}'")
            if FF_combine_method == "max":
                FFx = np.maximum(c_x, g_x)
                FFy = np.maximum(c_y, g_y)
            elif FF_combine_method == "mean":
                FFx = 0.5 * (c_x + g_x)
                FFy = 0.5 * (c_y + g_y)
            elif FF_combine_method == "sum":
                FFx = c_x + g_x
                FFy = c_y + g_y
            self.logger.info("combined field before optional clipping: "
                             f"FFx min/max/sum={np.nanmin(FFx):.4e}/{np.nanmax(FFx):.4e}/{np.nansum(FFx):.4e}; "
                             f"FFy min/max/sum={np.nanmin(FFy):.4e}/{np.nanmax(FFy):.4e}/{np.nansum(FFy):.4e}; "
                             f"nonzero cells={int(np.count_nonzero((FFx > 0) | (FFy > 0))):,}")
            if clip_max is not None:
                self.logger.info(f"clipping combined FFx/FFy fields to [0, {float(clip_max):.4e}]")
                FFx = np.clip(FFx, 0.0, float(clip_max))
                FFy = np.clip(FFy, 0.0, float(clip_max))
                self.logger.info("combined field after clipping: "
                                 f"FFx min/max/sum={np.nanmin(FFx):.4e}/{np.nanmax(FFx):.4e}/{np.nansum(FFx):.4e}; "
                                 f"FFy min/max/sum={np.nanmin(FFy):.4e}/{np.nanmax(FFy):.4e}/{np.nansum(FFy):.4e}; "
                                 f"nonzero cells={int(np.count_nonzero((FFx > 0) | (FFy > 0))):,}")
            else:
                self.logger.info("clip_max=None; preserving Liu-style combined values greater than 1 where present")
            # --------------------------------------------------------------
            # 5. Create output dataset.
            #
            #    Important: use _empty_FF_dataset(), not _empty_FF_dataset_cice().
            #    The production convention is xarray/CDL order ("nj", "ni").
            # --------------------------------------------------------------
            self.logger.info("creating combined output xarray Dataset")
            ds_out = self._empty_FF_dataset(tlon, tlat, FFx, FFy,
                                            attrs = {"title"             : "Combined coastline + grounded iceberg lateral drag form factors",
                                                     "form_factors_coast": str(P_FF_cst),
                                                     "form_factors_GIB"  : str(P_FF_GIB),
                                                     "FF_combine_method" : str(FF_combine_method),
                                                     "clip_max"          : "None" if clip_max is None else float(clip_max),
                                                     "F2x_varname"       : str(F2x_varname),
                                                     "F2y_varname"       : str(F2y_varname),
                                                     "file_order"        : "FFx(nj,ni), FFy(nj,ni)",
                                                     "method"            : ("component-wise combination of T-cell coastline and "
                                                                            "grounded-iceberg source fields; CICE maps T cells to E/N faces"),
                                                     "note"              : ("This file should be supplied to CICE as F2_file. "
                                                                            "It is not pre-mapped to velocity faces.")})
            # --------------------------------------------------------------
            # 6. Carry through 2-D diagnostic fields.
            #
            #    Avoid lazy references to datasets that will soon be closed by
            #    copying values and attrs explicitly.
            #
            #    If a diagnostic name collides with an existing output variable,
            #    prefix it by source label.
            # --------------------------------------------------------------
            self.logger.info("copying compatible 2-D diagnostic fields into combined Dataset")
            copied_diags = []
            for src_ds, label in [(ds_c, "coast"), (ds_g, "GIB")]:
                for v in src_ds.data_vars:
                    if v in {F2x_varname, F2y_varname, "lon", "lat"}:
                        continue
                    if src_ds[v].ndim != 2:
                        self.logger.info(f"skipping {label} diagnostic '{v}': ndim={src_ds[v].ndim}, expected 2")
                        continue
                    if src_ds[v].shape != FFx.shape:
                        self.logger.info(f"skipping {label} diagnostic '{v}': "
                                         f"shape={src_ds[v].shape}, expected {FFx.shape}")
                        continue
                    out_name = v
                    if out_name in ds_out:
                        out_name = f"{label}_{v}"
                    self.logger.info(f"copying {label} diagnostic '{v}' -> '{out_name}'")
                    data = src_ds[v].values
                    # Preserve integer diagnostics as integer, but clean floating
                    # diagnostics to avoid write problems.
                    if np.issubdtype(data.dtype, np.integer):
                        ds_out[out_name] = (("nj", "ni"), data.astype("int32"))
                    else:
                        data = np.asarray(data, dtype="float64")
                        data[~np.isfinite(data)] = 0.0
                        ds_out[out_name] = (("nj", "ni"), data.astype("float32"))
                    ds_out[out_name].attrs.update(self._netcdf_safe_attrs(dict(src_ds[v].attrs)))
                    ds_out[out_name].attrs["source_product"] = label
                    copied_diags.append(out_name)
            self.logger.info(f"diagnostic copy complete: n_copied={len(copied_diags)}; "
                             f"diagnostics={copied_diags}")
            # --------------------------------------------------------------
            # 7. Write safely.
            # --------------------------------------------------------------
            P_out.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"writing combined form factors to NetCDF: {P_out}")
            ds_out = self._write_FF_dataset(ds_out, P_out, mode="w")
            self.logger.info("final combined dataset summary: "
                             f"dims={dict(ds_out.sizes)}; "
                             f"data_vars={list(ds_out.data_vars)}")
            self._log(f"Wrote CICE-compatible combined F2 file: {P_out}")
            return ds_out
        finally:
            self.logger.info("closing input coastline and GIB form-factor datasets")
            ds_c.close()
            ds_g.close()
