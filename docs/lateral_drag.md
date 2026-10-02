# Lateral-drag form-factor generation in `shuga`

**Target file:** `shuga/docs/lateral_drag.md`
**Primary module:** `shuga/grid/lateral_drag.py`
**Primary class:** `FormFactors`
**CICE consumer:** `cicecore/cicedyn/infrastructure/ice_grid.F90` in `CICE_free-slip`

This document describes the practical and analytical workflow used by `shuga` to generate CICE lateral-drag form-factor files from Antarctic coastline and grounded-iceberg geometry. It covers the input requirements, the Liu-style projected coastline length-density calculation, the grounded-iceberg perimeter and area-fraction options, the coastline/GIB combination method, the CICE `ice_grid.F90` ingestion pathway, and a notebook-style workflow for building and plotting the products.

The examples below assume the present Antarctic landfast-ice workflow on Gadi, but the code is structured around a `hemisphere` option and could potentially be adapted to Arctic source geometry ... arghhhhhh!!!

---

## Table of contents

1. [Purpose and scope](#purpose-and-scope)
2. [Required inputs](#required-inputs)
3. [Output file convention](#output-file-convention)
4. [Coastline form factors: Liu-style projected length density](#coastline-form-factors-liu-style-projected-length-density)
5. [Grounded-iceberg form factors](#grounded-iceberg-form-factors)
6. [Combining coastline and grounded-iceberg products](#combining-coastline-and-grounded-iceberg-products)
7. [Diagnostics written to NetCDF](#diagnostics-written-to-netcdf)
8. [How CICE consumes the `shuga` form-factor files](#how-cice-consumes-the-shuga-form-factor-files)
9. [Practical notebook workflow](#practical-notebook-workflow)
10. [Plotting and regional diagnostics](#plotting-and-regional-diagnostics)
11. [Figure gallery links](#figure-gallery-links)
12. [Recommended checks before using a file in CICE](#recommended-checks-before-using-a-file-in-cice)
13. [Common pitfalls and design decisions](#common-pitfalls-and-design-decisions)
14. [References and source links](#references-and-source-links)

---

## Purpose and scope

The `FormFactors` class builds gridded geometric source fields for the CICE lateral-drag parameterisation. The generated NetCDF files contain two T-cell source components:

- `FFx(nj, ni)`: local model-$i$ or nominal $x$-direction source strength;
- `FFy(nj, ni)`: local model-$j$ or nominal $y$-direction source strength.

These are not velocity-face fields. They are T-cell source fields that are later mapped inside CICE to the east-face and north-face form factors `F2E` and `F2N`.

There are three production-level source products:

1. **CST-only:** high-resolution coastline or ice-front geometry mapped to the CICE T grid.
2. **GIB-only:** grounded-iceberg geometry mapped to the CICE T grid using grounded-iceberg perimeters, with optional area-fraction diagnostics.
3. **Combined:** component-wise combination of coastline and grounded-iceberg products using `max`, `mean`, or `sum`.

The present workflow intentionally separates the geometric source-strength calculation from any optional spatial spreading or tapering. The Liu-style products described here have **no taper distance**. A cell receives a non-zero value only where the high-resolution source geometry actually intersects the candidate CICE T-cell polygon.

---

## Required inputs

### Python and package requirements

The workflow uses the standard `shuga` analysis environment plus geospatial packages:

- `numpy`
- `pandas`
- `xarray`
- `geopandas`
- `shapely`
- `pyproj`
- `scipy` only if `use_coastal_neighbour_filter=True`
- `netCDF4` for robust NetCDF writing
- `pygmt` for figure generation

On Gadi this is intended to run within the configured `analysis3` environment used by the broader `mawsons-chest`/`shuga` workflow.

### CICE grid inputs

The class uses `CICEGridwork` to load the CICE grid. The important grid fields are:

| Field | Role |
|---|---|
| `TLON`, `TLAT` | T-cell longitude and latitude. |
| `ULON`, `ULAT` | Corner-like grid geometry used to build approximate T-cell polygons when `build_faces=True`. |
| `HTE` | Length of the eastern edge of the T cell, used as the local $x/i$ normalisation scale. |
| `HTN` | Length of the northern edge of the T cell, used as the local $y/j$ normalisation scale. |
| grid mask / `kmt`-derived mask | Used to keep ocean T cells as candidates. It is not used as a source of coastline geometry. |

The grid file must match the CICE grid used in the lateral-drag experiments. The workflow assumes the Python-side grid arrays are shaped `(nj, ni)`.

### Coastline source file

The coastline method expects a high-resolution coastline or ice-front vector file. In the current Antarctic workflow:

```python
P_Hres_cst = Path('/g/data/gv90/da1339/coastlines/high_res_coast/add_coastline_high_res_polygon_v7_9.shp')
```

Accepted source geometries include:

- `LineString`
- `MultiLineString`
- `Polygon`
- `MultiPolygon`
- `GeometryCollection` containing linework or polygons

Polygon and multipolygon sources are converted to boundary linework. Points are ignored.

### Grounded-iceberg source files

The GIB method expects a grounded-iceberg vector product, preferably polygon or multipolygon geometry. In the current workflow:

```python
D_GIB     = Path('/g/data/gv90/da1339/grounded_icebergs/Kaihong_Jiao')
GIB_SPECS = {'v0p9': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v0.9.gpkg',
             'v1p0': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.0.gpkg',
             'v1p1': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.1.gpkg',
             'v1p2': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.2.gpkg'}
```

The preferred GIB implementation uses polygon perimeters as contact-line sources. If `include_area_fraction=True`, it also clips GIB polygons to the same T-cell polygons and stores area-fraction diagnostics.

### Path and lateral-drag configuration

The workflow is configured through:

- `RunSpec`
- `ClassificationSpec`
- `ShugaPaths`
- `LateralDragSpec`
- `FormFactors`

A minimal setup is:

```python
from pathlib import Path
from shuga.core.types import RunSpec, ClassificationSpec, LateralDragSpec
from shuga.core.paths import ShugaPaths
from shuga.grid.lateral_drag import FormFactors

run_cfg = RunSpec(sim_name   = 'LD-static-Cs1e-3',
                  start_date = '1999-01-01',
                  end_date   = '1999-12-31',
                  hemisphere = 'SH',
                  project    = 'gv90',
                  user       = 'da1339')

cls_cfg = ClassificationSpec(ice_type     = 'FI',
                             grid_type    = 'Tc',
                             ispd_thresh  = 5.0e-4,
                             methods      = ('binary-days',),
                             bin_window   = 11,
                             bin_min_days = 9,
                             roll_window  = 15)
pth_cfg = ShugaPaths(run_cfg = run_cfg, cls_cfg = cls_cfg)
LD_cfg  = LateralDragSpec()
FF      = FormFactors(pth_cfg = pth_cfg, LD_cfg = LD_cfg)
```

---

## Output file convention

All production `shuga` form-factor files use the same NetCDF convention:

```text
FFx(nj, ni)
FFy(nj, ni)
lon(nj, ni)
lat(nj, ni)
```

This is deliberate. `ncks` and xarray display the variables in CDL/C order as `(nj, ni)`, matching the Python-side array shape. CICE then reads the NetCDF variables through the NetCDF Fortran interface into arrays allocated as `F2x_in(nx_global, ny_global)` and `F2y_in(nx_global, ny_global)`. Do not transpose these files to `FFx(ni, nj)`.

The output variables are:

| Variable | Dimensions | Description |
|---|---|---|
| `FFx` | `(nj, ni)` | T-cell $x/i$-direction form factor. |
| `FFy` | `(nj, ni)` | T-cell $y/j$-direction form factor. |
| `lon` | `(nj, ni)` | T-cell longitude. |
| `lat` | `(nj, ni)` | T-cell latitude. |

Diagnostic fields differ by product type and are described below.

---

## Coastline form factors: Liu-style projected length density

### Analytical basis

The coastline method follows the geometric intent of Liu et al. (2022): represent unresolved coastline complexity as a directional sub-grid length density on the model grid. In the original conceptual form, high-resolution coastline segments within a model cell are projected onto grid directions and normalised by the corresponding grid spacing.

In `shuga`, the practical CICE-grid form is:

$$
F_{2x}(i,j) = \frac{1}{\mathrm{HTE}_{ij}} \sum_{s \in S_{ij}} \left| \boldsymbol{\ell}_{s,ij} \cdot \hat{\boldsymbol{e}}_{i,ij} \right|,
$$

$$
F_{2y}(i,j) = \frac{1}{\mathrm{HTN}_{ij}} \sum_{s \in S_{ij}} \left| \boldsymbol{\ell}_{s,ij} \cdot \hat{\boldsymbol{e}}_{j,ij} \right|,
$$

where:

- $S_{ij}$ is the set of clipped coastline or ice-front line segments intersecting CICE T cell $(i,j)$;
- $\boldsymbol{\ell}_{s,ij}$ is a clipped high-resolution source-line segment in projected metre coordinates;
- $\hat{\boldsymbol{e}}_{i,ij}$ is the local model-grid $i$-direction unit vector;
- $\hat{\boldsymbol{e}}_{j,ij}$ is the local model-grid $j$-direction unit vector;
- $\mathrm{HTE}_{ij}$ is the local CICE T-cell eastern-edge length;
- $\mathrm{HTN}_{ij}$ is the local CICE T-cell northern-edge length.

The absolute value follows the length-density interpretation: both orientations of coastline complexity contribute positive unresolved boundary length. The result is dimensionless. Values can exceed 1 in cells with strongly convoluted coastline or multiple source segments. This is not an error.

### Source-driven algorithm

The computational path is source-driven rather than grid-cell-driven:

1. Read the high-resolution coastline vector file.
2. Assign EPSG:4326 if no CRS is provided.
3. Filter source features to the requested polar domain. For Antarctic builds, the default source threshold is south of $-60^\circ$ latitude.
4. Project the retained source geometries to `LateralDragSpec.proj_crs`.
5. Convert polygons to boundary linework and keep line geometries.
6. Load the CICE grid with `build_faces=True`.
7. Build a candidate T-cell mask using source bounds, polar-domain filtering, and optionally the CICE ocean mask.
8. Construct projected T-cell polygons only for candidate cells.
9. Loop over source line geometries and query the candidate-cell spatial index.
10. Clip each source line to each intersecting T-cell polygon.
11. Project clipped segment vectors onto the local model-grid $i$ and $j$ directions.
12. Sum projected lengths, normalise by `HTE` and `HTN`, and scatter the result back to the full CICE T grid.
13. Write a NetCDF file using the `(nj, ni)` convention.

The high-resolution source geometry, not the coarse CICE landmask, determines where form factors can exist. The landmask is only a validity and efficiency filter.

### Coastline build call

```python
ds_coast = FF.build_FF_from_Hres_coast_Liu(
    P_Hres_cst       = P_Hres_cst,
    P_out            = P_FF_cst,
    source_lat_limit = -60.0,
    overwrite        = True,
    clip_max         = None,
)
```

Recommended Antarctic defaults:

| Option | Recommended value | Rationale |
|---|---:|---|
| `hemisphere` | `'SH'` | Antarctic source filtering. |
| `source_lat_limit` | `None` or `-60.0` | Retain Antarctic source features; exclude non-Antarctic southern landmasses. |
| `grid_lat_pad_deg` | `3.0` | Conservative grid prefilter around source envelope. |
| `grid_lon_pad_deg` | `3.0` | Conservative grid prefilter around source envelope. |
| `use_ocean_mask` | `True` | Retain ocean T cells only. |
| `use_index_half_hint` | `False` | Avoid grid-layout-specific assumptions. |
| `use_coastal_neighbour_filter` | `False` | Avoid losing valid high-resolution source geometry displaced from coarse `kmt`. |
| `clip_max` | `None` | Preserve physically meaningful values greater than 1. |

---

## Grounded-iceberg form factors

### Perimeter-density interpretation

The GIB perimeter method treats each grounded iceberg polygon boundary as a sub-grid lateral-contact source. It applies the same projected length-density calculation used for the coastline product:

$$
F_{2x}^{\mathrm{GIB}}(i,j) = \frac{1}{\mathrm{HTE}_{ij}} \sum_{s \in G_{ij}} \left| \boldsymbol{\ell}_{s,ij}^{\mathrm{GIB}} \cdot \hat{\boldsymbol{e}}_{i,ij} \right|,
$$

$$
F_{2y}^{\mathrm{GIB}}(i,j) = \frac{1}{\mathrm{HTN}_{ij}} \sum_{s \in G_{ij}} \left| \boldsymbol{\ell}_{s,ij}^{\mathrm{GIB}} \cdot \hat{\boldsymbol{e}}_{j,ij} \right|,
$$

where $G_{ij}$ is the set of clipped grounded-iceberg perimeter segments within the CICE T cell.

This is a contact-line roughness metric. It measures how much sub-grid grounded-iceberg boundary length is available to provide unresolved lateral resistance.

### Area-fraction diagnostic

When `include_area_fraction=True`, the method also computes the grounded-iceberg area fraction:

$$
A_{\mathrm{frac}}^{\mathrm{GIB}}(i,j) = \frac{A_{\mathrm{GIB} \cap T}(i,j)}{A_T(i,j)},
$$

where $A_{\mathrm{GIB} \cap T}$ is the area of GIB polygons clipped to the CICE T cell, and $A_T$ is the T-cell polygon area.

This is not the same physical quantity as perimeter length density:

| Quantity | Interpretation | Primary role |
|---|---|---|
| GIB perimeter density | Potential lateral contact-boundary length. | Primary GIB form factor in the perimeter method. |
| GIB area fraction | Obstacle occupancy within the grid cell. | Diagnostic by default; optional isotropic contribution. |

A grid cell can have large perimeter density but modest area fraction if it contains many small grounded icebergs. Conversely, one large grounded iceberg can produce large area fraction with relatively less perimeter complexity.

### Area-component modes

The `area_component_mode` controls whether `GIB_area_frac` modifies `FFx` and `FFy`.

| Mode | Formula | Recommended use |
|---|---|---|
| `'diagnostic'` | $F_{2x}=F_{2x}^{\mathrm{perim}}$, $F_{2y}=F_{2y}^{\mathrm{perim}}$ | Recommended production default. Area fraction is written but not used in `FFx`/`FFy`. |
| `'add'` | $F_{2x}=F_{2x}^{\mathrm{perim}} + w A_{\mathrm{frac}}^{\mathrm{GIB}}$, $F_{2y}=F_{2y}^{\mathrm{perim}} + w A_{\mathrm{frac}}^{\mathrm{GIB}}$ | Experimental: perimeter roughness plus isotropic obstacle occupancy. |
| `'replace'` | $F_{2x}=w A_{\mathrm{frac}}^{\mathrm{GIB}}$, $F_{2y}=w A_{\mathrm{frac}}^{\mathrm{GIB}}$ | Experimental: area occupancy only. |

Here $w$ is `area_weight`.

The recommended first-pass Antarctic setting is:

```python
ds_gib = FF.build_FF_from_GIB_perimeter(
    P_GIB                 = GIB_SPECS['v1p2'],
    P_out                 = P_FF_gib,
    overwrite             = True,
    include_area_fraction = True,
    area_component_mode   = 'diagnostic',
    clip_max              = None,
)
```

### GIB-specific options

| Option | Default | Meaning |
|---|---:|---|
| `include_area_fraction` | `True` | Compute area-fraction diagnostics. |
| `area_component_mode` | `'diagnostic'` | Store area fraction without modifying `FFx`/`FFy`. |
| `area_weight` | `1.0` | Scalar applied in `add` or `replace` modes. |
| `clip_area_fraction` | `True` | Clip area fraction to $[0,1]$. |
| `clip_max` | `None` | Preserve perimeter-density values greater than 1. |
| `use_coastal_neighbour_filter` | `False` | Recommended, because grounded icebergs can occur offshore and should not be removed solely because they are not adjacent to the coarse continental landmask. |

---

## Combining coastline and grounded-iceberg products

The combined product reads two existing NetCDF files:

- coastline product, usually produced by `build_FF_from_Hres_coast_Liu()`;
- GIB product, usually produced by `build_FF_from_GIB_perimeter()`.

It then combines `FFx` and `FFy` component-wise. This is a Python-side source-field combination. It is distinct from the CICE `F2_map_method`, which later maps T-cell source fields to E/N faces.

### Combination formulas

For `max`:

$$
F_{2x}^{\mathrm{cmb}} = \max(F_{2x}^{\mathrm{cst}}, F_{2x}^{\mathrm{GIB}}),
$$

$$
F_{2y}^{\mathrm{cmb}} = \max(F_{2y}^{\mathrm{cst}}, F_{2y}^{\mathrm{GIB}}).
$$

For `mean`:

$$
F_{2x}^{\mathrm{cmb}} = \frac{1}{2}\left(F_{2x}^{\mathrm{cst}} + F_{2x}^{\mathrm{GIB}}\right),
$$

$$
F_{2y}^{\mathrm{cmb}} = \frac{1}{2}\left(F_{2y}^{\mathrm{cst}} + F_{2y}^{\mathrm{GIB}}\right).
$$

For `sum`:

$$
F_{2x}^{\mathrm{cmb}} = F_{2x}^{\mathrm{cst}} + F_{2x}^{\mathrm{GIB}},
$$

$$
F_{2y}^{\mathrm{cmb}} = F_{2y}^{\mathrm{cst}} + F_{2y}^{\mathrm{GIB}}.
$$

Recommended first-pass production choice is `max`: it preserves the strongest local source without amplifying cells where coastline and GIB contributions overlap. The `sum` method is useful as an upper-amplitude sensitivity. The `mean` method is a conservative lower-amplitude alternative where only one source class contributes strongly.

Example:

```python
ds_cmb = FF.build_FF_combined_CICE(
    P_FF_cst          = P_FF_cst,
    P_FF_GIB          = P_FF_gib,
    P_out             = P_FF_cmb,
    FF_combine_method = 'max',
    overwrite         = True,
    clip_max          = None,
)
```

---

## Diagnostics written to NetCDF

### CST-only product

| Variable | Meaning |
|---|---|
| `coast_line_i_m` | Summed clipped coastline length projected onto local model-$i$, in metres. |
| `coast_line_j_m` | Summed clipped coastline length projected onto local model-$j$, in metres. |
| `coast_n_source_hits` | Number of coastline source geometries intersecting each T cell. |

### GIB-only perimeter product

| Variable | Meaning |
|---|---|
| `GIB_perimeter_i_m` | Summed clipped GIB perimeter length projected onto local model-$i$, in metres. |
| `GIB_perimeter_j_m` | Summed clipped GIB perimeter length projected onto local model-$j$, in metres. |
| `GIB_n_perimeter_hits` | Number of GIB perimeter source geometries intersecting each T cell. |
| `GIB_area_frac` | Grounded-iceberg polygon area fraction in each T cell. |
| `GIB_area_m2` | Grounded-iceberg polygon area intersecting each T cell, in square metres. |
| `GIB_n_polygon_hits` | Number of GIB polygon features intersecting each T cell. |

### Combined product

The combined product carries through compatible 2-D diagnostics from the source products. For the current combined file this normally includes:

- `coast_n_source_hits`
- `GIB_area_frac`
- `GIB_n_perimeter_hits`
- plus supporting length and area diagnostics where retained.

---

## How CICE consumes the `shuga` form-factor files

The CICE-side consumer is `load_F2_form_factors()` in `ice_grid.F90` in the `CICE_free-slip` repository. The module declares public allocatable face fields `F2E` and `F2N`, plus namelist-controlled file/variable names and mapping method.

A CICE namelist should specify something like:

```fortran
F2_file       = '/g/data/gv90/da1339/form_factors/FF_combined_meth-max_cst-v7p9-Liu_GIB-v1p2-perimeter_Liu_CICE.nc'
F2x_varname   = 'FFx'
F2y_varname   = 'FFy'
F2_map_method = 'max'     ! or 'avg'
```

The CICE logic is:

1. Allocate global temporary arrays on the master task:

```fortran
F2x_in(nx_global, ny_global)
F2x_out(nx_global, ny_global)
F2y_in(nx_global, ny_global)
F2y_out(nx_global, ny_global)
```

2. Open `F2_file` and read `F2x_varname` and `F2y_varname` into `F2x_in` and `F2y_in`.
3. Replace NaNs with zero.
4. Map the T-cell $x$-projection to E faces:

For `F2_map_method='avg'`:

$$
F2E(i,j) = \frac{1}{2}\left[\max(F2x_{in}(i,j),0) + \max(F2x_{in}(i+1,j),0)\right].
$$

For `F2_map_method='max'`:

$$
F2E(i,j) = \max\left[\max(F2x_{in}(i,j),0), \max(F2x_{in}(i+1,j),0)\right].
$$

For cyclic east-west grids, the final east face wraps from $i=nx\_global$ to $i=1$. For non-cyclic grids, the final column is set to zero.

5. Map the T-cell $y$-projection to N faces:

For `F2_map_method='avg'`:

$$
F2N(i,j) = \frac{1}{2}\left[\max(F2y_{in}(i,j),0) + \max(F2y_{in}(i,j+1),0)\right].
$$

For `F2_map_method='max'`:

$$
F2N(i,j) = \max\left[\max(F2y_{in}(i,j),0), \max(F2y_{in}(i,j+1),0)\right].
$$

The final northern row is set to zero.

6. Scatter `F2x_out` to the distributed `F2E` field at `field_loc_Eface` and `F2y_out` to the distributed `F2N` field at `field_loc_Nface`.
7. Zero land velocity points using `emask` and `nmask`.
8. Apply halo updates to `F2E` and `F2N`.

This means the `shuga` file must remain a T-cell source product. It should not be pre-mapped to E or N faces in Python.

---

## Practical notebook workflow

A downloadable notebook accompanying this document is provided as `lateral_drag_form_factors_workflow.ipynb`. The key cells are reproduced below.

### Imports and repository setup

```python
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import Image, display

xr.set_options(keep_attrs=True)
warnings.filterwarnings('default')

repo_root = Path.home() / 'AFIM' / 'src' / 'mawsons-chest'
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
```

### Configure paths and shuga objects

```python
from shuga.core.types import RunSpec, ClassificationSpec, LateralDragSpec
from shuga.core.paths import ShugaPaths
from shuga.grid.lateral_drag import FormFactors

PROJECT    = 'gv90'
USER       = 'da1339'
HEMISPHERE = 'SH'
ICE_TYPE   = 'FI'
GRID_TYPE  = 'Tc'

D_root = Path('/g/data/gv90/da1339')
D_FF   = D_root / 'form_factors'
D_cst  = D_root / 'coastlines' / 'high_res_coast'
D_GIB  = D_root / 'grounded_icebergs' / 'Kaihong_Jiao'
D_FF.mkdir(parents=True, exist_ok=True)

P_Hres_cst = D_cst / 'add_coastline_high_res_polygon_v7_9.shp'

GIB_SPECS = {
    'v0p9': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v0.9.gpkg',
    'v1p0': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.0.gpkg',
    'v1p1': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.1.gpkg',
    'v1p2': D_GIB / 'Antarctic_Grounded_Iceberg_Dataset_Sentinel1_v1.2.gpkg',
}

P_FF_cst = D_FF / 'FF_ADD-Hres-cst-v7p9_Liu_CICE.nc'
P_FF_gib = D_FF / 'FF_GIB-v1p2_perimeter_Liu_CICE.nc'
P_FF_cmb = D_FF / 'FF_combined_meth-max_cst-v7p9-Liu_GIB-v1p2-perimeter_Liu_CICE.nc'

run_cfg = RunSpec(
    sim_name   = 'LD-static-Cs1e-3',
    start_date = '1999-01-01',
    end_date   = '1999-12-31',
    hemisphere = HEMISPHERE,
    project    = PROJECT,
    user       = USER,
)

cls_cfg = ClassificationSpec(
    ice_type     = ICE_TYPE,
    grid_type    = GRID_TYPE,
    ispd_thresh  = 5.0e-4,
    methods      = ('binary-days',),
    bin_window   = 11,
    bin_min_days = 9,
    roll_window  = 15,
)

pth_cfg = ShugaPaths(run_cfg=run_cfg, cls_cfg=cls_cfg)
LD_cfg  = LateralDragSpec()
FF      = FormFactors(pth_cfg=pth_cfg, LD_cfg=LD_cfg)
```

### Build coastline, GIB, and combined files

```python
ds_coast = FF.build_FF_from_Hres_coast_Liu(
    P_Hres_cst       = P_Hres_cst,
    P_out            = P_FF_cst,
    source_lat_limit = -60.0,
    overwrite        = True,
)

FF.assert_CICE_F2_file_compatibility(P_FF_cst, nx_global=1440, ny_global=1080)
```

```python
ds_gib = FF.build_FF_from_GIB_perimeter(
    P_GIB     = GIB_SPECS['v1p2'],
    P_out     = P_FF_gib,
    overwrite = True,
)

FF.assert_CICE_F2_file_compatibility(P_FF_gib, nx_global=1440, ny_global=1080)
```

```python
ds_cmb = FF.build_FF_combined_CICE(
    P_FF_cst          = P_FF_cst,
    P_FF_GIB          = P_FF_gib,
    P_out             = P_FF_cmb,
    FF_combine_method = 'max',
    overwrite         = True,
)

FF.assert_CICE_F2_file_compatibility(P_FF_cmb, nx_global=1440, ny_global=1080)
```

### Inspect products

```python
for path in [P_FF_cst, P_FF_gib, P_FF_cmb]:
    ds = xr.open_dataset(path)
    try:
        print('\n', path)
        print(ds)
        print('FFx max:', float(ds['FFx'].max()))
        print('FFy max:', float(ds['FFy'].max()))
        print('nonzero:', int(((ds['FFx'] > 0) | (ds['FFy'] > 0)).sum()))
    finally:
        ds.close()
```

---

## Plotting and regional diagnostics

The plotting workflow computes $|F_{2xy}|$ as:

$$
|F_{2xy}| = \sqrt{FFx^2 + FFy^2}.
$$

For regional diagnostic annotation, the current helper reports active-cell statistics for selected diagnostic fields:

| Product | Diagnostics |
|---|---|
| `cst-only` | `coast_n_source_hits` |
| `gib-only` | `GIB_area_frac`, `GIB_n_perimeter_hits` |
| `cmb` | `coast_n_source_hits`, `GIB_area_frac`, `GIB_n_perimeter_hits` |

Each diagnostic line reports:

- `n`: number of active regional cells;
- `max`: regional maximum over active cells;
- coordinate of the maximum cell;
- `p95`: 95th percentile over active cells;
- `sum`: regional sum over active cells.

The annotation is placed explicitly using `fig.text(x=..., y=...)`. For regional plots, the text is placed near the southern edge of the map using `y = lats[0] - 1`. For the full SH plot, it is placed at the centre of the Antarctic stereographic view.

Example plotting loop:

```python
from shuga import regions

D_root_out = Path('/g/data/gv90/da1339/GRAPHICAL/LD-pub-workspace/form-factors')
D_FF       = Path('/g/data/gv90/da1339/form_factors')

SH_REGION     = [-180, 180, -90, -60]
SH_PROJECTION = 'S0/-90/20c'
REGION_WIDTH_CM = 16
CPT_CMAP   = 'cmocean/matter'
CPT_SERIES = [0, 3]
POINT_STYLE = 's0.125c'

ANTARCTIC_REGIONS = regions.ANTARCTIC_8_REGIONS
ALL_REGIONS = ['SH'] + list(ANTARCTIC_REGIONS.keys())
```

The notebook contains the complete plotting helper functions used for the figure generation.

---

## Figure gallery links

After copying:

```text
/g/data/gv90/da1339/GRAPHICAL/LD-pub-workspace/form-factors/
```

to:

```text
shuga/docs/figs/
```

the following relative links should resolve from `shuga/docs/lateral_drag.md`.

| Product | Region | Figure link |
|---|---:|---|
| CST-only | AS | [FF_cst-only_cst-v7p9_AS.png](figs/cst_only/no_taper/AS/FF_cst-only_cst-v7p9_AS.png) |
| CST-only | Aus | [FF_cst-only_cst-v7p9_Aus.png](figs/cst_only/no_taper/Aus/FF_cst-only_cst-v7p9_Aus.png) |
| CST-only | BS | [FF_cst-only_cst-v7p9_BS.png](figs/cst_only/no_taper/BS/FF_cst-only_cst-v7p9_BS.png) |
| CST-only | DML | [FF_cst-only_cst-v7p9_DML.png](figs/cst_only/no_taper/DML/FF_cst-only_cst-v7p9_DML.png) |
| CST-only | EIO | [FF_cst-only_cst-v7p9_EIO.png](figs/cst_only/no_taper/EIO/FF_cst-only_cst-v7p9_EIO.png) |
| CST-only | SH | [FF_cst-only_cst-v7p9_SH.png](figs/cst_only/no_taper/SH/FF_cst-only_cst-v7p9_SH.png) |
| CST-only | VOL | [FF_cst-only_cst-v7p9_VOL.png](figs/cst_only/no_taper/VOL/FF_cst-only_cst-v7p9_VOL.png) |
| CST-only | WIO | [FF_cst-only_cst-v7p9_WIO.png](figs/cst_only/no_taper/WIO/FF_cst-only_cst-v7p9_WIO.png) |
| CST-only | WS | [FF_cst-only_cst-v7p9_WS.png](figs/cst_only/no_taper/WS/FF_cst-only_cst-v7p9_WS.png) |
| GIB-only | AS | [FF_gib-only_GIB-v1p2_AS.png](figs/gib_only/no_taper/AS/FF_gib-only_GIB-v1p2_AS.png) |
| GIB-only | Aus | [FF_gib-only_GIB-v1p2_Aus.png](figs/gib_only/no_taper/Aus/FF_gib-only_GIB-v1p2_Aus.png) |
| GIB-only | BS | [FF_gib-only_GIB-v1p2_BS.png](figs/gib_only/no_taper/BS/FF_gib-only_GIB-v1p2_BS.png) |
| GIB-only | DML | [FF_gib-only_GIB-v1p2_DML.png](figs/gib_only/no_taper/DML/FF_gib-only_GIB-v1p2_DML.png) |
| GIB-only | EIO | [FF_gib-only_GIB-v1p2_EIO.png](figs/gib_only/no_taper/EIO/FF_gib-only_GIB-v1p2_EIO.png) |
| GIB-only | SH | [FF_gib-only_GIB-v1p2_SH.png](figs/gib_only/no_taper/SH/FF_gib-only_GIB-v1p2_SH.png) |
| GIB-only | VOL | [FF_gib-only_GIB-v1p2_VOL.png](figs/gib_only/no_taper/VOL/FF_gib-only_GIB-v1p2_VOL.png) |
| GIB-only | WIO | [FF_gib-only_GIB-v1p2_WIO.png](figs/gib_only/no_taper/WIO/FF_gib-only_GIB-v1p2_WIO.png) |
| GIB-only | WS | [FF_gib-only_GIB-v1p2_WS.png](figs/gib_only/no_taper/WS/FF_gib-only_GIB-v1p2_WS.png) |
| Combined max | AS | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_AS.png](figs/cmb/no_taper/AS/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_AS.png) |
| Combined max | Aus | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_Aus.png](figs/cmb/no_taper/Aus/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_Aus.png) |
| Combined max | BS | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_BS.png](figs/cmb/no_taper/BS/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_BS.png) |
| Combined max | DML | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_DML.png](figs/cmb/no_taper/DML/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_DML.png) |
| Combined max | EIO | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_EIO.png](figs/cmb/no_taper/EIO/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_EIO.png) |
| Combined max | SH | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_SH.png](figs/cmb/no_taper/SH/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_SH.png) |
| Combined max | VOL | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_VOL.png](figs/cmb/no_taper/VOL/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_VOL.png) |
| Combined max | WIO | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_WIO.png](figs/cmb/no_taper/WIO/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_WIO.png) |
| Combined max | WS | [FF_cmb_meth-max_cst-v7p9_GIB-v1p2_WS.png](figs/cmb/no_taper/WS/FF_cmb_meth-max_cst-v7p9_GIB-v1p2_WS.png) |

---

## Recommended checks before using a file in CICE

### Check NetCDF metadata

```bash
ncks -m /g/data/gv90/da1339/form_factors/FF_combined_meth-max_cst-v7p9-Liu_GIB-v1p2-perimeter_Liu_CICE.nc
```

Expected core variables:

```text
float FFx(nj,ni)
float FFy(nj,ni)
float lat(nj,ni)
float lon(nj,ni)
```

### Check finite, non-negative fields in Python

```python
ds = xr.open_dataset(P_FF_cmb)
assert ds['FFx'].dims == ('nj', 'ni')
assert ds['FFy'].dims == ('nj', 'ni')
assert np.isfinite(ds['FFx']).all()
assert np.isfinite(ds['FFy']).all()
assert float(ds['FFx'].min()) >= 0.0
assert float(ds['FFy'].min()) >= 0.0
ds.close()
```

### Check CICE namelist consistency

```fortran
F2_file       = '/g/data/gv90/da1339/form_factors/FF_combined_meth-max_cst-v7p9-Liu_GIB-v1p2-perimeter_Liu_CICE.nc'
F2x_varname   = 'FFx'
F2y_varname   = 'FFy'
F2_map_method = 'max'
```

Make sure `F2_map_method` is not confused with the Python-side `FF_combine_method`. The former maps T cells to CICE E/N faces; the latter combines coastline and GIB source products.

---

## Common pitfalls and design decisions

### Do not write `FFx(ni, nj)`

The production convention is `FFx(nj, ni)` and `FFy(nj, ni)`. This matches the current CICE read pathway through the NetCDF Fortran interface.

### Do not pre-map to faces in Python

`shuga` writes T-cell source fields. CICE maps those fields to `F2E` and `F2N` internally.

### Do not use the coarse landmask as a geometry source

The CICE `kmt`/ocean mask is only a candidate-cell filter. The source geometry must come from the high-resolution coastline or GIB vector product. This avoids accidentally adding form factors to unrelated coastlines such as South America, Australia, or South Africa.

### Do not clip Liu-style fields to 1 by default

Length-density values can exceed 1. Use `clip_max` only for a deliberate bounded sensitivity experiment.

### Keep perimeter and area fraction analytically separate

For GIB products, perimeter density is a contact-boundary metric. Area fraction is an occupancy metric. The recommended default is `area_component_mode='diagnostic'`.

---

## References and source links

- Liu, Y. et al. (2022). *A new parameterization of coastal drag to simulate landfast ice in deep marginal seas in the Arctic*. Journal of Geophysical Research: Oceans. DOI: https://doi.org/10.1029/2022JC018413
- `shuga/grid/lateral_drag.py`: `FormFactors` class and form-factor generation workflow.
- `CICE_free-slip/cicecore/cicedyn/infrastructure/ice_grid.F90`: `load_F2_form_factors()` reads `FFx`/`FFy`, maps T-cell source fields to `F2E`/`F2N`, masks land velocity points, and applies halo updates. Source: https://github.com/dpath2o/CICE_free-slip/blob/main/cicecore/cicedyn/infrastructure/ice_grid.F90
- PyGMT `Figure.text` API used for regional diagnostic annotation: https://www.pygmt.org/latest/api/generated/pygmt.Figure.text.html
