# Static grid

`shuga` uses a universal CICE 1/4-degree C-grid static-coordinate store:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

This store is grid-level, not simulation-level. It is shared by classification, metrics, plotting, observation comparison, regridding, and NetCDF-to-Zarr conversion workflows.

## Why it exists

CICE history files can contain static grid variables such as `TLON`, `TLAT`, `tarea`, and `HTE`. Writing those repeatedly in every daily or hourly file wastes storage and slows analysis. `shuga` therefore stores dynamic fields in simulation-specific grouped Zarr stores and static fields once in the universal grid store.

## Expected fields

| Field | Grid | Meaning |
|---|---|---|
| `TLON`, `TLAT` | T | T-cell longitude/latitude |
| `ULON`, `ULAT` | U/corner | U/corner longitude/latitude |
| `ELON`, `ELAT` | E | East-face longitude/latitude |
| `NLON`, `NLAT` | N | North-face longitude/latitude |
| `tarea`, `uarea`, `earea`, `narea` | area | grid-cell/face areas |
| `dxt`, `dyt`, `dxu`, `dyu`, `dxe`, `dye`, `dxn`, `dyn` | metric | grid spacings |
| `HTE`, `HTN` | metric | CICE grid metrics |
| `ANGLE`, `ANGLET` | angle | grid orientation |
| `tmask`, `umask`, `emask`, `nmask` | mask | grid masks |
| `NCAT` | scalar | number of ice categories |

## Loading

```python
from shuga.core.paths import ShugaPaths
from shuga.grid.cice import CICEGridwork

paths = ShugaPaths()
static = CICEGridwork(paths=paths).load_cice_static(
    variables=["TLON", "TLAT", "tarea", "uarea", "earea", "narea"],
    require=("TLON", "TLAT"),
)
print(static)
```

The loader supports both proper xarray Zarr Dataset stores and existing loose per-variable Zarr-array directories.

## Building

Static-grid construction belongs in `shuga/grid/static.py`.

```python
from shuga import RunSpec, ClassificationSpec, CICEGridSpec, ShugaPaths
from shuga.grid.static import CICEStaticBuilder

run = RunSpec(sim_name="LD-blend-base", start_date="2000-01-01", end_date="2000-12-31", hemisphere="SH")
classify = ClassificationSpec(grid_type="Tc")
grid_spec = CICEGridSpec(grid_file="/path/to/CICE_grid.nc", kmt_file="/path/to/kmt.nc")
paths = ShugaPaths(run=run, classify=classify, cice_grid=grid_spec)

builder = CICEStaticBuilder(paths)
builder.write_zarr_from_resolved_assets(overwrite=False, require_metadata=False)
```

The default target is `paths.resolve_static_store_target()`.

## Shape safety

The static builder should not silently crop or reshape grid masks. A source mask with shape `(1152, 1440)` is not equivalent to a target CICE grid with shape `(1080, 1440)`. Silent cropping can corrupt spatial alignment.

## Development rules

- Static-grid construction belongs in `grid/static.py`.
- Static-grid loading belongs in `grid/cice.py`.
- Automatic static/dynamic merging belongs in `io/iceh_loading.py`.
- Do not add static-grid reconstruction logic to classification, metrics, plotting, or notebooks.
