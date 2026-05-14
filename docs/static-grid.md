# Static grid products

CICE history files may be written without static grid fields to save disk space. In that case, `shuga` can build a separate `iceh_static.zarr` store from grid assets and merge it during loading.

## Why `iceh_static.zarr` exists

Many diagnostics require static fields:

- `TLON`
- `TLAT`
- `ULON`
- `ULAT`
- `ELON`
- `ELAT`
- `NLON`
- `NLAT`
- `tarea`
- `uarea`
- `earea`
- `narea`
- `HTE`
- `HTN`
- `ANGLE`
- `ANGLET`
- `NCAT`

If these are disabled in `ice_in`, the dynamic history files remain usable for time-dependent variables but are insufficient for metrics, plotting, and regional reductions.

## Builder

Use:

```python
from shuga import RunSpec, ClassificationSpec, CICEGridSpec, ShugaPaths
from shuga.grid.static import CICEStaticBuilder

run = RunSpec(
    sim_name="LD-static-Cs2p5e-4",
    start_date="1993-01-01",
    end_date="1993-01-31",
    hemisphere="SH",
    project="gv90",
    user="da1339",
)

classify = ClassificationSpec(grid_type="Tc")

grid = CICEGridSpec(
    grid_file="/home/581/da1339/grids/ACCESS-OM3-025_ocean_hgrid.nc",
    kmt_file="/home/581/da1339/grids/ACCESS-OM3-025_kmt_super.nc",
    bathymetry_file="/g/data/gv90/da1339/grids/ACCESS-OM3-025_topog.nc",
)

paths = ShugaPaths(run=run, classify=classify, cice_grid=grid)

builder = CICEStaticBuilder(paths)
ds_static = builder.build_dataset_from_resolved_assets(require_metadata=False)
print(ds_static)
```

To write:

```python
builder.write_zarr_from_resolved_assets(
    overwrite=False,
    require_metadata=True,
)
```

The default target is:

```text
/g/data/<PROJECT>/<USER>/afim_output/<SIM_NAME>/zarr/iceh_static.zarr
```

## Metadata inputs

The static builder prefers run metadata files when available:

1. `ice_in`
2. `ice_diag.d`

These are used for provenance and fields such as `NCAT`.

If neither exists and `require_metadata=True`, the builder will not write the static store. For testing grid geometry only, use:

```python
require_metadata=False
```

## Shape safety

The builder does not crop or reshape masks to fit.

Example warning:

```text
Skipping /home/581/da1339/grids/ACCESS-OM3-025_kmt_super.nc mask because shape
(1152, 1440) does not match target T-grid shape (1080, 1440).
```

This is deliberate. A 1152-row supergrid mask is not equivalent to a 1080-row CICE T-grid mask. Silently cropping it would corrupt spatial alignment.

If an incompatible mask is skipped, the static store can still contain geometry fields such as `TLON`, `TLAT`, `tarea`, `HTE`, and `HTN`. Metrics that require area and coordinates can still work.

## Expected dataset structure

A successful static dataset may look like:

```text
Dimensions:
  nj:   1080
  ni:   1440
  nj_b: 1081
  ni_b: 1441

Coordinates:
  TLON(nj, ni)
  TLAT(nj, ni)
  ULON(nj_b, ni_b)
  ULAT(nj_b, ni_b)
  ELON(nj, ni_b)
  ELAT(nj, ni_b)
  NLON(nj_b, ni)
  NLAT(nj_b, ni)

Data variables:
  ANGLET(nj, ni)
  ANGLE(nj, ni)
  HTE(nj, ni)
  HTN(nj, ni)
  tarea(nj, ni)
  uarea(nj, ni)
  earea(nj, ni)
  narea(nj, ni)
  dxt/dyt/dxu/dyu/dxe/dye/dxn/dyn
  NCAT
```

`tmask` may be absent if no compatible mask was found.

## Loading with static fields

Once written, load normally:

```python
from shuga import load_cice

ds = load_cice(
    run=run,
    classify=classify,
    paths=paths,
    variables=["aice", "hi", "tarea", "TLON", "TLAT"],
)
```

`load_cice()` will merge static fields automatically.

## Development notes

Static-grid construction belongs in:

```text
shuga/grid/static.py
```

Pure coordinate and unit helpers belong in:

```text
shuga/grid/geometry.py
```

Do not add static-grid reconstruction logic to conversion scripts, loaders, classifier code, or metrics code.
