# Architecture

`shuga` is organised so that workflow classes orchestrate analysis, while reusable functions live in small modules. The package was refactored to reduce duplicated loader logic, remove private helper sprawl, and keep path conventions in one place.

## Core design rules

1. `ShugaPaths` owns path construction.
2. `IceHistoryLoader` owns CICE history loading.
3. `CICEStoreLocator` owns classification/metrics store resolution.
4. `CICEClassifier` owns classification workflow orchestration.
5. `CICEMetrics` owns metrics workflow orchestration.
6. Pure calculations live outside workflow classes.
7. Static-grid construction belongs in `grid/static.py`, not in conversion scripts.
8. Plotting, observations, regridding, and wave tools remain peer modules.

## Current package layout

```text
shuga/
├── classify/
│   └── cice.py
├── core/
│   ├── data_conversion.py
│   ├── logging.py
│   ├── naming.py
│   ├── paths.py
│   ├── regions.py
│   ├── reporting.py
│   ├── store_selection.py
│   └── types.py
├── grid/
│   ├── cice.py
│   ├── geometry.py
│   ├── lateral_drag.py
│   └── static.py
├── io/
│   ├── iceh_loading.py
│   ├── store_locator.py
│   ├── zarr_loading.py
│   └── zarr_writing.py
├── metrics/
│   ├── calculations.py
│   ├── cice.py
│   ├── dispatch.py
│   ├── io.py
│   ├── regional.py
│   ├── registry.py
│   ├── secondary.py
│   ├── skill.py
│   ├── stress.py
│   └── temporal.py
├── observations/
├── plotting/
├── regridding/
└── waves/
```

## Runtime specifications

### `RunSpec`

The run-level context:

- simulation name
- date window
- hemisphere
- project
- user
- CICE history frequency

### `ClassificationSpec`

The mask/classification context:

- ice type
- grid type
- speed threshold
- velocity variable names
- concentration threshold
- classification methods
- binary-days parameters
- rolling-mean parameters

### `MetricsSpec`

The metric context:

- requested metric groups
- area and volume scales
- observation metric store
- observation variable names
- optional distance/coast fields

### `CICEGridSpec`

The grid context:

- CICE/ocean grid file
- KMT or mask file
- bathymetry/topography file
- longitude convention

## Path ownership

`ShugaPaths` is the single source of truth for output layout. It is responsible for:

- CICE history store paths
- static store paths
- classification roots
- `data.zarr` paths
- `mets.zarr` paths
- graphics paths
- log paths
- grid asset resolution

No classification or metrics script should hand-build these paths. If a new output product needs a path, add a method or property to `ShugaPaths`.

## IO boundary

### `io/iceh_loading.py`

Owns CICE history loading. It handles daily/hourly grouped Zarr stores, flat Zarr stores, requested variable filtering, static/dynamic variable splitting, `iceh_static.zarr` merge, time slicing, and hemisphere masking.

### `io/zarr_loading.py`

Public façade for:

- `load_cice`
- `load_classified`
- `load_metrics`
- `open_cice_history`

This module should stay thin. It resolves user-facing context and delegates real history loading to `IceHistoryLoader` and store resolution to `CICEStoreLocator`.

### `io/store_locator.py`

Finds `data.zarr` and `mets.zarr` stores for a simulation/method/grid context. This avoids duplicating method-path logic in classification, metrics, plotting, and notebooks.

## Grid boundary

### `grid/cice.py`

Loads CICE-compatible grid geometry. It should not own conversion workflow or metric logic.

### `grid/geometry.py`

Pure geometry and unit helpers: longitude normalisation, degree conversion, angle conversion, metric/area conversion, and dimension-coordinate preservation.

### `grid/static.py`

Builds `iceh_static.zarr` from resolved CICE grid assets. This is used when CICE history was written with static grid fields disabled.

## Metrics boundary

`metrics/cice.py` contains the public `CICEMetrics` workflow. It should mainly load data, create dispatch contexts, request primary and secondary metrics, and write stores.

The supporting modules are:

| Module | Responsibility |
|---|---|
| `registry.py` | Metric names and groups. |
| `calculations.py` | Pure metric calculations. |
| `dispatch.py` | Primary metric dispatch and memoisation. |
| `secondary.py` | Seasonal summaries, FIPSI, observation skill, metadata. |
| `stress.py` | Stress diagnostics. |
| `temporal.py` | Extrema tables and seasonal growth/retreat rates. |
| `regional.py` | Region masks and spatial dimension helpers. |
| `skill.py` | Bias/RMSE/MAE/correlation. |
| `io.py` | Metrics-store IO helpers. |

## Extension pattern

When adding a new metric:

1. Add the metric name to `metrics/registry.py`.
2. Put any pure calculation in `metrics/calculations.py`, `metrics/stress.py`, or a new small module.
3. Add primary dispatch to `metrics/dispatch.py` if it is a primary metric.
4. Add secondary orchestration to `metrics/secondary.py` if it depends on another metric.
5. Keep `CICEMetrics` as the workflow layer, not the calculation layer.
