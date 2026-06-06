# Architecture

`shuga` is organised so workflow classes orchestrate analysis, while reusable calculations, path resolution, IO, grid handling, observations, regridding, and plotting live in focused modules.

## Design rules

1. `ShugaPaths` owns path construction.
2. `IceHistoryLoader` owns CICE history loading.
3. `CICEStoreLocator` owns classification/metrics store resolution.
4. `CICEClassifier` owns classification orchestration.
5. `CICEMetrics` owns metrics orchestration.
6. Pure calculations belong in small helper modules.
7. Static-grid construction belongs in `grid/static.py`.
8. Static-grid loading belongs in `grid/cice.py`.
9. Plotting reads existing products and prepares/renders figures.
10. Scripts are thin wrappers around package code.

## Package layout

```text
shuga/
├── classify/
├── core/
├── forcing/
├── grid/
├── io/
├── metrics/
├── observations/
├── plotting/
├── regridding/
└── scripts/
```

## Runtime specifications

| Spec | Purpose |
|---|---|
| `RunSpec` | Simulation name, dates, hemisphere, project/user, frequency. |
| `ClassificationSpec` | Ice domain, grid type, thresholds, methods, windows. |
| `MetricsSpec` | Metric groups, scaling, observation skill settings. |
| `PlottingSpec` | Common figure settings. |
| `ObservationSpec` | Observation roots and variables. |
| `CICEGridSpec` | CICE grid assets. |
| `ShugaPaths` | Path authority. |

## Boundaries

### Classification

`classify/cice.py` loads CICE fields, computes T-grid speed, classifies `FI`, derives `PI`, and writes `data.zarr`. It should not build paths manually or compute metrics.

### Metrics

`metrics/cice.py` loads CICE and classified products, creates a dispatch context, computes requested primary/secondary metrics, and writes `mets.zarr`. Pure mathematics belongs in supporting metrics modules.

### IO

`io/iceh_loading.py` owns grouped CICE Zarr loading, static/dynamic splitting, static-grid merge, time slicing, and hemisphere masking. `io/zarr_loading.py` is the public facade. `io/store_locator.py` resolves `data.zarr` and `mets.zarr`.

### Grid

`grid/cice.py` loads CICE grid/static products. `grid/static.py` builds the universal static-grid store. `grid/geometry.py` contains dimension, longitude, angle, metric, and area helpers.

### Plotting

`plotting/cice.py` provides common plots and PyGMT data preparation. Publication-specific figure composition should normally remain in notebooks.

## Extension pattern

When adding a metric: update `registry.py`, add pure calculation/dispatch, add required source fields, and update `docs/metrics.md`.

When adding a classification method: update method normalisation, path naming if needed, classification logic, scripts, and `docs/classification.md`.

When adding an observation product: add a focused module under `observations/`, keep regridding/comparison separate from plotting, and update `docs/observations.md`.
