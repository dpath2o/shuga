# shuga documentation

`shuga` is the post-processing package used in the Mawson's Chest workflow for CICE sea-ice output, with a particular focus on Antarctic fast-ice classification, metrics, plotting, experiment comparison, and preparation of selected forcing products.

The package is organised around a small number of public workflow objects:

- `RunSpec`, `ClassificationSpec`, `MetricsSpec`, `CICEGridSpec` and related spec classes define runtime configuration.
- `ShugaPaths` owns path construction and naming conventions.
- `IceHistoryLoader` and the `load_*` functions open CICE history, classified masks, and metrics stores.
- `CICEClassifier` builds fast-ice masks.
- `CICEMetrics` computes metric products from CICE history and classified masks.
- `CICEGridwork` and `CICEStaticBuilder` handle CICE grid assets and `iceh_static.zarr`.
- `WHACSRegridder` / `WHACSMultiSourceRegridder` prepare hourly WHACS wave spectra for the standalone CICE/Icepack wave-forcing pathway.

Most day-to-day usage should go through the public workflow classes or the loader functions. Lower-level modules exist so calculations can be tested and reused, but they are not normally required in notebooks or PBS scripts.

## Documentation map

| File | Purpose |
|---|---|
| `quickstart.md` | Minimal run-through from CICE history to classification and metrics. |
| `architecture.md` | Current module boundaries and design rules. |
| `io.md` | CICE history loading, classified-store loading, metrics-store loading, and store resolution. |
| `static-grid.md` | Building and using `iceh_static.zarr` when CICE history omits static fields. |
| `classification.md` | Fast-ice mask methods and classification outputs. |
| `metrics.md` | Metric groups, metric dispatch, secondary metrics, and outputs. |
| `plotting-observations.md` | Plotting and observation loader conventions. |
| [`WHACS_wave_forcing.md`](WHACS_wave_forcing.md) | Five-source WHACS directional spectra → CICE25 hourly forcing, spectral QC, spatial regridding, and output contract. |
| `forcing.md` | General forcing capability and design intent. |
| `gadi-workflows.md` | PBS wrapper use, Gadi paths, and common operational checks. |
| `developer-guide.md` | Maintenance rules, testing checks, and repo hygiene. |

## Naming

The package name is `shuga`.

Use:

```python
from shuga import RunSpec, ClassificationSpec, ShugaPaths
```

not older draft names such as `shugga` or `ShuggaPaths`.

## Typical working order

1. Convert or locate CICE history Zarr stores.
2. Ensure static fields are available in history or `iceh_static.zarr`.
3. Run classification for one or more methods.
4. Run metrics for the same methods.
5. Load metric stores into notebooks for figures and analysis.
6. Prepare external forcing products where required by an experiment, including WHACS wave spectra.
7. Use scripts/PBS wrappers for repeatable production runs.

## Public API reminder

Prefer these public imports in analysis notebooks and scripts:

```python
from shuga import (
    RunSpec,
    ClassificationSpec,
    MetricsSpec,
    CICEGridSpec,
    ShugaPaths,
    CICEClassifier,
    CICEMetrics,
    load_cice,
    load_classified,
    load_metrics,
)
from shuga.grid.static import CICEStaticBuilder
```

The internal helper modules are useful for tests and advanced development, but routine user code should not depend on private methods or implementation details.
