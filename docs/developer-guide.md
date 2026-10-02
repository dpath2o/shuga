# Developer guide

This guide is for maintaining `shuga` during active research development.

## Principles

1. Keep workflow classes readable.
2. Keep path rules centralised in `ShugaPaths`.
3. Keep store discovery centralised in `CICEStoreLocator`.
4. Keep CICE history loading centralised in `IceHistoryLoader`.
5. Prefer pure helper functions for reusable calculations.
6. Keep scripts thin.
7. Keep generated data out of Git.
8. Run small real-store smoke tests before pushing workflow changes.
9. Update documentation whenever a public workflow or store layout changes.

## Where code belongs

| Task | Location |
|---|---|
| Path rules | `core/paths.py` |
| Runtime dataclasses | `core/types.py` |
| Naming/method normalisation | `core/naming.py` |
| CICE history loading | `io/iceh_loading.py` |
| Public load facade | `io/zarr_loading.py` |
| Store discovery | `io/store_locator.py` |
| Zarr write cleanup | `io/zarr_writing.py` |
| CICE grid loading | `grid/cice.py` |
| Static-grid construction | `grid/static.py` |
| Geometry/unit helpers | `grid/geometry.py` |
| Classification | `classify/cice.py` |
| Metric names/groups | `metrics/registry.py` |
| Pure metrics | `metrics/calculations.py` |
| Metric dispatch | `metrics/dispatch.py` |
| Secondary summaries/skill | `metrics/secondary.py` |
| Stress diagnostics | `metrics/stress.py` |
| Temporal summaries | `metrics/temporal.py` |
| Regional masks | `metrics/regional.py` |
| Observations | `observations/<PRODUCT>.py` |
| Plotting/data prep | `plotting/*.py` |
| Regridding | `regridding/*.py` |
| PBS/CLI entry points | `scripts/*` |

## Versioning

For this documentation update, use version `0.5.0` in both `shuga/__init__.py` and `pyproject.toml` when ready.

## Smoke tests

```bash
python -m compileall shuga
```

Static-grid test:

```bash
python - <<'PY'
from shuga.core.paths import ShugaPaths
from shuga.grid.cice import CICEGridwork
paths = ShugaPaths()
print(paths.resolve_static_store())
print(CICEGridwork(paths=paths).load_cice_static(variables=["TLON", "TLAT", "tarea"]))
PY
```

Loader test:

```bash
python - <<'PY'
from shuga import RunSpec, ClassificationSpec, ShugaPaths, load_cice
run = RunSpec(sim_name="LD-blend-base", start_date="2000-04-01", end_date="2000-04-03", hemisphere="SH")
cls = ClassificationSpec(ice_type="FI", grid_type="Tc")
paths = ShugaPaths(run=run, classify=cls)
print(load_cice(run=run, classify=cls, paths=paths, variables=["aice", "TLON", "TLAT", "tarea"]))
PY
```

## Hygiene

```bash
find shuga -type d -name "__pycache__" -prune -exec rm -rf {} +
find shuga -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".#*" \) -delete
find shuga -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find shuga/scripts -type f \( -name "*.o[0-9]*" -o -name "*.e[0-9]*" -o -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
git status --short
```
