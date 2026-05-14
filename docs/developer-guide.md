# Developer guide

This guide is for maintaining `shuga` after the refactor.

## Development principles

1. Keep public workflow classes small enough to read.
2. Prefer pure helper functions for reusable calculations.
3. Avoid duplicate path logic.
4. Avoid duplicate loader logic.
5. Keep generated data and job outputs out of Git.
6. Test with a small real-store smoke test before pushing workflow changes.

## Where new code belongs

| New code | Location |
|---|---|
| Path rules | `core/paths.py` |
| Runtime configuration | `core/types.py` |
| Method/path naming | `core/naming.py` |
| Store discovery | `io/store_locator.py` |
| CICE history loading | `io/iceh_loading.py` |
| Public loaders | `io/zarr_loading.py` |
| Zarr write cleanup | `io/zarr_writing.py` |
| Grid loading | `grid/cice.py` |
| Static-grid store building | `grid/static.py` |
| Geometry/unit helpers | `grid/geometry.py` |
| Metric names/groups | `metrics/registry.py` |
| Pure metric calculations | `metrics/calculations.py` |
| Primary metric dispatch | `metrics/dispatch.py` |
| Secondary metric orchestration | `metrics/secondary.py` |
| Stress diagnostics | `metrics/stress.py` |
| Extrema/growth/retreat tables | `metrics/temporal.py` |
| Skill stats | `metrics/skill.py` |
| Regional masks | `metrics/regional.py` |
| Metrics-store helpers | `metrics/io.py` |

## Adding a metric

1. Add the metric name to `metrics/registry.py`.
2. Add calculation code to a pure helper module.
3. Add primary dispatch in `metrics/dispatch.py` if the metric is directly computed.
4. Add secondary orchestration in `metrics/secondary.py` if it derives from another metric.
5. Add any required source fields to `_get_cice()` in `metrics/cice.py`.
6. Add a smoke test using a small date range.
7. Update `docs/metrics.md`.

Do not add another long `if/elif` chain to `CICEMetrics`.

## Adding a classification method

1. Add method normalisation in `core/naming.py` if needed.
2. Add path construction to `ShugaPaths`.
3. Add classification logic to `classify/cice.py` or a focused helper module.
4. Update `CICEStoreLocator` if the store layout differs.
5. Update scripts and wrappers.
6. Update `docs/classification.md`.

## Testing checks

Always run:

```bash
python -m compileall shuga
```

Then run one lightweight Python smoke test. For metrics:

```bash
PYTHONPATH=$PWD python - <<'PY'
from shuga import RunSpec, ClassificationSpec, ShugaPaths, CICEMetrics

run = RunSpec(
    sim_name="LD-static-Cs2p5e-4",
    start_date="1993-01-01",
    end_date="1993-01-31",
    hemisphere="SH",
    project="gv90",
    user="da1339",
)

cls = ClassificationSpec(grid_type="Tc", methods=("binary-days",))
paths = ShugaPaths(run=run, classify=cls)

m = CICEMetrics(run=run, classify=cls, paths=paths)

ds = m._compute_requested_metrics(
    "binary-days",
    {"FIA", "FIT", "FIP", "SIA", "SIT"},
)

print(ds)
PY
```

## Repo hygiene

Before committing:

```bash
find shuga -type d -name "__pycache__" -prune -exec rm -rf {} +
find shuga -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".#*" \) -delete
find shuga -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find shuga/scripts -type f \( -name "*.o[0-9]*" -o -name "*.e[0-9]*" -o -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
```

Then check for tracked detritus:

```bash
git ls-files \
  '*/__pycache__/*' \
  '*.pyc' \
  '*.pyo' \
  '*.o[0-9]*' \
  '*.e[0-9]*' \
  '*.log' \
  '.#*' \
  '*/.#*' \
  '.ipynb_checkpoints/*' \
  '*/.ipynb_checkpoints/*'
```

This should print nothing.

## Version consistency

Keep these in sync:

- `shuga/pyproject.toml`
- `shuga/__init__.py`

Check:

```bash
grep -n "version" shuga/pyproject.toml
grep -n "__version__" shuga/__init__.py
```

## Documentation checks

Before writing or publishing docs:

```bash
grep -RIn "shugga\|ShuggaPaths" shuga/README.md shuga/docs || true
```

This should print nothing. Use:

- `shuga`
- `ShugaPaths`

## Commit style

Use small commits with a single purpose. Examples:

```text
Extract metrics stress helpers
Add static-grid documentation
Remove generated PBS output artifacts
Synchronise package version
```

Avoid mixed commits that combine generated output deletion, refactors, and documentation.
