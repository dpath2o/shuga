# Plotting and observations

Plotting and observation loading are separate from metrics. Metrics should produce reusable data products; plotting should read those products and render figures.

## Plotting

Primary plotting functionality lives in:

```text
shuga/plotting/cice.py
```

The public plotting object is usually available as:

```python
from shuga import CICEPlotter
```

or through convenience methods on `CICEMetrics`:

```python
metrics.plot_fip(...)
metrics.plot_timeseries(...)
metrics.plot_var_by_region(...)
```

The convenience methods create a `CICEPlotter` with the same `run`, `classify`, `metrics`, and `paths` objects.

## PyGMT

Several plotting methods use PyGMT. PyGMT is imported lazily, so importing `shuga` does not require a working PyGMT environment.

If a plotting call fails with a PyGMT import error, check the active conda/module environment rather than the metric computation.

## Common plotting products

Typical products include:

- FIP maps;
- FIA/FIT time series;
- regional panels;
- split-hemisphere maps;
- triptych maps for model/observation/difference-style plots.

## Recommended plotting pattern

Compute and save metrics first:

```python
metrics.compute_metrics("binary-days", overwrite=False)
```

Then load and plot from the existing store:

```python
ds = metrics.load_metrics("binary-days")
```

or through the public loader:

```python
from shuga import load_metrics

ds = load_metrics(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
)
```

This makes plotting reproducible and avoids recomputing metrics inside figure notebooks.

## Observations

Observation utilities live in:

```text
shuga/observations/cice.py
```

The public observation helper is:

```python
from shuga import SeaIceObservations
```

Observation support is used for:

- AF2020 fast-ice comparisons;
- NSIDC sea-ice area/extent comparisons;
- observation metric stores used by skill diagnostics;
- repeated daily climatologies for overlay plots.

## Skill metrics

Observation skill metrics are computed only when the configured observation metrics store and variable names are available through `MetricsSpec`.

Common skill outputs include:

- `FIA_Bias`
- `FIA_RMSE`
- `FIA_MAE`
- `FIA_Corr`
- `FIT_Bias`
- `FIT_RMSE`
- `FIT_MAE`
- `FIT_Corr`

If the observation store is absent, skill metrics are skipped with a warning.

## Separation rule

Do not put plotting code inside metric calculations. Use this split:

```text
metrics/
  compute reusable products

plotting/
  render products into figures

observations/
  load/prepare observational references
```

This keeps production metric jobs usable on headless compute nodes and keeps notebooks lightweight.
