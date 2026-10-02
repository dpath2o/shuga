# Plotting

`shuga.plotting` provides plotting helpers and PyGMT-oriented data-preparation utilities for common CICE/fast-ice products. The aim is not to replace publication-specific notebooks, but to provide reliable starting points for FIP maps, FIP difference maps, area/thickness time series, regional panels, and quicklook products.

## Philosophy

Metrics should be computed once and stored in `mets.zarr`. Plotting should read existing products. Figure notebooks can then use `CICEPlotter` to either make standard plots or prepare data for custom PyGMT figures.

A publication notebook should:

1. define experiments, metadata, date windows, and regions;
2. load precomputed classification/metric products;
3. use `CICEPlotter` for robust loading and PyGMT preparation;
4. customise final panels, labels, annotations, and colour scales in the notebook.

## PyGMT

Most map plotting uses [PyGMT](https://www.pygmt.org/). PyGMT is imported lazily, so non-plotting workflows can still run on compute nodes. If plotting fails at import time, check the active module/conda environment first.

## Primary class

```python
from shuga import CICEPlotter

plotter = CICEPlotter(run=run, classify=classify, metrics=metrics, plotting=plotting, paths=paths)
```

`CICEMetrics` also exposes convenience wrappers such as `metrics.plot_fip()` and `metrics.plot_timeseries()`.

## FIP maps

Simulation FIP is loaded from a precomputed metrics store:

```python
plotter.plot_fip(method="binary-days", source="sim", field="FIP", region_name="Aus")
```

AF2020 FIP can be plotted from a persistent common-grid store:

```python
plotter.plot_fip(
    source="af2020",
    af2020_store="~/AFIM_archive/FI_obs/AF2020_common_grid.zarr",
    af2020_start="2000-03-01",
    af2020_end="2018-02-15",
    region_name="Aus",
)
```

FIP differences can be plotted from the output of `FIP_differencing.py`:

```python
plotter.plot_fip(source="dataset", dataset="/path/to/FIP_difference.zarr", field="diff", region_name="Aus")
plotter.plot_fip(source="dataset", dataset="/path/to/FIP_difference.zarr", field="diff_cat", region_name="Aus")
```

## Time series

Common variables include `FIA`, `FIT`, `FIV`, `PIA`, `PIT`, `PIV`, `SIA`, `SIT`, and `SIV`.

```python
plotter.plot_timeseries("FIA", method="binary-days", region="total", add_obs=True)
plotter.plot_timeseries("FIT", method="binary-days", region="Aus")
```

Regional time series use products such as `FIA_by_region` and `FIT_by_region`.

## PyGMT data preparation

`CICEPlotter` includes helpers such as `pygmt_da_prep()` to flatten a 2-D `DataArray` with lon/lat into a table suitable for custom PyGMT calls.

```python
ds = load_metrics(run=run, classify=classify, paths=paths, classification="binary-days", variables=["FIP"])
static = plotter._load_static_lonlat()
lon, lat = plotter._detect_lonlat(static)

df = plotter.pygmt_da_prep(ds["FIP"], lon=lon, lat=lat, mask_zero=True, region=[90, 170, -70, -55])
```

## Observational overlays

| Variable | Observation source |
|---|---|
| `FIA` | AF2020 Antarctic fast-ice area |
| `SIA` | NSIDC sea-ice concentration/area |

Observation modules load/prepare references; metrics compute reusable products; plotting renders figures.

## Other plotting modules

| Module | Purpose |
|---|---|
| `plotting/cice.py` | CICE classification/metrics plotting and PyGMT prep. |
| `plotting/era5.py` | ERA5 forcing quicklook plotting. |
| `plotting/cawcr.py` | CAWCR wave-product plotting helpers. |

Waves and tides are intentionally not documented in detail yet.
