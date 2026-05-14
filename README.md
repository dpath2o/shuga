# shuga

`shuga` is a standalone CICE post-processing package for Antarctic fast-ice workflows. It replaces the older AFIM JSON-driven classification and metrics scripts with a shared, code-first package built around explicit runtime specifications, common path rules, and reusable loading logic.

The package has two primary peer workflows:

- `shuga.classify.CICEClassifier`: compute raw, binary-days, and rolling-mean fast-ice masks.
- `shuga.metrics.CICEMetrics`: compute fast-ice persistence, area, volume, thickness, and regional FIA/FIT from those masks.

The key design goal is to prevent path drift between modules. Classification and metrics both use the same `ShugaPaths` object, so directory naming is defined once and reused everywhere.

## Package layout

```text
shuga/
    core/
        logging.py
        naming.py
        paths.py
        regions.py
        types.py
    io/
        zarr_loading.py
    classify/
        cice.py
    metrics/
        cice.py

scripts/
    classification/
        classify.py
        classify.pbs
        classify_pbs_wrapper.sh
    metrics/
        metrics.py
        metrics.pbs
        metrics_pbs_wrapper.sh

notebooks/
    example_LD-waves-exp01_1993.ipynb
docs/
    quickstart.md
    architecture.md
```

## Defaults and path conventions

By default, `shuga` assumes the Gadi layout below:

- project: `gv90`
- user: `da1339`
- daily CICE store:
  `/g/data/[PROJECT]/[USER]/afim_output/[SIM_NAME]/zarr/iceh_daily.zarr`
- static store:
  `/g/data/[PROJECT]/[USER]/afim_output/[SIM_NAME]/zarr/iceh_static.zarr`
- classification root:
  `/g/data/[PROJECT]/[USER]/afim_output/[SIM_NAME]/zarr/[SH|NH]/ispd_thresh_[THRESH]/[ICE_TYPE]/[GRID_TYPE]/`
- classification stores:
  - `raw/data.zarr`
  - `bin-win-11_bin-min-09/data.zarr`
  - `roll-days-15/data.zarr`
- metrics stores:
  - `raw/mets.zarr`
  - `bin-win-11_bin-min-09/mets.zarr`
  - `roll-days-15/mets.zarr`
- FIP graphics:
  `/g/data/[PROJECT]/[USER]/GRAPHICAL/AFIM/[SIM_NAME]/FIP/`
- time-series graphics:
  `/g/data/[PROJECT]/[USER]/GRAPHICAL/AFIM/timeseries/`
- logs:
  `~/logs/classification/` and `~/logs/metrics/`

Threshold directories follow the shared formatting rule:

- `ispd_thresh_5.0e-4`

while log filenames use the compact form:

- `metrics_<SIM>_<ICE>_<GRID>_ispd_thresh5e-4_BW11_BM9_roll15.log`

## Fast-ice methods

`shuga` currently supports three mask products:

- `raw`: daily speed-threshold classification.
- `binary-days`: centered rolling count over the raw mask.
- `rolling-mean`: centered rolling mean of ice-speed magnitude before thresholding.

The classifier automatically pads the read window by half the binary or rolling window so edge dates are handled from available data and then cropped back to the requested interval.

## Regional outputs

The metrics workflow includes pan-Antarctic totals plus FIA/FIT for the eight Antarctic sectors:

- `DML`
- `WIO`
- `EIO`
- `Aus`
- `VOL`
- `AS`
- `BS`
- `WS`

## Command-line examples

### Classification

```bash
python scripts/classification/classify.py   --sim-name LD-waves-exp01   --start-date 1993-01-01   --end-date 1993-12-31   --hemisphere SH   --ice-type FI   --BorC2T-type Tc   --ispd-thresh 5e-4   --methods raw,binary-days,rolling-mean   --bin-window 11   --bin-min-days 9   --roll-window 15   --project gv90   --user da1339
```

### Metrics + plotting

```bash
python scripts/metrics/metrics.py   --sim-name LD-waves-exp01   --start-date 1993-01-01   --end-date 1993-12-31   --hemisphere SH   --ice-type FI   --BorC2T-type Tc   --ispd-thresh 5e-4   --methods raw,binary-days,rolling-mean   --bin-window 11   --bin-min-days 9   --roll-window 15   --project gv90   --user da1339   --plot-fip   --plot-fia   --plot-fit
```

### PBS wrapper usage

Classification:

```bash
./scripts/classification/classify_pbs_wrapper.sh   -s LD-waves-exp01 -b 1993-01-01 -e 1993-12-31 -H SH -i FI -g Tc   -t 5e-4 -m raw,binary-days,rolling-mean -B 11 -N 9 -R 15 -P gv90 -U da1339
```

Metrics:

```bash
./scripts/metrics/metrics_pbs_wrapper.sh   -s LD-waves-exp01 -b 1993-01-01 -e 1993-12-31 -H SH -i FI -g Tc   -t 5e-4 -m binary-days,rolling-mean -B 11 -N 9 -R 15 -P gv90 -U da1339   -f -a -T -r total
```

## Python API example

```python
from shuga import RunSpec, ClassificationSpec, MetricsSpec, ShugaPaths
from shuga.classify import CICEClassifier
from shuga.metrics import CICEMetrics

run = RunSpec(
    sim_name="LD-waves-exp01",
    start_date="1993-01-01",
    end_date="1993-12-31",
    hemisphere="SH",
)

classify = ClassificationSpec(
    ice_type="FI",
    grid_type="Tc",
    ispd_thresh=5e-4,
    methods=("raw", "binary-days", "rolling-mean"),
    bin_window=11,
    bin_min_days=9,
    roll_window=15,
)

paths = ShugaPaths(run=run, classify=classify)

classifier = CICEClassifier(run=run, classify=classify, paths=paths)
classifier.run_methods(overwrite=False)

metrics = CICEMetrics(run=run, classify=classify, paths=paths)
metrics.compute_metrics("binary-days", overwrite=False)
ds = metrics.load_metrics("binary-days")
```

## Notes

- The grouped-month `iceh_daily.zarr/YYYY-MM` layout is supported directly.
- `iceh_static.zarr` is merged automatically when present.
- Plotting methods use PyGMT and only import it when you actually request a plot.
- `load_classification()` and `load_metrics()` are included so downstream notebooks can read computed products quickly without rebuilding them.

See:
- `docs/quickstart.md`
- `docs/architecture.md`
- `notebooks/example_LD-waves-exp01_1993.ipynb`
