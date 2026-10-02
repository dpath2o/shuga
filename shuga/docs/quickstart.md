# Quickstart

This page gives a minimal end-to-end `shuga` workflow: load CICE history, classify fast ice, compute metrics, and plot/load outputs.

## 1. Put the repo on `PYTHONPATH`

```bash
cd ~/AFIM/src/mawsons-chest
export PYTHONPATH="$PWD:$PYTHONPATH"
```

For notebooks:

```python
from pathlib import Path
import sys
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
```

## 2. Define specs

```python
from shuga import RunSpec, ClassificationSpec, MetricsSpec, PlottingSpec, ShugaPaths

run = RunSpec(
    sim_name="LD-blend-base",
    start_date="2000-04-01",
    end_date="2003-06-30",
    hemisphere="SH",
    project="gv90",
    user="da1339",
    iceh_frequency="daily",
)

classify = ClassificationSpec(
    ice_type="FI",
    grid_type="Tc",
    ispd_thresh=5.0e-4,
    aice_thresh=0.15,
    methods=("raw", "binary-days", "rolling-mean"),
    bin_window=11,
    bin_min_days=9,
    roll_window=15,
)

metrics = MetricsSpec(methods=("binary-days",))
plotting = PlottingSpec(fig_size=20.0)
paths = ShugaPaths(run=run, classify=classify, metrics=metrics, plotting=plotting)
```

## 3. Check CICE loading

```python
from shuga import load_cice

ds = load_cice(run=run, classify=classify, paths=paths, variables=["aice", "hi", "TLON", "TLAT", "tarea"])
print(ds)
```

## 4. Convert NetCDF to grouped Zarr, if needed

```bash
python shuga/scripts/conversion/nc2zarr.py \
  --sim-name LD-blend-base \
  --start-date 2000-01-01 \
  --end-date 2003-12-31 \
  --iceh-frequency daily
```

## 5. Run classification

```python
from shuga import CICEClassifier
CICEClassifier(run=run, classify=classify, paths=paths).run_methods(overwrite=False)
```

or:

```bash
python shuga/scripts/classification/classify.py \
  --sim-name LD-blend-base \
  --start-date 2000-04-01 \
  --end-date 2003-06-30 \
  --hemisphere SH \
  --ice-type FI \
  --BorC2T-type Tc \
  --ispd-thresh 5e-4 \
  --methods raw,binary-days,rolling-mean \
  --bin-window 11 \
  --bin-min-days 9 \
  --roll-window 15 \
  --skip-history-conversion
```

## 6. Run metrics

```python
from shuga import CICEMetrics
runner = CICEMetrics(run=run, classify=classify, metrics=metrics, paths=paths)
runner.compute_metrics("binary-days", metric_groups=["fi_core", "fi_spatial", "fi_regional", "fi_summary"], update_missing_only=True)
```

or:

```bash
python shuga/scripts/metrics/metrics.py \
  --sim-name LD-blend-base \
  --start-date 2000-04-01 \
  --end-date 2003-06-30 \
  --hemisphere SH \
  --ice-type FI \
  --BorC2T-type Tc \
  --ispd-thresh 5e-4 \
  --methods binary-days \
  --metric-groups fi_core,fi_spatial,fi_regional,fi_summary \
  --update-missing-only
```

## 7. Load and plot

```python
from shuga import load_classified, load_metrics, CICEPlotter

classified = load_classified(run=run, classify=classify, paths=paths, classification="binary-days", variables=["FI_mask", "PI_mask"])
mets = load_metrics(run=run, classify=classify, metrics=metrics, paths=paths, classification="binary-days", variables=["FIA", "FIT", "FIP"])

plotter = CICEPlotter(run=run, classify=classify, metrics=metrics, plotting=plotting, paths=paths)
plotter.plot_fip(method="binary-days", source="sim", region_name="Aus")
plotter.plot_timeseries("FIA", method="binary-days", region="total")
```

## 8. PBS examples

```bash
qsub -v SIM_NAME=LD-blend-base,START_DATE=2000-04-01,END_DATE=2003-06-30,SKIP_HISTORY_CONVERSION=true \
  shuga/scripts/classification/classify.pbs
```

```bash
qsub -v SIM_NAME=LD-blend-base,START_DATE=2000-04-01,END_DATE=2003-06-30,METRIC_GROUPS=fi_core\|fi_spatial\|fi_regional \
  shuga/scripts/metrics/metrics.pbs
```
