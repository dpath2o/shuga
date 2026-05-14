# Quickstart

This page gives the shortest practical workflow for running `shuga` from CICE history to classified masks and metrics.

The examples assume the common Gadi layout:

```text
/g/data/gv90/da1339/afim_output/<SIM_NAME>/
```

with CICE history and Zarr products beneath that simulation directory.

## 1. Put the repository on `PYTHONPATH`

From the repository root:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

For Gadi jobs, the PBS scripts usually set this explicitly. For notebook sessions, set it once in the shell that launches Jupyter.

## 2. Define the run context

```python
from shuga import RunSpec, ClassificationSpec, ShugaPaths

run = RunSpec(
    sim_name="LD-static-Cs2p5e-4",
    start_date="1993-01-01",
    end_date="1993-12-31",
    hemisphere="SH",
    project="gv90",
    user="da1339",
    iceh_frequency="daily",
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
```

The same `run`, `classify`, and `paths` objects should be reused by classification, metrics, plotting, and loaders. This prevents path drift.

## 3. Check CICE history loading

```python
from shuga import load_cice

ds = load_cice(
    run=run,
    classify=classify,
    paths=paths,
    variables=["aice", "hi", "tarea", "TLON", "TLAT"],
)

print(ds)
```

For daily grouped stores, this opens monthly Zarr groups such as:

```text
iceh_daily.zarr/1993-01
iceh_daily.zarr/1993-02
...
```

If requested static variables are not in the history store, `load_cice()` will also use `iceh_static.zarr` when present.

## 4. Run classification

```python
from shuga import CICEClassifier

classifier = CICEClassifier(run=run, classify=classify, paths=paths)
classifier.run_methods(overwrite=False)
```

Expected products are method-specific `data.zarr` stores:

```text
/g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr/SH/ispd_thresh_5.0e-4/FI/Tc/
├── raw/
│   └── data.zarr
├── bin-win-11_bin-min-09/
│   └── data.zarr
└── roll-days-15/
    └── data.zarr
```

## 5. Run metrics

```python
from shuga import CICEMetrics

metrics = CICEMetrics(run=run, classify=classify, paths=paths)
metrics.compute_metrics("binary-days", overwrite=False)
```

Expected metric products are `mets.zarr` stores beside the classified data:

```text
bin-win-11_bin-min-09/
├── data.zarr
└── mets.zarr
```

## 6. Load outputs for analysis

```python
from shuga import load_classified, load_metrics

classified = load_classified(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FI_mask"],
)

mets = load_metrics(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP", "FIA_by_region", "FIT_by_region"],
)

print(classified)
print(mets)
```

## 7. Command-line workflows

Classification:

```bash
python shuga/scripts/classification/classify.py \
  --sim-name LD-static-Cs2p5e-4 \
  --start-date 1993-01-01 \
  --end-date 1993-12-31 \
  --hemisphere SH \
  --ice-type FI \
  --BorC2T-type Tc \
  --ispd-thresh 5e-4 \
  --methods raw,binary-days,rolling-mean \
  --bin-window 11 \
  --bin-min-days 9 \
  --roll-window 15 \
  --project gv90 \
  --user da1339
```

Metrics:

```bash
python shuga/scripts/metrics/metrics.py \
  --sim-name LD-static-Cs2p5e-4 \
  --start-date 1993-01-01 \
  --end-date 1993-12-31 \
  --hemisphere SH \
  --ice-type FI \
  --BorC2T-type Tc \
  --ispd-thresh 5e-4 \
  --methods binary-days \
  --bin-window 11 \
  --bin-min-days 9 \
  --roll-window 15 \
  --project gv90 \
  --user da1339
```

Use `--help` on the scripts for the authoritative list of flags.

## 8. Basic sanity checks

After a run:

```bash
python -m compileall shuga
```

and in Python:

```python
from shuga import load_metrics

ds = load_metrics(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP"],
)
print(ds)
```

Expected dimensions:

- `FIA(time)`
- `FIT(time)`
- `FIP(nj, ni)`
- regional variables such as `FIA_by_region(time, region)`
