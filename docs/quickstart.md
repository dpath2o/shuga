# Quickstart

This page shows the shortest path from a CICE experiment to classified fast-ice masks, metrics, and plots using `shugga`.

## 1. Put the repo on `PYTHONPATH`

From the repo root:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## 2. Run classification

```bash
python scripts/classification/classify.py   --sim-name LD-waves-exp01   --start-date 1993-01-01   --end-date 1993-12-31   --hemisphere SH   --ice-type FI   --BorC2T-type Tc   --ispd-thresh 5e-4   --methods raw,binary-days,rolling-mean   --bin-window 11   --bin-min-days 9   --roll-window 15
```

Expected output directories:

```text
/g/data/gv90/da1339/afim_output/LD-waves-exp01/zarr/SH/ispd_thresh_5.0e-4/FI/Tc/
├── raw/
│   └── data.zarr
├── bin-win-11_bin-min-09/
│   └── data.zarr
└── roll-days-15/
    └── data.zarr
```

## 3. Run metrics

```bash
python scripts/metrics/metrics.py   --sim-name LD-waves-exp01   --start-date 1993-01-01   --end-date 1993-12-31   --hemisphere SH   --ice-type FI   --BorC2T-type Tc   --ispd-thresh 5e-4   --methods raw,binary-days,rolling-mean   --bin-window 11   --bin-min-days 9   --roll-window 15   --plot-fip   --plot-fia   --plot-fit
```

Expected metric stores:

```text
/g/data/gv90/da1339/afim_output/LD-waves-exp01/zarr/SH/ispd_thresh_5.0e-4/FI/Tc/
├── raw/
│   ├── data.zarr
│   └── mets.zarr
├── bin-win-11_bin-min-09/
│   ├── data.zarr
│   └── mets.zarr
└── roll-days-15/
    ├── data.zarr
    └── mets.zarr
```

## 4. Load the results in Python

```python
from shugga import RunSpec, ClassificationSpec, ShuggaPaths
from shugga.metrics import CICEMetrics

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
    bin_window=11,
    bin_min_days=9,
    roll_window=15,
)

paths = ShuggaPaths(run=run, classify=classify)
metrics = CICEMetrics(run=run, classify=classify, paths=paths)

raw = metrics.load_metrics("raw")
binary = metrics.load_metrics("binary-days")
rolling = metrics.load_metrics("rolling-mean")
```

## 5. Use the PBS wrappers

Classification wrapper:

```bash
./scripts/classification/classify_pbs_wrapper.sh   -s LD-waves-exp01 -b 1993-01-01 -e 1993-12-31 -H SH -i FI -g Tc   -t 5e-4 -m raw,binary-days,rolling-mean -B 11 -N 9 -R 15
```

Metrics wrapper:

```bash
./scripts/metrics/metrics_pbs_wrapper.sh   -s LD-waves-exp01 -b 1993-01-01 -e 1993-12-31 -H SH -i FI -g Tc   -t 5e-4 -m raw,binary-days,rolling-mean -B 11 -N 9 -R 15 -f -a -T
```

## 6. Regional metrics

The metrics stores include:

- `FIA`
- `FIV`
- `FIT`
- `FIP`
- `FIA_by_region(time, region)`
- `FIT_by_region(time, region)`

with regions:

- `DML`
- `WIO`
- `EIO`
- `Aus`
- `VOL`
- `AS`
- `BS`
- `WS`
