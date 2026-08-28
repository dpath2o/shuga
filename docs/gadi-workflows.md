# Gadi workflows

This page describes how `shuga/scripts/*` connect package code to NCI Gadi PBS jobs. Scripts should remain thin: parse arguments, build runtime specs, construct `ShugaPaths`, call package code, and log outputs.

## Environment

Typical PBS setup:

```bash
module purge
module use /g/data/xp65/public/modules
module load conda/analysis3-25.12

REPO_ROOT="${REPO_ROOT:-$HOME/AFIM/src/mawsons-chest}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export DASK_TEMPORARY_DIRECTORY="/scratch/${PROJECT}/${RUN_USER}/dask_tmp/${PBS_JOBID}"
export TMPDIR="/scratch/${PROJECT}/${RUN_USER}/tmp/${PBS_JOBID}"
mkdir -p "$DASK_TEMPORARY_DIRECTORY" "$TMPDIR"
```

Individual production scripts may pin a newer `analysis3` module where required; check the PBS header before submission.

## Common roots

```text
~/AFIM_archive -> /g/data/gv90/da1339/afim_output
~/AFIM_archive/<SIM_NAME>/zarr/iceh_daily.zarr
~/AFIM_archive/<SIM_NAME>/zarr/iceh_hourly.zarr
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
/g/data/gv90/da1339/GRAPHICAL/
```

## Classification

Python entry point:

```text
shuga/scripts/classification/classify.py
```

Responsibilities:

1. build `RunSpec`, `ClassificationSpec`, `CICEGridSpec`, and `ShugaPaths`;
2. optionally convert NetCDF to grouped Zarr via `NC2Zarr`;
3. resolve the universal static-grid store;
4. run `CICEClassifier`;
5. write method-specific `data.zarr` stores.

Example:

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
  --iceh-frequency daily \
  --skip-history-conversion
```

PBS:

```bash
qsub -v SIM_NAME=LD-blend-base,START_DATE=2000-04-01,END_DATE=2003-06-30,SKIP_HISTORY_CONVERSION=true \
  shuga/scripts/classification/classify.pbs
```

## Metrics

Python entry point:

```text
shuga/scripts/metrics/metrics.py
```

Responsibilities:

1. build runtime spec objects;
2. resolve CICE, classification, static-grid, metrics, and graphics paths;
3. run `CICEMetrics.compute_metrics()`;
4. optionally call `CICEPlotter` for common plots.

Example:

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

PBS uses `|` as a safe delimiter for comma lists:

```bash
qsub -v SIM_NAME=LD-blend-base,START_DATE=2000-04-01,END_DATE=2003-06-30,METRIC_GROUPS=fi_core\|fi_spatial\|fi_regional \
  shuga/scripts/metrics/metrics.pbs
```

## Conversion

```text
shuga/scripts/conversion/nc2zarr.py
```

Daily output:

```text
iceh_daily.zarr/YYYY-MM
```

Hourly output:

```text
iceh_hourly.zarr/YYYY_MM_DD
```

Example:

```bash
python shuga/scripts/conversion/nc2zarr.py \
  --sim-name LD-blend-base \
  --start-date 2000-01-01 \
  --end-date 2003-12-31 \
  --iceh-frequency daily \
  --overwrite-static
```

## Observations and comparisons

Current scripts include:

```text
shuga/scripts/observations/build_af2020_fip_fic_common_grid.py
shuga/scripts/comparisons/FIP_differencing.py
shuga/scripts/plotting/plot_FIC_side_by_side.py
```

Guiding rules:

- AF2020 native time sampling is 15-day.
- AF2020 common-grid products should be persistent Zarr stores.
- Simulation FIC can be computed on the fly from `FI_mask * aice`.
- FIP differencing must be done on a shared grid.

## Forcing

Atmospheric forcing scripts live primarily in:

```text
shuga/scripts/forcing/
```

Wave-forcing production scripts live in:

```text
shuga/scripts/waves/
```

Package capability is described in [`forcing.md`](forcing.md). The WHACS-specific scientific and technical workflow is documented in [`WHACS_wave_forcing.md`](WHACS_wave_forcing.md).

### WHACS → CICE25

The production WHACS workflow combines the five hourly full-spectrum archives:

```text
GRID,GLOB,BUOYS,NIWA,SCHISM
```

and writes monthly CICE forcing under:

```text
/g/data/gv90/da1339/afim_input/CAWCR/CAWCR_efreq_for_CICE6_YYYYMM.nc
```

A January-1995 pilot is submitted as a normal job:

```bash
qsub \
  -P au88 \
  -v START_YEAR=1995,END_YEAR=1995 \
  shuga/scripts/waves/whacs_regrid.pbs
```

The PBS worker defaults to index 0 when `PBS_ARRAY_INDEX` is absent, so this processes January of `START_YEAR`.

A multi-year run uses the wrapper:

```bash
./shuga/scripts/waves/whacs_regrid_wrapper.sh 1995 2005
```

Operational defaults:

```text
PBS accounting project = au88
data/output project     = gv90
WHACS source project    = ia39
queue                   = normalbw
ncpus                   = 8
memory                  = 64 GB
walltime                = 24 h
target latitude         = <= 35°S
IDW                     = k=8, p=2.5, radius=1000 km
time chunk              = 4 hours
ice mask                = none
```

The shared five-source station-weight file is:

```text
/g/data/gv90/da1339/grids/weights/
map_WHACS_grid-glob-buoys-niwa-schism_to_ACCESS-OM3-025_idw_k8.npz
```

Before a long production submission, inspect the monthly log and verify:

```text
WHACS spectral source sets: GRID,GLOB,BUOYS,NIWA,SCHISM
Combined WHACS source stations: ...
Spectral QC retained m0 ...
Ice mask    : none ...
Processing hourly chunk ...
```

and verify the resulting NetCDF contains:

```text
source_sets = GRID,GLOB,BUOYS,NIWA,SCHISM
completed   = true
```

Daily PyGMT regridding QC scripts live in:

```text
shuga/scripts/plotting/plot_whacs_daily.py
shuga/scripts/plotting/plot_whacs_daily.pbs
shuga/scripts/plotting/plot_whacs_daily_wrapper.sh
```

See [`WHACS_wave_forcing.md`](WHACS_wave_forcing.md) before interpreting native-versus-regridded plots: once production uses all five source sets, a `GRID`-only native panel is not a complete before/after comparison.

## Job checks

```bash
qstat -u $USER
find ~/AFIM_archive/<SIM_NAME>/zarr -maxdepth 8 \( -name data.zarr -o -name mets.zarr \)
```

Static-grid check:

```bash
python - <<'PY'
from shuga.core.paths import ShugaPaths
from shuga.grid.cice import CICEGridwork
paths = ShugaPaths()
print(paths.resolve_static_store())
print(CICEGridwork(pth_cfg=paths).load_cice_static(variables=["TLON", "TLAT", "tarea"]))
PY
```

## Hygiene

Do not commit PBS outputs, Zarr stores, NetCDF files, generated figures, animations, or scratch products.

```bash
python -m compileall shuga
find shuga -type d -name "__pycache__" -prune -exec rm -rf {} +
find shuga/scripts -type f \( -name "*.o[0-9]*" -o -name "*.e[0-9]*" -o -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
git status --short
```
