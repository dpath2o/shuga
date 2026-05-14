# Gadi workflows

This page records practical operating conventions for running `shuga` on Gadi.

## Environment

A typical shell setup is:

```bash
module use /g/data/hh5/public/modules
module load conda/analysis3-25.12
export PYTHONPATH=/home/581/da1339/AFIM/src/mawsons-chest:$PYTHONPATH
```

Use whichever `analysis3` version is current for the project. The important packages are:

- `numpy`
- `pandas`
- `xarray`
- `dask`
- `zarr`
- `matplotlib`
- `pygmt` for plotting workflows

## Common data roots

Typical roots are:

```text
/g/data/gv90/da1339/afim_output/<SIM_NAME>/
/g/data/gv90/da1339/AFIM_archive/<SIM_NAME>/
/g/data/gv90/da1339/GRAPHICAL/AFIM/
```

The package uses `RunSpec(project=..., user=...)` and `ShugaPaths` to construct paths. Avoid hard-coding these roots in scripts unless adding a new path rule.

## Classification wrapper

From:

```bash
cd shuga/scripts/classification
```

run:

```bash
./classify_pbs_wrapper.sh \
  -s LD-static-Cs2p5e-4 \
  -b 1993-01-01 \
  -e 1993-12-31 \
  -H SH \
  -i FI \
  -g Tc \
  -t 5e-4 \
  -m raw,binary-days,rolling-mean \
  -B 11 \
  -N 9 \
  -R 15 \
  -P gv90 \
  -U da1339
```

Useful options may include overwrite/history conversion controls depending on the script version. Run:

```bash
./classify_pbs_wrapper.sh -h
```

or inspect the script for the current flags.

## Metrics wrapper

From:

```bash
cd shuga/scripts/metrics
```

run:

```bash
./metrics_pbs_wrapper.sh \
  -s LD-static-Cs2p5e-4 \
  -b 1993-01-01 \
  -e 1993-12-31 \
  -H SH \
  -i FI \
  -g Tc \
  -t 5e-4 \
  -m binary-days \
  -B 11 \
  -N 9 \
  -R 15 \
  -P gv90 \
  -U da1339
```

Use the wrapper rather than manually editing PBS scripts for each simulation where possible.

## Checking job outputs

PBS outputs should not be committed to the repository. They are useful locally for debugging but should be cleaned before commit.

Find local job outputs:

```bash
find shuga/scripts -type f \( -name "*.o[0-9]*" -o -name "*.e[0-9]*" -o -name "*.log" -o -name "*.out" -o -name "*.err" \)
```

Remove after extracting useful information:

```bash
find shuga/scripts -type f \( -name "*.o[0-9]*" -o -name "*.e[0-9]*" -o -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
```

## Basic run status checks

```bash
qstat -u $USER
```

Check produced stores:

```bash
ls -lah /g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr
find /g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr -maxdepth 6 -name "data.zarr" -o -name "mets.zarr"
```

## Python smoke test

```bash
PYTHONPATH=/home/581/da1339/AFIM/src/mawsons-chest python - <<'PY'
from shuga import RunSpec, ClassificationSpec, ShugaPaths
from shuga import load_metrics

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

ds = load_metrics(
    run=run,
    classify=cls,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP"],
)

print(ds)
PY
```

## Repository hygiene before pushing

```bash
python -m compileall shuga

git status --short

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

The final `git ls-files` command should print nothing.
