# shuga

`shuga` is a CICE-focused Python toolbox for sea-ice post-processing, with an emphasis on Antarctic landfast sea ice, standalone CICE6 experiments, observational comparison, and reproducible NCI Gadi workflows.

It has evolved from an earlier notebook- and script-heavy AFIM analysis workflow into a package structured around explicit runtime specifications, common path rules, grouped Zarr stores, reusable classification and metrics logic, observational adapters, PyGMT plotting helpers, and PBS entry points.

The present documentation target is **version 0.5**, but `shuga` remains an active research codebase and APIs/store layouts may still change.

## Current scope

The package now supports four main analysis streams:

1. **Standalone CICE6**
   - grouped NetCDF → Zarr conversion;
   - universal static-grid handling;
   - SI/FI/PI classification;
   - metric-group and explicit metric execution;
   - regional, temporal, stress, deformation, persistence, and thickness diagnostics.

2. **CMEMS/ORAS**
   - native-grid sea-ice loading;
   - SI/FI/PI classification from `siconc`, `usi`, and `vsi`;
   - metrics comparable with standalone CICE6;
   - SIA and SIT climatology comparisons.

3. **Observational products**
   - AF2020 landfast ice;
   - NSIDC sea-ice area/extent;
   - OSI-SAF-450 sea-ice area;
   - ESA-CCI sea-ice thickness/concentration archive mirror;
   - AWI sea-ice thickness;
   - ESA/AWI hemispheric SIT processing.

4. **Publication workflows**
   - SIA climatologies using NSIDC, OSI-SAF-450, CMEMS, and CICE6;
   - SIT climatologies using ESA-CCI, AWI, CMEMS, and CICE6;
   - FIP/FIC maps and differencing;
   - regional and time-series diagnostics;
   - figure-generation scripts and PBS controls under `shuga/scripts/`.

## High-level workflow

```text
CICE iceh*.nc
    ↓
grouped Zarr conversion
    ↓
CICE history loading + universal static-grid merge
    ↓
SI / FI / PI classification
    ↓
metrics
    ↓
publication plotting / observation comparison
```

CMEMS follows a parallel native-grid pathway:

```text
CMEMS annual daily NetCDF
    ↓
native lazy loading
    ↓
SI / FI / PI classification
    ↓
CMEMS metric runner
    ↓
comparison against observations and standalone CICE6
```

Observation-specific workflows sit alongside these model workflows:

```text
AF2020 / NSIDC / OSI-SAF-450 / ESA-CCI / AWI
    ↓
product-specific download / processing
    ↓
standardised Zarr time series or gridded stores
    ↓
comparison plots
```

## Package structure

```text
shuga/
├── classify/
│   ├── cice.py
│   └── cmems.py
├── core/
│   ├── data_conversion.py
│   ├── logging.py
│   ├── naming.py
│   ├── paths.py
│   ├── regions.py
│   ├── reporting.py
│   ├── store_selection.py
│   └── types.py
├── cpt/
├── docs/
├── forcing/
├── grid/
├── io/
├── metrics/
│   ├── calculations.py
│   ├── cice.py
│   ├── cmems.py
│   ├── diagnostics.py
│   ├── dispatch.py
│   ├── regional.py
│   ├── registry.py
│   ├── secondary.py
│   ├── skill.py
│   ├── stress.py
│   └── temporal.py
├── observations/
│   ├── AF2020.py
│   ├── CMEMS.py
│   ├── NSIDC.py
│   ├── OSI_SAF450.py
│   ├── sea_ice_thickness.py
│   └── legacy.py
├── plotting/
│   └── cice.py
├── regridding/
├── notebooks/
└── scripts/
    ├── admin/
    ├── classification/
    ├── comparisons/
    ├── conversion/
    ├── downloading/
    ├── forcing/
    ├── metrics/
    ├── observations/
    ├── plotting/
    ├── tides_analysis/
    └── waves/
```

## Core runtime objects

| Object | Purpose |
|---|---|
| `RunSpec` | Simulation name, analysis dates, hemisphere, project/user, history frequency. |
| `ClassificationSpec` | Ice domain, speed/concentration thresholds, classification method/window, and CICE grid reconstruction where relevant. |
| `MetricsSpec` | Metric selection, groups, scales, diagnostics, and optional observational skill settings. |
| `ObservationSpec` | Observation roots and product-specific settings. |
| `CICEGridSpec` | Static CICE grid/mask/bathymetry/form-factor resources. |
| `PlottingSpec` | Shared figure/PyGMT settings. |
| `ShugaPaths` | Central authority for model stores, classifications, metrics, logs, observations, and graphics. |

The preferred rule is:

> use `ShugaPaths` and public shuga loaders instead of hand-constructing paths in notebooks or plotting scripts.

For **SI**, grid type is not part of the physical classification. FI/PI remain grid-type dependent because the ice-speed reconstruction depends on the chosen CICE grid location.

## Standalone CICE6 layout

A typical standalone experiment lives under:

```text
/g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr/
```

For FI/PI:

```text
SH/
└── ispd_thresh_5.0e-4/
    ├── FI/
    │   └── Tc/
    │       └── bin-win-11_bin-min-09/
    │           ├── data.zarr
    │           └── mets.zarr
    └── PI/
        └── Tc/
            └── bin-win-11_bin-min-09/
                ├── data.zarr
                └── mets.zarr
```

For SI:

```text
SH/
└── SI/
    ├── data.zarr
    └── mets.zarr
```

Paper1 FI products use `Tb`; newer paper2/paper3 workflows generally use `Tc`. SI does not require a `Tb`/`Tc` path component.

The universal CICE static store is:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

and contains reusable static fields such as `TLON`, `TLAT`, `tarea`, velocity-grid areas, masks, metrics, and grid angles.

## Ice domains and classification

| Domain | Meaning | Classification |
|---|---|---|
| `SI` | Sea ice above the concentration threshold | concentration only |
| `FI` | Fast/landfast-ice candidate cells | concentration + low non-zero speed |
| `PI` | Pack ice | `SI_mask & ~FI_mask` |

Current FI semantics are:

```python
FI_raw = (
    (aice > aice_thresh)
    & finite(speed)
    & (speed > 0)
    & (speed <= ispd_thresh)
)
```

Typical defaults:

```text
aice_thresh = 0.15
ispd_thresh = 5.0e-4 m s^-1
```

### Binary-days

```text
window = 11 days
minimum raw FI days = 9
```

using a centred rolling window.

### Rolling-mean

```text
15-day centred mean speed
```

followed by the standard FI speed and concentration thresholds.

## Shared metric definitions

Low-level metric formulae live in:

```text
shuga/metrics/calculations.py
```

and are reused by standalone CICE and CMEMS where possible.

### Area

```text
A = Σ(C × cell_area)
```

### Volume

```text
V = Σ(C × h × cell_area)
```

### Mean thickness

```text
SIT = Σ(C × h × cell_area) / Σ(C × cell_area)
```

This concentration/area-weighted thickness definition is used for model/reanalysis SIT comparison.

# CMEMS / ORAS workflow

`shuga` now includes a native-grid CMEMS pathway through:

```text
shuga/observations/CMEMS.py
shuga/classify/cmems.py
shuga/metrics/cmems.py
```

Source mapping:

```text
siconc  -> aice
sithick -> hi
usi     -> eastward ice velocity
vsi     -> northward ice velocity
```

and:

```python
ice_speed = hypot(usi, vsi)
```

No CICE velocity-grid reconstruction is applied.

## CMEMS output layout

```text
/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily/SH/
├── static.zarr
├── SI/
│   ├── data.zarr
│   └── mets.zarr
└── ispd_thresh_5.0e-4/
    ├── FI/
    │   └── native/
    │       ├── raw/
    │       │   ├── data.zarr
    │       │   └── mets.zarr
    │       ├── bin-win-11_bin-min-09/
    │       │   ├── data.zarr
    │       │   └── mets.zarr
    │       └── roll-days-15/
    │           ├── data.zarr
    │           └── mets.zarr
    └── PI/
        └── native/
            └── ...
```

`native` replaces CICE grid labels such as `Tb` or `Tc`.

The static store includes:

```text
latitude
longitude
TLAT
TLON
tarea
```

## CMEMS classification

Classification mirrors standalone CICE semantics:

- SI from concentration threshold.
- FI from concentration + low non-zero speed.
- binary-days: centred 11-day window with at least 9 FI days.
- rolling-mean: centred 15-day speed mean.
- PI = SI minus FI.

## CMEMS metric groups

```text
cmems_fi_core : FIA FIV FIT FIP FIHI
cmems_pi_core : PIA PIV PIT PIP PIHI
cmems_si_core : SIA SIV SIT SIP SIHI
cmems_core    : all of the above
```

Explicit metric-name requests are independent of groups. Example:

```bash
./metrics_CMEMS_pbs_wrapper.sh \
    -b 1995-01-01 \
    -e 2005-12-31 \
    -M SIA \
    -o
```

computes only `SIA`.

If neither `-M` nor `-G` is supplied, the default remains `cmems_core`.

## CMEMS write-time chunking

Metric outputs are rechunked immediately before Zarr writing to avoid irregular Dask chunks produced by alignment between annual CMEMS source files and classification masks.

Typical write chunks:

```text
time      = 31
latitude  = 256
longitude = 540
```

## Recommended CMEMS execution

Classification:

```bash
cd ~/AFIM/src/mawsons-chest/shuga/scripts/classification

./classify_CMEMS_pbs_wrapper.sh \
    -b 2002-01-01 \
    -e 2002-12-31 \
    -H SH \
    -m raw,binary-days,rolling-mean
```

Metrics:

```bash
cd ~/AFIM/src/mawsons-chest/shuga/scripts/metrics

./metrics_CMEMS_pbs_wrapper.sh \
    -b 2002-01-01 \
    -e 2002-12-31 \
    -H SH \
    -m binary-days \
    -G cmems_core
```

# Observation workflows

## AF2020

AF2020 is the primary landfast-ice observational product used for FIC/FIP/FIA comparison.

## NSIDC

NSIDC supports hemispheric SIA/SIE comparison and is used as one of the primary SIA climatology references.

## OSI-SAF-450

`shuga/observations/OSI_SAF450.py` provides download and processing support for the OSI-SAF-450 sea-ice concentration record.

## ESA-CCI sea-ice archive

Current downloading is handled through:

```text
shuga/scripts/downloading/download_ESA_CCI.py
shuga/scripts/downloading/download_ESA_CCI.pbs
```

The downloader is product-aware and uses declarative THREDDS roots rather than recursively traversing obsolete product versions.

Local mirror:

```text
/g/data/gv90/da1339/SeaIce/ESA/CCI/
├── thickness/
├── concentration/
└── thickness_drift_aware/
```

### ESA thickness

```text
version : v4.0
levels  : L2P and L3C
sensors : Envisat, CryoSat-2, Sentinel-3A, Sentinel-3B
hems    : NH and SH
```

### ESA concentration

Only:

```text
L4/ssmi_ssmis/12.5km/v3.0/NH
L4/ssmi_ssmis/12.5km/v3.0/SH
```

is mirrored.

### ESA thickness-drift-aware

The drift-aware thickness product is NH-only.

Download:

```bash
qsub shuga/scripts/downloading/download_ESA_CCI.pbs
```

Dry run:

```bash
qsub -v DRY_RUN=true \
    shuga/scripts/downloading/download_ESA_CCI.pbs
```

Existing complete files are skipped and partial `.part` downloads are resumed where supported.

## AWI SIT

AWI SIT downloading is handled through:

```text
shuga/scripts/downloading/download_AWI_SIT.py
shuga/scripts/downloading/download_AWI_SIT.pbs
```

The downloader skips files already present and resumes partial downloads.

# Sea-ice thickness processing

The revised SIT workflow keeps **ESA-CCI and AWI separate** rather than treating them as one continuous primary observation record.

This is deliberate because the products have different mission histories and overlapping missions need explicit combination.

## ESA discovery

ESA L3C files are discovered from:

```text
/g/data/gv90/da1339/SeaIce/ESA/CCI/
└── thickness/
    └── L3C/
        └── <sensor>/
            └── v4.0/
                └── <SH|NH>/
                    └── .../*.nc
```

Sensors:

```text
envisat
cryosat2
sentinel3a
sentinel3b
```

## AWI discovery

```text
/g/data/gv90/da1339/SeaIce/AWI/
└── l3cp_release/
    └── <sh|nh>/
        └── <sensor>/
            └── .../*.nc
```

## SIT processing products

```text
/g/data/gv90/da1339/SeaIce/SIT/processed/SH/
├── ESA/
│   ├── envisat.zarr
│   ├── envisat_SIT_timeseries.zarr
│   ├── cryosat2.zarr
│   ├── cryosat2_SIT_timeseries.zarr
│   ├── sentinel3a.zarr
│   ├── sentinel3a_SIT_timeseries.zarr
│   ├── sentinel3b.zarr
│   ├── sentinel3b_SIT_timeseries.zarr
│   └── SIT_timeseries.zarr
└── AWI/
    ├── ...
    └── SIT_timeseries.zarr
```

Overlapping sensor months are combined by the median across valid sensor-level hemispheric SIT estimates.

Run:

```bash
qsub -v HEMISPHERE=SH,OVERWRITE=true \
    shuga/scripts/observations/process_SIT.pbs
```

The old `continuous/SIT_timeseries.zarr` splice is no longer the preferred publication input.

# SIA climatology comparison

The SIA plotting workflow compares:

```text
NSIDC
OSI-SAF-450
CMEMS
standalone CICE6 experiments
```

## Paper1

```bash
export PAPER_ROOT=/g/data/gv90/da1339/afim_output/paper1
export START_DATE=1994-01-01
export END_DATE=1999-12-31
export OUTPUT=/g/data/gv90/da1339/GRAPHICAL/paper1/SIA_SH_climatology_1994-1999.png
export EXPERIMENTS="AOM2-ERA5=ACCESS-OM2-ERA5,notensnogi=notens-nogi,ry93=ry93,elps-min=elps-min"

qsub -V shuga/scripts/plotting/plot_SIA_climatology.pbs
```

## Paper2

```bash
export PAPER_ROOT=/g/data/gv90/da1339/afim_output
export START_DATE=2000-01-01
export END_DATE=2005-12-31
export OUTPUT=/g/data/gv90/da1339/GRAPHICAL/paper2/SIA_SH_climatology_2000-2005.png
export EXPERIMENTS="no-slip-LFI=LFI rheology without lateral drag,Cs-high=static high Cs,Cq-high=quadratic high Cq"

qsub -V shuga/scripts/plotting/plot_SIA_climatology.pbs
```

## Paper3

```bash
export PAPER_ROOT=/g/data/gv90/da1339/afim_output/paper3
export START_DATE=2000-01-01
export END_DATE=2005-12-31
export OUTPUT=/g/data/gv90/da1339/GRAPHICAL/paper3/SIA_SH_climatology_2000-2005.png
export EXPERIMENTS="LD-tides=LD tides"

qsub -V shuga/scripts/plotting/plot_SIA_climatology.pbs
```

For experiment definitions containing commas and spaces, exporting variables and using `qsub -V` is safer than putting everything directly into `qsub -v`.

# SIT climatology comparison

The revised SIT plotting workflow compares:

```text
ESA-CCI
AWI
CMEMS
standalone CICE6 experiments
```

Temporal treatment:

```text
ESA-CCI : full available remote-sensing record
AWI     : full available remote-sensing record
CMEMS   : requested model comparison period
CICE6   : requested model comparison period
```

For paper2:

```text
2000-01-01 to 2005-12-31
```

All sources are reduced to **calendar-month means** before constructing a Jan–Dec climatology.

The default envelope is:

```text
p10-p90
```

Run with the paper2 defaults:

```bash
qsub shuga/scripts/plotting/plot_SIT_climatology.pbs
```

Expected plotting order/style:

```text
ESA-CCI     black
AWI         dashed grey
CMEMS       green
no-slip-LFI blue
Cs-high     orange
Cq-high     purple
```

The satellite SIT climatology is intended as a broad hemispheric reference, not direct validation of Antarctic coastal fast ice.

# Typical notebook usage

```python
from pathlib import Path
import sys

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga import (
    RunSpec,
    ClassificationSpec,
    MetricsSpec,
    PlottingSpec,
    ShugaPaths,
    load_metrics,
    CICEPlotter,
)

run = RunSpec(
    sim_name="Cs-high",
    start_date="2000-01-01",
    end_date="2005-12-31",
    hemisphere="SH",
)

classify = ClassificationSpec(
    ice_type="FI",
    grid_type="Tc",
    ispd_thresh=5.0e-4,
    methods=("binary-days",),
    bin_window=11,
    bin_min_days=9,
)

metrics = MetricsSpec(methods=("binary-days",))
plotting = PlottingSpec()

paths = ShugaPaths(
    run_cfg=run,
    cls_cfg=classify,
)

ds = load_metrics(
    run_cfg=run,
    cls_cfg=classify,
    met_cfg=metrics,
    pth_cfg=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP", "FIHI"],
)

plotter = CICEPlotter(
    run_cfg=run,
    cls_cfg=classify,
    met_cfg=metrics,
    plt_cfg=plotting,
    pth_cfg=paths,
)
```

Publication notebooks should define experiments, scientific windows, and final figure composition. Reusable loading, classification, metrics, masking, and plot-data preparation should remain in `shuga`.

# Gadi workflow guidance

Primary environment:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-26.02
```

PBS entry points live under:

```text
shuga/scripts/
```

with `copyq` used for download/mirror jobs and `normalbw` for most analysis jobs.

For large metrics runs, explicit metrics are often preferable to large all-in-one groups when memory pressure is significant.

# Documentation map

| Page | Purpose |
|---|---|
| `docs/quickstart.md` | Minimal end-to-end workflow. |
| `docs/architecture.md` | Package structure and responsibilities. |
| `docs/io.md` | CICE/Zarr loading and store discovery. |
| `docs/data_conversion.md` | NetCDF → Zarr conversion. |
| `docs/static-grid.md` | Universal static-grid store. |
| `docs/classification.md` | FI/PI/SI classification semantics. |
| `docs/metrics.md` | Metric definitions/groups/units. |
| `docs/plotting.md` | PyGMT plotting architecture. |
| `docs/observations.md` | Observation adapters and processed stores. |
| `docs/regridding.md` | Velocity grids and regridding. |
| `docs/forcing.md` | Forcing utilities. |
| `docs/gadi-workflows.md` | PBS/Gadi workflows. |
| `docs/developer-guide.md` | Extension and maintenance rules. |

# Development status

`shuga` is now beyond the original CICE-only refactor. The current codebase contains:

- standalone CICE6 classification and metrics;
- native-grid CMEMS classification and metrics;
- observation download/processing workflows;
- paper-agnostic SIA plotting;
- hemispheric ESA/AWI SIT processing;
- SIT climatology comparison;
- publication-focused landfast-ice diagnostics;
- tides and waves script areas under active development.

The package remains research software. Store layouts, product adapters, and plotting interfaces may continue to evolve as paper2/paper3 analyses mature.

A useful future consolidation would be a shared model/product adapter layer so standalone CICE and CMEMS can share more orchestration without duplicating high-level runner logic.
