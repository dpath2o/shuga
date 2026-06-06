# shuga

`shuga` is a CICE-focused Python toolbox for sea-ice post-processing, with an emphasis on Antarctic landfast sea ice, C-grid free-slip/lateral-drag experiments, and reproducible NCI Gadi workflows. It replaces older notebook- and JSON-driven AFIM analysis scripts with explicit runtime specifications, common path rules, reusable CICE/Zarr loading, classification, metrics, plotting, observations, regridding, and forcing utilities.

The present documentation target is **version 0.5**.

## What shuga does

```text
CICE iceh*.nc
    ↓
NetCDF → grouped Zarr conversion
    ↓
CICE history loading with universal static-grid merge
    ↓
Fast-ice / pack-ice / sea-ice classification
    ↓
Metrics: area, volume, thickness, persistence, rates, regions, stresses, diagnostics
    ↓
Plotting, observational comparison, publication notebooks, Gadi workflows
```

The package is designed around large CICE history workflows. Zarr is used because it allows lazy loading, Dask chunking, month/day group access, and storage of static CICE grid fields only once.

## Core runtime objects

| Object | Purpose |
|---|---|
| `RunSpec` | Simulation name, date window, hemisphere, project/user, and CICE history frequency. |
| `ClassificationSpec` | Ice domain, grid type, speed threshold, concentration threshold, and classification-window parameters. |
| `MetricsSpec` | Metric groups, scaling factors, observation-skill settings, and optional diagnostic controls. |
| `ObservationSpec` | Observation roots and product-specific settings. |
| `CICEGridSpec` | CICE grid, mask, bathymetry, form-factor, and grid-asset paths. |
| `PlottingSpec` | Common figure and PyGMT plotting settings. |
| `ShugaPaths` | Single path authority for CICE stores, static grid, classifications, metrics, graphics, and logs. |

The central rule is: **use `ShugaPaths` and public loaders instead of hand-building paths in notebooks or scripts**.

## Package structure

```text
shuga/
├── classify/       # CICEClassifier workflow
├── core/           # dataclasses, paths, naming, regions, conversion
├── forcing/        # ERA5 forcing helpers; ORAS support planned
├── grid/           # CICE grid/static-grid/lateral-drag helpers
├── io/             # public loaders and store discovery
├── metrics/        # CICEMetrics workflow and calculations
├── observations/   # AF2020, NSIDC, legacy compatibility
├── plotting/       # PyGMT/data-prep plotting helpers
├── regridding/     # CICE velocity grid handling, pyresample, xESMF
└── scripts/        # command-line and PBS entry points
```

Waves and tides tooling exists in the repository but is deliberately not documented here yet.

## Data layout

Current default layout assumes Gadi and AFIM-style directories:

```text
~/AFIM_archive -> /g/data/gv90/da1339/afim_output    # often a symlink

~/AFIM_archive/<SIM_NAME>/zarr/
├── iceh_daily.zarr/YYYY-MM/              # daily dynamic history
├── iceh_hourly.zarr/YYYY_MM_DD/          # optional hourly dynamic history
└── SH/ispd_thresh_5.0e-4/
    ├── FI/Tc/bin-win-11_bin-min-09/data.zarr
    ├── FI/Tc/bin-win-11_bin-min-09/mets.zarr
    ├── PI/Tc/...
    └── SI/Tc/...
```

The **universal CICE 1/4-degree C-grid static store** is grid-level rather than simulation-level:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

It is used when workflows need static variables such as `TLON`, `TLAT`, `tarea`, `uarea`, `earea`, `narea`, masks, metrics, and grid angles.

## Ice domains

| Domain | Meaning | Speed classified? | Stored product? |
|---|---|---:|---:|
| `FI` | Fast/landfast ice candidate cells | Yes | `data.zarr` |
| `PI` | Pack ice, defined as sea ice that is not fast ice | Indirectly | derived/written with FI workflow |
| `SI` | Sea ice defined by concentration threshold in a hemisphere | No | metrics domain; no speed classification |

Fast ice uses raw, binary-days, or rolling-mean classification. Pack ice is derived as sea ice minus fast ice. Sea ice is not speed-thresholded.

## Typical notebook usage

```python
from pathlib import Path
import sys

repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shuga import (
    RunSpec, ClassificationSpec, MetricsSpec, PlottingSpec,
    ShugaPaths, load_metrics, CICEPlotter,
)

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
    methods=("binary-days",),
    bin_window=11,
    bin_min_days=9,
    roll_window=15,
)

metrics = MetricsSpec(methods=("binary-days",))
plotting = PlottingSpec(fig_size=20.0)
paths = ShugaPaths(run=run, classify=classify, metrics=metrics, plotting=plotting)

ds = load_metrics(
    run=run,
    classify=classify,
    metrics=metrics,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP", "FIHI", "FIST"],
)

plotter = CICEPlotter(run=run, classify=classify, metrics=metrics, plotting=plotting, paths=paths)
plotter.plot_fip(method="binary-days", region_name="Aus")
plotter.plot_timeseries("FIA", method="binary-days", region="total")
```

Publication notebooks should define experiments, scientific windows, and final figure composition, while `shuga` owns reusable loading, masks, metrics, and plotting/data-prep logic.

## Command-line usage

Convert CICE NetCDF history to grouped Zarr:

```bash
python shuga/scripts/conversion/nc2zarr.py \
  --sim-name LD-blend-base \
  --start-date 2000-01-01 \
  --end-date 2003-12-31 \
  --iceh-frequency daily
```

Classify fast ice:

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

Compute metrics:

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

On Gadi, prefer PBS wrappers under `shuga/scripts/*`.

## Documentation map

| Page | Purpose |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Minimal end-to-end workflow. |
| [`docs/architecture.md`](docs/architecture.md) | Package design and module boundaries. |
| [`docs/io.md`](docs/io.md) | CICE/Zarr loading and store resolution. |
| [`docs/data_conversion.md`](docs/data_conversion.md) | NetCDF-to-Zarr conversion and storage design. |
| [`docs/static-grid.md`](docs/static-grid.md) | Universal CICE static-grid store. |
| [`docs/classification.md`](docs/classification.md) | FI/PI/SI classification mathematics and outputs. |
| [`docs/metrics.md`](docs/metrics.md) | Metric definitions, groups, units, and relevance. |
| [`docs/plotting.md`](docs/plotting.md) | PyGMT plotting/data-prep philosophy and examples. |
| [`docs/observations.md`](docs/observations.md) | AF2020 and NSIDC observation support. |
| [`docs/regridding.md`](docs/regridding.md) | CICE velocity grid handling, pyresample, and xESMF. |
| [`docs/forcing.md`](docs/forcing.md) | ERA5 forcing support and future ORAS direction. |
| [`docs/gadi-workflows.md`](docs/gadi-workflows.md) | PBS workflows and Gadi usage. |
| [`docs/developer-guide.md`](docs/developer-guide.md) | Extension and maintenance rules. |

## External references

- [CICE Consortium documentation](https://cice-consortium-cice.readthedocs.io/)
- [CICE source code](https://github.com/CICE-Consortium/CICE)
- [Icepack source code](https://github.com/CICE-Consortium/Icepack)
- [xarray documentation](https://docs.xarray.dev/)
- [Dask documentation](https://docs.dask.org/)
- [Zarr Python documentation](https://zarr.readthedocs.io/)
- [PyGMT documentation](https://www.pygmt.org/)
- [pyresample documentation](https://pyresample.readthedocs.io/)
- [xESMF documentation](https://xesmf.readthedocs.io/)
- [NCO documentation](https://nco.sourceforge.net/nco.html)
- [NCI Gadi user guide](https://opus.nci.org.au/display/Help/Gadi)

## Status

`shuga` is an active research toolbox. APIs and store layouts may still evolve as lateral-drag, observation-comparison, and forcing-sensitivity workflows mature.
