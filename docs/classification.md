# Fast-ice classification

`CICEClassifier` builds method-specific fast-ice masks from CICE history output.

The classifier consumes CICE velocity/concentration fields and writes `data.zarr` stores under the path controlled by `ShugaPaths`.

## Public workflow

```python
from shuga import RunSpec, ClassificationSpec, ShugaPaths, CICEClassifier

run = RunSpec(
    sim_name="LD-static-Cs2p5e-4",
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
```

## Methods

### `raw`

Daily speed-threshold classification. A cell is classified as fast ice when ice speed is below the configured threshold and ice concentration satisfies the configured concentration threshold.

Typical threshold:

```python
ispd_thresh = 5e-4  # m s^-1
```

### `binary-days`

Persistence classification based on a rolling count of raw fast-ice days.

Typical parameters:

```python
bin_window = 11
bin_min_days = 9
```

This means a cell is classified as fast ice when at least 9 of the 11 centred days satisfy the raw speed-threshold condition.

### `rolling-mean`

Speed is first smoothed with a centred rolling window, then thresholded.

Typical parameter:

```python
roll_window = 15
```

## Window padding

Centred windows require extra days at the beginning and end of the requested analysis period. The classifier pads the read window, computes the rolling/binary product, and then crops back to the requested dates.

This matters when comparing timing of annual maxima/minima. The output store should contain the requested period, not the padded read period.

## Output variables

Classification stores normally include:

| Variable | Dimensions | Meaning |
|---|---|---|
| `FI_mask` | `time, nj, ni` | Boolean fast-ice mask. |
| `FI_ispd` | `time, nj, ni` | Ice speed used by the method, masked to fast-ice cells. |
| `FI_aice` | `time, nj, ni` | Ice concentration in classified cells. |

Exact variables depend on the method and available source fields.

## Output paths

Common layout:

```text
/g/data/<PROJECT>/<USER>/afim_output/<SIM_NAME>/zarr/<HEMISPHERE>/
└── ispd_thresh_5.0e-4/
    └── FI/
        └── Tc/
            ├── raw/
            │   └── data.zarr
            ├── bin-win-11_bin-min-09/
            │   └── data.zarr
            └── roll-days-15/
                └── data.zarr
```

Do not construct these paths manually. Use `ShugaPaths` or public loaders.

## Loading classified masks

```python
from shuga import load_classified

ds = load_classified(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FI_mask"],
)
```

## Operational checks

After classification:

```python
ds = load_classified(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FI_mask"],
)

print(ds)
print(ds["FI_mask"].mean())
```

A zero or all-NaN mask usually indicates wrong velocity variables, wrong grid type, a threshold that is too strict, a missing concentration field, or a path resolving to an old/empty store.

## Development notes

`CICEClassifier` should remain a workflow class. Avoid placing reusable Zarr-writing, path-resolution, or static-grid helper functions inside it. Those belong in `io/`, `grid/`, or small calculation modules.
