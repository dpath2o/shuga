# IO and store loading

The `shuga.io` layer opens CICE history, classified masks, and metrics products without duplicating path logic across notebooks, scripts, classification, metrics, and plotting.

## Public loaders

```python
from shuga import load_cice, load_classified, load_metrics, open_cice_history
```

## `load_cice`

```python
ds = load_cice(
    run=run,
    classify=classify,
    paths=paths,
    variables=["aice", "hi", "tarea", "TLON", "TLAT"],
    chunks={"time": 31},
)
```

Responsibilities:

1. resolve run context;
2. identify daily/hourly CICE history stores;
3. open grouped or flat Zarr stores;
4. split requested variables into dynamic and static fields;
5. merge static fields from the universal static store;
6. apply time slices;
7. apply hemisphere masks when latitude is available.

## History layouts

Daily:

```text
iceh_daily.zarr/YYYY-MM
```

Hourly:

```text
iceh_hourly.zarr/YYYY_MM_DD
```

Use `RunSpec(iceh_frequency="daily")` or `RunSpec(iceh_frequency="hourly")`.

## Universal static-grid merge

Static CICE fields are stored once in:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

Typical fields include `TLON`, `TLAT`, `ULON`, `ULAT`, `ELON`, `ELAT`, `NLON`, `NLAT`, `tarea`, `uarea`, `earea`, `narea`, grid metrics, masks, angles, and `NCAT`.

When a user requests one of these variables, `IceHistoryLoader` merges it from the static store if it is absent from the dynamic grouped store.

## `load_classified`

```python
classified = load_classified(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FI_mask", "PI_mask"],
)
```

Store resolution is handled by `CICEStoreLocator`, which understands simulation name, hemisphere, threshold, ice domain, grid type, method, binary-days settings, rolling settings, and optional grid-type maps.

## `load_metrics`

```python
mets = load_metrics(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP"],
)
```

Metric stores can cover a longer period than the requested `RunSpec`; loading can return the overlap.

## Layout

```text
~/AFIM_archive/<SIM_NAME>/zarr/<HEMISPHERE>/ispd_thresh_5.0e-4/<ICE_TYPE>/<GRID_TYPE>/
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

## Failure modes

Missing CICE store:

```python
print(paths.resolve_cice_store())
```

Missing static variable:

```python
from shuga.grid.cice import CICEGridwork
ds_static = CICEGridwork(paths=paths).load_cice_static(variables=["TLON", "TLAT", "tarea"])
print(ds_static)
```

Ambiguous method:

```python
load_metrics(..., classification="binary-days")
```

Wrong domain:

```python
ClassificationSpec(ice_type="FI")
ClassificationSpec(ice_type="PI")
ClassificationSpec(ice_type="SI")
```

## Development rule

Do not add new path-search branches to notebooks, classifier, metrics, plotting, or scripts. Add path logic to `ShugaPaths` or store-discovery logic to `CICEStoreLocator`.
