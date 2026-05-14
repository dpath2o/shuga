# IO and store loading

The IO layer is responsible for opening CICE history, classified masks, and metrics products without duplicating path logic across modules.

## Public loaders

Use these functions in notebooks and scripts:

```python
from shuga import load_cice, load_classified, load_metrics
```

### `load_cice`

Loads CICE history data.

```python
ds = load_cice(
    run=run,
    classify=classify,
    paths=paths,
    variables=["aice", "hi", "tarea", "TLON", "TLAT"],
)
```

Responsibilities:

- resolve run context;
- open daily or hourly history stores;
- handle grouped Zarr layouts;
- filter requested variables;
- merge `iceh_static.zarr` when needed;
- apply hemisphere masks when static latitude fields are available.

### `load_classified`

Loads a method-specific classification store.

```python
ds = load_classified(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FI_mask"],
)
```

The store is resolved through `CICEStoreLocator`, not by manually constructing paths.

### `load_metrics`

Loads a method-specific metrics store.

```python
ds = load_metrics(
    run=run,
    classify=classify,
    paths=paths,
    classification="binary-days",
    variables=["FIA", "FIT", "FIP"],
)
```

The metrics loader is overlap-aware for time slicing, because metric stores may cover a longer range than the current analysis request.

## History store layouts

### Daily grouped layout

```text
iceh_daily.zarr/
├── 1993-01/
├── 1993-02/
├── 1993-03/
└── ...
```

This is the preferred layout for daily CICE products.

### Hourly grouped layout

Hourly products may be grouped more finely. Use the same public `load_cice()` entry point; select frequency with `RunSpec(iceh_frequency="hourly")`.

### Flat layout

Flat Zarr stores are supported where present.

## Static field merge

Static variables such as `TLON`, `TLAT`, `tarea`, `HTE`, `HTN`, and `NCAT` may be absent from CICE history if namelist output flags were disabled to save space.

When requested variables are not in the history store, the loader checks for:

```text
iceh_static.zarr
```

and merges static variables where possible.

## Store resolution

`CICEStoreLocator` resolves classification and metrics stores. It understands:

- simulation name;
- classification method;
- grid type;
- ice type;
- threshold path;
- binary-days and rolling-mean path fragments;
- optional grid-type maps across simulations.

This avoids hard-coded path branches in notebooks.

## Common classified store layout

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

## Common metrics store layout

```text
/g/data/<PROJECT>/<USER>/afim_output/<SIM_NAME>/zarr/<HEMISPHERE>/
└── ispd_thresh_5.0e-4/
    └── FI/
        └── Tc/
            └── bin-win-11_bin-min-09/
                ├── data.zarr
                └── mets.zarr
```

## Loader failure modes

### Missing store

Check:

```python
print(paths.resolve_cice_store())
print(paths.classification_store("binary-days"))
print(paths.metrics_store("binary-days"))
```

Then check the filesystem:

```bash
ls -lah /g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr
```

### Missing static variable

If `load_cice(..., variables=["tarea"])` fails, either the history store does not contain static fields, or `iceh_static.zarr` is missing or incomplete.

Build `iceh_static.zarr` using `CICEStaticBuilder`.

### Ambiguous method

If multiple method stores exist and no method was specified, pass:

```python
classification="binary-days"
```

or set:

```python
ClassificationSpec(methods=("binary-days",))
```

## Development rule

Do not add new path-search code to scripts, notebooks, classifier, metrics, or plotting classes. Extend `ShugaPaths` or `CICEStoreLocator` instead.
