# Architecture

## Why `shuga` exists

The immediate motivation was to stop path drift between classification and metrics. In the older AFIM workflow, the PBS wrappers, CLI scripts, and classes each carried pieces of the naming logic. That made it easy for one module to write:

```text
ispd_thresh_5.0e-4 / bin-win-11_bin-min-09 / data.zarr
```

while another tried to read:

```text
ispd_thresh5e-4 / BW11_BM9 / classification.zarr
```

`shuga` fixes that by making `ShugaPaths` the single source of truth.

## Core objects

### `RunSpec`

Pure runtime context:

- simulation name
- date window
- hemisphere
- project
- user

### `ClassificationSpec`

Defines how masks are built:

- ice type label
- grid type / BorC2T label
- speed threshold
- binary-days window and minimum count
- rolling-mean window
- velocity variable names
- concentration threshold

### `MetricsSpec`

Defines optional metric extras:

- observation store and variable names
- coast-distance variable
- area and volume scaling constants

### `ShugaPaths`

Encodes all directory and filename rules:

- CICE daily and static stores
- classification root and `data.zarr`
- metrics `mets.zarr`
- log filenames
- FIP and time-series graphics

## IO model

The shared loader in `shuga.io.zarr_loading` supports:

1. grouped monthly stores:
   `iceh_daily.zarr/YYYY-MM`
2. flat Zarr datasets
3. optional `iceh_static.zarr` merge
4. request-window padding for centered rolling methods

That keeps classification and metrics on the same dataset loading path.

## Workflow model

Classification and metrics are siblings, not subclasses of one another.

- `CICEClassifier` consumes daily CICE history and writes masks.
- `CICEMetrics` consumes daily CICE history plus those masks and writes derived metrics.

This avoids forcing a false inheritance relationship between them.

## Future extension points

The package layout leaves room for:

- observation loaders
- more plotting frontends
- regridding tools
- additional metrics
- wave/ice comparison products
- notebook helpers
