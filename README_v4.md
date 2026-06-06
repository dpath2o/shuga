# shuga AF2020/FIP/FIC revisions

This bundle implements the revised workflow:

1. Build AF2020 FIC/FIP once on a persistent common EPSG:3031 grid.
2. Keep AF2020 FIC on the native 15-day AF2020 timestamps.
3. Compute continuous FIP difference as `mod - obs`, in `[-1, 1]`, with both-zero cells masked.
4. Keep `diff_cat` as a secondary categorical product.
5. Plot simulation, AF2020, `diff`, or `diff_cat` via `CICEPlotter.plot_fip()` without recomputing simulation FIP.

## Copy targets

```text
shuga/regridder/pyresample_fip_difference_replacement.py
    Copy the two functions into shuga/regridder/pyresample.py:
      area_definition_from_xy
      fip_difference_dataset

shuga/plotting/CICEPlotter_plot_fip_replacement.py
    Replace CICEPlotter.plot_fip() in shuga/plotting/cice.py with this method.

shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.py
shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.pbs
shuga/scripts/comparisons/FIP_differencing.py
shuga/scripts/comparisons/FIP_differencing.pbs
shuga/scripts/comparisons/plot_FIC_side_by_side.py
```

## Build persistent AF2020 store

```bash
qsub -v SIM_NAME=LD-blend-base,OVERWRITE=0 \
  shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.pbs
```

Default output:

```text
/g/data/gv90/da1339/SeaIce/FI_obs/AF-FI-2020db_common-5km_pyresample.zarr
```

The script will skip if the output exists unless `OVERWRITE=1`.

## Compute FIP difference and plot all 8 regions

```bash
qsub -v SIM_NAME=LD-blend-base,START_DATE=2000-01-01,END_DATE=2003-12-31,FIP_START=2000-04-01,FIP_END=2003-06-30,OVERWRITE=1,PLOT=1 \
  shuga/scripts/comparisons/FIP_differencing.pbs
```

Outputs go to:

```text
~/AFIM_archive/SIM_NAME/zarr/comparisons/FIPdiff_SIM_minus_AF2020_...
```

## Plot AF2020 and simulation FIC side by side

```bash
python shuga/scripts/comparisons/plot_FIC_side_by_side.py \
  -s LD-blend-base \
  -b 2000-03-01 \
  -e 2003-12-31 \
  --regions Aus
```

AF2020 FIC is read from the persistent common-grid store. Simulation FIC is computed on the fly as:

```text
FI_mask * aice
```

and downsampled to AF2020 native 15-day timestamps via nearest-time selection. Simulation FIC is not stored and is not regridded.
