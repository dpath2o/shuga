# AF2020/CICE FIP and FIC comparison on a shared pyresample grid

This bundle replaces the earlier xESMF-based comparison scaffold with a workflow that follows the AFIM `FI-sensitivity-study.ipynb` FIP-difference cells and `AFIM/src/sea_ice_regridder.py` pyresample helpers.

## Files to copy into shuga

```text
shuga/observations/AF2020.py
shuga/observations/NSIDC.py
shuga/observations/__init__.py
shuga/regridder/__init__.py
shuga/regridder/pyresample.py
shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.py
shuga/scripts/comparisons/plot_fic_pair_from_common_grid.py
shuga/scripts/comparisons/README_AF2020_pyresample.md
```

After this, remove or retire:

```text
shuga/observations/cice.py
```

The old `cice.py` mixed NSIDC and AF2020 observation logic. The replacement layout is:

```text
shuga.observations.AF2020.AF2020Observations
shuga.observations.NSIDC.NSIDCObservations
```

Any imports of `SeaIceObservations` should be revised to one of those explicit classes.

## Scientific/data design

### FIP

FIP is fast-ice persistence over a multi-year or seasonal period.

Default behaviour reproduces the notebook logic:

1. AF2020 native 15-day rasters are opened from `FastIce_70_YYYY.nc`.
2. AF2020 fast ice is defined as `Fast_Ice_Time_series >= 4`.
3. AF2020 FIP is computed on the native AF2020 grid using the 15-day samples.
4. CICE daily `FI_mask` is reindexed to the AF2020 native times using nearest-time matching.
5. CICE FIP is computed on the native CICE grid.
6. AF2020 FIP and CICE FIP are both pyresample-nearest regridded to the same EPSG:3031 grid.
7. FIP difference is computed on the shared grid as `model - obs`.

The categorical difference follows the notebook convention:

```text
0 = agreement              -0.5 <= model - obs <= 0.5
1 = model-dominant          model - obs > 0.5
2 = observation-dominant    model - obs < -0.5
NaN = both zero / unclassified
```

### FIC

FIC here means daily binary-days `FI_mask`-masked model concentration:

```text
SIM_FIC = SIM_FI_mask * aice
```

AF2020 is natively 15-day, so AF2020 FIC-like daily fields are produced by temporal interpolation of the native AF2020 binary mask:

```text
AF_FIC = interp_daily(AF_FI_mask_15day)
```

Use `--af2020-daily-method linear` for fractional transition fields or `nearest` for stepwise fields.

The build script does **not** write full daily circum-Antarctic FIC fields by default because a 5-km whole-Antarctic daily store can become enormous. Instead, pass `--fic-dates` for specific AMJ days or `--write-daily-fic` when you intentionally want every daily field.

## Example: FIP difference plus selected AMJ FIC dates

```bash
python shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.py \
  -s LD-blend-base \
  -b 2000-04-01 \
  -e 2003-06-30 \
  --fip-start 2000-04-01 \
  --fip-end 2003-06-30 \
  -m binary-days \
  --grid-type Tc \
  --af2020-raw-root /g/data/gv90/da1339/SeaIce/FI_obs/org \
  --pixel-size-m 5000 \
  --radius-of-influence-m 10000 \
  --buffer-m 20000 \
  --cice-lon-shift-deg 0.25 \
  --fic-dates 2000-04-01,2000-05-01,2000-06-01 \
  --include-weight \
  --overwrite
```

Outputs default to:

```text
~/AFIM_archive/LD-blend-base/zarr/comparisons/
  af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_FIP.zarr
  af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_AF2020_native_FIP.zarr
  af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_FIP_diff_stats.csv
  af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_FIP_diff_stats.tex
  af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_FIC_samples.zarr
```

## Quick-look FIC plot

```bash
python shuga/scripts/comparisons/plot_fic_pair_from_common_grid.py \
  ~/AFIM_archive/LD-blend-base/zarr/comparisons/af2020_pyresample_LD-blend-base_binary-days_2000-04-01_2003-06-30_FIC_samples.zarr \
  --date 2000-04-01 \
  --region 97.5 142.5 -67.5 -63 \
  --out /g/data/gv90/da1339/GRAPHICAL/LD-blend-base/Aus/FIC_pair_20000401.png
```

## Notes on the CICE longitude shift

The notebook pyresample section used `CICE_SO["TLON"] + 0.25` before resampling model FIP. The script preserves that via the default:

```bash
--cice-lon-shift-deg 0.25
```

Set `--cice-lon-shift-deg 0.0` if later inspection shows the shift is not appropriate for a particular grid.
