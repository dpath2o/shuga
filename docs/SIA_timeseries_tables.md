# CICE / NSIDC / OSI-SAF SIA comparison

`shuga/scripts/comparisons/SIA_timeseries_tables.py` creates a multi-experiment Antarctic sea-ice area comparison against two observational products:

- `NSIDC`: plotted as a solid black line.
- `OSI-SAF-450`: plotted as a dashed black line.

The script writes:

```text
SIA_daily.csv
SIA_monthly_mean.csv
SIA_annual_mean.csv
SIA_full_period_<years>.png
SIA_daily_climatology_envelope_<years>.png
```

The first figure is a full-period daily SIA time series. The second figure is a day-of-year climatological view: daily mean SIA as the main line and min/max envelope as shading for each CICE experiment.

## Expected CICE input

By default the script looks for shuga metric stores at:

```text
/g/data/gv90/da1339/afim_output/<SIM_NAME>/zarr/SH/ispd_thresh_5.0e-4/SI/Tc/bin-win-11_bin-min-09/mets.zarr
```

It first looks for an existing SIA variable (`sia`, `SIA`, or `sea_ice_area`). If that is absent, it attempts to compute SIA from a concentration or mask variable plus a cell-area variable.

If your local store differs, pass a template containing `{sim_name}`:

```bash
--cice-store-template '/path/to/{sim_name}/some/store.zarr'
```

## Interactive usage

```bash
python shuga/scripts/comparisons/SIA_timeseries_tables.py \
  --sim-names LD-NIL Cs-high-ktens-high Cq-high Cl-mid \
  --start-date 2000-01-01 \
  --end-date 2005-12-31 \
  --nsidc-sia-store /g/data/gv90/da1339/observations/NSIDC/processed/NSIDC_SH_SIA.zarr \
  --osisaf-sia-store /g/data/gv90/da1339/observations/OSI-SAF-450/processed/OSI-SAF-450_SH_SIA.zarr \
  --out-dir /g/data/gv90/da1339/GRAPHICAL/AFIM/SIA_comparisons/LD_pub_2000_2005
```

## PBS usage

```bash
cd /g/data/gv90/da1339/src/mawsons-chest
shuga/scripts/comparisons/SIA_timeseries_tables_pbs_wrapper.sh \
  -s "LD-NIL Cs-high-ktens-high Cq-high Cl-mid" \
  -b 2000-01-01 \
  -e 2005-12-31 \
  -n /g/data/gv90/da1339/observations/NSIDC/processed/NSIDC_SH_SIA.zarr \
  -o /g/data/gv90/da1339/observations/OSI-SAF-450/processed/OSI-SAF-450_SH_SIA.zarr \
  -d /g/data/gv90/da1339/GRAPHICAL/AFIM/SIA_comparisons/LD_pub_2000_2005
```

Additional script arguments can be passed after the wrapper options:

```bash
shuga/scripts/comparisons/SIA_timeseries_tables_pbs_wrapper.sh \
  -s "LD-NIL Cs-high-ktens-high" \
  -b 2000-01-01 -e 2005-12-31 \
  -n /path/nsidc.zarr \
  -- --method raw --ice-type SI --grid-type Tc
```

## Caveats

- The observational SIA products should be checked for consistent concentration threshold, land mask, pole-hole treatment, and temporal coverage before publication-quality interpretation.
- OSI-SAF streams differ by sensor family and CDR/ICDR status. Do not silently join OSI-450-a1 and AMSR-based streams without documenting the product transition.
- The seasonal envelope drops 29 February to produce a stable 365-day day-of-year axis.
