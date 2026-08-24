# BEPSII-TS processing v0p2

This revision changes the operational BEPSII ice-type classification after
cross-checking the raw `i_type` field against independently documented
Southern Hemisphere fast-ice campaigns.

Observed cross-tab:

```text
documented FAST campaigns:
i_type = 1 : 190
i_type = 2 :   0
```

Operational mapping:

```text
i_type = 1 -> fast ice
i_type = 2 -> pack ice
i_type = 0 -> unknown / other
```

The campaign whitelist is retained only as an independent QA check.

## Output

```text
/g/data/gv90/da1339/SeaIce/InSitu/processed/BEPSII_TS/
├── BEPSII_TS_processing_summary.json
└── SH/
    ├── all/
    │   └── BEPSII_TS_points.zarr
    ├── fast/
    │   └── BEPSII_TS_fast_points.zarr
    ├── pack/
    │   └── BEPSII_TS_pack_points.zarr
    └── unknown/
        └── BEPSII_TS_unknown_points.zarr
```

## QA fields

No plausible thickness observations are removed merely because they are thick.

The processor adds:

```text
thickness_gt_3m
thickness_gt_5m
missing_snow
max_core_length
core_length_fraction_of_thickness
partial_core_lt80pct
duplicate_observation_key
is_documented_fast_campaign
type_campaign_consistent
```

`hi` remains the authoritative observational ice thickness.

`cl_S` and `cl_T` remain core lengths and are never substituted for `hi`.

## Date handling

The early Japanese records are retained. Campaign labels such as
`JARE 11 & 16` refer to expedition numbers, not necessarily calendar years.
No arbitrary 1980 cutoff is applied.

## Dry test

```bash
qsub shuga/scripts/observations/test_insitu_BEPSII.pbs
```

## Process

```bash
qsub -v OVERWRITE=true \
    shuga/scripts/observations/insitu_thickness.pbs
```

## Inspect

```bash
qsub shuga/scripts/observations/inspect_insitu_BEPSII.pbs
```

The fast-ice Zarr is the primary BEPSII input for the later CICE matchup
workflow, but the pack store is deliberately retained for independent testing
of CICE FI/PI classification.
