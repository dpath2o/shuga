# Observations

`shuga.observations` provides product-specific observation loaders and preparation utilities. The current focus is Antarctic fast ice from AF2020 and hemispheric sea-ice concentration/area from NSIDC.

## Current modules

```text
shuga/observations/
├── AF2020.py
├── NSIDC.py
└── legacy.py
```

`legacy.py` provides a compatibility facade for older code that imported `SeaIceObservations`.

## AF2020

AF2020 support is for Alex Fraser's Antarctic fast-ice dataset. Current use cases include:

- fast-ice area time series;
- AF2020 climatology overlays for `FIA`;
- native 15-day fast-ice concentration (`FIC`);
- common-grid AF2020 `FIC`/`FIP` products;
- FIP differencing against model simulations.

### FIC

`FIC` is used as a fast-ice concentration-like binary or fractional field from AF2020 and as `FI_mask * aice` for model products where needed. AF2020 native output is 15-day. The preferred design is to keep AF2020 FIC on native timestamps and downsample/select model FIC to those timestamps for side-by-side comparison.

### FIP

\[
\mathrm{FIP}_{\mathrm{obs}}(i,j)=\frac{1}{N_{ij}}\sum_t\mathrm{FIC}_{\mathrm{obs}}(t,i,j).
\]

When differencing model and observation FIP, both products must be on a shared grid:

\[
\Delta\mathrm{FIP}=\mathrm{FIP}_{\mathrm{model}}-\mathrm{FIP}_{\mathrm{obs}}.
\]

The continuous difference should remain in `[-1, 1]`, with both-zero cells masked before differencing.

## NSIDC

NSIDC support is used primarily for sea-ice area and extent style diagnostics, especially `SIA` overlays in time-series plots.

```python
from shuga.observations import NSIDCObs

obs = NSIDCObs(run=run, observations=obs_spec, paths=paths)
ds = obs.compute_sia_sie(start_date="2000-01-01", end_date="2003-12-31", hemisphere="SH")
```

## Future support: ESA-CCI sea-ice thickness

ESA-CCI sea-ice thickness support is planned. Likely uses include gridded thickness comparison, regional SIT/FIT context, seasonal sea-ice thickness climatologies, and model-observation skill metrics.

## Separation from plotting

Observation modules should load and prepare observational references. Plotting modules should consume those prepared products. This keeps observation processing reusable for both metrics and figures.
