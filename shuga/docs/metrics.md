# Metrics

`shuga.metrics.CICEMetrics` computes diagnostics from CICE history fields and classified masks. Metrics are written as method-specific `mets.zarr` stores beside the corresponding `data.zarr` classification product.

Metric names and groups are defined in `shuga/metrics/registry.py`.

## Notation

| Symbol | Meaning |
|---|---|
| `a(t,i,j)` | sea-ice concentration, usually `aice` |
| `h(t,i,j)` | ice thickness, usually `hi` |
| `A(i,j)` | T-grid cell area, usually `tarea` |
| `M_F` | fast-ice mask |
| `M_P` | pack-ice mask |
| `M_S` | sea-ice mask |

Fast-ice metrics use `M_F`, pack-ice metrics use `M_P`, and sea-ice metrics use `M_S`.

## Metric groups

| Group | Description |
|---|---|
| `fi_core` | FIA, FIV, FIT, FIP, FIS, thermodynamic/mechanical rates. |
| `fi_spatial` | FIHI, FIST, and annualised spatial rate fields. |
| `fi_regional` | FIA/FIT by Antarctic sector. |
| `fi_summary` | Seasonal maxima, minima, and timing summaries. |
| `fi_stress` | Stress/form-factor diagnostics where source variables exist. |
| `fi_diags` | Mean fast-ice diagnostic fields such as speed and stress budgets. |
| `fi_spec` | FIPSI, winter-persistence, and observation skill metrics. |
| `pi_*`, `si_*` | Pack-ice and sea-ice analogues. |
| `fi_all`, `pi_all`, `si_all` | Domain-specific full groups. |

## Fast-ice core metrics

### `FIA`: fast-ice area

\[
\mathrm{FIA}(t)=10^{-9}\sum_{i,j} M_F(t,i,j)a(t,i,j)A(i,j).
\]

Units: `10^3 km^2`. This is the primary scalar diagnostic for fast-ice extent, growth, maximum, retreat, and bias.

### `FIV`: fast-ice volume

\[
\mathrm{FIV}(t)=10^{-12}\sum_{i,j}M_F(t,i,j)a(t,i,j)h(t,i,j)A(i,j).
\]

Units: commonly `10^3 km^3`, depending on `MetricsSpec.volume_scale`.

### `FIT`: fast-ice mean thickness

\[
\mathrm{FIT}(t)=
\frac{\sum_{i,j}M_F a h A}{\sum_{i,j}M_F a A}.
\]

Units: `m`. Interpret with `FIA`, because a compact thick fast-ice region and a broader thin region can have similar mean thickness.

### `FIP`: fast-ice persistence

\[
\mathrm{FIP}(i,j)=\frac{1}{N_{ij}}\sum_{t\in\mathcal{T}}M_F(t,i,j).
\]

Units: `1`. A value of 1 means fast ice for all valid timesteps; 0.5 means fast ice for half of them. This is the main map diagnostic for spatial fast-ice comparison.

### `FIS`: fast-ice strength

\[
\mathrm{FIS}(t)=
\frac{\sum M_F S A}{\sum M_F A},
\]

where `S` is CICE `strength`. Units follow CICE, typically `N m-1`.

### Rate metrics

| Variable | Meaning | Typical relevance |
|---|---|---|
| `FITVR` | fast-ice thermodynamic volume-rate diagnostic | thermodynamic growth/melt contribution |
| `FIMVR` | fast-ice mechanical/dynamic volume-rate diagnostic | convergence, divergence, export, mechanical redistribution |
| `FITAR` | fast-ice thermodynamic area-rate diagnostic | thermodynamic concentration/area change |
| `FIMAR` | fast-ice mechanical/dynamic area-rate diagnostic | mechanical area change |

These help diagnose whether missing FIA growth is thermodynamic, mechanical, or a combination.

## Spatial fast-ice metrics

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `FIHI` | `nj, ni` | `m` | Time-mean fast-ice thickness field. |
| `FIST` | `nj, ni` | `N m-1` or source units | Time-mean fast-ice strength field. |
| `FITVR_YR` | `nj, ni` | tendency units | Annualised thermodynamic volume/thickness-rate field. |
| `FIMVR_YR` | `nj, ni` | tendency units | Annualised mechanical/dynamic volume/thickness-rate field. |
| `FITAR_YR` | `nj, ni` | tendency units | Annualised thermodynamic area-rate field. |
| `FIMAR_YR` | `nj, ni` | tendency units | Annualised mechanical/dynamic area-rate field. |

`FIHI`, `FIST`, and `FIMAR_YR` are particularly useful for separating persistent fast-ice locations from thermodynamic or mechanical growth mechanisms.

## Regional metrics

The Antarctic sectors are `DML`, `WIO`, `EIO`, `Aus`, `VOL`, `AS`, `BS`, and `WS`.

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `FIA_by_region` | `time, region` | `10^3 km^2` | Regional fast-ice area. |
| `FIT_by_region` | `time, region` | `m` | Regional area-weighted fast-ice thickness. |

## Seasonal summaries

| Variable pattern | Units | Meaning |
|---|---|---|
| `FIA_max_mean`, `FIA_max_std` | `10^3 km^2` | Mean and spread of seasonal/annual maxima. |
| `FIA_min_mean`, `FIA_min_std` | `10^3 km^2` | Mean and spread of seasonal/annual minima. |
| `FIA_doy_max_mean`, `FIA_doy_max_std` | day / days | Timing and spread of maximum FIA. |
| `FIA_doy_min_mean`, `FIA_doy_min_std` | day / days | Timing and spread of minimum FIA. |
| `FIT_*` analogues | `m` or days | Thickness extrema and timing. |

## Observation skill and persistence-stability

| Variable | Units | Meaning |
|---|---|---|
| `FIPSI` | `1` or area-scaled diagnostic | Fast-ice persistence stability index. |
| `persistent_winter_area` | `10^3 km^2` | Area persistent through winter window. |
| `ever_winter_area` | `10^3 km^2` | Area ever classified as fast ice during winter. |
| `FIA_Bias`, `FIA_RMSE`, `FIA_MAE`, `FIA_Corr` | area units / 1 | FIA skill metrics. |
| `FIT_Bias`, `FIT_RMSE`, `FIT_MAE`, `FIT_Corr` | m / 1 | FIT skill metrics. |

## Stress and diagnostics

Examples include `FIKuxE_mean`, `FIKuxE_abs_mean`, `FIKuE_mag_mean`, `FI_tau_air_mean`, `FI_tau_ocean_mean`, `FI_tau_internal_mean`, `FI_tau_ld_est_mean`, `FI_strain_invariant_mean`, and `FI_ld_mag_proxy_mean`. Units follow the underlying source fields. These variables diagnose why the model produces or fails to produce fast ice.

## Pack-ice and sea-ice metric names

Pack-ice core metrics: `PIA`, `PIV`, `PIT`, `PIP`, `PIS`, `PITVR`, `PIMVR`, `PITAR`, `PIMAR`.

Sea-ice core metrics: `SIA`, `SIV`, `SIT`, `SIP`, `SIS`, `SITVR`, `SIMVR`, `SITAR`, `SIMAR`.

Spatial, regional, summary, stress, and diagnostic products follow the same naming pattern with `P` or `S` prefixes.

## Usage

```python
from shuga import RunSpec, ClassificationSpec, MetricsSpec, ShugaPaths, CICEMetrics

run = RunSpec(sim_name="LD-blend-base", start_date="2000-04-01", end_date="2003-06-30", hemisphere="SH")
classify = ClassificationSpec(ice_type="FI", grid_type="Tc", methods=("binary-days",), ispd_thresh=5.0e-4)
metrics_spec = MetricsSpec(methods=("binary-days",))
paths = ShugaPaths(run=run, classify=classify, metrics=metrics_spec)

runner = CICEMetrics(run=run, classify=classify, metrics=metrics_spec, paths=paths)
runner.compute_metrics("binary-days", metric_groups=["fi_core", "fi_spatial", "fi_regional"], update_missing_only=True)
```
