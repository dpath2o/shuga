# WHACS wave forcing for standalone CICE6

This page documents the `shuga` workflow used to convert hourly WHACS directional wave spectra into monthly CICE6 wave-forcing files on the ACCESS-OM3 0.25° CICE T grid.

The workflow is designed for standalone CICE/Icepack experiments in which externally prescribed open-ocean wave spectra are read by CICE and the model wave/FSD machinery subsequently handles propagation, attenuation, fracture, and floe-size evolution.

The relevant package code is:

```text
shuga/waves/whacs.py
shuga/waves/whacs_multi.py
shuga/waves/cawcr.py
```

The production entry points are:

```text
shuga/scripts/waves/whacs_regrid.py
shuga/scripts/waves/whacs_regrid.pbs
shuga/scripts/waves/whacs_regrid_wrapper.sh
```

Daily PyGMT QC plotting is handled separately under:

```text
shuga/plotting/cawcr.py
shuga/scripts/plotting/plot_whacs_daily.py
shuga/scripts/plotting/plot_whacs_daily.pbs
shuga/scripts/plotting/plot_whacs_daily_wrapper.sh
```

## Scientific purpose

The target CICE forcing variable is a one-dimensional frequency spectrum

\[
E(f) \quad [\mathrm{m^2\,s}],
\]

stored hourly on the CICE T grid. CICE/Icepack does not require the original WHACS directional dimension for this forcing pathway.

The preprocessing therefore performs three scientifically distinct transformations:

1. integrate the WHACS directional variance spectrum over direction;
2. conservatively remap the native WHACS frequency bins onto the 25-bin frequency grid expected by the CICE/Icepack wave implementation;
3. spatially interpolate the resulting station spectra onto the 0.25° CICE T grid.

These operations should be kept conceptually separate. The directional and frequency transformations modify the spectral representation. The spatial interpolation only maps those spectra from WHACS output points onto the CICE grid.

## WHACS source archive

The source root currently used on Gadi is:

```text
/g/data/ia39/WP3/release/ACS_hindcast/spec/release/WP3/WHACS/
BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/spectra/1hr/efth/
```

Each calendar month is represented by five parallel directional-spectrum files:

```text
efth_WHACS_hindcast_spec_GRID_1hr_YYYYMM010000-YYYYMMDD2300.nc
efth_WHACS_hindcast_spec_GLOB_1hr_YYYYMM010000-YYYYMMDD2300.nc
efth_WHACS_hindcast_spec_BUOYS_1hr_YYYYMM010000-YYYYMMDD2300.nc
efth_WHACS_hindcast_spec_NIWA_1hr_YYYYMM010000-YYYYMMDD2300.nc
efth_WHACS_hindcast_spec_SCHISM_1hr_YYYYMM010000-YYYYMMDD2300.nc
```

All five files expose the same fundamental spectral structure:

```text
efth(time, station, frequency, direction)
frequency(frequency)
frequency1(frequency)
frequency2(frequency)
direction(direction)
latitude(station)
longitude(station)
time(time)
```

with native dimensions:

```text
frequency = 28
direction = 30
time      = hourly calendar month
```

The number and spatial distribution of `station` points differs between source sets.

### Why all five source sets are used

The WHACS `station` dimension is a collection of **fixed model-output locations**. It is not a set of drifting buoys and it does not move or advect through the month: `latitude(station)` and `longitude(station)` do not carry a time dimension.

This matters for circum-Antarctic forcing. The `GRID` archive is dense in selected regional/coastal parts of the WHACS domain but does not provide adequate spectral-point coverage around the entire Southern Ocean. A `GRID`-only regrid therefore leaves major source-coverage gaps, most visibly in parts of the Atlantic and eastern South Pacific.

The production workflow consequently combines all five full-spectrum archives:

```text
GRID + GLOB + BUOYS + NIWA + SCHISM
```

before spatial interpolation. In particular, `GLOB` supplies coarse global spectral points that substantially improve circum-Antarctic coverage.

The source-set order is currently:

```python
WHACS_SPECTRAL_SETS = (
    "GRID",
    "GLOB",
    "BUOYS",
    "NIWA",
    "SCHISM",
)
```

This order also defines duplicate-location priority: when exactly duplicated fixed coordinates occur in more than one source set, the first occurrence is retained.

## Multi-source assembly

`WHACSMultiSourceRegridder` in `shuga/waves/whacs_multi.py` extends the tested `WHACSRegridder` pathway while replacing the original `GRID`-only source loader with a composite five-source loader.

For each month it:

1. opens all five WHACS files lazily with xarray/Dask;
2. validates that their hourly time, frequency, frequency-bound, and direction axes agree;
3. retains only the fields required for spectral forcing;
4. concatenates their fixed station locations;
5. normalises station indexing;
6. removes duplicate station coordinates;
7. passes the resulting composite station spectrum into the established direction-integration, frequency-remapping and CICE-grid interpolation workflow.

Station coordinates are currently de-duplicated after longitude normalisation by rounding longitude and latitude to five decimal degrees. This removes exact/practical duplicates without intentionally merging merely nearby stations.

### Example: January 1995

A successful January 1995 pilot reported:

```text
GRID   = 10055
GLOB   =   370
BUOYS  =   286
NIWA   =    80
SCHISM =   443
----------------
total  = 11234
unique = 11111
duplicates removed = 123
```

These numbers are useful as a QC reference for that month, not as a permanent archive invariant.

## Direction integration

WHACS stores directional variance spectral density as

\[
E(f,\theta) \quad [\mathrm{m^2\,s\,rad^{-1}}].
\]

`shuga` integrates over the periodic directional bins to obtain the one-dimensional frequency spectrum required by the CICE forcing pathway:

\[
E(f) = \int E(f,\theta)\,d\theta.
\]

In discrete form:

\[
E_j = \sum_m E_{j,m}\,\Delta\theta_m.
\]

The code derives periodic angular-bin widths from the supplied WHACS directional centres and verifies that

\[
\sum_m \Delta\theta_m \approx 2\pi.
\]

Negative spectral densities are clipped defensively to zero before integration.

After directional integration, the station spectrum has dimensions:

```text
(time, station, frequency)
```

with units `m2 s`.

## Conservative 28 → 25 frequency remapping

The native WHACS spectrum contains 28 frequency bins. The CICE/Icepack wave implementation used by this workflow expects the following 25 frequency centres:

```text
0.04118000  0.04529800  0.04982780  0.05481058  0.06029164
0.06632081  0.07295289  0.08024818  0.08827299  0.09710029
0.10681032  0.11749136  0.12924050  0.14216454  0.15638101
0.17201911  0.18922101  0.20814312  0.22895744  0.25185317
0.27703848  0.30474234  0.33521661  0.36873826  0.40561208
```

The target bins use the geometric ratio 1.1, with bounds

\[
f^-_k = \frac{f_k}{\sqrt{1.1}},
\qquad
f^+_k = f_k\sqrt{1.1},
\]

and widths

\[
\Delta f_k = f^+_k - f^-_k.
\]

The native 28-bin spectrum is **not** simply sampled at the 25 target centres. Spectral variance is remapped using bin overlap:

\[
E^{\mathrm{CICE}}_k =
\frac{\sum_j E^{\mathrm{WHACS}}_j\,\Delta f^{\mathrm{overlap}}_{jk}}
     {\Delta f^{\mathrm{CICE}}_k}.
\]

This formulation conserves integrated variance over the frequency range common to WHACS and CICE25:

\[
m_0 = \int E(f)\,df.
\]

Significant wave height follows from

\[
H_s = 4\sqrt{m_0}
    = 4\sqrt{\sum_k E_k\Delta f_k}.
\]

Because the two frequency ranges are not identical, exact retention of total native-WHACS `m0` is not expected where appreciable energy lies outside the CICE25 support. The production log therefore reports the sampled ratio

```text
CICE25 m0 / native WHACS m0
```

as a direct spectral QC.

For the January 1995 five-source pilot:

```text
median = 0.99103
p05    = 0.92189
p95    = 0.99791
n      = 130 sampled stations
```

The approximately 99% median retention is consistent with most wave-energy variance lying within the CICE25 spectral support, while the lower tail reflects spectra containing a larger fraction of energy outside that range.

## Spatial interpolation to the CICE T grid

After frequency remapping, all unique WHACS station spectra are interpolated to the ACCESS-OM3 0.25° CICE T grid.

Default production settings are:

```text
method       = inverse-distance weighting (IDW)
k nearest    = 8
power        = 2.5
radius       = 1000 km
target north = 35°S
```

The interpolation uses great-circle geometry via unit-sphere Cartesian coordinates and a `scipy.spatial.cKDTree`. Distances are converted to kilometres before IDW weights are constructed.

For target cell `i` and source station `j`, the unnormalised weight is

\[
w_{ij} = d_{ij}^{-p},
\]

for neighbours within the configured search radius. The retained weights are normalised to sum to one for each valid target cell.

If a target point exactly coincides with a WHACS station, the coincident source receives the interpolation weight rather than allowing the inverse-distance singularity to dominate numerically.

### Static weights

The station geometry is fixed, so the sparse station-to-CICE mapping can be built once and reused by subsequent months.

The five-source default weight filename is:

```text
/g/data/gv90/da1339/grids/weights/
map_WHACS_grid-glob-buoys-niwa-schism_to_ACCESS-OM3-025_idw_k8.npz
```

This deliberately differs from the former `GRID`-only weight filename. Reusing a `GRID`-only matrix with the five-source station vector would be scientifically and dimensionally invalid.

The sparse weight build is protected by a filesystem lock so concurrently submitted PBS array jobs do not race while creating the shared weight file.

## No sea-ice mask in the forcing product

The production WHACS forcing is deliberately **not masked by NSIDC or by model sea-ice concentration**.

The forcing file represents the externally prescribed WHACS wave field on the CICE grid. Subsequent wave propagation, attenuation, fracture and FSD evolution belong to the CICE/Icepack model state.

The production log should explicitly contain:

```text
Ice mask    : none (WHACS spectrum retained independently of sea-ice concentration)
```

and the NetCDF contains the global attribute:

```text
ice_mask = "none; WHACS forcing is not masked by observed or model sea-ice concentration"
```

This separation is important experimentally. Masking the external forcing with observed SIC would introduce an observational ice-state constraint into an otherwise prognostic standalone-CICE wave experiment.

## Output contract

Monthly files are currently written under the historical CAWCR-compatible naming convention:

```text
/g/data/gv90/da1339/afim_input/CAWCR/
CAWCR_efreq_for_CICE6_YYYYMM.nc
```

The filename is retained for downstream compatibility even though the production source is WHACS.

The principal forcing field is:

```text
efreq(time, nfreq, nj, ni)
```

with:

```text
nfreq = 25
```

Frequency metadata are included as:

```text
wavefreq(nfreq)
wavefreq_lo(nfreq)
wavefreq_hi(nfreq)
dwavefreq(nfreq)
```

Grid coordinates are included as:

```text
TLON(nj, ni)
TLAT(nj, ni)
```

Useful provenance/global attributes include:

```text
source_product
source_file
source_sets
source_geometry
frequency_grid
spectral_remap
station_regrid
station_weights
target_lat_max
ice_mask
completed
completed_utc
```

A five-source file identifies itself with:

```text
source_sets = "GRID,GLOB,BUOYS,NIWA,SCHISM"
```

The completion check requires this attribute. An older completed `GRID`-only file is therefore treated as stale and rebuilt by the five-source production workflow.

## Streaming and restart behaviour

The native directional spectrum is large. The production workflow is therefore monthly and Dask-backed rather than materialising multi-year spectra in memory.

The regridded CICE forcing is written incrementally in hourly chunks. The default PBS setting is:

```text
TIME_CHUNK=4
```

Output is first written to a process-specific partial file:

```text
CAWCR_efreq_for_CICE6_YYYYMM.nc.partial.<pid>
```

Only after all hourly records are successfully written is the file marked:

```text
completed = "true"
```

and atomically moved onto the final monthly filename. A failed job therefore should not masquerade as a valid completed forcing month.

Completed five-source files are skipped on rerun unless explicit overwrite is requested.

## Gadi execution

A single January-1995 pilot can be submitted as a normal PBS job:

```bash
qsub \
  -P au88 \
  -v START_YEAR=1995,END_YEAR=1995 \
  shuga/scripts/waves/whacs_regrid.pbs
```

When no `PBS_ARRAY_INDEX` is present, the PBS worker defaults to index 0, corresponding to January of `START_YEAR`.

The production wrapper is:

```bash
shuga/scripts/waves/whacs_regrid_wrapper.sh
```

A multi-year submission uses one PBS array element per month, for example:

```bash
./shuga/scripts/waves/whacs_regrid_wrapper.sh 1995 2005
```

Important distinction:

```text
PBS accounting project = au88
shuga data project      = gv90
WHACS source project    = ia39
analysis3 modules       = xp65
```

Default PBS resources are presently:

```text
queue    = normalbw
ncpus    = 8
memory   = 64 GB
walltime = 24 h
storage  = gdata/gv90 + gdata/ia39 + gdata/xp65
```

See [`gadi-workflows.md`](gadi-workflows.md) for operational details.

## Production logging and QC

Each monthly run writes a log under the shuga logs hierarchy, normally:

```text
~/logs/waves/whacs_regrid_YYYYMM.log
```

A healthy five-source startup should show all of the following:

```text
WHACS spectral source sets: GRID,GLOB,BUOYS,NIWA,SCHISM
Opening WHACS GRID: ...
Opening WHACS GLOB: ...
Opening WHACS BUOYS: ...
Opening WHACS NIWA: ...
Opening WHACS SCHISM: ...
Combined WHACS source stations: ...
Integrating WHACS directional spectrum over theta
Conservatively remapping WHACS frequency bins -> CICE25
Spectral QC retained m0 ...
Opening grid geometry: ...
Building/loading static WHACS-station -> CICE weights
Processing hourly chunk ...
```

Before launching a long production run, check at least:

1. all five source files were found;
2. the combined and unique station counts are plausible;
3. duplicate removal is non-zero but small relative to the total source set;
4. the CICE25/native-WHACS `m0` retention distribution is plausible;
5. the combined static weight filename is being used;
6. `Ice mask: none` is present;
7. the final NetCDF contains `source_sets=GRID,GLOB,BUOYS,NIWA,SCHISM` and `completed=true`;
8. daily spatial QC maps show sensible circum-Antarctic coverage and no obvious interpolation discontinuities.

## Daily PyGMT QC

The plotting workflow diagnoses significant wave height and spectral peak period from both native and regridded spectra.

For each hourly spectrum:

\[
H_s = 4\sqrt{\sum_f E(f)\Delta f}
\]

and

\[
T_p = \frac{1}{f_{\mathrm{peak}}}.
\]

Daily maps are produced from the daily mean of the hourly diagnosed quantities rather than diagnosing them from a pre-averaged daily spectrum.

The intended 2 × 2 diagnostic is:

| | Significant wave height | Peak period |
|---|---|---|
| upper row | native WHACS spectral points | native WHACS spectral points |
| lower row | regridded CICE T grid | regridded CICE T grid |

with PyGMT colourmaps:

```text
Hs : cmocean/amp
Tp : cmocean/phase
```

For a strict five-source before/after comparison, the upper-row native diagnostic must use the same combined `GRID,GLOB,BUOYS,NIWA,SCHISM` source set as the production regridder. A `GRID`-only upper row should not be interpreted as the complete source geometry once the lower row has been generated from all five sets.

## Design boundaries

This workflow intentionally does **not**:

- propagate waves between WHACS station points during preprocessing;
- infer station motion or advect source locations;
- mask waves using NSIDC concentration;
- reconstruct the full directional spectrum after direction integration;
- reproduce a dynamically coupled WW3/CICE system;
- reconstruct spectra from the native WHACS SMC partition fields.

The five-source interpolation is a pragmatic full-spectrum forcing pathway. A future, more comprehensive alternative could reconstruct `E(f)` from native WHACS gridded/SMC partition output and use the full-spectrum station archive as an independent validation dataset.

## External references

- WHACS data catalogue: <https://data.csiro.au/>
- WHACS dataset description / publication: search the CSIRO WHACS release associated with the BoM–CSIRO WWIII-v6.07 ERA5-forced hindcast.
- CICE Consortium: <https://github.com/CICE-Consortium/CICE>
- Icepack: <https://github.com/CICE-Consortium/Icepack>

For the exact model implementation used by the current wave experiments, also see the corresponding CICE/Icepack experiment branches and commit SHAs recorded with the experiment configuration.
