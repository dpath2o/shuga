# Classification

`shuga.classify.CICEClassifier` creates classified ice-domain products from CICE history output. The current workflow supports fast ice (`FI`), pack ice (`PI`), and sea ice (`SI`).

## Inputs

Classification uses CICE history variables loaded through `load_cice()`:

| Quantity | Typical variable | Purpose |
|---|---|---|
| Sea-ice concentration | `aice` | Parent sea-ice mask. |
| C-grid edge velocities | `uvelE`, `uvelN`, `vvelE`, `vvelN` | C-grid to T-grid speed reconstruction for `grid_type="Tc"`. |
| B-grid velocities | `uvel`, `vvel` | Legacy B-grid/T-grid reconstruction modes. |
| Static coordinates | `TLON`, `TLAT` | Hemisphere masking and plotting. |

Static variables are merged from the universal static store when needed:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

## Ice domains

### Fast ice (`FI`)

Fast ice is the speed-thresholded ice domain. A cell can become `FI` only if it first satisfies a sea-ice concentration threshold and then satisfies one of the velocity-based classification methods.

### Pack ice (`PI`)

Pack ice is defined as sea ice that is not fast ice:

\[
M_{\mathrm{PI}}(t,i,j)=M_{\mathrm{SI}}(t,i,j)\land\neg M_{\mathrm{FI}}(t,i,j).
\]

`PI` is derived from the parent sea-ice mask and the selected fast-ice mask. It is not independently speed-thresholded.

### Sea ice (`SI`)

Sea ice is the hemispheric concentration-threshold domain:

\[
M_{\mathrm{SI}}(t,i,j)=\left[a_{\mathrm{ice}}(t,i,j)\ge a_{\min}\right].
\]

`SI` does **not** undergo speed classification. It is used for sea-ice area, volume, thickness, stress, and diagnostic metrics. It is not stored in the same speed-classification manner as `FI` and `PI`.

## Parent sea-ice mask

\[
M_{\mathrm{SI}}(t,i,j)=
\begin{cases}
1, & a_{\mathrm{ice}}(t,i,j)>a_{\min},\\
0, & \text{otherwise}.
\end{cases}
\]

The default threshold is normally:

\[
a_{\min}=0.15.
\]

## T-grid speed

Classification uses a T-grid speed magnitude:

\[
s_T(t,i,j)=\sqrt{u_T(t,i,j)^2+v_T(t,i,j)^2}.
\]

For `grid_type="Tc"`, `u_T` and `v_T` are reconstructed from C-grid E/N face velocities. The details are described in [`regridding.md`](regridding.md).

## Raw fast-ice classification

\[
M_{\mathrm{FI,raw}}(t,i,j)=
M_{\mathrm{SI}}(t,i,j)
\land
\left[0<s_T(t,i,j)\le s_{\max}\right]
\land
\mathrm{finite}(s_T).
\]

The current default fast-ice speed threshold is:

\[
s_{\max}=5.0\times10^{-4}\ \mathrm{m\ s^{-1}}.
\]

The lower bound avoids classifying zero-filled or invalid velocity cells as fast ice.

## Binary-days classification

Binary-days classification applies a centred persistence filter to the raw mask. For a centred window of `W` days and minimum count `N`:

\[
M_{\mathrm{FI,bin}}(t,i,j)=
\left[
\sum_{\tau\in\mathcal{W}(t)} M_{\mathrm{FI,raw}}(\tau,i,j)\ge N
\right].
\]

The standard setting is:

\[
W=11,\qquad N=9.
\]

This means a cell is classified as fast ice when it is raw fast ice on at least **9 out of 11** centred days. The classifier pads the read window by half the required rolling window and then crops back to the requested output period.

## Rolling-mean classification

Rolling-mean classification smooths speed before thresholding:

\[
\bar{s}_T(t,i,j)=\frac{1}{R}\sum_{\tau\in\mathcal{R}(t)}s_T(\tau,i,j).
\]

The rolling-mean mask is:

\[
M_{\mathrm{FI,roll}}(t,i,j)=
M_{\mathrm{SI}}(t,i,j)
\land
\left[0<\bar{s}_T(t,i,j)\le s_{\max}\right]
\land
\mathrm{finite}(\bar{s}_T).
\]

The standard setting is:

\[
R=15\ \mathrm{days}.
\]

## Output variables

| Variable | Dimensions | Meaning |
|---|---|---|
| `FI_mask` | `time, nj, ni` | Boolean final fast-ice mask. |
| `FI_ispd` | `time, nj, ni` | Speed used by the method, masked to fast-ice cells. |
| `FI_aice` | `time, nj, ni` | Concentration masked to fast-ice cells. |
| `PI_mask` | `time, nj, ni` | Pack-ice mask, `SI_mask & ~FI_mask`. |
| `PI_ispd` | `time, nj, ni` | Speed used by the method, masked to pack-ice cells. |
| `PI_aice` | `time, nj, ni` | Concentration masked to pack-ice cells. |

## Output paths

```text
~/AFIM_archive/<SIM_NAME>/zarr/<HEMISPHERE>/ispd_thresh_5.0e-4/FI/Tc/
├── raw/data.zarr
├── bin-win-11_bin-min-09/data.zarr
└── roll-days-15/data.zarr
```

## Usage

```python
from shuga import RunSpec, ClassificationSpec, ShugaPaths, CICEClassifier

run = RunSpec(sim_name="LD-blend-base", start_date="2000-04-01", end_date="2003-06-30", hemisphere="SH")
classify = ClassificationSpec(
    ice_type="FI", grid_type="Tc", ispd_thresh=5.0e-4, aice_thresh=0.15,
    methods=("raw", "binary-days", "rolling-mean"), bin_window=11, bin_min_days=9, roll_window=15,
)
paths = ShugaPaths(run=run, classify=classify)
CICEClassifier(run=run, classify=classify, paths=paths).run_methods(overwrite=False)
```
