# Regridding

`shuga.regridding` contains tools for grid transformations used by classification, comparisons, and model-observation workflows.

Current modules:

```text
shuga/regridding/
├── cice.py
├── pyresample.py
└── xesmf.py
```

## CICE velocity grid handling

`shuga/regridding/cice.py` contains CICE-specific velocity reconstruction helpers. These are central to fast-ice classification because the speed threshold must be applied on a consistent target grid.

### T-grid speed

\[
s_T(t,i,j)=\sqrt{u_T(t,i,j)^2+v_T(t,i,j)^2}.
\]

Fast ice is then classified by thresholding `s_T`, as described in [`classification.md`](classification.md).

### C-grid to T-grid reconstruction: `grid_type="Tc"`

For CICE C-grid output, velocity components can be available on E and N faces:

```text
uvelE, vvelE
uvelN, vvelN
```

A T-cell velocity is reconstructed by averaging adjacent face velocities to the T-cell centre. Schematically:

\[
u_T(i,j)=\frac{1}{2}\left[u_E(i,j)+u_E(i-1,j)\right],
\]

\[
v_T(i,j)=\frac{1}{2}\left[v_N(i,j)+v_N(i,j-1)\right].
\]

The implementation handles masks, periodic wrapping, and missing values according to the selected grid mode.

### B-grid to T-grid reconstruction

Legacy modes use 2-by-2 corner averaging:

\[
u_T(i,j)=\frac{1}{4}\left[u_B(i,j)+u_B(i+1,j)+u_B(i,j+1)+u_B(i+1,j+1)\right],
\]

and similarly for `v_T`. `Ta` propagates missing values; `Tb` fills missing values as zero before averaging.

### Explicit regridder mode

`Tx` supports an explicit regridder callable. Use this when the velocity source grid and target grid require an externally constructed mapping.

## pyresample

`shuga/regridding/pyresample.py` supports swath-to-area resampling and common-grid comparison workflows, especially AF2020/CICE FIP and FIC products. Algorithm details are provided by the [pyresample documentation](https://pyresample.readthedocs.io/).

## xESMF

`shuga/regridding/xesmf.py` supports ESMF-backed regridding. Use it for structured model-grid interpolation/regridding and some forcing workflows. See the [xESMF documentation](https://xesmf.readthedocs.io/).

## Choosing a method

| Task | Recommended method |
|---|---|
| CICE C-grid velocity to T-grid speed | `regridding/cice.py` |
| Fast-ice classification speed reconstruction | `regridding/cice.py` |
| AF2020 to common polar stereographic grid | `pyresample.py` |
| FIP differencing on a shared grid | `pyresample.py` |
| Structured model-grid interpolation | `xesmf.py` |
| Forcing product regridding | usually `xesmf.py`, depending on variable type |

## Development note

CICE velocity reconstruction is scientific logic and should be tested carefully. It is not interchangeable with generic regridding because mask treatment, C-grid staggering, and boundary-condition interpretation affect the resulting fast-ice classification.
