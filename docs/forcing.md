# Forcing

`shuga` provides support for preparing CICE-compatible external forcing products. Current package capability includes ERA5 atmospheric forcing and WHACS wave spectra. ORAS and related ocean/reanalysis forcing support remains a developing area.

This page describes package capability at a high level. PBS workflow details for building or plotting forcing products belong in [`gadi-workflows.md`](gadi-workflows.md).

## ERA5 atmospheric forcing

ERA5 support is focused on preparing CICE-compatible forcing products, including monthly ERA5-to-CICE regridding workflows.

Typical fields of interest include:

| Category | Examples |
|---|---|
| radiation | shortwave and longwave surface radiation |
| winds | 10 m winds, optional 100 m winds, gusts |
| thermodynamic state | air temperature, specific humidity, surface pressure |
| precipitation | total precipitation, rainfall, snowfall |
| boundary layer | boundary-layer height where available |

## WHACS wave forcing

`shuga.waves` supports conversion of the BoM–CSIRO WHACS WWIII-v6.07 hourly directional-spectrum archive into monthly CICE6 wave-forcing files.

The production workflow currently combines all five WHACS full-spectral point archives:

```text
GRID + GLOB + BUOYS + NIWA + SCHISM
```

The fixed source points are de-duplicated, directional spectra are integrated to `E(f)`, the native 28 frequency bins are conservatively remapped to the 25-bin CICE/Icepack wave grid, and the resulting spectra are interpolated to the ACCESS-OM3 0.25° CICE T grid.

No NSIDC or model sea-ice mask is applied to the resulting external wave-forcing field. Wave propagation, attenuation, fracture and FSD evolution remain part of the subsequent CICE/Icepack model state.

See [`WHACS_wave_forcing.md`](WHACS_wave_forcing.md) for the full scientific and technical specification, including:

- the five WHACS source geometries;
- fixed-station versus drifting-location interpretation;
- directional integration;
- conservative 28→25 frequency remapping;
- `m0` / significant-wave-height QC;
- IDW station-to-CICE interpolation;
- duplicate handling and static weights;
- output metadata and restart/completion semantics;
- daily PyGMT QC.

## Design intent

Forcing helpers should:

- keep raw forcing-product naming separate from CICE-ready naming;
- support chunked/monthly workflows;
- avoid loading multi-decade products into memory;
- provide reproducible CICE-grid outputs;
- keep scientific transformations explicit;
- retain enough provenance in output metadata to identify source geometry and transformation choices;
- distinguish external forcing preparation from physics that should remain prognostic inside CICE/Icepack.

## Future support: ORAS

ORAS support is planned. Likely use cases include SST/SSS comparison or forcing products, mixed-layer-depth diagnostics, ocean-current fields relevant to ice dynamics, and ocean restoring fields.

## Scripts

Atmospheric forcing scripts live primarily in:

```text
shuga/scripts/forcing/
```

WHACS wave-forcing production scripts live in:

```text
shuga/scripts/waves/
```

and associated QC plotting scripts live in:

```text
shuga/scripts/plotting/
```

Scripts should remain thin wrappers around package code.

## External documentation

- [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [CICE Consortium](https://github.com/CICE-Consortium/CICE)
- [Icepack](https://github.com/CICE-Consortium/Icepack)
- [xarray documentation](https://docs.xarray.dev/)
- [Dask documentation](https://docs.dask.org/)
- [xESMF documentation](https://xesmf.readthedocs.io/)
- [NCO documentation](https://nco.sourceforge.net/nco.html)
