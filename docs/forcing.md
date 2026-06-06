# Forcing

`shuga.forcing` currently provides support for ERA5 forcing workflows. The long-term intention is to add comparable support for ORAS and other ocean/reanalysis forcing products where useful.

This page describes package capability. PBS workflow details for building or plotting forcing products belong in [`gadi-workflows.md`](gadi-workflows.md).

## Current support: ERA5

ERA5 support is focused on preparing CICE-compatible forcing products, including monthly ERA5-to-CICE regridding workflows.

Typical fields of interest include:

| Category | Examples |
|---|---|
| radiation | shortwave and longwave surface radiation |
| winds | 10 m winds, optional 100 m winds, gusts |
| thermodynamic state | air temperature, specific humidity, surface pressure |
| precipitation | total precipitation, rainfall, snowfall |
| boundary layer | boundary-layer height where available |

## Design intent

Forcing helpers should:

- keep raw forcing-product naming separate from CICE-ready naming;
- support chunked/monthly workflows;
- avoid loading multi-decade products into memory;
- provide reproducible CICE-grid outputs;
- keep scientific transformations explicit.

## Future support: ORAS

ORAS support is planned. Likely use cases include SST/SSS comparison or forcing products, mixed-layer-depth diagnostics, ocean-current fields relevant to ice dynamics, and ocean restoring fields.

## Scripts

Current forcing scripts live in:

```text
shuga/scripts/forcing/
```

These include ERA5 CICE-ready monthly products and quicklook plotting. The scripts should remain thin wrappers around package code.

## External documentation

- [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [xarray documentation](https://docs.xarray.dev/)
- [Dask documentation](https://docs.dask.org/)
- [xESMF documentation](https://xesmf.readthedocs.io/)
- [NCO documentation](https://nco.sourceforge.net/nco.html)
