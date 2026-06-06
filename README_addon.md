# shuga AF2020/NSIDC observation split and pyresample FIP/FIC comparisons

This bundle implements the requested refactor and comparison workflow:

- `shuga/observations/AF2020.py`: AF2020 raw 15-day rasters, daily FIC interpolation, native FIP, daily FIA helper methods.
- `shuga/observations/NSIDC.py`: NSIDC SIC/SIA/SIE methods split out of the old observation module.
- `shuga/regridder/pyresample.py`: pyresample/EPSG:3031 common-grid helpers based on AFIM `sea_ice_regridder.py`.
- `shuga/scripts/comparisons/build_af2020_fip_fic_common_grid.py`: builds common-grid FIP differences and optional FIC sample stores.
- `shuga/scripts/comparisons/plot_fic_pair_from_common_grid.py`: quick-look side-by-side FIC PNG from a sample store.

See `shuga/scripts/comparisons/README_AF2020_pyresample.md` for usage.
