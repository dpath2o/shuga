# OSI-SAF-450 SIA workflow

This patch adds a new observation module, `shuga.observations.OSI_SAF450`, for downloading and processing Antarctic sea-ice area (SIA) from the Copernicus Marine / EUMETSAT OSI-SAF product `SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009`.

The Copernicus product is the global OSI-SAF passive-microwave sea-ice concentration record. The product page lists it as daily, Level-4, NetCDF-4, 25 km resolution, and includes the long SMMR/SSMI/SSMIS OSI-450-a1/OSI-430-a stream and the AMSR OSI-458/OSI-438 stream. For the Antarctic SIA comparison here, the default dataset is the southern OSI-450-a1 stream:

```text
OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS
```

For post-2020 extension, use one of the listed southern interim/AMSR dataset IDs when appropriate, for example:

```text
OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-SH-LA-OBS
osisaf_obs-si_glo_phy_sic-south_my_amsr_icdr_P1D-m
```

## Authentication

Install/configure the Copernicus Marine Toolbox in the analysis environment. The downloader calls the `copernicusmarine subset` CLI. It can use either an existing Copernicus Marine login configuration or these environment variables:

```bash
export COPERNICUSMARINE_SERVICE_USERNAME='your_username'
export COPERNICUSMARINE_SERVICE_PASSWORD='your_password'
```

## Interactive usage

From the repository root:

```bash
python shuga/scripts/downloading/download_OSI_SAF450.py \
  --raw-dir /g/data/gv90/da1339/observations/OSI-SAF-450/raw \
  --processed-dir /g/data/gv90/da1339/observations/OSI-SAF-450/processed \
  --start-date 2000-01-01 \
  --end-date 2005-12-31 \
  --dataset-id OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS \
  --hemisphere SH \
  --download \
  --process
```

The processed output is:

```text
/g/data/gv90/da1339/observations/OSI-SAF-450/processed/OSI-SAF-450_SH_SIA.zarr
```

Variables:

- `sia`: daily sea-ice area in `10^6 km^2`.
- `sia_m2`: daily sea-ice area in `m2`.

The reduction uses a 15% concentration threshold and sums fractional ice-covered cell area, not extent. The code attempts to use a cell-area variable if present, then grid x/y spacing, and finally the nominal 25 km × 25 km OSI-SAF cell area.

## Gadi copyq usage

```bash
cd /g/data/gv90/da1339/src/mawsons-chest
shuga/scripts/downloading/download_OSI_SAF450_pbs_wrapper.sh \
  -b 2000-01-01 \
  -e 2005-12-31 \
  -r /g/data/gv90/da1339/observations/OSI-SAF-450/raw \
  -p /g/data/gv90/da1339/observations/OSI-SAF-450/processed
```

The PBS script uses Gadi `copyq`, one CPU, and modest memory because the expensive part is data movement rather than numerical reduction.
