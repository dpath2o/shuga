# Data conversion

`shuga.core.data_conversion.NC2Zarr` converts CICE `iceh*.nc` history files into grouped Zarr stores.

The main purpose is to make multi-year CICE workflows practical: dynamic fields are chunked into month/day groups, while static grid fields are written once to the universal static-grid store. This can produce very large storage savings compared with repeatedly storing static coordinates and grid metrics in every daily or hourly history file.

## Why Zarr?

Zarr is useful for CICE post-processing because:

- data can be chunked for Dask;
- grouped stores can be opened lazily;
- monthly/daily groups allow partial reads;
- static fields can be separated from dynamic fields;
- repeated notebooks and PBS jobs avoid reopening thousands of NetCDF files.

## Supported input

Daily CICE history:

```text
iceh.YYYY-MM-DD.nc
```

Output:

```text
iceh_daily.zarr/YYYY-MM
```

Hourly CICE history:

```text
iceh_inst.YYYY-MM-DD-SSSSS.nc
iceh_00h.YYYY-MM-DD-SSSSS.nc
iceh_01h.YYYY-MM-DD-SSSSS.nc
```

Output:

```text
iceh_hourly.zarr/YYYY_MM_DD
```

## Static/dynamic split

Dynamic fields are written to simulation-specific grouped stores:

```text
~/AFIM_archive/<SIM_NAME>/zarr/iceh_daily.zarr/
~/AFIM_archive/<SIM_NAME>/zarr/iceh_hourly.zarr/
```

Static fields are written once to:

```text
~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
```

unless `--static-store` is explicitly supplied.

## Python usage

```python
from shuga import RunSpec, ClassificationSpec, ShugaPaths
from shuga.core.data_conversion import NC2Zarr

run = RunSpec(sim_name="LD-blend-base", start_date="2000-01-01", end_date="2003-12-31", hemisphere="SH", iceh_frequency="daily")
classify = ClassificationSpec(ice_type="FI", grid_type="Tc")
paths = ShugaPaths(run=run, classify=classify)

converter = NC2Zarr(paths=paths, chunks={"time": 31}, netcdf_engine="scipy")
result = converter.ensure_iceh_stores(dt0_str=run.start_date, dtN_str=run.end_date, overwrite=False, overwrite_static=False)
print(result)
```

## Command-line usage

Daily:

```bash
python shuga/scripts/conversion/nc2zarr.py \
  --sim-name LD-blend-base \
  --start-date 2000-01-01 \
  --end-date 2003-12-31 \
  --iceh-frequency daily
```

Hourly:

```bash
python shuga/scripts/conversion/nc2zarr.py \
  --sim-name LD-blend-base \
  --start-date 2000-01-01 \
  --end-date 2000-01-31 \
  --iceh-frequency hourly \
  --hourly-root ~/AFIM_archive/LD-blend-base/history/hourly \
  --chunks-time 24
```

## Safety flags

| Flag | Meaning |
|---|---|
| `--overwrite` | Rewrite dynamic grouped stores. |
| `--overwrite-static` | Rewrite the static-grid store. |
| `--delete-original` | Delete source NetCDF files after successful conversion. Use carefully. |
| `--static-store` | Override the universal static-grid path. |

## Relationship to NCO

`shuga` conversion does not replace NCO. NCO remains useful for inspecting and editing NetCDF files before or after conversion: `ncks`, `ncap2`, `ncrename`, and `nccopy` are all useful. See the [NCO documentation](https://nco.sourceforge.net/nco.html).
