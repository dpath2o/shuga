from __future__       import annotations
import pygmt
from pathlib          import Path
from typing           import Optional
import numpy          as np
import pandas         as pd
import xarray         as xr
from shuga.core.paths import ShugaPaths
from shuga.grid.cice  import CICEGridwork

#------------------------------------------------------------------
# helper functions
def _normalise_station_coord(coord: xr.DataArray) -> xr.DataArray:
    if coord.dims == ("station",):
        return coord
    if "time" in coord.dims:
        coord = coord.isel(time=0, drop=True)
    squeeze_dims = [d for d in coord.dims if d != "station" and coord.sizes[d] == 1]
    if squeeze_dims:
        coord = coord.squeeze(squeeze_dims, drop=True)
    if coord.dims != ("station",):
        raise ValueError(f"Expected station coordinate to reduce to ('station',); got {coord.dims}")
    return coord

def _direction_integrated_efreq(ds_raw: xr.Dataset, source_var: str = "Efth") -> xr.DataArray:
    efth_name = source_var if source_var in ds_raw else "efth"
    theta = ds_raw["direction"].astype(float)
    theta_vals = theta.values
    theta_rad = np.deg2rad(theta_vals) if np.nanmax(np.abs(theta_vals)) > (2 * np.pi + 1e-6) else theta_vals
    dtheta = xr.DataArray(np.gradient(theta_rad), dims=("direction",), coords={"direction": ds_raw["direction"]})
    efreq = (ds_raw[efth_name] * dtheta).sum("direction", skipna=True)
    efreq = efreq.transpose("time", "station", "frequency")
    return efreq

def _dwavefreq(ds_raw: xr.Dataset) -> xr.DataArray:
    if "frequency1" in ds_raw and "frequency2" in ds_raw:
        out = (ds_raw["frequency2"] - ds_raw["frequency1"]).astype(np.float32)
    else:
        freq = ds_raw["frequency"].values.astype(np.float64)
        out = xr.DataArray(np.gradient(freq).astype(np.float32), dims=("frequency",), coords={"frequency": ds_raw["frequency"]})
    return out

def _station_hs(ds_raw: xr.Dataset, source_var: str = "Efth") -> xr.DataArray:
    efreq = _direction_integrated_efreq(ds_raw, source_var=source_var)
    dw    = _dwavefreq(ds_raw)
    m0    = (efreq * dw).sum("frequency", skipna=True)
    hs    = 4.0 * np.sqrt(xr.where(m0 > 0, m0, 0.0))
    hs    = hs.assign_coords(station_lon=_normalise_station_coord(ds_raw["longitude"] if "longitude" in ds_raw else ds_raw["station_lon"]),
                             station_lat=_normalise_station_coord(ds_raw["latitude"] if "latitude" in ds_raw else ds_raw["station_lat"]))
    hs.name = "hs_station"
    return hs

def _station_dataframe(da_station: xr.DataArray, time_index: int = 0) -> pd.DataFrame:
    da0 = da_station.isel(time=time_index)
    return pd.DataFrame({"lon": da0["station_lon"].values.astype(float),
                         "lat": da0["station_lat"].values.astype(float),
                         "value": da0.values.astype(float)}).dropna(subset=["lon", "lat", "value"])

def _south_region() -> list[float]:
    return [-180, 180, -90, -45]

def _resolve_plot_grid(ds_wave: xr.Dataset,
                       paths  : ShugaPaths | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    lon  = None
    lat  = None
    mask = None
    if "TLON" in ds_wave and "TLAT" in ds_wave:
        lon = ds_wave["TLON"].values.astype(np.float64)
        lat = ds_wave["TLAT"].values.astype(np.float64)
    else:
        if paths is None:
            raise ValueError("Could not find TLON/TLAT in ds_wave and no ShugaPaths was provided.")
        gridwork = CICEGridwork(paths=paths)
        bundle = gridwork.load_cice_grid(build_faces=False)
        lon = bundle.tgrid["TLON"].values.astype(np.float64)
        lat = bundle.tgrid["TLAT"].values.astype(np.float64)
    if "target_active_mask" in ds_wave:
        mask = ds_wave["target_active_mask"].values.astype(bool)
    elif "ocean_mask" in ds_wave:
        mask = ds_wave["ocean_mask"].values.astype(bool)
    elif paths is not None:
        gridwork = CICEGridwork(paths=paths)
        bundle = gridwork.load_cice_grid(build_faces=False)
        if bundle.mask is not None:
            mask = bundle.mask.values.astype(bool)
    return lon, lat, mask

def _grid_dataframe(da: xr.DataArray, lon2d: np.ndarray, lat2d: np.ndarray, *,
                    mask2d: np.ndarray | None = None,
                    stride: int = 3,
                    region: list[float] | None = None) -> pd.DataFrame:
    if tuple(da.dims) != ("nj", "ni"):
        raise ValueError(f"Expected 2D field with dims ('nj', 'ni'); got {da.dims}")
    vals = da.values.astype(np.float64)
    lon  = lon2d[::stride, ::stride].copy()
    lat  = lat2d[::stride, ::stride].copy()
    val  = vals[::stride, ::stride].copy()
    keep = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    if mask2d is not None:
        keep &= mask2d[::stride, ::stride].astype(bool)
    if region is not None:
        west, east, south, north = region
        if east <= 180.0 and np.nanmax(lon) > 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0
        keep &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    if not np.any(keep):
        return pd.DataFrame(columns=["lon", "lat", "value"])
    return pd.DataFrame({"lon": lon[keep].ravel(),
                         "lat": lat[keep].ravel(),
                         "value": val[keep].ravel()})

def _safe_positive_series(values: np.ndarray, *, fallback_max: float, min_step: float,
                          pct: float = 99.0) -> list[float] | None:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmax = float(np.nanpercentile(vals, pct))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(fallback_max)
    vmax = max(vmax, float(fallback_max))
    step = max(vmax / 20.0, float(min_step))
    return [0.0, vmax, step]

def _draw_empty_panel(fig: pygmt.Figure, *, region : list[float], projection : str, title : str,
                      message : str = "No finite data") -> None:
    fig.basemap(region=region, projection=projection, frame=["af", f"+t{title}"])
    fig.coast(shorelines="0.25p,black", land="lightgray", water="white")
    lon_c = 0.5 * (region[0] + region[1])
    lat_c = 0.5 * (region[2] + region[3])
    fig.text(x=lon_c, y=lat_c, text=message, font="12p,Helvetica,black", justify="CM")

#--------------------------------------------------------------------------
# main functions
def plot_original_hs_map(ds_raw: xr.Dataset,
                         time_index : int                   = 0,
                         output     : str | Path | None     = None,
                         title      : str | None            = None,
                         projection : str                   = "S0/-90/14c",
                         region     : Optional[list[float]] = None,
                         source_var : str                   = "Efth") -> pygmt.Figure:
    hs     = _station_hs(ds_raw, source_var=source_var)
    df     = _station_dataframe(hs, time_index=time_index)
    region = region or _south_region()
    title  = title or f"CAWCR station Hs | {pd.to_datetime(hs.time.values[time_index])}"
    fig    = pygmt.Figure()
    pygmt.makecpt(cmap="viridis", series=[0, max(0.5, float(df.value.quantile(0.99))), 0.1], continuous=True)
    fig.basemap(region=region, projection=projection, frame=["af", f"+t{title}"])
    fig.coast(shorelines="0.5p,black", land="lightgray", water="white")
    fig.plot(x=df.lon, y=df.lat, style="c0.16c", fill=df.value, cmap=True, pen="0.05p,black")
    fig.colorbar(frame=['xaf+l"Hs"', 'y+l"m"'])
    if output is not None:
        fig.savefig(str(output))
    return fig

def plot_original_band_map(ds_raw: xr.Dataset,
                           time_index : int                   = 0,
                           freq_index : int                   = 0,
                           output     : str | Path | None     = None,
                           title      : str | None            = None,
                           projection : str                   = "S0/-90/14c",
                           region     : Optional[list[float]] = None,
                           source_var : str                   = "Efth") -> pygmt.Figure:
    efreq = _direction_integrated_efreq(ds_raw, source_var=source_var)
    efreq = efreq.assign_coords(
        station_lon=_normalise_station_coord(ds_raw["longitude"] if "longitude" in ds_raw else ds_raw["station_lon"]),
        station_lat=_normalise_station_coord(ds_raw["latitude"] if "latitude" in ds_raw else ds_raw["station_lat"]),
    )
    band = efreq.isel(frequency=freq_index)
    df = _station_dataframe(band, time_index=time_index)
    fval = float(ds_raw["frequency"].values[freq_index])
    region = region or _south_region()
    title = title or f"CAWCR E(f) band | f={fval:.4f} s^-1 | {pd.to_datetime(ds_raw.time.values[time_index])}"

    vmax = max(float(df.value.quantile(0.99)), 1.0e-4)
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="viridis", series=[0, vmax, vmax / 20.0], continuous=True)
    fig.basemap(region=region, projection=projection, frame=["af", f"+t{title}"])
    fig.coast(shorelines="0.5p,black", land="lightgray", water="white")
    fig.plot(x=df.lon, y=df.lat, style="c0.16c", fill=df.value, cmap=True, pen="0.05p,black")
    fig.colorbar(frame=['xaf+l"E(f)"'])
    if output is not None:
        fig.savefig(str(output))
    return fig

def plot_regridded_hs_sic_panel(ds_wave         : xr.Dataset,
                                time_index      : int                   = 0,
                                output          : str | Path | None     = None,
                                projection      : str                   = "S0/-90/10c",
                                region          : Optional[list[float]] = None,
                                stride          : int                   = 3,
                                point_size_cm   : float                 = 0.045,
                                paths           : ShugaPaths | None     = None,
                                use_active_mask : bool                  = True) -> pygmt.Figure:
    """
    Two-panel plot from regridded CAWCR output:
      1. hs_masked
      2. sic
    Safe against all-NaN panels and empty sampled dataframes.
    """
    region = region or _south_region()
    tstamp = pd.to_datetime(ds_wave["time"].values[time_index])
    lon2d, lat2d, mask2d = _resolve_plot_grid(ds_wave, paths=paths)
    if not use_active_mask:
        mask2d = None
    hs_m = ds_wave["hs_masked"].isel(time=time_index)
    sic  = ds_wave["sic"].isel(time=time_index)
    fig  = pygmt.Figure()
    panel_specs = [(hs_m, f"{tstamp:%Y-%m-%d %H:%M}: Hs (masked)", "cmocean/amp", 0.5, 0.01, ['xaf+l"Hs"', 'y+l"m"']),
                   (sic,  f"NSIDC SIC", "ice",     1.0, 0.05, ['xaf+l"SIC"'])]
    with fig.subplot(nrows=1, ncols=2, figsize=("20c", "10c"), margins=["0.2c", "0.2c"]):
        for panel, (da, panel_title, cmap, fallback_max, min_step, cbar_frame) in enumerate(panel_specs):
            df = _grid_dataframe(da, lon2d, lat2d, mask2d=mask2d, stride=stride, region=region)
            with fig.set_panel(panel=panel):
                if df.empty:
                    _draw_empty_panel(fig, region=region, projection=projection, title=panel_title, message="No finite sampled data")
                    continue
                if da.name == "sic":
                    series = [0.0, 1.0, 0.05]
                else:
                    series = _safe_positive_series(df["value"].to_numpy(), fallback_max=fallback_max, min_step=min_step, pct=99.0)
                if series is None:
                    _draw_empty_panel(fig, region=region, projection=projection, title=panel_title, message="No finite sampled data")
                    continue
                pygmt.makecpt(cmap=cmap, series=series, continuous=True)
                fig.basemap(region=region, projection=projection, frame=["af", f"+t{panel_title}"])
                fig.coast(shorelines="0.25p,black", land="lightgray", water="white")
                fig.plot(x=df["lon"].to_numpy(), y=df["lat"].to_numpy(), style=f"c{point_size_cm}c", fill=df["value"].to_numpy(), cmap=True, pen=None)
                fig.colorbar(frame=cbar_frame)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output))
        print(f"saved: {output}")
    return fig
