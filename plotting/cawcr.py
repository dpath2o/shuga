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


def _direction_bin_widths(direction: xr.DataArray) -> xr.DataArray:
    """Return periodic directional-bin widths in radians."""
    theta_vals = np.asarray(direction.values, dtype=np.float64)
    if theta_vals.ndim != 1 or theta_vals.size < 2:
        raise ValueError("WHACS direction coordinate must be one-dimensional with at least two bins.")
    theta = np.deg2rad(theta_vals) if np.nanmax(np.abs(theta_vals)) > (2.0 * np.pi + 1.0e-6) else theta_vals.copy()
    theta = np.mod(theta, 2.0 * np.pi)
    order = np.argsort(theta)
    theta_sorted = theta[order]
    edges = np.empty(theta_sorted.size + 1, dtype=np.float64)
    mids = 0.5 * (theta_sorted[:-1] + theta_sorted[1:])
    edges[1:-1] = mids
    edges[0] = 0.5 * ((theta_sorted[-1] - 2.0 * np.pi) + theta_sorted[0])
    edges[-1] = edges[0] + 2.0 * np.pi
    widths_sorted = np.diff(edges)
    widths = np.empty_like(widths_sorted)
    widths[order] = widths_sorted
    if not np.isclose(widths.sum(), 2.0 * np.pi, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"Directional-bin widths sum to {widths.sum()}, expected 2*pi.")
    return xr.DataArray(widths.astype(np.float32), dims=("direction",), coords={"direction": direction})


def _direction_integrated_efreq(ds_raw: xr.Dataset, source_var: str = "Efth") -> xr.DataArray:
    efth_name = source_var if source_var in ds_raw else "efth"
    if efth_name not in ds_raw:
        raise KeyError(f"Could not find directional spectrum variable {source_var!r} or 'efth'.")
    dtheta = _direction_bin_widths(ds_raw["direction"])
    efreq = (ds_raw[efth_name].clip(min=0.0) * dtheta).sum("direction", skipna=True)
    efreq = efreq.transpose("time", "station", "frequency")
    efreq.name = "efreq_station"
    efreq.attrs["units"] = "m2 s"
    return efreq


def _dwavefreq(ds_raw: xr.Dataset) -> xr.DataArray:
    if "frequency1" in ds_raw and "frequency2" in ds_raw:
        out = (ds_raw["frequency2"] - ds_raw["frequency1"]).astype(np.float32)
    else:
        freq = ds_raw["frequency"].values.astype(np.float64)
        edges = np.empty(freq.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (freq[:-1] + freq[1:])
        edges[0] = freq[0] - 0.5 * (freq[1] - freq[0])
        edges[-1] = freq[-1] + 0.5 * (freq[-1] - freq[-2])
        out = xr.DataArray(np.diff(edges).astype(np.float32), dims=("frequency",), coords={"frequency": ds_raw["frequency"]})
    out.name = "dwavefreq"
    return out


def _hs_from_spectrum(efreq: xr.DataArray, dwavefreq: xr.DataArray, freq_dim: str) -> xr.DataArray:
    m0 = (efreq.clip(min=0.0) * dwavefreq).sum(freq_dim, skipna=True)
    hs = 4.0 * np.sqrt(xr.where(m0 > 0.0, m0, 0.0))
    hs = hs.where(m0 > 0.0)
    hs.name = "Hs"
    hs.attrs.update({"long_name": "significant wave height", "units": "m"})
    return hs


def _peak_period_from_spectrum(efreq: xr.DataArray, wavefreq: xr.DataArray, freq_dim: str) -> xr.DataArray:
    spec = efreq.clip(min=0.0)
    energy = spec.sum(freq_dim, skipna=True)
    peak = spec.max(freq_dim, skipna=True)
    freq_broadcast = wavefreq.broadcast_like(spec)
    peak_freq = xr.where(spec == peak, freq_broadcast, np.nan).max(freq_dim, skipna=True)
    tp = xr.where((energy > 0.0) & (peak_freq > 0.0), 1.0 / peak_freq, np.nan)
    tp.name = "Tp"
    tp.attrs.update({"long_name": "spectral peak period", "units": "s"})
    return tp


def _station_hs(ds_raw: xr.Dataset, source_var: str = "Efth") -> xr.DataArray:
    efreq = _direction_integrated_efreq(ds_raw, source_var=source_var)
    dw    = _dwavefreq(ds_raw)
    hs    = _hs_from_spectrum(efreq, dw, "frequency")
    hs    = hs.assign_coords(station_lon=_normalise_station_coord(ds_raw["longitude"] if "longitude" in ds_raw else ds_raw["station_lon"]),
                             station_lat=_normalise_station_coord(ds_raw["latitude"] if "latitude" in ds_raw else ds_raw["station_lat"]))
    hs.name = "hs_station"
    return hs


def _station_dataframe(da_station: xr.DataArray, time_index: int = 0) -> pd.DataFrame:
    da0 = da_station.isel(time=time_index)
    return pd.DataFrame({"lon": da0["station_lon"].values.astype(float),
                         "lat": da0["station_lat"].values.astype(float),
                         "value": da0.values.astype(float)}).dropna(subset=["lon", "lat", "value"])


def _south_region(north: float = -35.0) -> list[float]:
    return [-180, 180, -90, float(north)]


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


def _daily_station_dataframe(da_station: xr.DataArray, region: list[float]) -> pd.DataFrame:
    lon = np.asarray(da_station["station_lon"].values, dtype=np.float64)
    lat = np.asarray(da_station["station_lat"].values, dtype=np.float64)
    val = np.asarray(da_station.values, dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0
    west, east, south, north = region
    keep = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    keep &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    return pd.DataFrame({"lon": lon[keep], "lat": lat[keep], "value": val[keep]})


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


def _shared_positive_series(*values: np.ndarray, explicit_max: float | None,
                            fallback_max: float, min_step: float, pct: float = 99.5) -> list[float]:
    chunks: list[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr) & (arr >= 0.0)]
        if arr.size:
            chunks.append(arr)
    if explicit_max is not None:
        vmax = float(explicit_max)
    elif chunks:
        vmax = float(np.nanpercentile(np.concatenate(chunks), pct))
        vmax = max(vmax, fallback_max)
    else:
        vmax = fallback_max
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = fallback_max
    step = max(vmax / 20.0, min_step)
    return [0.0, vmax, step]


def _draw_empty_panel(fig: pygmt.Figure, *, region : list[float], projection : str, title : str,
                      message : str = "No finite data") -> None:
    fig.basemap(region=region, projection=projection, frame=["afg", f"+t{title}"])
    fig.coast(shorelines="0.25p,black", land="gray90", water="white")
    lon_c = 0.5 * (region[0] + region[1])
    lat_c = 0.5 * (region[2] + region[3])
    fig.text(x=lon_c, y=lat_c, text=message, font="12p,Helvetica,black", justify="CM")


#--------------------------------------------------------------------------
# daily WHACS diagnostics
def daily_station_wave_diagnostics(ds_raw: xr.Dataset,
                                   day: str | pd.Timestamp,
                                   source_var: str = "efth") -> xr.Dataset:
    """
    Daily mean hourly Hs and spectral peak period on native WHACS stations.

    Hs and Tp are diagnosed independently for each hourly spectrum and then
    averaged over the UTC day. This preserves the conventional interpretation
    of a daily mean significant wave height / peak period diagnostic.
    """
    day0 = pd.Timestamp(day).normalize()
    day1 = day0 + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sub = ds_raw.sel(time=slice(day0, day1))
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No native WHACS records available for {day0:%Y-%m-%d}.")

    efreq = _direction_integrated_efreq(sub, source_var=source_var)
    hs_hourly = _hs_from_spectrum(efreq, _dwavefreq(sub), "frequency")
    tp_hourly = _peak_period_from_spectrum(efreq, sub["frequency"], "frequency")

    lon = _normalise_station_coord(sub["longitude"] if "longitude" in sub else sub["station_lon"])
    lat = _normalise_station_coord(sub["latitude"] if "latitude" in sub else sub["station_lat"])
    hs = hs_hourly.mean("time", skipna=True).assign_coords(station_lon=lon, station_lat=lat)
    tp = tp_hourly.mean("time", skipna=True).assign_coords(station_lon=lon, station_lat=lat)
    hs.name = "Hs"
    tp.name = "Tp"
    return xr.Dataset({"Hs": hs, "Tp": tp}, attrs={"date": day0.strftime("%Y-%m-%d"), "source": "native WHACS stations"})


def daily_regridded_wave_diagnostics(ds_wave: xr.Dataset,
                                     day: str | pd.Timestamp) -> xr.Dataset:
    """Daily mean hourly Hs and spectral peak period on the regridded CICE T grid."""
    day0 = pd.Timestamp(day).normalize()
    day1 = day0 + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sub = ds_wave.sel(time=slice(day0, day1))
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No regridded WHACS records available for {day0:%Y-%m-%d}.")
    if "efreq" not in sub:
        raise KeyError("Regridded WHACS dataset does not contain efreq.")

    efreq = sub["efreq"]
    freq_dim = "nfreq" if "nfreq" in efreq.dims else "frequency"
    if "wavefreq" in sub:
        wavefreq = sub["wavefreq"]
    elif freq_dim in sub.coords:
        wavefreq = sub[freq_dim]
    else:
        raise KeyError("Regridded WHACS dataset has no wavefreq/frequency coordinate.")
    if "dwavefreq" in sub:
        dwavefreq = sub["dwavefreq"]
    else:
        freq = np.asarray(wavefreq.values, dtype=np.float64)
        edges = np.empty(freq.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (freq[:-1] + freq[1:])
        edges[0] = freq[0] - 0.5 * (freq[1] - freq[0])
        edges[-1] = freq[-1] + 0.5 * (freq[-1] - freq[-2])
        dwavefreq = xr.DataArray(np.diff(edges).astype(np.float32), dims=(freq_dim,), coords={freq_dim: efreq[freq_dim]})

    hs_hourly = _hs_from_spectrum(efreq, dwavefreq, freq_dim)
    tp_hourly = _peak_period_from_spectrum(efreq, wavefreq, freq_dim)
    hs = hs_hourly.mean("time", skipna=True)
    tp = tp_hourly.mean("time", skipna=True)
    hs.name = "Hs"
    tp.name = "Tp"

    coords = {}
    for name in ("TLON", "TLAT"):
        if name in sub:
            coords[name] = sub[name]
    return xr.Dataset({"Hs": hs, "Tp": tp}, coords=coords,
                      attrs={"date": day0.strftime("%Y-%m-%d"), "source": "WHACS regridded to CICE T grid"})


def plot_whacs_daily_comparison(ds_raw: xr.Dataset,
                                ds_wave: xr.Dataset,
                                day: str | pd.Timestamp,
                                output: str | Path | None = None,
                                projection: str = "S0/-90/10.5c",
                                region: Optional[list[float]] = None,
                                grid_stride: int = 3,
                                station_point_size_cm: float = 0.075,
                                grid_point_size_cm: float = 0.055,
                                hs_max: float | None = None,
                                tp_max: float | None = None,
                                paths: ShugaPaths | None = None,
                                source_var: str = "efth",
                                dpi: int = 300) -> pygmt.Figure:
    """
    Plot a 2x2 daily WHACS regridding diagnostic with PyGMT.

    Upper row
        Native WHACS station data: Hs and Tp.
    Lower row
        WHACS spectrum after spatial/frequency regridding to the CICE T grid:
        Hs and Tp.

    Colour scales are shared vertically so the source and regridded maps are
    directly comparable. Hs uses ``cmocean/amp`` and Tp uses
    ``cmocean/phase`` as requested.
    """
    day0 = pd.Timestamp(day).normalize()
    region = region or _south_region(-35.0)

    native = daily_station_wave_diagnostics(ds_raw, day0, source_var=source_var)
    regrid = daily_regridded_wave_diagnostics(ds_wave, day0)

    native_hs = _daily_station_dataframe(native["Hs"], region)
    native_tp = _daily_station_dataframe(native["Tp"], region)

    lon2d, lat2d, ocean_mask = _resolve_plot_grid(ds_wave, paths=paths)
    regrid_hs = _grid_dataframe(regrid["Hs"], lon2d, lat2d, mask2d=ocean_mask,
                                stride=grid_stride, region=region)
    regrid_tp = _grid_dataframe(regrid["Tp"], lon2d, lat2d, mask2d=ocean_mask,
                                stride=grid_stride, region=region)

    hs_series = _shared_positive_series(native_hs["value"].to_numpy(), regrid_hs["value"].to_numpy(),
                                         explicit_max=hs_max, fallback_max=6.0, min_step=0.25, pct=99.5)
    tp_series = _shared_positive_series(native_tp["value"].to_numpy(), regrid_tp["value"].to_numpy(),
                                         explicit_max=tp_max, fallback_max=18.0, min_step=0.5, pct=99.5)

    panel_specs = [
        (native_hs, "Native WHACS stations — Hs", "cmocean/amp", hs_series, station_point_size_cm, ['xaf+l"Hs"', 'y+l"m"']),
        (native_tp, "Native WHACS stations — peak period", "cmocean/phase", tp_series, station_point_size_cm, ['xaf+l"Tp"', 'y+l"s"']),
        (regrid_hs, "Regridded CICE T-grid — Hs", "cmocean/amp", hs_series, grid_point_size_cm, ['xaf+l"Hs"', 'y+l"m"']),
        (regrid_tp, "Regridded CICE T-grid — peak period", "cmocean/phase", tp_series, grid_point_size_cm, ['xaf+l"Tp"', 'y+l"s"']),
    ]

    fig = pygmt.Figure()
    with pygmt.config(MAP_FRAME_TYPE="plain", FONT_TITLE="11p,Helvetica-Bold", FONT_LABEL="9p", FONT_ANNOT_PRIMARY="8p"):
        with fig.subplot(nrows=2, ncols=2, figsize=("23.5c", "23.5c"),
                         margins=["0.25c", "0.35c"],
                         title=f"WHACS wave-forcing regridding QC — {day0:%Y-%m-%d} UTC daily mean"):
            for panel, (df, title, cmap, series, point_size, cbar_frame) in enumerate(panel_specs):
                with fig.set_panel(panel=panel):
                    if df.empty:
                        _draw_empty_panel(fig, region=region, projection=projection, title=title)
                        continue
                    pygmt.makecpt(cmap=cmap, series=series, continuous=True)
                    fig.basemap(region=region, projection=projection, frame=["afg", f"+t{title}"])
                    fig.coast(shorelines="0.25p,black", land="gray90", water="white")
                    fig.plot(x=df["lon"].to_numpy(), y=df["lat"].to_numpy(),
                             style=f"c{point_size}c", fill=df["value"].to_numpy(),
                             cmap=True, pen=None)
                    fig.colorbar(frame=cbar_frame, position="JBC+w6.8c/0.28c+o0c/0.35c+h")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=int(dpi))
        print(f"saved: {output}")
    return fig


#--------------------------------------------------------------------------
# legacy / single-time plotting functions
def plot_original_hs_map(ds_raw: xr.Dataset,
                         time_index : int                   = 0,
                         output     : str | Path | None     = None,
                         title      : str | None            = None,
                         projection : str                   = "S0/-90/14c",
                         region     : Optional[list[float]] = None,
                         source_var : str                   = "Efth") -> pygmt.Figure:
    hs     = _station_hs(ds_raw, source_var=source_var)
    df     = _station_dataframe(hs, time_index=time_index)
    region = region or _south_region(-45.0)
    title  = title or f"CAWCR station Hs | {pd.to_datetime(hs.time.values[time_index])}"
    fig    = pygmt.Figure()
    pygmt.makecpt(cmap="cmocean/amp", series=[0, max(0.5, float(df.value.quantile(0.99))), 0.1], continuous=True)
    fig.basemap(region=region, projection=projection, frame=["afg", f"+t{title}"])
    fig.coast(shorelines="0.5p,black", land="gray90", water="white")
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
    region = region or _south_region(-45.0)
    title = title or f"CAWCR E(f) band | f={fval:.4f} s^-1 | {pd.to_datetime(ds_raw.time.values[time_index])}"

    vmax = max(float(df.value.quantile(0.99)), 1.0e-4)
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="viridis", series=[0, vmax, vmax / 20.0], continuous=True)
    fig.basemap(region=region, projection=projection, frame=["afg", f"+t{title}"])
    fig.coast(shorelines="0.5p,black", land="gray90", water="white")
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
    """Legacy two-panel helper for pre-WHACS files containing hs_masked and sic."""
    region = region or _south_region(-45.0)
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
                fig.basemap(region=region, projection=projection, frame=["afg", f"+t{panel_title}"])
                fig.coast(shorelines="0.25p,black", land="gray90", water="white")
                fig.plot(x=df["lon"].to_numpy(), y=df["lat"].to_numpy(), style=f"c{point_size_cm}c", fill=df["value"].to_numpy(), cmap=True, pen=None)
                fig.colorbar(frame=cbar_frame)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output))
        print(f"saved: {output}")
    return fig
