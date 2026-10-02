from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
import xarray as xr

@dataclass(frozen=True)
class ERA5PlotSpec:
    cmap  : str
    series: tuple[float, float, float]
    label : str
    scale : float = 1.0
    offset: float = 0.0
    units : str | None = None

ERA5_PLOT_SPECS: dict[str, ERA5PlotSpec] = {# Thermodynamics
                                            "airtmp"   : ERA5PlotSpec("turbo",   (-40.0,  10.0,  2.0)   , "2 m air temperature"         , scale=1.0, offset=-273.15, units="\[^{\circ}\[C"),
                                            "spchmd"   : ERA5PlotSpec("batlow",  (0.0,   0.006, 0.00025), "2 m specific humidity"       , units="kg kg\[^{-1}\["),
                                            "pair"     : ERA5PlotSpec("roma",    (930.0, 1030.0, 5.0)   , "Surface pressure"            , scale=0.01, units="hPa"),
                                            # Radiation
                                            "glbrad"   : ERA5PlotSpec("lajolla", (0.0,  600.0,  25.0)   , "Downward shortwave radiation", units="W m\[^{-2}\["),
                                            "dlwsfc"   : ERA5PlotSpec("bilbao",  (100.0, 400.0, 10.0)   , "Downward longwave radiation" , units="W m\[^{-2}\["),
                                            # Precipitation. Rates are converted from kg m-2 s-1 to mm day-1.
                                            "ttlpcp"   : ERA5PlotSpec("lapaz",   (0.0,  20.0,  1.0)     , "Total precipitation"         , scale=86400.0, units="mm day\[^{-1}\["),
                                            "snowfall" : ERA5PlotSpec("lapaz",   (0.0,  20.0,  1.0)     , "Snowfall"                    , scale=86400.0, units="mm day\[^{-1}\["),
                                            "rainfall" : ERA5PlotSpec("lapaz",   (0.0,  20.0,  1.0)     , "Rainfall"                    , scale=86400.0, units="mm day\[^{-1}\["),
                                            # Winds
                                            "wndewd"   : ERA5PlotSpec("vik",     (-25.0, 25.0, 2.5)     , "10 m eastward wind"          , units="m s\[^{-1}\["),
                                            "wndnwd"   : ERA5PlotSpec("vik",     (-25.0, 25.0, 2.5)     , "10 m northward wind"         , units="m s\[^{-1}\["),
                                            "wnd100ewd": ERA5PlotSpec("vik",     (-35.0, 35.0, 2.5)     , "100 m eastward wind"         , units="m s\[^{-1}\["),
                                            "wnd100nwd": ERA5PlotSpec("vik",     (-35.0, 35.0, 2.5)     , "100 m northward wind"        , units="m s\[^{-1}\["),
                                            "windgust" : ERA5PlotSpec("batlowK", (0.0,  40.0,  2.0)     , "10 m wind gust"              , units="m s\[^{-1}\["),
                                            # Boundary layer
                                            "blh"      : ERA5PlotSpec("batlow",  (0.0, 2000.0, 100.0)   , "Boundary layer height"       , units="m")}


class ERA5Plotter:
    """
    PyGMT plotting helpers for monthly ERA5 -> CICE forcing products.

    Expected input file variables:
        LON/LAT or lon/lat
        time
        one or more ERA5 forcing variables on (time, ny, nx)
    """

    def __init__(self, *,
                 default_region    : Sequence[float] = (-180.0, 180.0, -90.0, -45.0),
                 default_projection: str = "S0/-90/16c",
                 point_style       : str = "s0.018c",
                 frame             : str | Sequence[str] = "af",
                 shorelines        : str = "0.25p,black",
                 land              : str = "gray85",
                 water             : str = "white") -> None:
        self.default_region     = tuple(float(v) for v in default_region)
        self.default_projection = default_projection
        self.point_style        = point_style
        self.frame              = frame
        self.shorelines         = shorelines
        self.land               = land
        self.water              = water

    @staticmethod
    def require_pygmt():
        try:
            import pygmt
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyGMT is required for ERA5 plotting.") from exc
        return pygmt

    @staticmethod
    def detect_lonlat(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        lon_name = next((n for n in ("LON", "lon", "TLON", "longitude") if n in ds), None)
        lat_name = next((n for n in ("LAT", "lat", "TLAT", "latitude") if n in ds), None)
        if lon_name is None or lat_name is None:
            raise KeyError("Could not find LON/LAT or lon/lat coordinates in ERA5 forcing file.")
        return ds[lon_name], ds[lat_name]

    @staticmethod
    def lon_to_180(lon: xr.DataArray) -> xr.DataArray:
        return ((lon + 180.0) % 360.0) - 180.0

    @staticmethod
    def region_mask(lon: xr.DataArray, lat: xr.DataArray, region: Sequence[float]) -> xr.DataArray:
        lon180 = ERA5Plotter.lon_to_180(lon)
        lon_min, lon_max, lat_min, lat_max = [float(v) for v in region]
        if lon_min <= lon_max:
            lon_mask = (lon180 >= lon_min) & (lon180 <= lon_max)
        else:
            lon_mask = (lon180 >= lon_min) | (lon180 <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        return lon_mask & lat_mask

    @staticmethod
    def time_label(da: xr.DataArray) -> str:
        if "time" not in da.coords:
            return "no-time"
        try:
            return pd.Timestamp(da["time"].values).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return str(da["time"].values)

    @staticmethod
    def time_filename(da: xr.DataArray) -> str:
        if "time" not in da.coords:
            return "notime"
        try:
            return pd.Timestamp(da["time"].values).strftime("%Y%m%dT%H%M")
        except Exception:
            return str(da["time"].values).replace(":", "").replace("-", "").replace(" ", "_")

    def prep_xyz(self, da: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray, *,
                 region: Sequence[float],
                 scale : float = 1.0,
                 offset: float = 0.0,
                 stride: int = 1) -> pd.DataFrame:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        work   = da * scale + offset
        lon180 = self.lon_to_180(lon)
        mask   = self.region_mask(lon180, lat, region)
        work   = work.where(mask)
        if stride > 1:
            work   = work.isel({work.dims[-2]: slice(None, None, stride), work.dims[-1]: slice(None, None, stride)})
            lon180 = lon180.isel({lon180.dims[-2]: slice(None, None, stride), lon180.dims[-1]: slice(None, None, stride)})
            lat    = lat.isel({lat.dims[-2]: slice(None, None, stride), lat.dims[-1]: slice(None, None, stride)})
        lon_flat = lon180.values.ravel()
        lat_flat = lat.values.ravel()
        z_flat   = work.values.ravel()
        good     = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(z_flat)
        return pd.DataFrame({"lon": lon_flat[good], "lat": lat_flat[good], "z": z_flat[good]})

    def plot_hour(self, ds: xr.Dataset, variable: str, time_index: int, *,
                  D_out     : str | Path,
                  region    : Sequence[float] | None = None,
                  projection: str | None = None,
                  spec      : ERA5PlotSpec | None = None,
                  stride    : int = 1,
                  show      : bool = False) -> Path:
        if variable not in ds:
            raise KeyError(f"Variable {variable!r} not found in dataset.")
        pygmt      = self.require_pygmt()
        region     = tuple(region or self.default_region)
        projection = projection or self.default_projection
        spec       = spec or ERA5_PLOT_SPECS.get(variable, ERA5PlotSpec("turbo", (0.0, 1.0, 0.1), variable))
        lon, lat   = self.detect_lonlat(ds)
        da         = ds[variable].isel(time=time_index)
        plt_data   = self.prep_xyz(da, lon, lat, region = region, scale = spec.scale, offset = spec.offset, stride = stride)
        if plt_data.empty:
            raise ValueError(f"No finite data for {variable} at time index {time_index} in region {region}.")
        D_out = Path(D_out).expanduser()
        D_out.mkdir(parents=True, exist_ok=True)
        P_png = D_out / variable / f"{self.time_filename(da)}_{variable}.png"
        P_png.parent.mkdir(parents=True, exist_ok=True)
        title = f"{variable}: {spec.label} | {self.time_label(da)}"
        fig   = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="dddF", MAP_FRAME_TYPE="plain")
        pygmt.makecpt(cmap=spec.cmap, series=list(spec.series), continuous=True)
        fig.basemap(region=list(region), projection=projection, frame=["af", f"+t{title}"])
        fig.coast(shorelines=self.shorelines, land=self.land, water=self.water)
        fig.plot(x = plt_data["lon"], y = plt_data["lat"], style = self.point_style, fill = plt_data["z"], cmap = True, pen = "none")
        label = spec.label
        if spec.units:
            label = f"{label} ({spec.units})"
        fig.colorbar(position="JMB+w9c/0.35c+o0c/0.7c", frame=[f"xaf+l{label}"])
        fig.savefig(P_png)
        if show:
            fig.show()
        return P_png

    def plot_hours(self, ds: xr.Dataset, *,
                   variables : Iterable[str] | None = None,
                   hours     : int = 48,
                   D_out     : str | Path,
                   region    : Sequence[float] | None = None,
                   projection: str | None = None,
                   stride    : int = 1,
                   show      : bool = False) -> list[Path]:
        if "time" not in ds.dims:
            raise ValueError("ERA5 forcing dataset must have a time dimension.")
        if variables is None:
            variables = [v for v in ds.data_vars if v not in {"LON", "LAT", "lon", "lat", "TLON", "TLAT"} and "time" in ds[v].dims]
        ntime = min(int(hours), int(ds.sizes["time"]))
        saved: list[Path] = []
        for variable in variables:
            if variable not in ds:
                continue
            if "time" not in ds[variable].dims:
                continue
            for i_t in range(ntime):
                saved.append(self.plot_hour(ds, variable, i_t, D_out = D_out, region = region,  projection = projection, stride = stride, show = show))
        return saved
