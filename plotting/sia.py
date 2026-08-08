"""Reusable PyGMT helpers for sea-ice-area time series and climatologies."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
import xarray as xr

@dataclass(frozen=True)
class SIAStyle:
    """Line and envelope styling for one SIA series."""
    pen: str
    fill: str

DEFAULT_SIA_STYLES: dict[str, SIAStyle] = {"NSIDC":            SIAStyle("2.4p,black",          "gray40@82"),
                                           "OSI-SAF-450":      SIAStyle("2.4p,black,5_2",      "gray65@82"),
                                           "ORAS":             SIAStyle("2.4p,#61D97B",        "#61D97B@80"),
                                           "ACCESS-OM2-ERA5":  SIAStyle("2.4p,#E69F00",        "#E69F00@80"),
                                           "notens-nogi":      SIAStyle("2.4p,#FF99CC",        "#FF99CC@80"),
                                           "ry93":             SIAStyle("2.4p,#D55E00",        "#D55E00@80"),
                                           "elps-min":         SIAStyle("2.4p,#0072B2",        "#0072B2@80")}

def sia_to_million_km2(da: xr.DataArray) -> xr.DataArray:
    """
    Convert a sea-ice-area DataArray to 10^6 km^2.

    Recognised units include m^2, km^2, 10^3 km^2, and 10^6 km^2.
    Shuga CICE SIA currently uses 10^3 km^2; NSIDC and OSI-SAF use
    10^6 km^2.
    """
    units = (
        str(da.attrs.get("units", ""))
        .strip()
        .lower()
        .replace("²", "^2")
        .replace(" ", "")
    )

    if units in {"m2", "m^2", "m**2"}:
        out = da / 1.0e12
    elif units in {"km2", "km^2", "km**2"}:
        out = da / 1.0e6
    elif any(token in units for token in ("10^3km^2", "10^3km2", "10^3*km^2", "10^3*km2")):
        out = da / 1.0e3
    elif any(token in units for token in ("10^6km^2", "10^6km2", "10^6*km^2", "10^6*km2")):
        out = da
    elif units == "":
        # Do not silently guess from values. The caller must explicitly assign units.
        raise ValueError(
            f"SIA variable {da.name!r} has no units attribute. "
            "Assign one of: m^2, km^2, 10^3 km^2, or 10^6 km^2."
        )
    else:
        raise ValueError(
            f"Unsupported SIA units {da.attrs.get('units')!r} "
            f"for variable {da.name!r}."
        )

    out = out.rename(da.name)
    out.attrs.update(da.attrs)
    out.attrs["units"] = "10^6 km^2"
    return out


def dataarray_to_series(da: xr.DataArray, name: str) -> pd.Series:
    """Convert a one-dimensional time DataArray to a sorted pandas Series."""
    if "time" not in da.dims:
        raise ValueError(f"{name}: expected a 'time' dimension; got {da.dims}")

    extra_dims = [dim for dim in da.dims if dim != "time"]
    if extra_dims:
        raise ValueError(
            f"{name}: expected a one-dimensional SIA series, "
            f"but additional dimensions remain: {extra_dims}"
        )

    values = da.compute().to_series()
    values.index = pd.DatetimeIndex(values.index)
    values = values[~values.index.duplicated(keep="first")].sort_index()
    values.name = name
    return values.astype(float)

def _noleap_doy(index: pd.DatetimeIndex) -> np.ndarray:
    ref = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": np.full(len(index), 2001, dtype=int),
                "month": index.month,
                "day": index.day,
            }
        )
    )

    return ref.dayofyear.to_numpy()


def _circular_rolling(values: pd.Series, window: int) -> pd.Series:
    """Centred cyclic rolling mean across the 31-Dec/1-Jan boundary."""
    window = int(window)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("smooth_days must be odd so the rolling window is centred.")

    pad = window // 2
    arr = values.to_numpy(dtype=float)
    extended = np.concatenate([arr[-pad:], arr, arr[:pad]])
    smoothed = (
        pd.Series(extended)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .iloc[pad : pad + len(arr)]
        .to_numpy()
    )
    return pd.Series(smoothed, index=values.index, name=values.name)


def daily_climatology_envelopes(
    df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    envelope: str = "minmax",
    smooth_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Calculate 365-day means and interannual envelopes for every DataFrame column.

    Parameters
    ----------
    df
        Daily SIA time series indexed by datetime.
    start_date, end_date
        Inclusive climatology period.
    envelope
        ``minmax`` for annual minimum/maximum at each calendar day,
        ``std`` for mean +/- one standard deviation, or
        ``p10-p90`` for the 10th/90th percentiles.
    smooth_days
        Optional odd-width cyclic rolling mean applied after aggregation.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must use a pandas.DatetimeIndex")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    subset = df.loc[start:end].copy()
    subset = subset[~((subset.index.month == 2) & (subset.index.day == 29))]
    if subset.empty:
        raise ValueError(f"No SIA data overlap {start_date} to {end_date}")

    subset["__noleap_doy__"] = _noleap_doy(subset.index)
    full_doy = pd.Index(np.arange(1, 366), name="doy")
    envelope_key = envelope.strip().lower()

    output: dict[str, pd.DataFrame] = {}
    for name in df.columns:
        grouped = subset.groupby("__noleap_doy__")[name]
        mean = grouped.mean().reindex(full_doy)

        if envelope_key == "minmax":
            lower = grouped.min().reindex(full_doy)
            upper = grouped.max().reindex(full_doy)
        elif envelope_key == "std":
            std = grouped.std(ddof=1).reindex(full_doy)
            lower = mean - std
            upper = mean + std
        elif envelope_key in {"p10-p90", "p10p90"}:
            lower = grouped.quantile(0.10).reindex(full_doy)
            upper = grouped.quantile(0.90).reindex(full_doy)
        else:
            raise ValueError("envelope must be one of: minmax, std, p10-p90")

        if int(smooth_days) > 1:
            mean = _circular_rolling(mean, int(smooth_days))
            lower = _circular_rolling(lower, int(smooth_days))
            upper = _circular_rolling(upper, int(smooth_days))

        output[name] = pd.DataFrame(
            {
                "doy": full_doy.to_numpy(dtype=int),
                "mean": mean.to_numpy(dtype=float),
                "lower": lower.to_numpy(dtype=float),
                "upper": upper.to_numpy(dtype=float),
            }
        )

    return output


def plot_sia_full_period_pygmt(
    df: pd.DataFrame,
    out_png: str | Path,
    title: str = "Sea-ice area",
) -> None:
    """Plot full-period SIA time series."""
    import pygmt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    long = (
        df.reset_index(names="time")
        .melt(id_vars="time", var_name="series", value_name="sia")
        .dropna()
    )
    t0 = long.time.min()
    long["days"] = (long.time - t0).dt.total_seconds() / 86400.0
    region = [
        0,
        float(long.days.max()),
        max(0, np.floor(long.sia.min() * 2) / 2),
        np.ceil(long.sia.max() * 2) / 2,
    ]

    fig = pygmt.Figure()
    fig.basemap(
        region=region,
        projection="X20c/9c",
        frame=[
            f"WSen+t{title}",
            "xaf+lDays from start",
            "yaf+lSea Ice Area (10@+6@+ km@+2@+)",
        ],
    )

    for series in df.columns:
        sub = long[long.series == series]
        style = DEFAULT_SIA_STYLES.get(series, SIAStyle("1.4p,black", "gray80@80"))
        fig.plot(x=sub.days, y=sub.sia, pen=style.pen, label=series)

    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
    fig.savefig(out_png, dpi=600)


def plot_sia_daily_climatology_envelope_pygmt(
    df: pd.DataFrame,
    out_file: str | Path,
    *,
    start_date: str,
    end_date: str,
    title: str | None = None,
    envelope: str = "minmax",
    smooth_days: int = 1,
    order: Sequence[str] | None = None,
    styles: Mapping[str, SIAStyle] | None = None,
    y_min: float = 0.0,
    y_max: float = 20.0,
    projection: str = "X20c/14c",
    legend_position: str = "JTL+jTL+o0.2c",
    write_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Recreate the Antarctic daily SIA climatology figure with PyGMT.

    Lines are daily climatological means. Shading is the selected interannual
    envelope. Month labels use a fixed 365-day calendar.
    """
    import pygmt

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    clim = daily_climatology_envelopes(
        df,
        start_date=start_date,
        end_date=end_date,
        envelope=envelope,
        smooth_days=smooth_days,
    )

    series_order = list(order) if order is not None else list(df.columns)
    missing = [name for name in series_order if name not in clim]
    if missing:
        raise KeyError(f"Requested plot series are missing: {missing}")

    style_map = dict(DEFAULT_SIA_STYLES)
    if styles:
        style_map.update(styles)

    month_starts = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    month_mids = 0.5 * (month_starts[:-1] + month_starts[1:] - 1)
    month_labels = list("JFMAMJJASOND")

    frame = ["WSen", "x0", "ya2f1+lSea Ice Area (10@+6@+ km@+2@+)"]
    if title:
        frame[0] += f"+t{title}"

    with pygmt.config(
        FONT_ANNOT_PRIMARY="12p,Helvetica",
        FONT_LABEL="14p,Helvetica",
        FONT_HEADING="12p,Helvetica",
        MAP_FRAME_PEN="1p,black",
        MAP_TICK_PEN_PRIMARY="0.8p,black",
    ):
        fig = pygmt.Figure()
        fig.basemap(
            region=[1, 365, float(y_min), float(y_max)],
            projection=projection,
            frame=frame,
        )

        # Grid behind data: month boundaries and 5-unit horizontal guides.
        for x in month_starts[1:-1]:
            fig.plot(x=[x, x], y=[y_min, y_max], pen="0.35p,gray55")
        for y in np.arange(5.0, y_max + 0.001, 5.0):
            fig.plot(x=[1, 365], y=[y, y], pen="0.35p,gray55")

        for name in series_order:
            table = clim[name]
            style = style_map.get(name, SIAStyle("1.6p,black", "gray80@80"))
            valid = (
                np.isfinite(table["doy"])
                & np.isfinite(table["lower"])
                & np.isfinite(table["upper"])
            )

            x = table.loc[valid, "doy"].to_numpy()
            lower = table.loc[valid, "lower"].to_numpy()
            upper = table.loc[valid, "upper"].to_numpy()

            if len(x):
                fig.plot(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    close=True,
                    fill=style.fill,
                    pen="0p",
                )

        # Lines are drawn after all fills so every mean remains visible.
        for name in series_order:
            table = clim[name]
            style = style_map.get(name, SIAStyle("1.6p,black", "gray80@80"))
            valid = np.isfinite(table["doy"]) & np.isfinite(table["mean"])
            fig.plot(
                x=table.loc[valid, "doy"],
                y=table.loc[valid, "mean"],
                pen=style.pen,
                label=name,
            )

        fig.text(
            x=month_mids,
            y=np.full(12, y_min - 0.55),
            text=month_labels,
            font="12p,Helvetica",
            justify="TC",
            no_clip=True,
        )

        fig.legend(
            position=legend_position,
            box="+gwhite+p0.7p,black",
        )
        fig.savefig(out_file, dpi=600)

    if write_csv:
        rows = []
        for name in series_order:
            table = clim[name].copy()
            table.insert(0, "series", name)
            rows.append(table)
        pd.concat(rows, ignore_index=True).to_csv(
            out_file.with_suffix(".csv"),
            index=False,
        )

    return clim
