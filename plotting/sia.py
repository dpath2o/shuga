"""SIA plotting helpers for CICE/NSIDC/OSI-SAF comparisons."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def plot_sia_full_period_pygmt(df: pd.DataFrame, out_png: str | Path, title: str = "Sea-ice area") -> None:
    """Plot a full-period SIA time series with NSIDC/OSI-SAF black-line convention."""
    import pygmt
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    long = df.reset_index().melt(id_vars="time", var_name="series", value_name="sia").dropna()
    t0 = long.time.min()
    long["days"] = (long.time - t0).dt.total_seconds() / 86400.0
    region = [0, float(long.days.max()), max(0, np.floor(long.sia.min() * 2) / 2), np.ceil(long.sia.max() * 2) / 2]
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="X20c/9c", frame=[f"WSen+t{title}", "xaf+lDays from start", "yaf+lSIA (10@+6@+ km@+2@+)"])
    for series in df.columns:
        sub = long[long.series == series]
        pen = "1.4p,black" if series == "NSIDC" else "1.4p,black,-" if series == "OSI-SAF-450" else "0.8p"
        fig.plot(x=sub.days, y=sub.sia, pen=pen, label=series)
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
    fig.savefig(out_png)

def plot_sia_daily_climatology_envelope_pygmt(df: pd.DataFrame, out_png: str | Path, title: str = "Sea-ice area seasonal envelope") -> None:
    """Plot daily mean SIA with min/max envelope by day of year."""
    import pygmt
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    tmp = tmp[~((tmp.index.month == 2) & (tmp.index.day == 29))]
    tmp["doy"] = tmp.index.dayofyear
    clim = {}
    for col in df.columns:
        g = tmp.groupby("doy")[col]
        clim[col] = pd.DataFrame({"doy": g.mean().index, "mean": g.mean().values, "min": g.min().values, "max": g.max().values})
    allvals = pd.concat([v[["min", "max"]] for v in clim.values()])
    region = [1, 365, max(0, np.floor(allvals.min().min() * 2) / 2), np.ceil(allvals.max().max() * 2) / 2]
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="X20c/9c", frame=[f"WSen+t{title}", "xa30f15+lDay of year", "yaf+lSIA (10@+6@+ km@+2@+)"])
    for series, c in clim.items():
        poly_x = np.r_[c.doy.values, c.doy.values[::-1]]
        poly_y = np.r_[c["max"].values, c["min"].values[::-1]]
        if series not in {"NSIDC", "OSI-SAF-450"}:
            fig.plot(x=poly_x, y=poly_y, close=True, fill="gray85", pen="0.1p,gray70")
        pen = "1.8p,black" if series == "NSIDC" else "1.8p,black,-" if series == "OSI-SAF-450" else "1.2p"
        fig.plot(x=c.doy, y=c["mean"], pen=pen, label=series)
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
    fig.savefig(out_png)
