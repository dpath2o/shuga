# shuga/metrics/temporal.py
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def month_window_bounds(
    year_start: int,
    start_month: int,
    end_month: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return inclusive start/end dates for a month window.

    If end_month < start_month, the window is treated as crossing into
    year_start + 1. Example: Dec-Mar for year_start=1993 gives
    1993-12-01 to 1994-03-31.
    """
    start = pd.Timestamp(year=int(year_start), month=int(start_month), day=1)
    end_year = int(year_start) if end_month >= start_month else int(year_start) + 1
    end_day = pd.Period(f"{end_year}-{int(end_month):02d}", freq="M").days_in_month
    end = pd.Timestamp(year=end_year, month=int(end_month), day=int(end_day))
    return start, end


def linear_rate_per_day(series: pd.Series) -> float:
    """
    Least-squares linear slope in series units per day.
    """
    s = series.dropna()
    if len(s) < 2:
        return np.nan

    x = np.asarray((s.index - s.index[0]).days, dtype=float)
    y = np.asarray(s.values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(y) < 2 or np.unique(x).size < 2:
        return np.nan

    x0 = x - x.mean()
    den = float(np.sum(x0 * x0))

    if den == 0.0:
        return np.nan

    return float(np.sum(x0 * (y - y.mean())) / den)


def seasonal_rate_record(
    series: pd.Series,
    *,
    year_start: int,
    start_month: int,
    end_month: int,
    prefix: str,
    require_full_window: bool = True,
    min_points: int = 20,
) -> dict:
    """
    Compute linear seasonal rate diagnostics for one month window.
    """
    window_start, window_end = month_window_bounds(
        year_start=year_start,
        start_month=start_month,
        end_month=end_month,
    )

    out = {
        f"{prefix}_window_start": window_start.date().isoformat(),
        f"{prefix}_window_end": window_end.date().isoformat(),
        f"{prefix}_start_date": "",
        f"{prefix}_end_date": "",
        f"{prefix}_n_time": 0,
        f"{prefix}_n_days": np.nan,
        f"{prefix}_value_start": np.nan,
        f"{prefix}_value_end": np.nan,
        f"{prefix}_delta_value": np.nan,
        f"{prefix}_rate_per_day": np.nan,
    }

    if series.empty:
        return out

    if require_full_window:
        if pd.Timestamp(series.index.min()) > window_start:
            return out
        if pd.Timestamp(series.index.max()) < window_end:
            return out

    s = series.loc[(series.index >= window_start) & (series.index <= window_end)].dropna()

    if len(s) < min_points:
        return out

    n_days = int((s.index[-1] - s.index[0]).days)
    if n_days <= 0:
        return out

    out.update(
        {
            f"{prefix}_start_date": pd.Timestamp(s.index[0]).date().isoformat(),
            f"{prefix}_end_date": pd.Timestamp(s.index[-1]).date().isoformat(),
            f"{prefix}_n_time": int(len(s)),
            f"{prefix}_n_days": n_days,
            f"{prefix}_value_start": float(s.iloc[0]),
            f"{prefix}_value_end": float(s.iloc[-1]),
            f"{prefix}_delta_value": float(s.iloc[-1] - s.iloc[0]),
            f"{prefix}_rate_per_day": linear_rate_per_day(s),
        }
    )

    return out


def compute_extrema_table(
    da: xr.DataArray,
    *,
    variable: str | None = None,
    sim_name: str | None = None,
    year_mode: str = "calendar",
    include_mean: bool = True,
    include_overall: bool = True,
    growth_window: tuple[int, int] | None = (4, 7),
    retreat_window: tuple[int, int] | None = (12, 3),
    require_full_rate_window: bool = True,
    rate_min_points: int = 20,
    drop_partial_periods: bool = False,
) -> pd.DataFrame:
    """
    Build a per-year extrema table for a 1D metric time series.

    Also optionally reports seasonal linear growth and retreat rates.

    growth_window defaults to Apr-Jul.
    retreat_window defaults to Dec-Mar and is assigned to the December year.
    """
    if "time" not in da.dims:
        raise ValueError("Input metric must have a 'time' dimension.")

    non_time_dims = [d for d in da.dims if d != "time"]
    if non_time_dims:
        raise ValueError(
            f"Input metric must be 1D over time. Found extra dimensions: {non_time_dims}"
        )

    variable = variable or da.name or "metric"
    units = da.attrs.get("units", "")
    rate_units = f"{units} day^-1" if units else "metric units day^-1"

    series = da.load().to_series().dropna()
    if series.empty:
        raise ValueError(f"No finite data found for metric {variable!r}.")

    series.index = pd.to_datetime(series.index)
    idx = pd.DatetimeIndex(series.index)

    mode = year_mode.lower().strip()

    if mode in {"calendar", "cal", "year"}:
        year_start = idx.year
        period_labels = [str(y) for y in year_start]
        year_mode_out = "calendar"
        period_bounds = {
            int(y): month_window_bounds(int(y), 1, 12)
            for y in np.unique(year_start)
        }
    elif mode in {"antarctic", "ant", "ant-year", "ant_year"}:
        year_start = np.where(idx.month >= 3, idx.year, idx.year - 1)
        period_labels = [f"{int(y)}/{str(int(y) + 1)[-2:]}" for y in year_start]
        year_mode_out = "antarctic"
        period_bounds = {
            int(y): month_window_bounds(int(y), 3, 2)
            for y in np.unique(year_start)
        }
    else:
        raise ValueError(f"year_mode must be either 'calendar' or 'antarctic'. Got {year_mode!r}.")

    grouped = series.groupby(period_labels)

    label_to_year_start: dict[str, int] = {}
    for lab, ys in zip(period_labels, year_start):
        label_to_year_start.setdefault(lab, int(ys))

    rows: list[dict] = []

    for period, grp in grouped:
        if grp.empty:
            continue

        ys = int(label_to_year_start[period])

        if drop_partial_periods:
            p0, p1 = period_bounds[ys]
            if pd.Timestamp(series.index.min()) > p0:
                continue
            if pd.Timestamp(series.index.max()) < p1:
                continue

        t_min = pd.Timestamp(grp.idxmin())
        t_max = pd.Timestamp(grp.idxmax())

        row = {
            "sim_name": sim_name,
            "metric": variable,
            "units": units,
            "rate_units": rate_units,
            "year_mode": year_mode_out,
            "period": period,
            "year_start": ys,
            "n_time": int(grp.count()),
            "n_years": 1,
            "start_date": pd.Timestamp(grp.index.min()).date().isoformat(),
            "end_date": pd.Timestamp(grp.index.max()).date().isoformat(),
            "date_min": t_min.date().isoformat(),
            "doy_min": int(t_min.dayofyear),
            "value_min": float(grp.min()),
            "date_max": t_max.date().isoformat(),
            "doy_max": int(t_max.dayofyear),
            "value_max": float(grp.max()),
            "mean_value": float(grp.mean()),
            "std_value": float(grp.std(ddof=0)),
        }

        if growth_window is not None:
            row.update(
                seasonal_rate_record(
                    series,
                    year_start=ys,
                    start_month=int(growth_window[0]),
                    end_month=int(growth_window[1]),
                    prefix="growth",
                    require_full_window=require_full_rate_window,
                    min_points=rate_min_points,
                )
            )

        if retreat_window is not None:
            row.update(
                seasonal_rate_record(
                    series,
                    year_start=ys,
                    start_month=int(retreat_window[0]),
                    end_month=int(retreat_window[1]),
                    prefix="retreat",
                    require_full_window=require_full_rate_window,
                    min_points=rate_min_points,
                )
            )

        rows.append(row)

    if not rows:
        raise ValueError(f"No annual groups could be built for {variable!r}.")

    df = pd.DataFrame(rows).sort_values(["year_start", "period"]).reset_index(drop=True)

    def _mean_numeric(col: str) -> float:
        if col not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    def _sum_numeric(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    aggregate_rows: list[dict] = []

    if include_mean:
        mean_row = {
            "sim_name": sim_name,
            "metric": variable,
            "units": units,
            "rate_units": rate_units,
            "year_mode": year_mode_out,
            "period": "MEAN",
            "year_start": np.nan,
            "n_time": int(series.count()),
            "n_years": int(len(df)),
            "start_date": pd.Timestamp(series.index.min()).date().isoformat(),
            "end_date": pd.Timestamp(series.index.max()).date().isoformat(),
            "date_min": "",
            "doy_min": _mean_numeric("doy_min"),
            "value_min": _mean_numeric("value_min"),
            "date_max": "",
            "doy_max": _mean_numeric("doy_max"),
            "value_max": _mean_numeric("value_max"),
            "mean_value": float(series.mean()),
            "std_value": float(series.std(ddof=0)),
        }

        for prefix in ("growth", "retreat"):
            if f"{prefix}_rate_per_day" in df.columns:
                mean_row.update(
                    {
                        f"{prefix}_window_start": "",
                        f"{prefix}_window_end": "",
                        f"{prefix}_start_date": "",
                        f"{prefix}_end_date": "",
                        f"{prefix}_n_time": _sum_numeric(f"{prefix}_n_time"),
                        f"{prefix}_n_days": _mean_numeric(f"{prefix}_n_days"),
                        f"{prefix}_value_start": _mean_numeric(f"{prefix}_value_start"),
                        f"{prefix}_value_end": _mean_numeric(f"{prefix}_value_end"),
                        f"{prefix}_delta_value": _mean_numeric(f"{prefix}_delta_value"),
                        f"{prefix}_rate_per_day": _mean_numeric(f"{prefix}_rate_per_day"),
                    }
                )

        aggregate_rows.append(mean_row)

    if include_overall:
        t_min = pd.Timestamp(series.idxmin())
        t_max = pd.Timestamp(series.idxmax())

        overall_row = {
            "sim_name": sim_name,
            "metric": variable,
            "units": units,
            "rate_units": rate_units,
            "year_mode": year_mode_out,
            "period": "OVERALL",
            "year_start": np.nan,
            "n_time": int(series.count()),
            "n_years": int(len(df)),
            "start_date": pd.Timestamp(series.index.min()).date().isoformat(),
            "end_date": pd.Timestamp(series.index.max()).date().isoformat(),
            "date_min": t_min.date().isoformat(),
            "doy_min": int(t_min.dayofyear),
            "value_min": float(series.min()),
            "date_max": t_max.date().isoformat(),
            "doy_max": int(t_max.dayofyear),
            "value_max": float(series.max()),
            "mean_value": float(series.mean()),
            "std_value": float(series.std(ddof=0)),
        }

        aggregate_rows.append(overall_row)

    if aggregate_rows:
        df = pd.concat([df, pd.DataFrame(aggregate_rows)], ignore_index=True)

    return df
