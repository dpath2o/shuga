#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from shuga.core.paths import ShugaPaths
from shuga.core.types import ClassificationSpec, MetricsSpec, RunSpec
from shuga.metrics.cice import CICEMetrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Report annual min/max timing and values for a 1D shuga metric "
            "from method-specific mets.zarr stores."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "sim_names",
        nargs="+",
        help="One or more simulation names to compare.",
    )

    p.add_argument(
        "-v",
        "--variable",
        "--metric",
        dest="variable",
        default="FIA",
        help="Metric variable to report, e.g. FIA, SIA, FIT, SIT.",
    )

    p.add_argument(
        "-b",
        "--start-date",
        "--dt0-str",
        dest="start_date",
        default=None,
        help="Optional start date, YYYY-MM-DD.",
    )

    p.add_argument(
        "-e",
        "--end-date",
        "--dtN-str",
        dest="end_date",
        default=None,
        help="Optional end date, YYYY-MM-DD.",
    )

    p.add_argument(
        "-m",
        "--method",
        "--classification",
        dest="method",
        default="binary-days",
        help="Classification/metrics method branch: raw, binary-days, rolling-mean.",
    )

    p.add_argument(
        "-g",
        "--grid-type",
        default="Tc",
        help="Classification grid branch used to resolve mets.zarr.",
    )

    p.add_argument(
        "-H",
        "--hemisphere",
        default="SH",
        help="Hemisphere: SH or NH.",
    )

    p.add_argument(
        "--ice-type",
        default="FI",
        help="Ice type branch used in the classification/metrics path.",
    )

    p.add_argument(
        "--project",
        default="gv90",
        help="NCI project used in default shuga path resolution.",
    )

    p.add_argument(
        "--user",
        default="da1339",
        help="NCI username used in default shuga path resolution.",
    )

    p.add_argument(
        "--afim-output-root",
        default=None,
        help=(
            "Optional explicit AFIM output root. "
            "Example: /g/data/gv90/da1339/afim_output"
        ),
    )

    p.add_argument(
        "--ispd-thresh",
        type=float,
        default=None,
        help="Optional fast-ice speed threshold override.",
    )

    p.add_argument(
        "--bin-window",
        type=int,
        default=None,
        help="Optional binary-days window override.",
    )

    p.add_argument(
        "--bin-min-days",
        type=int,
        default=None,
        help="Optional binary-days minimum-days override.",
    )

    p.add_argument(
        "--roll-window",
        type=int,
        default=None,
        help="Optional rolling-mean window override.",
    )

    p.add_argument(
        "--year-mode",
        choices=["calendar", "antarctic"],
        default="calendar",
        help=(
            "Year grouping. 'calendar' is Jan-Dec. "
            "'antarctic' is Mar-Feb, labelled by season start year."
        ),
    )

    p.add_argument(
        "--compute-missing",
        action="store_true",
        help=(
            "Compute the requested metric if it is missing from mets.zarr. "
            "Requires the cice.py _expand_metric_names() patch above to avoid "
            "building the full default metric set."
        ),
    )

    p.add_argument(
        "--no-mean",
        action="store_true",
        help="Do not append the MEAN row.",
    )

    p.add_argument(
        "--no-overall",
        action="store_true",
        help="Do not append the OVERALL row.",
    )

    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    p.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number of decimal places for terminal output.",
    )

    p.add_argument(
        "--growth-start-month",
        type=int,
        default=4,
        help="Start month for growth-rate window.",
    )

    p.add_argument(
        "--growth-end-month",
        type=int,
        default=7,
        help="End month for growth-rate window.",
    )

    p.add_argument(
        "--retreat-start-month",
        type=int,
        default=12,
        help="Start month for retreat-rate window.",
    )

    p.add_argument(
        "--retreat-end-month",
        type=int,
        default=3,
        help="End month for retreat-rate window.",
    )

    p.add_argument(
        "--no-seasonal-rates",
        action="store_true",
        help="Disable growth/retreat rate diagnostics.",
    )

    p.add_argument(
        "--allow-partial-rate-window",
        action="store_true",
        help=(
            "Allow rate calculation from partial seasonal windows. "
            "By default full Apr-Jul and Dec-Mar windows are required."
        ),
    )

    p.add_argument(
        "--rate-min-points",
        type=int,
        default=20,
        help="Minimum number of valid time points required for a seasonal rate.",
    )

    p.add_argument(
        "--drop-partial-years",
        action="store_true",
        help=(
            "Drop partial annual periods from the extrema table. "
            "Useful if you extend end-date to March only to capture Dec-Mar retreat."
        ),
    )

    return p


def _classification_spec(args: argparse.Namespace) -> ClassificationSpec:
    kwargs = {
        "ice_type": args.ice_type,
        "grid_type": args.grid_type,
        "methods": (args.method,),
    }

    if args.ispd_thresh is not None:
        kwargs["ispd_thresh"] = args.ispd_thresh
    if args.bin_window is not None:
        kwargs["bin_window"] = args.bin_window
    if args.bin_min_days is not None:
        kwargs["bin_min_days"] = args.bin_min_days
    if args.roll_window is not None:
        kwargs["roll_window"] = args.roll_window

    return ClassificationSpec(**kwargs)


def _metrics_for_sim(args: argparse.Namespace, sim_name: str) -> pd.DataFrame:
    run = RunSpec(
        sim_name=sim_name,
        start_date=args.start_date or "1900-01-01",
        end_date=args.end_date or "2100-12-31",
        hemisphere=args.hemisphere,
        project=args.project,
        user=args.user,
    )

    classify = _classification_spec(args)
    metrics = MetricsSpec()

    paths = ShugaPaths(
        run=run,
        classify=classify,
        metrics=metrics,
        afim_output_root=args.afim_output_root,
    )

    reporter = CICEMetrics(
        run=run,
        classify=classify,
        metrics=metrics,
        paths=paths,
        chunks={"time": 31},
    )

    if args.no_seasonal_rates:
        growth_window = None
        retreat_window = None
    else:
        growth_window = (args.growth_start_month, args.growth_end_month)
        retreat_window = (args.retreat_start_month, args.retreat_end_month)

    return reporter.report_metric_extrema(
        args.method,
        variable=args.variable,
        year_mode=args.year_mode,
        compute_missing=args.compute_missing,
        include_mean=not args.no_mean,
        include_overall=not args.no_overall,
        growth_window=growth_window,
        retreat_window=retreat_window,
        require_full_rate_window=not args.allow_partial_rate_window,
        rate_min_points=args.rate_min_points,
        drop_partial_periods=args.drop_partial_years,
    )


DISPLAY_RENAMES = {
    "period": "PERIOD",
    "year_start": "YEAR_START",
    "n_time": "N",
    "n_years": "N_YEARS",
    "start_date": "START",
    "end_date": "END",

    "date_min": "MIN_DATE",
    "doy_min": "MIN_DOY",
    "value_min": "MIN_VAL",

    "date_max": "MAX_DATE",
    "doy_max": "MAX_DOY",
    "value_max": "MAX_VAL",

    "mean_value": "MEAN",
    "std_value": "STD",

    "growth_start_date": "GROW_START",
    "growth_end_date": "GROW_END",
    "growth_n_time": "GROW_N",
    "growth_n_days": "GROW_DAYS",
    "growth_value_start": "GROW_START_VAL",
    "growth_value_end": "GROW_END_VAL",
    "growth_delta_value": "GROW_DELTA",
    "growth_rate_per_day": "GROW_RATE",

    "retreat_start_date": "RETREAT_START",
    "retreat_end_date": "RETREAT_END",
    "retreat_n_time": "RETREAT_N",
    "retreat_n_days": "RETREAT_DAYS",
    "retreat_value_start": "RETREAT_START_VAL",
    "retreat_value_end": "RETREAT_END_VAL",
    "retreat_delta_value": "RETREAT_DELTA",
    "retreat_rate_per_day": "RETREAT_RATE",
}


HEADER_COLUMNS = [
    "sim_name",
    "method",
    "grid_type",
    "ice_type",
    "hemisphere",
    "metric",
    "units",
    "rate_units",
    "year_mode",
]


WINDOW_COLUMNS = [
    "growth_window_start",
    "growth_window_end",
    "retreat_window_start",
    "retreat_window_end",
]


PREFERRED_TABLE_COLUMNS = [
    "period",
    "year_start",
    "n_time",
    "start_date",
    "end_date",

    "date_min",
    "doy_min",
    "value_min",

    "date_max",
    "doy_max",
    "value_max",

    "mean_value",
    "std_value",

    "growth_start_date",
    "growth_end_date",
    "growth_n_time",
    "growth_n_days",
    "growth_value_start",
    "growth_value_end",
    "growth_delta_value",
    "growth_rate_per_day",

    "retreat_start_date",
    "retreat_end_date",
    "retreat_n_time",
    "retreat_n_days",
    "retreat_value_start",
    "retreat_value_end",
    "retreat_delta_value",
    "retreat_rate_per_day",
]


def _is_blank_series(s: pd.Series) -> bool:
    if s.empty:
        return True
    return s.isna().all() or s.astype(str).str.strip().isin(["", "nan", "NaT"]).all()


def _format_value(value, *, precision: int, integer_like: bool = False) -> str:
    if pd.isna(value):
        return ""

    if integer_like:
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value)

    if isinstance(value, float):
        if abs(value - round(value)) < 1.0e-10:
            return str(int(round(value)))
        return f"{value:.{precision}f}"

    return str(value)


def _format_display_df(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    out = df.copy()

    integer_cols = {
        "year_start",
        "n_time",
        "n_years",
        "doy_min",
        "doy_max",
        "growth_n_time",
        "growth_n_days",
        "retreat_n_time",
        "retreat_n_days",
    }

    for col in out.columns:
        if col in integer_cols:
            out[col] = out[col].map(
                lambda x: _format_value(x, precision=precision, integer_like=True)
            )
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: _format_value(x, precision=precision, integer_like=False)
            )
        else:
            out[col] = out[col].fillna("").astype(str)

    return out


def _unique_nonblank(values: pd.Series) -> list[str]:
    vals = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "nat"}:
            vals.append(text)
    return list(dict.fromkeys(vals))


def _window_header_value(g: pd.DataFrame, col: str) -> str:
    if col not in g.columns:
        return ""

    rows = g[~g["period"].isin(["MEAN", "OVERALL"])] if "period" in g.columns else g
    vals = _unique_nonblank(rows[col])

    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        return vals[0]

    # Multi-year tables have year-specific dates, e.g. 1993-04-01,
    # 1994-04-01, ... Collapse this to month-day notation.
    mmdd = list(dict.fromkeys([v[5:] if len(v) >= 10 else v for v in vals]))
    if len(mmdd) == 1:
        return f"{mmdd[0]} each period"

    return f"{vals[0]} ... {vals[-1]}"


def _print_one_group(g: pd.DataFrame, *, precision: int) -> None:
    first = g.iloc[0]

    header = {
        "SIM_NAME": first.get("sim_name", ""),
        "METHOD": first.get("method", ""),
        "GRID_TYPE": first.get("grid_type", ""),
        "ICE_TYPE": first.get("ice_type", ""),
        "HEMISPHERE": first.get("hemisphere", ""),
        "METRIC": first.get("metric", ""),
        "VAR_UNITS": first.get("units", ""),
        "RATE_UNITS": first.get("rate_units", ""),
        "YEAR_MODE": first.get("year_mode", ""),
        "GROWTH WINDOW START DATE": _window_header_value(g, "growth_window_start"),
        "GROWTH WINDOW STOP DATE": _window_header_value(g, "growth_window_end"),
        "RETREAT WINDOW START DATE": _window_header_value(g, "retreat_window_start"),
        "RETREAT WINDOW STOP DATE": _window_header_value(g, "retreat_window_end"),
    }

    print("\n" + "=" * 100)
    for key, value in header.items():
        value = "" if pd.isna(value) else str(value)
        if value.strip():
            print(f"{key:<30}: {value}")
    print("-" * 100)

    excluded = set(HEADER_COLUMNS + WINDOW_COLUMNS)

    cols = [c for c in PREFERRED_TABLE_COLUMNS if c in g.columns and c not in excluded]
    extras = [c for c in g.columns if c not in excluded and c not in cols]
    cols.extend(extras)

    table = g[cols].copy()

    # Drop columns that are entirely blank for this specific sim table.
    table = table.loc[:, [c for c in table.columns if not _is_blank_series(table[c])]]

    table = _format_display_df(table, precision=precision)
    table = table.rename(columns={c: DISPLAY_RENAMES.get(c, c.upper()) for c in table.columns})

    print(table.to_string(index=False))
    print(f"\n[{len(table)} rows x {len(table.columns)} columns]")


def print_grouped_terminal_tables(df: pd.DataFrame, *, precision: int = 3) -> None:
    group_cols = [c for c in HEADER_COLUMNS if c in df.columns]

    if not group_cols:
        _print_one_group(df, precision=precision)
        return

    for _, g in df.groupby(group_cols, dropna=False, sort=False):
        _print_one_group(g.reset_index(drop=True), precision=precision)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, Exception]] = []

    for sim_name in args.sim_names:
        try:
            frames.append(_metrics_for_sim(args, sim_name))
        except Exception as exc:
            failures.append((sim_name, exc))

    if not frames:
        print("No reports were generated.", file=sys.stderr)
        for sim_name, exc in failures:
            print(f"[FAILED] {sim_name}: {exc}", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Wrote CSV: {args.csv}")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print_grouped_terminal_tables(df, precision=args.precision)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for sim_name, exc in failures:
            print(f"[FAILED] {sim_name}: {exc}", file=sys.stderr)

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
