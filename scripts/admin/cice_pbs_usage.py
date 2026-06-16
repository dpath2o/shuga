#!/usr/bin/env python3
"""
Summarise NCI/Gadi PBS resource usage for CICE experiment directories.

Supports two layouts:

1. Legacy CICE case-directory layout:
       <root>/free-slip*/*.o*
       <root>/no-slip*/*.o*

   Example:
       cice_pbs_usage.py ~/CICE_free-slip --layout case-dirs

2. AFIM archive metadata layout:
       <root>/<EXPERIMENT_NAME>/run_metadata/<CICE_RUN_CASE>/*.o*

   Example:
       cice_pbs_usage.py ~/AFIM_archive --layout archive-metadata

Auto-detection is used by default.

Examples:
    cice_pbs_usage.py
    cice_pbs_usage.py ~/CICE_free-slip
    cice_pbs_usage.py ~/AFIM_archive
    cice_pbs_usage.py ~/AFIM_archive --layout archive-metadata
    cice_pbs_usage.py ~/AFIM_archive --group-by experiment
    cice_pbs_usage.py ~/AFIM_archive --group-by run-case
    cice_pbs_usage.py ~/AFIM_archive --group-by experiment-run-case
    cice_pbs_usage.py ~/AFIM_archive --csv
    cice_pbs_usage.py ~/AFIM_archive --show-skipped
"""

import argparse
import os
import re
import sys


RE_SERVICE_UNITS = re.compile(r"Service Units:\s*([0-9.]+)")
RE_EXIT_STATUS   = re.compile(r"Exit Status:\s*(-?[0-9]+)")
RE_MEMORY_USED   = re.compile(r"Memory Used:\s*([0-9.]+)\s*([KMGT]?B)")
RE_JOBFS_USED    = re.compile(r"JobFS Used:\s*([0-9.]+)\s*([KMGT]?B)")
RE_WALL_USED     = re.compile(r"Walltime Used:\s*([0-9:]+)")

ARCHIVE_AVOID_DIRS = {
    "CICE_0p25_Cgrid_coords.zarr",
    "future_work",
    "paper1",
    "paper3",
}


def natural_key(text):
    """Sort free-slip2 before free-slip10."""
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"([0-9]+)", str(text))]


def size_to_gb(value, unit):
    """Convert PBS size strings to GB."""
    value = float(value)
    unit = unit.upper()

    if unit == "TB":
        return value * 1024.0
    if unit == "GB":
        return value
    if unit == "MB":
        return value / 1024.0
    if unit == "KB":
        return value / 1024.0 / 1024.0
    if unit == "B":
        return value / 1024.0 / 1024.0 / 1024.0

    raise ValueError(f"Unknown size unit: {unit}")


def hms_to_seconds(text):
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = [int(x) for x in text.strip().split(":")]

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 1:
        return parts[0]

    raise ValueError(f"Could not parse walltime: {text}")


def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS."""
    seconds = int(round(seconds))
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_pbs_output(path):
    """
    Parse one PBS output file.

    Returns a dict if a resource-usage block is found, otherwise None.
    """
    data = {
        "service_units": None,
        "memory_used_gb": None,
        "walltime_used_sec": None,
        "jobfs_used_gb": None,
        "exit_status": None,
    }

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = RE_SERVICE_UNITS.search(line)
                if match:
                    data["service_units"] = float(match.group(1))
                    continue

                match = RE_EXIT_STATUS.search(line)
                if match:
                    data["exit_status"] = int(match.group(1))
                    continue

                match = RE_MEMORY_USED.search(line)
                if match:
                    data["memory_used_gb"] = size_to_gb(match.group(1), match.group(2))
                    continue

                match = RE_WALL_USED.search(line)
                if match:
                    data["walltime_used_sec"] = hms_to_seconds(match.group(1))
                    continue

                match = RE_JOBFS_USED.search(line)
                if match:
                    data["jobfs_used_gb"] = size_to_gb(match.group(1), match.group(2))
                    continue

    except OSError as err:
        print(f"WARNING: could not read {path}: {err}", file=sys.stderr)
        return None

    if data["service_units"] is None:
        return None

    return data


def looks_like_pbs_stdout(name):
    """
    Match PBS stdout files such as:
        free-slip01-1996.o170694827
        free-slip01.o170788132
        no-slip-LFI-2005.o171164829

    Deliberately excludes cice.runlog.*.
    """
    return ".o" in name and not name.startswith("cice.runlog")


def find_pbs_files_in_dir(directory):
    """Return PBS stdout files in a single directory."""
    files = []

    try:
        entries = list(os.scandir(directory))
    except OSError as err:
        print(f"WARNING: could not scan {directory}: {err}", file=sys.stderr)
        return files

    for entry in entries:
        if not entry.is_file():
            continue

        if looks_like_pbs_stdout(entry.name):
            files.append(entry.path)

    return sorted(files, key=lambda p: natural_key(os.path.basename(p)))


def find_case_dirs(root):
    """
    Legacy layout:
        <root>/free-slip*
        <root>/no-slip*
    """
    case_dirs = []

    try:
        entries = list(os.scandir(root))
    except OSError as err:
        print(f"ERROR: could not scan root directory {root}: {err}", file=sys.stderr)
        sys.exit(1)

    for entry in entries:
        if not entry.is_dir():
            continue

        name = entry.name

        if name.startswith("free-slip") or name.startswith("no-slip"):
            case_dirs.append(entry.path)

    return sorted(case_dirs, key=lambda p: natural_key(os.path.basename(p)))


def find_archive_metadata_dirs(root):
    """
    Archive layout:
        <root>/<EXPERIMENT_NAME>/run_metadata/<CICE_RUN_CASE>/
    """
    metadata_dirs = []

    try:
        experiments = list(os.scandir(root))
    except OSError as err:
        print(f"ERROR: could not scan archive root {root}: {err}", file=sys.stderr)
        sys.exit(1)

    for exp_entry in experiments:
        if not exp_entry.is_dir():
            continue

        experiment = exp_entry.name

        if experiment in ARCHIVE_AVOID_DIRS:
            continue

        run_metadata = os.path.join(exp_entry.path, "run_metadata")

        if not os.path.isdir(run_metadata):
            continue

        try:
            run_entries = list(os.scandir(run_metadata))
        except OSError as err:
            print(f"WARNING: could not scan {run_metadata}: {err}", file=sys.stderr)
            continue

        for run_entry in run_entries:
            if not run_entry.is_dir():
                continue

            metadata_dirs.append({
                "experiment": experiment,
                "run_case": run_entry.name,
                "path": run_entry.path,
            })

    return sorted(
        metadata_dirs,
        key=lambda rec: (natural_key(rec["experiment"]), natural_key(rec["run_case"])),
    )


def detect_layout(root):
    """
    Return 'archive-metadata' if root contains any */run_metadata/* directories;
    otherwise return 'case-dirs'.
    """
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue

        if entry.name in ARCHIVE_AVOID_DIRS:
            continue

        run_metadata = os.path.join(entry.path, "run_metadata")

        if not os.path.isdir(run_metadata):
            continue

        try:
            for child in os.scandir(run_metadata):
                if child.is_dir():
                    return "archive-metadata"
        except OSError:
            continue

    return "case-dirs"


def iter_pbs_sources(root, layout, group_by):
    """
    Yield:
        group_key, path

    group_by options:
        case
        experiment
        run-case
        experiment-run-case
    """
    if layout == "case-dirs":
        for case_dir in find_case_dirs(root):
            case = os.path.basename(case_dir)

            for path in find_pbs_files_in_dir(case_dir):
                yield case, path

        return

    if layout == "archive-metadata":
        for rec in find_archive_metadata_dirs(root):
            experiment = rec["experiment"]
            run_case = rec["run_case"]
            metadata_dir = rec["path"]

            if group_by == "experiment":
                group_key = experiment
            elif group_by == "run-case":
                group_key = run_case
            elif group_by == "experiment-run-case":
                group_key = f"{experiment}/{run_case}"
            else:
                # Historical/default case-style grouping.
                group_key = experiment

            for path in find_pbs_files_in_dir(metadata_dir):
                yield group_key, path

        return

    raise ValueError(f"Unknown layout: {layout}")


def empty_stats():
    return {
        "n": 0,
        "n_exit0": 0,
        "service_units_sum": 0.0,
        "memory_used_gb_sum": 0.0,
        "memory_used_n": 0,
        "walltime_used_sec_sum": 0.0,
        "walltime_used_n": 0,
        "jobfs_used_gb_sum": 0.0,
        "jobfs_used_n": 0,
    }


def add_record(stats, record):
    stats["n"] += 1
    stats["service_units_sum"] += record["service_units"]

    if record["exit_status"] == 0:
        stats["n_exit0"] += 1

    if record["memory_used_gb"] is not None:
        stats["memory_used_gb_sum"] += record["memory_used_gb"]
        stats["memory_used_n"] += 1

    if record["walltime_used_sec"] is not None:
        stats["walltime_used_sec_sum"] += record["walltime_used_sec"]
        stats["walltime_used_n"] += 1

    if record["jobfs_used_gb"] is not None:
        stats["jobfs_used_gb_sum"] += record["jobfs_used_gb"]
        stats["jobfs_used_n"] += 1


def safe_avg(total, n):
    if n == 0:
        return None
    return total / n


def format_float_or_na(value, width, precision=2):
    if value is None:
        return f"{'NA':>{width}}"
    return f"{value:{width}.{precision}f}"


def format_hms_or_na(seconds):
    if seconds is None:
        return f"{'NA':>14}"
    return f"{seconds_to_hms(seconds):>14}"


def format_table(rows, csv=False):
    if csv:
        print(
            "group,jobs,exit0,total_service_units,"
            "avg_memory_used_GB,avg_walltime_used,avg_jobfs_used_MB"
        )

        for group, stats in rows:
            avg_mem = safe_avg(stats["memory_used_gb_sum"], stats["memory_used_n"])
            avg_wall = safe_avg(stats["walltime_used_sec_sum"], stats["walltime_used_n"])
            avg_jobfs_mb = safe_avg(stats["jobfs_used_gb_sum"] * 1024.0, stats["jobfs_used_n"])

            print(
                f"{group},{stats['n']},{stats['n_exit0']},"
                f"{stats['service_units_sum']:.2f},"
                f"{'' if avg_mem is None else f'{avg_mem:.2f}'},"
                f"{'' if avg_wall is None else seconds_to_hms(avg_wall)},"
                f"{'' if avg_jobfs_mb is None else f'{avg_jobfs_mb:.2f}'}"
            )

        return

    print(
        f"{'group':<32} {'jobs':>6} {'exit0':>6} "
        f"{'total_SU':>14} {'avg_mem_GB':>12} "
        f"{'avg_walltime':>14} {'avg_JobFS_MB':>14}"
    )
    print(
        f"{'-' * 32:<32} {'-' * 6:>6} {'-' * 6:>6} "
        f"{'-' * 14:>14} {'-' * 12:>12} "
        f"{'-' * 14:>14} {'-' * 14:>14}"
    )

    for group, stats in rows:
        avg_mem = safe_avg(stats["memory_used_gb_sum"], stats["memory_used_n"])
        avg_wall = safe_avg(stats["walltime_used_sec_sum"], stats["walltime_used_n"])
        avg_jobfs_mb = safe_avg(stats["jobfs_used_gb_sum"] * 1024.0, stats["jobfs_used_n"])

        print(
            f"{group:<32} {stats['n']:6d} {stats['n_exit0']:6d} "
            f"{stats['service_units_sum']:14.2f} "
            f"{format_float_or_na(avg_mem, 12)} "
            f"{format_hms_or_na(avg_wall)} "
            f"{format_float_or_na(avg_jobfs_mb, 14)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarise PBS resource usage for CICE/AFIM experiment logs."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=os.path.expanduser("~/AFIM_archive"),
        help=(
            "Root directory to scan. For archive layout, use ~/AFIM_archive. "
            "For legacy layout, use ~/CICE_free-slip. Default: ~/AFIM_archive"
        ),
    )

    parser.add_argument(
        "--layout",
        choices=["auto", "case-dirs", "archive-metadata"],
        default="auto",
        help=(
            "Input layout. 'case-dirs' scans <root>/free-slip*/*.o* and "
            "<root>/no-slip*/*.o*. 'archive-metadata' scans "
            "<root>/<experiment>/run_metadata/<case>/*.o*. Default: auto."
        ),
    )

    parser.add_argument(
        "--group-by",
        choices=["experiment", "run-case", "experiment-run-case"],
        default="experiment",
        help=(
            "How to group archive-metadata results. Ignored for case-dirs. "
            "Default: experiment."
        ),
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print comma-separated output instead of an aligned table.",
    )

    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print files skipped because no PBS resource block was found.",
    )

    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print each parsed PBS output file and its assigned group.",
    )

    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))

    if not os.path.isdir(root):
        print(f"ERROR: root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    layout = args.layout

    if layout == "auto":
        layout = detect_layout(root)

    if not args.csv:
        print(f"Root    : {root}")
        print(f"Layout  : {layout}")
        if layout == "archive-metadata":
            print(f"Group by: {args.group_by}")
        print()

    group_stats = {}
    all_stats = empty_stats()
    skipped = 0
    parsed = 0

    for group, path in iter_pbs_sources(root, layout, args.group_by):
        record = parse_pbs_output(path)

        if record is None:
            skipped += 1
            if args.show_skipped:
                print(f"SKIPPED: {path}", file=sys.stderr)
            continue

        if args.show_files:
            print(f"PARSED: {group}: {path}", file=sys.stderr)

        if group not in group_stats:
            group_stats[group] = empty_stats()

        add_record(group_stats[group], record)
        add_record(all_stats, record)
        parsed += 1

    rows = sorted(group_stats.items(), key=lambda item: natural_key(item[0]))

    if all_stats["n"] > 0:
        rows.append(("ALL", all_stats))

    if not rows:
        print(f"No PBS resource-usage blocks found under {root}", file=sys.stderr)
        sys.exit(1)

    format_table(rows, csv=args.csv)

    if skipped > 0:
        print(f"\nSkipped {skipped} file(s) with no Service Units line.", file=sys.stderr)

    if parsed == 0:
        print(f"\nNo parsed PBS files found under {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
