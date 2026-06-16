#!/usr/bin/env python3
"""
Administrative staging/check script for CICE run directories and AFIM archive.

Default behaviour is DRY-RUN.

This version uses ONLY the explicit MAPPED_CASES dictionary below to map:

    CICE run directory name -> AFIM archive experiment directory name

No automatic ice_in parsing, signature matching, heuristic matching, --map,
or --strict logic is used.

Primary tasks
-------------
1. Scan CICE run directories under DEFAULT_RUNS_ROOT / --runs-root:
       free-slip*
       no-slip*

   while excluding:
       free-slip-tides*

2. Use MAPPED_CASES to determine the corresponding archive experiment.

3. Compare archive zarr monthly groups:
       <archive-root>/<EXPERIMENT_NAME>/zarr/iceh_daily.zarr/YYYY_MM

   with model daily history files:
       <runs-root>/<RUN_CASE>/history/iceh*.nc

   Files whose YYYY_MM group is absent from the zarr archive are treated as
   unprocessed / not analysed.

4. With --send-to-archive, stage unprocessed history files to:
       <archive-root>/<EXPERIMENT_NAME>/history/daily/

   By default history files are copied.
   With --move-history-files, history files are moved instead.

5. With --send-to-archive, copy root run metadata:
       cice.runlog*
       ice*

   to:
       <archive-root>/<EXPERIMENT_NAME>/run_metadata/<RUN_CASE>/

6. With --send-to-archive, copy the latest restart file to:
       <archive-root>/<EXPERIMENT_NAME>/restart/last/<RUN_CASE>/

7. With --clean-cice-run-dirs, clean the CICE run directory by deleting:
       cice.runlog*
       ice*
       restart/iced*

   but only after verifying that each file already exists in its mapped archive
   counterpart location.

Examples
--------
Dry-run:
    afim_archive_admin.py --verbose

Actually copy history files, metadata, and latest restart:
    afim_archive_admin.py --send-to-archive --verbose

Move unprocessed history files instead of copying them:
    afim_archive_admin.py --send-to-archive --move-history-files --verbose

Archive then clean CICE run directories:
    afim_archive_admin.py --send-to-archive --clean-cice-run-dirs --verbose
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


###########################################################################
# User-editable configuration
###########################################################################

DEFAULT_RUNS_ROOT    = "/g/data/gv90/da1339/cice-dirs/runs"
DEFAULT_ARCHIVE_ROOT = "/g/data/gv90/da1339/afim_output"

# Explicit CICE run case -> AFIM archive experiment mapping.
#
# This dictionary is authoritative. Any run directory not listed here is skipped.
#
MAPPED_CASES = {"free-slip"   : "Cs-high-ktens-mid",
                "free-slip01" : "Cs-high",
                "free-slip02" : "Cs-high-ktens-high",
                "free-slip03" : "Cs-high-eDef",
                "free-slip04" : "Cs-mid",
                "free-slip05" : "Cs-low",
                "free-slip06" : "Cq-high",
                "free-slip07" : "Cq-mid",
                "free-slip08" : "Cq-low",
                "free-slip09" : "Cl-mid",
                "free-slip10" : "Cl-low",
                "free-slip11" : "blend-strain-mid",
                "free-slip12" : "blend-strain-low",
                "free-slip13" : "blend-strain-high",
                "free-slip14" : "no-lateral-drag",
                "no-slip-def" : "no-slip-def",
                "no-slip-LFI" : "no-slip-LFI"}
D_AVOIDS = {"CICE_0p25_Cgrid_coords.zarr", "future_work", "paper1", "paper3"}
DATE_RE = re.compile(r"(?P<year>[12][0-9]{3})[-_](?P<month>[01][0-9])[-_](?P<day>[0-3][0-9])")
MONTH_RE = re.compile(r"^[12][0-9]{3}_[01][0-9]$")

###########################################################################
def natural_key(text):
    """Human/natural sorting: free-slip2 before free-slip10."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", str(text))]

def list_run_dirs(runs_root):
    """
    List CICE run dirs:

        free-slip*
        no-slip*

    excluding:

        free-slip-tides*
    """
    if not runs_root.exists():
        print(f"ERROR: runs root does not exist: {runs_root}", file=sys.stderr)
        sys.exit(1)
    run_dirs = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("free-slip-tides"):
            continue
        if name.startswith("free-slip") or name.startswith("no-slip"):
            run_dirs.append(path)
    return sorted(run_dirs, key = lambda p: natural_key(p.name))

def is_experiment_archive_dir(path):
    if not path.is_dir():
        return False
    return path.name not in D_AVOIDS

def list_archive_experiments(archive_root):
    """
    Return valid experiment directories under archive_root.
    """
    if not archive_root.exists():
        print(f"ERROR: archive root does not exist: {archive_root}", file=sys.stderr)
        sys.exit(1)
    experiments = []
    for path in archive_root.iterdir():
        if is_experiment_archive_dir(path):
            experiments.append(path)
    return sorted(experiments, key=lambda p: natural_key(p.name))

def validate_mapped_cases(run_dirs, archive_names):
    """
    Validate MAPPED_CASES against available run and archive directories.
    """
    if not MAPPED_CASES:
        print("ERROR: MAPPED_CASES is empty. Populate it before running.", file=sys.stderr)
        sys.exit(2)
    run_names = {path.name for path in run_dirs}
    errors = 0
    for run_case, exp_name in sorted(MAPPED_CASES.items(), key=lambda item: natural_key(item[0])):
        if run_case not in run_names:
            print(f"WARNING: MAPPED_CASES contains run case not present under runs root: {run_case}", file = sys.stderr)
        if exp_name not in archive_names:
            print(f"ERROR: MAPPED_CASES maps {run_case} -> {exp_name}, but archive directory does not exist.", file = sys.stderr)
            errors += 1
    if errors:
        sys.exit(2)

def month_key_from_history_file(path):
    """
    Extract YYYY_MM from a daily CICE history filename.
    """
    match = DATE_RE.search(path.name)
    if not match:
        return None
    return f"{match.group('year')}_{match.group('month')}"

def list_history_files(history_dir):
    """
    Return daily history files from a CICE run history directory.
    """
    if not history_dir.is_dir():
        return []
    files = []
    for path in history_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if not name.startswith("iceh"):
            continue
        if not (name.endswith(".nc") or ".nc." in name):
            continue
        if month_key_from_history_file(path) is None:
            continue
        files.append(path)
    return sorted(files, key=lambda p: natural_key(p.name))

def archive_processed_months(exp_dir):
    """
    Read existing YYYY_MM groups from zarr/iceh_daily.zarr.
    """
    zarr_dir = exp_dir / "zarr" / "iceh_daily.zarr"
    months = set()
    if not zarr_dir.is_dir():
        return months
    for path in zarr_dir.iterdir():
        if path.is_dir() and MONTH_RE.match(path.name):
            months.add(path.name)
    return months

def ensure_parent(path):
    path.parent.mkdir(parents = True, exist_ok = True)

def copy_file(src, dst, dry_run = True, overwrite = False, verbose = False):
    """
    Copy src to dst.

    Returns one of:
        copied
        exists
        dry-run
        error
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"  exists: {dst}")
        return "exists"
    if dry_run:
        print(f"  would copy: {src} -> {dst}")
        return "dry-run"
    try:
        ensure_parent(dst)
        shutil.copy2(src, dst)
        if verbose:
            print(f"  copied: {src} -> {dst}")
        return "copied"
    except OSError as err:
        print(f"ERROR: failed to copy {src} -> {dst}: {err}", file=sys.stderr)
        return "error"

def move_file(src, dst, dry_run = True, overwrite = False, verbose = False):
    """
    Move src to dst.

    Returns one of:
        moved
        exists
        dry-run
        error
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"  exists: {dst}")
        return "exists"
    if dry_run:
        print(f"  would move: {src} -> {dst}")
        return "dry-run"
    try:
        ensure_parent(dst)
        if dst.exists() and overwrite:
            dst.unlink()
        shutil.move(str(src), str(dst))
        if verbose:
            print(f"  moved: {src} -> {dst}")
        return "moved"
    except OSError as err:
        print(f"ERROR: failed to move {src} -> {dst}: {err}", file=sys.stderr)
        return "error"

def delete_file(path, dry_run = True, verbose = False):
    """
    Delete one file.

    Returns one of:
        deleted
        dry-run
        error
    """
    if dry_run:
        print(f"  would delete: {path}")
        return "dry-run"
    try:
        path.unlink()
        if verbose:
            print(f"  deleted: {path}")
        return "deleted"
    except OSError as err:
        print(f"ERROR: failed to delete {path}: {err}", file=sys.stderr)
        return "error"

def root_metadata_files(run_dir):
    """
    Files in the run directory root to archive as metadata and optionally clean:

        cice.runlog*
        ice*
    """
    files = []
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("cice.runlog") or name.startswith("ice"):
            files.append(path)
    return sorted(files, key=lambda p: natural_key(p.name))

def restart_iced_files(run_dir):
    """
    Return restart/iced* files for optional archiving before clean-up.
    """
    restart_dir = run_dir / "restart"
    if not restart_dir.is_dir():
        return []
    files = []
    for path in restart_dir.iterdir():
        if path.is_file() and path.name.startswith("iced"):
            files.append(path)
    return sorted(files, key=lambda p: natural_key(p.name))

def newest_restart_file(run_dir):
    """
    Return the newest restart/iced* file.
    """
    files = restart_iced_files(run_dir)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def history_archive_target(exp_dir, src):
    return exp_dir / "history" / "daily" / src.name

def metadata_archive_target(exp_dir, run_case, src):
    return exp_dir / "run_metadata" / run_case / src.name

def latest_restart_archive_target(exp_dir, run_case, src):
    return exp_dir / "restart" / "last" / run_case / src.name

def iced_restart_archive_target(exp_dir, run_case, src):
    return exp_dir / "restart" / "iced" / run_case / src.name

def archive_history_files(unprocessed, exp_dir, dry_run = True, overwrite = False, move_history_files = False, verbose = False):
    """
    Archive unprocessed daily history files by copy or move.
    """
    counts = {"history_copied": 0,
              "history_moved": 0,
              "history_existing": 0,
              "history_dry": 0,
              "errors": 0}
    action = move_file if move_history_files else copy_file
    for src in unprocessed:
        dst = history_archive_target(exp_dir, src)
        result = action(src, dst, dry_run = dry_run, overwrite = overwrite, verbose = verbose)
        if result == "copied":
            counts["history_copied"] += 1
        elif result == "moved":
            counts["history_moved"] += 1
        elif result == "exists":
            counts["history_existing"] += 1
        elif result == "dry-run":
            counts["history_dry"] += 1
        elif result == "error":
            counts["errors"] += 1
    return counts

def archive_metadata_files(run_dir, exp_dir, dry_run = True, overwrite = False, verbose = False):
    """
    Copy cice.runlog* and ice* files to run_metadata/<RUN_CASE>/.
    """
    files = root_metadata_files(run_dir)
    counts = {"metadata_files": len(files),
              "metadata_archived": 0,
              "metadata_existing": 0,
              "errors": 0}
    for src in files:
        dst = metadata_archive_target(exp_dir, run_dir.name, src)
        result = copy_file(src, dst, dry_run = dry_run, overwrite = overwrite, verbose = verbose)
        if result in ("copied", "dry-run"):
            counts["metadata_archived"] += 1
        elif result == "exists":
            counts["metadata_existing"] += 1
        elif result == "error":
            counts["errors"] += 1
    return counts

def archive_latest_restart(run_dir, exp_dir, dry_run = True, overwrite = False, verbose = False):
    """
    Copy newest restart/iced* file to restart/last/<RUN_CASE>/.
    """
    latest = newest_restart_file(run_dir)
    counts = {"latest_restart_found": latest is not None,
              "latest_restart_archived": False,
              "errors": 0}
    if latest is None:
        return counts
    dst = latest_restart_archive_target(exp_dir, run_dir.name, latest)
    result = copy_file(latest, dst, dry_run = dry_run, overwrite = overwrite, verbose = verbose)
    if result in ("copied", "dry-run", "exists"):
        counts["latest_restart_archived"] = True
    elif result == "error":
        counts["errors"] += 1
    return counts

def archive_all_iced_restarts_for_cleaning(run_dir, exp_dir, dry_run = True, overwrite = False, verbose = False):
    """
    Copy all restart/iced* files to restart/iced/<RUN_CASE>/.

    This is only needed when --clean-cice-run-dirs is requested, because the
    cleaning safety check requires every restart/iced* file to exist in archive
    before deletion.
    """
    files = restart_iced_files(run_dir)
    counts = {"restart_iced_files": len(files),
              "restart_iced_archived": 0,
              "restart_iced_existing": 0,
              "errors": 0}
    for src in files:
        dst = iced_restart_archive_target(exp_dir, run_dir.name, src)
        result = copy_file(src, dst, dry_run = dry_run, overwrite = overwrite, verbose = verbose)
        if result in ("copied", "dry-run"):
            counts["restart_iced_archived"] += 1
        elif result == "exists":
            counts["restart_iced_existing"] += 1
        elif result == "error":
            counts["errors"] += 1
    return counts

def clean_targets(run_dir, exp_dir):
    """
    Return (source, archive_counterpart) pairs for files eligible for clean-up.
    """
    pairs = []
    for src in root_metadata_files(run_dir):
        pairs.append((src, metadata_archive_target(exp_dir, run_dir.name, src)))
    for src in restart_iced_files(run_dir):
        pairs.append((src, iced_restart_archive_target(exp_dir, run_dir.name, src)))
    return pairs

def verify_clean_preconditions(run_dir, exp_dir):
    """
    Return list of files whose archive counterpart is missing.
    """
    missing = []
    for src, archived in clean_targets(run_dir, exp_dir):
        if not archived.is_file():
            missing.append((src, archived))
    return missing

def clean_cice_run_dir(run_dir, exp_dir, dry_run=True, verbose=False):
    """
    Delete root cice.runlog*, root ice*, and restart/iced* files, but only if
    every file has an archive counterpart already present.
    """
    missing = verify_clean_preconditions(run_dir, exp_dir)
    if missing:
        print(f"ERROR: refusing to clean {run_dir.name}; "
              f"{len(missing)} file(s) do not exist in mapped archive location.",
              file = sys.stderr)
        for src, archived in missing[:20]:
            print(f"  missing archive counterpart: {src} -> {archived}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more missing files", file=sys.stderr)
        return {"clean_deleted": 0,
                "clean_errors": 1,
                "clean_missing": len(missing)}
    deleted = 0
    errors = 0
    for src, _archived in clean_targets(run_dir, exp_dir):
        result = delete_file(src, dry_run=dry_run, verbose=verbose)
        if result == "deleted":
            deleted += 1
        elif result == "error":
            errors += 1
    return {"clean_deleted": deleted,
            "clean_errors": errors,
            "clean_missing": 0}


def process_run_dir(
    run_dir,
    exp_name,
    archive_root,
    dry_run=True,
    overwrite=False,
    move_history_files=False,
    clean_run_dirs=False,
    verbose=False,
):
    """
    Compare one run directory with one archive experiment and optionally stage files.
    """
    exp_dir = archive_root / exp_name

    processed_months = archive_processed_months(exp_dir)
    history_files = list_history_files(run_dir / "history")

    unprocessed = []

    for path in history_files:
        month_key = month_key_from_history_file(path)

        if month_key not in processed_months:
            unprocessed.append(path)

    row = {
        "run_case": run_dir.name,
        "experiment": exp_name,
        "history_total": len(history_files),
        "processed_months": len(processed_months),
        "unprocessed_history": len(unprocessed),
        "history_copied": 0,
        "history_moved": 0,
        "history_existing": 0,
        "history_dry": 0,
        "metadata_files": 0,
        "metadata_archived": 0,
        "metadata_existing": 0,
        "latest_restart_found": False,
        "latest_restart_archived": False,
        "restart_iced_files": len(restart_iced_files(run_dir)),
        "restart_iced_archived": 0,
        "restart_iced_existing": 0,
        "clean_deleted": 0,
        "clean_missing": 0,
        "errors": 0,
    }

    hist_counts = archive_history_files(
        unprocessed,
        exp_dir,
        dry_run=dry_run,
        overwrite=overwrite,
        move_history_files=move_history_files,
        verbose=verbose,
    )

    row.update({k: hist_counts.get(k, row.get(k, 0)) for k in hist_counts if k != "errors"})
    row["errors"] += hist_counts["errors"]

    metadata_counts = archive_metadata_files(
        run_dir,
        exp_dir,
        dry_run=dry_run,
        overwrite=overwrite,
        verbose=verbose,
    )

    for key, val in metadata_counts.items():
        if key == "errors":
            row["errors"] += val
        else:
            row[key] = val

    latest_counts = archive_latest_restart(
        run_dir,
        exp_dir,
        dry_run=dry_run,
        overwrite=overwrite,
        verbose=verbose,
    )

    for key, val in latest_counts.items():
        if key == "errors":
            row["errors"] += val
        else:
            row[key] = val

    if clean_run_dirs:
        # To safely clean restart/iced*, every restart/iced* file must first
        # exist in archive/restart/iced/<RUN_CASE>/.
        restart_counts = archive_all_iced_restarts_for_cleaning(
            run_dir,
            exp_dir,
            dry_run=dry_run,
            overwrite=overwrite,
            verbose=verbose,
        )

        for key, val in restart_counts.items():
            if key == "errors":
                row["errors"] += val
            else:
                row[key] = val

        clean_counts = clean_cice_run_dir(
            run_dir,
            exp_dir,
            dry_run=dry_run,
            verbose=verbose,
        )

        row["clean_deleted"] = clean_counts["clean_deleted"]
        row["clean_missing"] = clean_counts["clean_missing"]
        row["errors"] += clean_counts["clean_errors"]

    return row


def print_summary(rows, dry_run, move_history_files, clean_run_dirs):
    """
    Print compact summary table.
    """
    if dry_run:
        history_action_label = "would_move" if move_history_files else "would_copy"
    else:
        history_action_label = "moved" if move_history_files else "copied"

    print()
    print(
        f"{'run_case':<16} {'experiment':<24} "
        f"{'hist':>6} {'zarr_mon':>8} {'unproc':>8} "
        f"{history_action_label:>11} {'meta':>6} "
        f"{'rst':>5} {'clean':>6} {'miss':>5} {'err':>4}"
    )
    print(
        f"{'-' * 16:<16} {'-' * 24:<24} "
        f"{'-' * 6:>6} {'-' * 8:>8} {'-' * 8:>8} "
        f"{'-' * 11:>11} {'-' * 6:>6} "
        f"{'-' * 5:>5} {'-' * 6:>6} {'-' * 5:>5} {'-' * 4:>4}"
    )

    totals = {
        "history_total": 0,
        "unprocessed_history": 0,
        "history_action": 0,
        "metadata_files": 0,
        "restart_iced_files": 0,
        "clean_deleted": 0,
        "clean_missing": 0,
        "errors": 0,
    }

    for row in rows:
        if dry_run:
            history_action = row["history_dry"]
        elif move_history_files:
            history_action = row["history_moved"]
        else:
            history_action = row["history_copied"]

        print(
            f"{row['run_case']:<16} {row['experiment']:<24} "
            f"{row['history_total']:6d} "
            f"{row['processed_months']:8d} "
            f"{row['unprocessed_history']:8d} "
            f"{history_action:11d} "
            f"{row['metadata_files']:6d} "
            f"{row['restart_iced_files']:5d} "
            f"{row['clean_deleted']:6d} "
            f"{row['clean_missing']:5d} "
            f"{row['errors']:4d}"
        )

        totals["history_total"] += row["history_total"]
        totals["unprocessed_history"] += row["unprocessed_history"]
        totals["history_action"] += history_action
        totals["metadata_files"] += row["metadata_files"]
        totals["restart_iced_files"] += row["restart_iced_files"]
        totals["clean_deleted"] += row["clean_deleted"]
        totals["clean_missing"] += row["clean_missing"]
        totals["errors"] += row["errors"]

    print(
        f"{'-' * 16:<16} {'-' * 24:<24} "
        f"{'-' * 6:>6} {'-' * 8:>8} {'-' * 8:>8} "
        f"{'-' * 11:>11} {'-' * 6:>6} "
        f"{'-' * 5:>5} {'-' * 6:>6} {'-' * 5:>5} {'-' * 4:>4}"
    )

    print(
        f"{'ALL':<16} {'':<24} "
        f"{totals['history_total']:6d} "
        f"{'':>8} "
        f"{totals['unprocessed_history']:8d} "
        f"{totals['history_action']:11d} "
        f"{totals['metadata_files']:6d} "
        f"{totals['restart_iced_files']:5d} "
        f"{totals['clean_deleted']:6d} "
        f"{totals['clean_missing']:5d} "
        f"{totals['errors']:4d}"
    )

    if clean_run_dirs and dry_run:
        print()
        print("NOTE: clean-run mode was requested in dry-run, which should not happen.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage CICE run history/metadata into AFIM archive using MAPPED_CASES."
    )

    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help=f"Root containing CICE run directories. Default: {DEFAULT_RUNS_ROOT}",
    )

    parser.add_argument(
        "--archive-root",
        default=DEFAULT_ARCHIVE_ROOT,
        help=f"AFIM archive root. Default: {DEFAULT_ARCHIVE_ROOT}",
    )

    parser.add_argument(
        "--send-to-archive",
        action="store_true",
        help="Actually archive files. Default is dry-run.",
    )

    parser.add_argument(
        "--move-history-files",
        action="store_true",
        help="Move unprocessed ice history files instead of copying them. Requires --send-to-archive to actually move.",
    )

    parser.add_argument(
        "--clean-cice-run-dirs",
        action="store_true",
        help=(
            "After archiving, delete cice.runlog*, ice*, and restart/iced* files "
            "from each mapped CICE run directory. Requires --send-to-archive and "
            "requires archive counterparts to exist."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archive files. Default: skip existing files.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file copy/move/delete actions.",
    )

    args = parser.parse_args()

    if args.clean_cice_run_dirs and not args.send_to_archive:
        parser.error("--clean-cice-run-dirs requires --send-to-archive; it is not allowed in dry-run mode.")

    runs_root = Path(args.runs_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()

    dry_run = not args.send_to_archive

    archive_dirs = list_archive_experiments(archive_root)
    archive_names = {path.name for path in archive_dirs}

    run_dirs = list_run_dirs(runs_root)

    validate_mapped_cases(run_dirs, archive_names)

    if dry_run:
        print("Mode: DRY-RUN. No files will be copied, moved, or deleted.")
        print("Use --send-to-archive to archive files.")
    else:
        print("Mode: SEND-TO-ARCHIVE. Files may be copied or moved.")

    if args.move_history_files:
        print("History file action: MOVE unprocessed history files.")
    else:
        print("History file action: COPY unprocessed history files.")

    if args.clean_cice_run_dirs:
        print("Clean CICE run dirs: ENABLED.")
    else:
        print("Clean CICE run dirs: disabled.")

    print(f"Runs root   : {runs_root}")
    print(f"Archive root: {archive_root}")
    print()

    print("MAPPED_CASES:")
    for run_case in sorted(MAPPED_CASES, key=natural_key):
        print(f"  {run_case} -> {MAPPED_CASES[run_case]}")
    print()

    rows = []
    skipped = []

    for run_dir in run_dirs:
        run_case = run_dir.name

        if run_case not in MAPPED_CASES:
            skipped.append(run_case)

            if args.verbose:
                print(f"SKIP: {run_case}: not present in MAPPED_CASES")

            continue

        exp_name = MAPPED_CASES[run_case]
        exp_dir = archive_root / exp_name

        if not exp_dir.is_dir():
            print(
                f"ERROR: mapped archive directory does not exist: "
                f"{run_case} -> {exp_dir}",
                file=sys.stderr,
            )
            continue

        if args.verbose:
            print(f"{run_case} -> {exp_name}")

        row = process_run_dir(
            run_dir,
            exp_name,
            archive_root,
            dry_run=dry_run,
            overwrite=args.overwrite,
            move_history_files=args.move_history_files,
            clean_run_dirs=args.clean_cice_run_dirs,
            verbose=args.verbose,
        )

        rows.append(row)

    print_summary(
        rows,
        dry_run=dry_run,
        move_history_files=args.move_history_files,
        clean_run_dirs=args.clean_cice_run_dirs,
    )

    if skipped:
        print()
        print("Skipped run directories not present in MAPPED_CASES:", file=sys.stderr)
        for run_case in skipped:
            print(f"  {run_case}", file=sys.stderr)

    if dry_run:
        print()
        print("Dry-run complete. Nothing was copied, moved, or deleted.")


if __name__ == "__main__":
    main()
