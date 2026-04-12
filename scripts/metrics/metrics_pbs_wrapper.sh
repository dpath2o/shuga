#!/bin/bash
set -euo pipefail

SIM_NAME=""
START_DATE="1993-01-01"
END_DATE="1999-12-31"
HEMISPHERE="SH"
ICE_TYPE="FI"
GRID_TYPE="Tc"
ISPD_THRESH="5e-4"
METHODS="binary-days,rolling-mean"
BIN_WINDOW="11"
BIN_MIN_DAYS="9"
ROLL_WINDOW="15"
PROJECT="gv90"
RUN_USER="da1339"
OVERWRITE="false"
UPDATE_MISSING_ONLY="true"
REBUILD_ON_INDEX_MISMATCH="false"
METRIC_GROUPS="default"
METRIC_NAMES=""
DRY_RUN="false"
PLOT_FIP="false"
PLOT_FIA="false"
PLOT_FIT="false"
PLOT_SIA="false"
PLOT_SIT="false"
PLOT_REGION="total"
OBS_FIA_VAR="FIA"
OBS_FIT_VAR="FIT"
LOG_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/metrics.pbs"

print_help() {
  cat <<EOF
Usage: $0 -s SIM_NAME [-b START_DATE] [-e END_DATE] [-H HEMISPHERE] [-i ICE_TYPE] [-g GRID_TYPE]
          [-t ISPD_THRESH] [-m METHODS] [-B BIN_WINDOW] [-N BIN_MIN_DAYS] [-R ROLL_WINDOW]
          [-G METRIC_GROUPS] [-M METRIC_NAMES] [-u] [-x] [-o] [-P PROJECT] [-U USER] [-L LOG_LEVEL]
          [-f] [-a] [-T] [-A] [-S] [-r REGION] [-n]

Short flags:
  -s  Simulation name
  -b  Start date (YYYY-MM-DD)
  -e  End date (YYYY-MM-DD)
  -H  Hemisphere (SH/NH)
  -i  Ice type (default: FI)
  -g  Grid type / BorC2T type (default: Tc)
  -t  Ice-speed threshold (default: 5e-4)
  -m  Methods, comma-separated
  -B  Binary-days window
  -N  Binary-days minimum valid days
  -R  Rolling-mean window
  -G  Metric groups, comma-separated (default, all, fi_core, si_core, regional, spatial, stress, summary)
  -M  Explicit metric names, comma-separated
  -u  Update missing metrics only
  -o  Overwrite existing metrics store for requested set
  -x  If existing mets.zarr has incompatible indexes, back it up and rebuild it
  -P  Gadi project (default: gv90)
  -U  Username / run owner (default: da1339)
  -L  Log level (default: INFO)
  -f  Plot FIP
  -a  Plot FIA
  -T  Plot FIT
  -A  Plot SIA
  -S  Plot SIT
  -r  Plot region (total, DML, WIO, EIO, Aus, VOL, AS, BS, WS)
  -n  Dry run
EOF
}

while getopts ":s:b:e:H:i:g:t:m:B:N:R:G:M:uxP:U:L:faTASr:onh" opt; do
  case "$opt" in
    s) SIM_NAME="$OPTARG" ;;
    b) START_DATE="$OPTARG" ;;
    e) END_DATE="$OPTARG" ;;
    H) HEMISPHERE="$OPTARG" ;;
    i) ICE_TYPE="$OPTARG" ;;
    g) GRID_TYPE="$OPTARG" ;;
    t) ISPD_THRESH="$OPTARG" ;;
    m) METHODS="$OPTARG" ;;
    B) BIN_WINDOW="$OPTARG" ;;
    N) BIN_MIN_DAYS="$OPTARG" ;;
    R) ROLL_WINDOW="$OPTARG" ;;
    G) METRIC_GROUPS="$OPTARG" ;;
    M) METRIC_NAMES="$OPTARG" ;;
    u) UPDATE_MISSING_ONLY="true" ;;
    x) REBUILD_ON_INDEX_MISMATCH="true" ;;
    P) PROJECT="$OPTARG" ;;
    U) RUN_USER="$OPTARG" ;;
    L) LOG_LEVEL="$OPTARG" ;;
    f) PLOT_FIP="true" ;;
    a) PLOT_FIA="true" ;;
    T) PLOT_FIT="true" ;;
    A) PLOT_SIA="true" ;;
    S) PLOT_SIT="true" ;;
    r) PLOT_REGION="$OPTARG" ;;
    o) OVERWRITE="true"; UPDATE_MISSING_ONLY="false" ;;
    n) DRY_RUN="true" ;;
    h) print_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    :)  echo "Missing argument for -$OPTARG" >&2; print_help; exit 1 ;;
  esac
done

[[ -z "$SIM_NAME" ]] && { echo "Simulation name is required (-s)." >&2; exit 1; }
[[ ! -f "$PBS_SCRIPT" ]] && { echo "PBS script not found: $PBS_SCRIPT" >&2; exit 1; }

JOB_NAME="${SIM_NAME}_${ICE_TYPE}_${GRID_TYPE}_metrics"

# PBS-safe encoding: commas inside a single variable value break qsub -v parsing.
METHODS_SAFE="${METHODS//,/|}"
METRIC_GROUPS_SAFE="${METRIC_GROUPS//,/|}"
METRIC_NAMES_SAFE="${METRIC_NAMES//,/|}"

QSUB_VARS="SIM_NAME=${SIM_NAME},START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},ICE_TYPE=${ICE_TYPE},GRID_TYPE=${GRID_TYPE},ISPD_THRESH=${ISPD_THRESH},METHODS=${METHODS_SAFE},BIN_WINDOW=${BIN_WINDOW},BIN_MIN_DAYS=${BIN_MIN_DAYS},ROLL_WINDOW=${ROLL_WINDOW},PROJECT=${PROJECT},RUN_USER=${RUN_USER},OVERWRITE=${OVERWRITE},UPDATE_MISSING_ONLY=${UPDATE_MISSING_ONLY},REBUILD_ON_INDEX_MISMATCH=${REBUILD_ON_INDEX_MISMATCH},METRIC_GROUPS=${METRIC_GROUPS_SAFE},METRIC_NAMES=${METRIC_NAMES_SAFE},PLOT_FIP=${PLOT_FIP},PLOT_FIA=${PLOT_FIA},PLOT_FIT=${PLOT_FIT},PLOT_SIA=${PLOT_SIA},PLOT_SIT=${PLOT_SIT},PLOT_REGION=${PLOT_REGION},OBS_FIA_VAR=${OBS_FIA_VAR},OBS_FIT_VAR=${OBS_FIT_VAR},LOG_LEVEL=${LOG_LEVEL}"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] qsub -P ${PROJECT} -N ${JOB_NAME} -v ${QSUB_VARS} ${PBS_SCRIPT}"
  exit 0
fi

qsub -P "${PROJECT}" -N "${JOB_NAME}" -v "${QSUB_VARS}" "${PBS_SCRIPT}"
