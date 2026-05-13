#!/bin/bash
set -euo pipefail

SIM_NAME=""
START_DATE="1993-01-01"
END_DATE="1999-12-31"
HEMISPHERE="SH"
ICE_TYPE="FI"
GRID_TYPE="Tc"
ISPD_THRESH="5e-4"
METHODS_CSV="raw,binary-days,rolling-mean"
BIN_WINDOW="11"
BIN_MIN_DAYS="9"
ROLL_WINDOW="15"
ICEH_FREQUENCY="daily"
DAILY_ROOT=""
HOURLY_ROOT=""
CHUNKS_TIME=""
SKIP_HISTORY_CONVERSION="false"
OVERWRITE_HISTORY="false"
OVERWRITE_STATIC="false"
DELETE_ORIGINAL="false"
NETCDF_ENGINE="scipy"
AFIM_OUTPUT_ROOT=""
CICE_STORE=""
STATIC_STORE=""
CLASSIFICATION_ROOT=""
ARCHIVE_ROOT=""
LOGS_ROOT=""
PROJECT="gv90"
RUN_USER="da1339"
OVERWRITE="false"
DRY_RUN="false"
GRID_FILE=""
KMT_FILE=""
BATHYMETRY_FILE=""
F2_FILE=""
GRIDCPL_FILE=""
ICE_IN_FILE=""
PERSIST_GRID_ASSETS="false"

usage() {
    cat <<'EOF'
Usage: ./classify_pbs_wrapper.sh -s SIM_NAME [-b START_DATE] [-e END_DATE] [-H HEMISPHERE] [-i ICE_TYPE] [-g GRID_TYPE]
          [-t ISPD_THRESH] [-m METHODS] [-B BIN_WINDOW] [-N BIN_MIN_DAYS] [-R ROLL_WINDOW]
          [-P PROJECT] [-U USER] [-G GRID_FILE] [-K KMT_FILE] [-A BATHYMETRY_FILE] [-F F2_FILE]
          [-C GRIDCPL_FILE] [-I ICE_IN_FILE] [-S] [-o] [-n]
          [--iceh-frequency daily|hourly] [--daily-root DIR] [--hourly-root DIR]
          [--chunks-time N] [--skip-history-conversion]
          [--overwrite-history] [--overwrite-static] [--delete-original]
          [--netcdf-engine ENGINE]
          [--afim-output-root DIR] [--cice-store DIR] [--static-store DIR]
          [--classification-root DIR] [--archive-root DIR] [--logs-root DIR]

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
  -P  Gadi project (default: gv90)
  -U  Username / run owner (default: da1339)
  -G  Non-default CICE grid file
  -K  Non-default CICE KMT file
  -A  Optional bathymetry file
  -F  Optional F2 file
  -C  Optional gridcpl file
  -I  Optional ice_in file
  -S  Persist chosen grid assets for downstream metrics / plotting
  -o  Overwrite existing outputs
  -n  Dry run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SIM_NAME="$2"; shift 2 ;;
        -b) START_DATE="$2"; shift 2 ;;
        -e) END_DATE="$2"; shift 2 ;;
        -H) HEMISPHERE="$2"; shift 2 ;;
        -i) ICE_TYPE="$2"; shift 2 ;;
        -g) GRID_TYPE="$2"; shift 2 ;;
        -t) ISPD_THRESH="$2"; shift 2 ;;
        -m) METHODS_CSV="$2"; shift 2 ;;
        -B) BIN_WINDOW="$2"; shift 2 ;;
        -N) BIN_MIN_DAYS="$2"; shift 2 ;;
        -R) ROLL_WINDOW="$2"; shift 2 ;;
        -P) PROJECT="$2"; shift 2 ;;
        -U) RUN_USER="$2"; shift 2 ;;
        -G) GRID_FILE="$2"; shift 2 ;;
        -K) KMT_FILE="$2"; shift 2 ;;
        -A) BATHYMETRY_FILE="$2"; shift 2 ;;
        -F) F2_FILE="$2"; shift 2 ;;
        -C) GRIDCPL_FILE="$2"; shift 2 ;;
        -I) ICE_IN_FILE="$2"; shift 2 ;;
        -S) PERSIST_GRID_ASSETS="true"; shift ;;
        -o) OVERWRITE="true"; shift ;;
        -n) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        --iceh-frequency) ICEH_FREQUENCY="$2"; shift 2 ;;
        --daily-root) DAILY_ROOT="$2"; shift 2 ;;
        --hourly-root) HOURLY_ROOT="$2"; shift 2 ;;
        --chunks-time) CHUNKS_TIME="$2"; shift 2 ;;
        --skip-history-conversion) SKIP_HISTORY_CONVERSION="true"; shift ;;
        --overwrite-history) OVERWRITE_HISTORY="true"; shift ;;
        --overwrite-static) OVERWRITE_STATIC="true"; shift ;;
        --delete-original) DELETE_ORIGINAL="true"; shift ;;
        --netcdf-engine) NETCDF_ENGINE="$2"; shift 2 ;;
        --afim-output-root) AFIM_OUTPUT_ROOT="$2"; shift 2 ;;
        --cice-store) CICE_STORE="$2"; shift 2 ;;
        --static-store) STATIC_STORE="$2"; shift 2 ;;
        --classification-root) CLASSIFICATION_ROOT="$2"; shift 2 ;;
        --archive-root) ARCHIVE_ROOT="$2"; shift 2 ;;
        --logs-root) LOGS_ROOT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$SIM_NAME" ]]; then
    echo "Error: -s SIM_NAME is required" >&2
    usage
    exit 1
fi

case "$ICEH_FREQUENCY" in
    daily|hourly) ;;
    *)
        echo "Error: --iceh-frequency must be 'daily' or 'hourly'; got '$ICEH_FREQUENCY'" >&2
        exit 1
        ;;
esac

PBS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/classify.pbs"

# PBS-safe encoding: commas are separators in qsub -v, so do not pass METHODS with commas.
METHODS_SAFE="${METHODS_CSV//,/|}"
QSUB_VARS="SIM_NAME=${SIM_NAME},START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},ICE_TYPE=${ICE_TYPE},GRID_TYPE=${GRID_TYPE},ISPD_THRESH=${ISPD_THRESH},METHODS=${METHODS_SAFE},BIN_WINDOW=${BIN_WINDOW},BIN_MIN_DAYS=${BIN_MIN_DAYS},ROLL_WINDOW=${ROLL_WINDOW},PROJECT=${PROJECT},RUN_USER=${RUN_USER},OVERWRITE=${OVERWRITE},ICEH_FREQUENCY=${ICEH_FREQUENCY},CHUNKS_TIME=${CHUNKS_TIME},OVERWRITE_HISTORY=${OVERWRITE_HISTORY},OVERWRITE_STATIC=${OVERWRITE_STATIC},DELETE_ORIGINAL=${DELETE_ORIGINAL},SKIP_HISTORY_CONVERSION=${SKIP_HISTORY_CONVERSION},NETCDF_ENGINE=${NETCDF_ENGINE}"
[[ -n "$GRID_FILE" ]] && QSUB_VARS+=",GRID_FILE=${GRID_FILE}"
[[ -n "$KMT_FILE" ]] && QSUB_VARS+=",KMT_FILE=${KMT_FILE}"
[[ -n "$BATHYMETRY_FILE" ]] && QSUB_VARS+=",BATHYMETRY_FILE=${BATHYMETRY_FILE}"
[[ -n "$F2_FILE" ]] && QSUB_VARS+=",F2_FILE=${F2_FILE}"
[[ -n "$GRIDCPL_FILE" ]] && QSUB_VARS+=",GRIDCPL_FILE=${GRIDCPL_FILE}"
[[ -n "$ICE_IN_FILE" ]] && QSUB_VARS+=",ICE_IN_FILE=${ICE_IN_FILE}"
[[ -n "$DAILY_ROOT" ]] && QSUB_VARS+=",DAILY_ROOT=${DAILY_ROOT}"
[[ -n "$HOURLY_ROOT" ]] && QSUB_VARS+=",HOURLY_ROOT=${HOURLY_ROOT}"
[[ -n "$AFIM_OUTPUT_ROOT" ]] && QSUB_VARS+=",AFIM_OUTPUT_ROOT=${AFIM_OUTPUT_ROOT}"
[[ -n "$CICE_STORE" ]] && QSUB_VARS+=",CICE_STORE=${CICE_STORE}"
[[ -n "$STATIC_STORE" ]] && QSUB_VARS+=",STATIC_STORE=${STATIC_STORE}"
[[ -n "$CLASSIFICATION_ROOT" ]] && QSUB_VARS+=",CLASSIFICATION_ROOT=${CLASSIFICATION_ROOT}"
[[ -n "$ARCHIVE_ROOT" ]] && QSUB_VARS+=",ARCHIVE_ROOT=${ARCHIVE_ROOT}"
[[ -n "$LOGS_ROOT" ]] && QSUB_VARS+=",LOGS_ROOT=${LOGS_ROOT}"
[[ "$PERSIST_GRID_ASSETS" == "true" ]] && QSUB_VARS+=",PERSIST_GRID_ASSETS=true"

JOB_NAME="${SIM_NAME}_${ICE_TYPE}_${GRID_TYPE}_${ICEH_FREQUENCY}_classify"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY_RUN] qsub -N ${JOB_NAME} -v ${QSUB_VARS} ${PBS_SCRIPT}"
    exit 0
fi

qsub -N "${JOB_NAME}" -v "${QSUB_VARS}" "${PBS_SCRIPT}"
