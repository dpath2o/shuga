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

while getopts ":s:b:e:H:i:g:t:m:B:N:R:P:U:G:K:A:F:C:I:Sonh" opt; do
    case "$opt" in
        s) SIM_NAME="$OPTARG" ;;
        b) START_DATE="$OPTARG" ;;
        e) END_DATE="$OPTARG" ;;
        H) HEMISPHERE="$OPTARG" ;;
        i) ICE_TYPE="$OPTARG" ;;
        g) GRID_TYPE="$OPTARG" ;;
        t) ISPD_THRESH="$OPTARG" ;;
        m) METHODS_CSV="$OPTARG" ;;
        B) BIN_WINDOW="$OPTARG" ;;
        N) BIN_MIN_DAYS="$OPTARG" ;;
        R) ROLL_WINDOW="$OPTARG" ;;
        P) PROJECT="$OPTARG" ;;
        U) RUN_USER="$OPTARG" ;;
        G) GRID_FILE="$OPTARG" ;;
        K) KMT_FILE="$OPTARG" ;;
        A) BATHYMETRY_FILE="$OPTARG" ;;
        F) F2_FILE="$OPTARG" ;;
        C) GRIDCPL_FILE="$OPTARG" ;;
        I) ICE_IN_FILE="$OPTARG" ;;
        S) PERSIST_GRID_ASSETS="true" ;;
        o) OVERWRITE="true" ;;
        n) DRY_RUN="true" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
        :)  echo "Missing argument for -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$SIM_NAME" ]]; then
    echo "Error: -s SIM_NAME is required" >&2
    usage
    exit 1
fi

PBS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/classify.pbs"

# PBS-safe encoding: commas are separators in qsub -v, so do not pass METHODS with commas.
METHODS_SAFE="${METHODS_CSV//,/|}"

QSUB_VARS="SIM_NAME=${SIM_NAME},START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},ICE_TYPE=${ICE_TYPE},GRID_TYPE=${GRID_TYPE},ISPD_THRESH=${ISPD_THRESH},METHODS=${METHODS_SAFE},BIN_WINDOW=${BIN_WINDOW},BIN_MIN_DAYS=${BIN_MIN_DAYS},ROLL_WINDOW=${ROLL_WINDOW},PROJECT=${PROJECT},RUN_USER=${RUN_USER},OVERWRITE=${OVERWRITE}"
[[ -n "$GRID_FILE" ]] && QSUB_VARS+=",GRID_FILE=${GRID_FILE}"
[[ -n "$KMT_FILE" ]] && QSUB_VARS+=",KMT_FILE=${KMT_FILE}"
[[ -n "$BATHYMETRY_FILE" ]] && QSUB_VARS+=",BATHYMETRY_FILE=${BATHYMETRY_FILE}"
[[ -n "$F2_FILE" ]] && QSUB_VARS+=",F2_FILE=${F2_FILE}"
[[ -n "$GRIDCPL_FILE" ]] && QSUB_VARS+=",GRIDCPL_FILE=${GRIDCPL_FILE}"
[[ -n "$ICE_IN_FILE" ]] && QSUB_VARS+=",ICE_IN_FILE=${ICE_IN_FILE}"
[[ "$PERSIST_GRID_ASSETS" == "true" ]] && QSUB_VARS+=",PERSIST_GRID_ASSETS=true"

JOB_NAME="${SIM_NAME}_${ICE_TYPE}_${GRID_TYPE}_classify"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY_RUN] qsub -N ${JOB_NAME} -v ${QSUB_VARS} ${PBS_SCRIPT}"
    exit 0
fi

qsub -N "${JOB_NAME}" -v "${QSUB_VARS}" "${PBS_SCRIPT}"
