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
PBS_SCRIPT="metrics.pbs"
OVERWRITE="false"
DRY_RUN="false"
PLOT_FIP="false"
PLOT_FIA="false"
PLOT_FIT="false"
PLOT_REGION="total"
OBS_FIA_VAR="FIA"
OBS_FIT_VAR="FIT"
LOG_LEVEL="INFO"

print_help() {
  cat <<EOF
Usage: $0 -s SIM_NAME [-b START_DATE] [-e END_DATE] [-H HEMISPHERE] [-i ICE_TYPE] [-g GRID_TYPE]
          [-t ISPD_THRESH] [-m METHODS] [-B BIN_WINDOW] [-N BIN_MIN_DAYS] [-R ROLL_WINDOW]
          [-P PROJECT] [-U USER] [-f] [-a] [-T] [-r REGION] [-o] [-n]

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
  -f  Plot FIP
  -a  Plot FIA
  -T  Plot FIT
  -r  Plot region (total, DML, WIO, EIO, Aus, VOL, AS, BS, WS)
  -o  Overwrite existing outputs
  -n  Dry run
EOF
}

while getopts "s:b:e:H:i:g:t:m:B:N:R:P:U:faTr:onh" opt; do
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
    P) PROJECT="$OPTARG" ;;
    U) RUN_USER="$OPTARG" ;;
    f) PLOT_FIP="true" ;;
    a) PLOT_FIA="true" ;;
    T) PLOT_FIT="true" ;;
    r) PLOT_REGION="$OPTARG" ;;
    o) OVERWRITE="true" ;;
    n) DRY_RUN="true" ;;
    h) print_help; exit 0 ;;
    *) print_help; exit 1 ;;
  esac
done

[[ -z "$SIM_NAME" ]] && { echo "Simulation name is required (-s)." >&2; exit 1; }

JOB_NAME="${SIM_NAME}_${ICE_TYPE}_${GRID_TYPE}_metrics"

QSUB_ARGS=(
  -P "$PROJECT"
  -N "$JOB_NAME"
  -v "SIM_NAME=$SIM_NAME,START_DATE=$START_DATE,END_DATE=$END_DATE,HEMISPHERE=$HEMISPHERE,ICE_TYPE=$ICE_TYPE,GRID_TYPE=$GRID_TYPE,ISPD_THRESH=$ISPD_THRESH,METHODS=$METHODS,BIN_WINDOW=$BIN_WINDOW,BIN_MIN_DAYS=$BIN_MIN_DAYS,ROLL_WINDOW=$ROLL_WINDOW,PROJECT=$PROJECT,RUN_USER=$RUN_USER,OVERWRITE=$OVERWRITE,PLOT_FIP=$PLOT_FIP,PLOT_FIA=$PLOT_FIA,PLOT_FIT=$PLOT_FIT,PLOT_REGION=$PLOT_REGION,OBS_FIA_VAR=$OBS_FIA_VAR,OBS_FIT_VAR=$OBS_FIT_VAR,LOG_LEVEL=$LOG_LEVEL"
  "$PBS_SCRIPT"
)

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] qsub ${QSUB_ARGS[*]}"
else
  qsub "${QSUB_ARGS[@]}"
fi
