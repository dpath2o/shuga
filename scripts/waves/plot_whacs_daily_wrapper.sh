#!/bin/bash
set -euo pipefail

PBS_SCRIPT="${PBS_SCRIPT:-/home/581/da1339/AFIM/src/mawsons-chest/shuga/scripts/waves/plot_whacs_daily.pbs}"

usage() {
    cat <<EOF
Usage:
  $0 [START_YEAR [END_YEAR]] [options]

Defaults:
  START_YEAR=1995
  END_YEAR=2005

For a single year argument, submits January of that year as a normal non-array
pilot job. For a year range, submits one PBS array element per month.

Options:
  --account-project NAME    PBS accounting project, default: au88
  --data-project NAME       shuga /g/data project, default: gv90
  --user NAME               default: da1339
  --sim-name NAME           default: LD-waves-exp01
  --north FLOAT             northern plot limit, default: -35.0
  --grid-stride N           plot every Nth CICE grid cell, default: 3
  --hs-max FLOAT            optional fixed Hs colour maximum (m)
  --tp-max FLOAT            optional fixed Tp colour maximum (s)
  --dpi N                   output PNG dpi, default: 300
  --overwrite               recreate existing daily figures
  --dry-run                 print qsub command only
  -h, --help                show this help

Examples:
  # January 1995 pilot:
  $0 1995

  # Full 1995--2005 monthly plotting array:
  $0 1995 2005
EOF
}

START_YEAR="1995"
END_YEAR="2005"
PILOT_SINGLE_YEAR="false"
if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
    START_YEAR="$1"
    shift
    if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
        END_YEAR="$1"
        shift
    else
        END_YEAR="${START_YEAR}"
        PILOT_SINGLE_YEAR="true"
    fi
fi

ACCOUNT_PROJECT="au88"
DATA_PROJECT="gv90"
RUN_USER="da1339"
SIM_NAME="LD-waves-exp01"
NORTH="-35.0"
GRID_STRIDE="3"
HS_MAX=""
TP_MAX=""
DPI="300"
OVERWRITE="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account-project) ACCOUNT_PROJECT="$2"; shift 2 ;;
        --data-project) DATA_PROJECT="$2"; shift 2 ;;
        --user) RUN_USER="$2"; shift 2 ;;
        --sim-name) SIM_NAME="$2"; shift 2 ;;
        --north) NORTH="$2"; shift 2 ;;
        --grid-stride) GRID_STRIDE="$2"; shift 2 ;;
        --hs-max) HS_MAX="$2"; shift 2 ;;
        --tp-max) TP_MAX="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        --overwrite) OVERWRITE="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if (( END_YEAR < START_YEAR )); then
    echo "END_YEAR must be >= START_YEAR" >&2
    exit 2
fi

VARS="START_YEAR=${START_YEAR},END_YEAR=${END_YEAR},DATA_PROJECT=${DATA_PROJECT},RUN_USER=${RUN_USER},SIM_NAME=${SIM_NAME},NORTH=${NORTH},GRID_STRIDE=${GRID_STRIDE},HS_MAX=${HS_MAX},TP_MAX=${TP_MAX},DPI=${DPI},OVERWRITE=${OVERWRITE}"

if [[ "${PILOT_SINGLE_YEAR}" == "true" ]]; then
    CMD=(qsub -P "${ACCOUNT_PROJECT}" -v "${VARS}" "${PBS_SCRIPT}")
    DESC="January ${START_YEAR} non-array pilot"
else
    TOTAL_MONTHS=$(( (END_YEAR - START_YEAR + 1) * 12 ))
    LAST_INDEX=$(( TOTAL_MONTHS - 1 ))
    CMD=(qsub -P "${ACCOUNT_PROJECT}" -J "0-${LAST_INDEX}" -v "${VARS}" "${PBS_SCRIPT}")
    DESC="${START_YEAR}--${END_YEAR}, array 0-${LAST_INDEX}"
fi

echo "Submitting WHACS daily plotting: ${DESC}"
echo "  accounting project = ${ACCOUNT_PROJECT}"
echo "  data project       = ${DATA_PROJECT}"
echo "  north limit        = ${NORTH}"
echo "  grid stride        = ${GRID_STRIDE}"
echo "  Hs max             = ${HS_MAX:-auto}"
echo "  Tp max             = ${TP_MAX:-auto}"

if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
