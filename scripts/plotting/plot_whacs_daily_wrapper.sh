#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PBS_SCRIPT="${PBS_SCRIPT:-${SCRIPT_DIR}/plot_whacs_daily.pbs}"

usage() {
    cat <<EOF
Usage:
  $0 [START_YEAR [END_YEAR]] [options]

Defaults:
  START_YEAR=1995
  END_YEAR=2005

Submission behaviour:
  * A single year argument (e.g. 1995) submits January of that year only.
  * A year range (e.g. 1995 2005) checks each month and submits one ordinary
    PBS job for every regridded WHACS file that already exists.
  * No PBS arrays are used.

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
  --dry-run                 print qsub command(s) only
  -h, --help                show this help

Examples:
  # January 1995 pilot:
  $0 1995 --north -50

  # Submit plotting for every available regridded month in 1995:
  $0 1995 1995 --north -50

  # Submit plotting for every available regridded month from 1995--2005:
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

REGRID_ROOT="/g/data/${DATA_PROJECT}/${RUN_USER}/afim_input/CAWCR"

submit_month() {
    local year="$1"
    local month_num="$2"
    local month
    month="$(printf '%02d' "${month_num}")"
    local reg_file="${REGRID_ROOT}/CAWCR_efreq_for_CICE6_${year}${month}.nc"

    if [[ ! -s "${reg_file}" ]]; then
        echo "skipping ${year}-${month}: no regridded file: ${reg_file}"
        return 0
    fi

    local vars
    vars="YEAR=${year},MONTH_NUM=${month_num},DATA_PROJECT=${DATA_PROJECT},RUN_USER=${RUN_USER},SIM_NAME=${SIM_NAME},NORTH=${NORTH},GRID_STRIDE=${GRID_STRIDE},HS_MAX=${HS_MAX},TP_MAX=${TP_MAX},DPI=${DPI},OVERWRITE=${OVERWRITE},SCRIPT_ROOT=${SCRIPT_ROOT},REGRID_ROOT=${REGRID_ROOT}"
    local cmd=(qsub -P "${ACCOUNT_PROJECT}" -v "${vars}" "${PBS_SCRIPT}")

    if [[ "${DRY_RUN}" == "true" ]]; then
        printf 'DRY RUN %s-%s: ' "${year}" "${month}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        printf 'submitting %s-%s: ' "${year}" "${month}"
        "${cmd[@]}"
    fi
}

echo "WHACS daily plotting submission"
echo "  accounting project = ${ACCOUNT_PROJECT}"
echo "  data project       = ${DATA_PROJECT}"
echo "  regridded root     = ${REGRID_ROOT}"
echo "  north limit        = ${NORTH}"
echo "  grid stride        = ${GRID_STRIDE}"
echo "  Hs max             = ${HS_MAX:-auto}"
echo "  Tp max             = ${TP_MAX:-auto}"
echo "  PBS arrays         = disabled"

if [[ "${PILOT_SINGLE_YEAR}" == "true" ]]; then
    submit_month "${START_YEAR}" 1
else
    for (( year=START_YEAR; year<=END_YEAR; year++ )); do
        for month_num in {1..12}; do
            submit_month "${year}" "${month_num}"
        done
    done
fi
