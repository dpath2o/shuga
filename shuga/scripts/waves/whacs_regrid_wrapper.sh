#!/bin/bash
set -euo pipefail

PBS_SCRIPT="${PBS_SCRIPT:-/home/581/da1339/AFIM/src/mawsons-chest/shuga/scripts/waves/whacs_regrid.pbs}"

usage() {
    cat <<EOF
Usage:
  $0 [START_YEAR [END_YEAR]] [options]

Defaults:
  START_YEAR=1995
  END_YEAR=2005

The PBS jobs are charged to au88 by default, while shuga input/output paths
remain under /g/data/gv90 unless --data-project is changed.

The WHACS spectrum is regridded without any NSIDC/model sea-ice mask.
For multi-month production this wrapper submits one ordinary PBS job per month
rather than a PBS array, avoiding Gadi queue/server array-size limits.

Options:
  --account-project NAME    PBS accounting project, default: au88
  --data-project NAME       shuga /g/data project, default: gv90
  --user NAME               default: da1339
  --sim-name NAME           default: LD-waves-exp01
  --target-lat-max FLOAT    default: -35.0
  --k-nearest N             default: 8
  --idw-power FLOAT         default: 2.5
  --radius-km FLOAT         default: 1000.0
  --time-chunk N            default: 4
  --compression-level N     default: 3
  --ow_nc                   rebuild completed output files too
  --ow_wgt                  rebuild station weights
  --dry-run                 print qsub commands only
  -h, --help                show this help

Examples:
  # January 1995 only:
  $0 1995

  # Full 1995--2005 production submission:
  $0 1995 2005
EOF
}

START_YEAR="1995"
END_YEAR="2005"
SINGLE_MONTH="false"
if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
    START_YEAR="$1"
    shift
    if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
        END_YEAR="$1"
        shift
    else
        END_YEAR="${START_YEAR}"
        SINGLE_MONTH="true"
    fi
fi

ACCOUNT_PROJECT="au88"
DATA_PROJECT="gv90"
RUN_USER="da1339"
SIM_NAME="LD-waves-exp01"
TARGET_LAT_MAX="-35.0"
K_NEAREST="8"
IDW_POWER="2.5"
RADIUS_KM="1000.0"
TIME_CHUNK="4"
COMPRESSION_LEVEL="3"
OW_NC="false"
OW_WGT="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account-project) ACCOUNT_PROJECT="$2"; shift 2 ;;
        --data-project) DATA_PROJECT="$2"; shift 2 ;;
        --user) RUN_USER="$2"; shift 2 ;;
        --sim-name) SIM_NAME="$2"; shift 2 ;;
        --target-lat-max) TARGET_LAT_MAX="$2"; shift 2 ;;
        --k-nearest) K_NEAREST="$2"; shift 2 ;;
        --idw-power) IDW_POWER="$2"; shift 2 ;;
        --radius-km) RADIUS_KM="$2"; shift 2 ;;
        --time-chunk) TIME_CHUNK="$2"; shift 2 ;;
        --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
        --ow_nc) OW_NC="true"; shift ;;
        --ow_wgt) OW_WGT="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if (( END_YEAR < START_YEAR )); then
    echo "END_YEAR must be >= START_YEAR" >&2
    exit 2
fi

OUTPUT_ROOT="/g/data/${DATA_PROJECT}/${RUN_USER}/afim_input/CAWCR"
TOTAL_MONTHS=$(( (END_YEAR - START_YEAR + 1) * 12 ))
if [[ "${SINGLE_MONTH}" == "true" ]]; then
    TOTAL_MONTHS=1
fi

echo "Submitting WHACS monthly regridding: ${START_YEAR}--${END_YEAR}"
echo "  accounting project = ${ACCOUNT_PROJECT}"
echo "  data project       = ${DATA_PROJECT}"
echo "  jobs               = ${TOTAL_MONTHS} ordinary monthly PBS job(s)"
echo "  ice mask           = none"
echo "  target latitude    = <= ${TARGET_LAT_MAX} deg"
echo "  station weights    = IDW k=${K_NEAREST}, p=${IDW_POWER}, radius=${RADIUS_KM} km"
echo "  PBS script         = ${PBS_SCRIPT}"

submitted=0
skipped=0

for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    month_start=1
    month_end=12
    if [[ "${SINGLE_MONTH}" == "true" ]]; then
        month_end=1
    fi

    for (( month=month_start; month<=month_end; month++ )); do
        mm="$(printf '%02d' "${month}")"
        out="${OUTPUT_ROOT}/CAWCR_efreq_for_CICE6_${year}${mm}.nc"

        # Avoid knowingly queuing an already-produced month. The Python worker
        # still performs the authoritative provenance/completion check.
        if [[ -f "${out}" && "${OW_NC}" != "true" ]]; then
            echo "  existing ${year}-${mm}: submitting anyway only if worker finds it stale"
        fi

        VARS="YEAR=${year},MONTH_NUM=${month},START_YEAR=${year},END_YEAR=${year},ACCOUNT_PROJECT=${ACCOUNT_PROJECT},DATA_PROJECT=${DATA_PROJECT},RUN_USER=${RUN_USER},SIM_NAME=${SIM_NAME},TARGET_LAT_MAX=${TARGET_LAT_MAX},K_NEAREST=${K_NEAREST},IDW_POWER=${IDW_POWER},RADIUS_KM=${RADIUS_KM},TIME_CHUNK=${TIME_CHUNK},COMPRESSION_LEVEL=${COMPRESSION_LEVEL},OW_NC=${OW_NC},OW_WGT=${OW_WGT}"
        CMD=(qsub -P "${ACCOUNT_PROJECT}" -v "${VARS}" "${PBS_SCRIPT}")

        if [[ "${DRY_RUN}" == "true" ]]; then
            printf 'DRY RUN %04d-%02d: ' "${year}" "${month}"
            printf '%q ' "${CMD[@]}"
            printf '\n'
        else
            printf '  submitting %04d-%02d: ' "${year}" "${month}"
            "${CMD[@]}"
        fi
        submitted=$((submitted + 1))
    done

done

echo "Submission complete: submitted=${submitted}, skipped=${skipped}"
