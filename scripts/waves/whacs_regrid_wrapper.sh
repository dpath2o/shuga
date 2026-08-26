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

Submits one PBS array element per calendar month. For 1995--2005 inclusive
this is 132 monthly jobs. Array concurrency defaults to 2 to avoid saturating
WHACS/gv90 I/O and can be increased after a pilot month succeeds.

Options:
  --max-concurrent N        PBS array concurrency, default: 2
  --project NAME            default: gv90
  --user NAME               default: da1339
  --sim-name NAME           default: LD-waves-exp01
  --target-lat-max FLOAT    default: -45.0
  --sic-threshold FLOAT     default: 0.15
  --k-nearest N             default: 8
  --idw-power FLOAT         default: 2.5
  --radius-km FLOAT         default: 1000.0
  --time-chunk N            default: 1
  --compression-level N     default: 3
  --ow_nc                   rebuild completed output files too
  --ow_wgt                  rebuild station weights (not recommended for a multi-month array)
  --ow_sic_wgt              rebuild NSIDC weights (not recommended for a multi-month array)
  --dry-run                 print qsub command only
  -h, --help                show this help

Examples:
  # Pilot January 1995 before the full array:
  qsub -J 0-0 -v START_YEAR=1995,END_YEAR=1995 ${PBS_SCRIPT}

  # Full 1995--2005 run, two months concurrently:
  $0 1995 2005 --max-concurrent 2
EOF
}

START_YEAR="1995"
END_YEAR="2005"
if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
    START_YEAR="$1"
    shift
fi
if [[ $# -gt 0 && "$1" =~ ^[0-9]{4}$ ]]; then
    END_YEAR="$1"
    shift
else
    END_YEAR="${START_YEAR}"
fi

PROJECT="gv90"
RUN_USER="da1339"
SIM_NAME="LD-waves-exp01"
MAX_CONCURRENT="2"
TARGET_LAT_MAX="-45.0"
SIC_THRESHOLD="0.15"
K_NEAREST="8"
IDW_POWER="2.5"
RADIUS_KM="1000.0"
TIME_CHUNK="1"
COMPRESSION_LEVEL="3"
OW_NC="false"
OW_WGT="false"
OW_SIC_WGT="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --user) RUN_USER="$2"; shift 2 ;;
        --sim-name) SIM_NAME="$2"; shift 2 ;;
        --target-lat-max) TARGET_LAT_MAX="$2"; shift 2 ;;
        --sic-threshold) SIC_THRESHOLD="$2"; shift 2 ;;
        --k-nearest) K_NEAREST="$2"; shift 2 ;;
        --idw-power) IDW_POWER="$2"; shift 2 ;;
        --radius-km) RADIUS_KM="$2"; shift 2 ;;
        --time-chunk) TIME_CHUNK="$2"; shift 2 ;;
        --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
        --ow_nc) OW_NC="true"; shift ;;
        --ow_wgt) OW_WGT="true"; shift ;;
        --ow_sic_wgt) OW_SIC_WGT="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if (( END_YEAR < START_YEAR )); then
    echo "END_YEAR must be >= START_YEAR" >&2
    exit 2
fi
if (( MAX_CONCURRENT < 1 )); then
    echo "--max-concurrent must be >= 1" >&2
    exit 2
fi
if [[ "${OW_WGT}" == "true" || "${OW_SIC_WGT}" == "true" ]]; then
    echo "WARNING: overwrite-weight flags apply to every array element; normally build shared weights once and leave these false." >&2
fi

TOTAL_MONTHS=$(( (END_YEAR - START_YEAR + 1) * 12 ))
LAST_INDEX=$(( TOTAL_MONTHS - 1 ))
ARRAY_SPEC="0-${LAST_INDEX}%${MAX_CONCURRENT}"

VARS="START_YEAR=${START_YEAR},END_YEAR=${END_YEAR},PROJECT=${PROJECT},RUN_USER=${RUN_USER},SIM_NAME=${SIM_NAME},TARGET_LAT_MAX=${TARGET_LAT_MAX},SIC_THRESHOLD=${SIC_THRESHOLD},K_NEAREST=${K_NEAREST},IDW_POWER=${IDW_POWER},RADIUS_KM=${RADIUS_KM},TIME_CHUNK=${TIME_CHUNK},COMPRESSION_LEVEL=${COMPRESSION_LEVEL},OW_NC=${OW_NC},OW_WGT=${OW_WGT},OW_SIC_WGT=${OW_SIC_WGT}"

CMD=(qsub -J "${ARRAY_SPEC}" -v "${VARS}" "${PBS_SCRIPT}")

echo "Submitting WHACS monthly regridding: ${START_YEAR}--${END_YEAR}"
echo "  months            = ${TOTAL_MONTHS}"
echo "  array             = ${ARRAY_SPEC}"
echo "  max concurrent    = ${MAX_CONCURRENT}"
echo "  target latitude   = <= ${TARGET_LAT_MAX} deg"
echo "  station weights   = IDW k=${K_NEAREST}, p=${IDW_POWER}, radius=${RADIUS_KM} km"
echo "  PBS script        = ${PBS_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
