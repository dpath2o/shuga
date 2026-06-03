#!/bin/bash
set -euo pipefail

# Submit monthly ERA5 -> CICE forcing regrid jobs.
#
# Usage:
#   ./submit_era5_cice_months.sh START_YYYY-MM STOP_YYYY-MM [options]
#
# Defaults:
#   INCLUDE_BL=1
#   OVERWRITE=0
#   REBUILD_WEIGHTS=0
#   REGRID_METHOD=bilinear
#   EXTRAP_METHOD=nearest_s2d
#   CICE_GRID_FILE=/g/data/gv90/da1339/grids/ACCESS-OM3-025_Cgrid.nc
#   PBS_SCRIPT=./build_era5_cice_month.pbs
#
# Examples:
#   # Default production use: assumes weights already exist.
#   ./submit_era5_cice_months.sh 1994-10 1994-12
#
#   # Build weights once on the first month, then release remaining jobs after it succeeds.
#   ./submit_era5_cice_months.sh 1994-10 1994-12 --rebuild-first
#
#   # No boundary-layer optional fields.
#   ./submit_era5_cice_months.sh 1994-10 1994-12 --no-boundary-layer
#
#   # Overwrite existing monthly files.
#   ./submit_era5_cice_months.sh 1994-10 1994-12 --overwrite
#
#   # Dry run only.
#   ./submit_era5_cice_months.sh 1994-10 1994-12 --dry-run

usage() {
  sed -n '1,45p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

parse_ym() {
  local ym="$1"
  [[ "${ym}" =~ ^[0-9]{4}-[0-9]{2}$ ]] || die "Date must be YYYY-MM, got '${ym}'"

  local y="${ym%-*}"
  local m="${ym#*-}"

  (( 10#${m} >= 1 && 10#${m} <= 12 )) || die "Month out of range in '${ym}'"

  # month index from year 0, Jan = y*12 + 0
  echo $((10#${y} * 12 + 10#${m} - 1))
}

ym_from_index() {
  local idx="$1"
  local y=$((idx / 12))
  local m=$((idx % 12 + 1))
  printf "%04d %02d\n" "${y}" "${m}"
}

grid_stem_from_path() {
  local p="$1"
  local b
  b="$(basename "${p}")"
  b="${b%.nc}"
  echo "${b}"
}

START_YM="${1:-}"
STOP_YM="${2:-}"

[[ -n "${START_YM}" ]] || { usage; exit 1; }
[[ -n "${STOP_YM}"  ]] || { usage; exit 1; }

shift 2

PBS_SCRIPT="./build_era5_cice_month.pbs"
INCLUDE_BL=1
OVERWRITE=0
REBUILD_FIRST=0
DRY_RUN=0

REGRID_METHOD="bilinear"
EXTRAP_METHOD="nearest_s2d"
CICE_GRID_FILE="/g/data/gv90/da1339/grids/ACCESS-OM3-025_Cgrid.nc"
WEIGHT_FILE=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pbs-script)
      PBS_SCRIPT="${2:?--pbs-script requires a path}"
      shift 2
      ;;
    --include-boundary-layer)
      INCLUDE_BL=1
      shift
      ;;
    --no-boundary-layer)
      INCLUDE_BL=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    --rebuild-first)
      REBUILD_FIRST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --regrid-method)
      REGRID_METHOD="${2:?--regrid-method requires a value}"
      shift 2
      ;;
    --extrap-method)
      EXTRAP_METHOD="${2:?--extrap-method requires a value}"
      shift 2
      ;;
    --cice-grid-file)
      CICE_GRID_FILE="${2:?--cice-grid-file requires a path}"
      shift 2
      ;;
    --weight-file)
      WEIGHT_FILE="${2:?--weight-file requires a filename}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -f "${PBS_SCRIPT}" ]] || die "PBS script not found: ${PBS_SCRIPT}"
[[ -f "${CICE_GRID_FILE}" ]] || die "CICE grid file not found: ${CICE_GRID_FILE}"

START_IDX="$(parse_ym "${START_YM}")"
STOP_IDX="$(parse_ym "${STOP_YM}")"

(( START_IDX <= STOP_IDX )) || die "START_YYYY-MM must be <= STOP_YYYY-MM"

GRID_STEM="$(grid_stem_from_path "${CICE_GRID_FILE}")"

if [[ -z "${WEIGHT_FILE}" ]]; then
  EXPECTED_WEIGHT_FILE="map_ERA5_to_${GRID_STEM}_${REGRID_METHOD}_${EXTRAP_METHOD}.nc"
else
  EXPECTED_WEIGHT_FILE="${WEIGHT_FILE}"
fi

EXPECTED_WEIGHT_PATH="/g/data/gv90/da1339/grids/weights/${EXPECTED_WEIGHT_FILE}"

if [[ "${REBUILD_FIRST}" == "0" && ! -f "${EXPECTED_WEIGHT_PATH}" ]]; then
  cat >&2 <<EOF
ERROR: Expected weight file does not exist:

  ${EXPECTED_WEIGHT_PATH}

Because --rebuild-first was not requested, this wrapper will not launch
multiple jobs that may race to create the same missing weight file.

Either run the first month manually with REBUILD_WEIGHTS=1, or use:

  $0 ${START_YM} ${STOP_YM} --rebuild-first

EOF
  exit 1
fi

echo "============================================================"
echo "Submit ERA5 -> CICE monthly forcing jobs"
echo "Start month      : ${START_YM}"
echo "Stop month       : ${STOP_YM}"
echo "PBS script       : ${PBS_SCRIPT}"
echo "Include BL       : ${INCLUDE_BL}"
echo "Overwrite        : ${OVERWRITE}"
echo "Rebuild first    : ${REBUILD_FIRST}"
echo "Regrid method    : ${REGRID_METHOD}"
echo "Extrap method    : ${EXTRAP_METHOD}"
echo "CICE grid file   : ${CICE_GRID_FILE}"
echo "Weight file      : ${EXPECTED_WEIGHT_FILE}"
echo "Weight path      : ${EXPECTED_WEIGHT_PATH}"
echo "Dry run          : ${DRY_RUN}"
echo "============================================================"

first_job_id=""
submitted_count=0

for idx in $(seq "${START_IDX}" "${STOP_IDX}"); do
  read -r YEAR MONTH < <(ym_from_index "${idx}")

  REBUILD_WEIGHTS=0
  depend_args=()

  if [[ "${REBUILD_FIRST}" == "1" && "${idx}" == "${START_IDX}" ]]; then
    REBUILD_WEIGHTS=1
  elif [[ "${REBUILD_FIRST}" == "1" && -n "${first_job_id}" ]]; then
    depend_args=(-W "depend=afterok:${first_job_id}")
  fi

  varlist="YEAR=${YEAR},MONTH=${MONTH},OVERWRITE=${OVERWRITE},INCLUDE_BL=${INCLUDE_BL},REBUILD_WEIGHTS=${REBUILD_WEIGHTS},REGRID_METHOD=${REGRID_METHOD},EXTRAP_METHOD=${EXTRAP_METHOD},CICE_GRID_FILE=${CICE_GRID_FILE}"

  if [[ -n "${WEIGHT_FILE}" ]]; then
    varlist="${varlist},WEIGHT_FILE=${WEIGHT_FILE}"
  fi

  echo
  echo "Submitting ${YEAR}-${MONTH}"
  echo "  REBUILD_WEIGHTS=${REBUILD_WEIGHTS}"
  if [[ "${#depend_args[@]}" -gt 0 ]]; then
    echo "  Dependency      : ${depend_args[*]}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  qsub -v \"${varlist}\" ${depend_args[*]} \"${PBS_SCRIPT}\""
    continue
  fi

  job_id="$(qsub -v "${varlist}" "${depend_args[@]}" "${PBS_SCRIPT}")"
  echo "  Submitted job   : ${job_id}"

  if [[ "${idx}" == "${START_IDX}" ]]; then
    first_job_id="${job_id}"
  fi

  submitted_count=$((submitted_count + 1))
done

echo
echo "============================================================"
echo "Submitted ${submitted_count} jobs"
if [[ "${REBUILD_FIRST}" == "1" && -n "${first_job_id}" ]]; then
  echo "First/rebuild job : ${first_job_id}"
  echo "Remaining jobs depend on successful completion of first job."
fi
echo "============================================================"
