#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<USAGE
Usage:
  $0 -b YYYY-MM-DD -e YYYY-MM-DD [options]

Required:
  -b  Start date
  -e  End date

Options:
  -r  Raw NetCDF directory
  -p  Processed-output directory
  -C  Time chunk size                    [default: 8]
  -o  Overwrite processed SIA Zarr
  -h  Show help
USAGE
}

RAW_DIR="/g/data/gv90/da1339/SeaIce/OSI-SAF-450/raw"
PROCESSED_DIR="/g/data/gv90/da1339/SeaIce/OSI-SAF-450/processed"
CHUNKS_TIME=8
OVERWRITE=""

while getopts ":b:e:r:p:C:oh" opt; do
    case "${opt}" in
        b) START_DATE="${OPTARG}" ;;
        e) END_DATE="${OPTARG}" ;;
        r) RAW_DIR="${OPTARG}" ;;
        p) PROCESSED_DIR="${OPTARG}" ;;
        C) CHUNKS_TIME="${OPTARG}" ;;
        o) OVERWRITE=1 ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

: "${START_DATE:?Missing -b START_DATE}"
: "${END_DATE:?Missing -e END_DATE}"

if [[ ! -d "${RAW_DIR}" ]]; then
    echo "ERROR: raw directory does not exist: ${RAW_DIR}" >&2
    exit 1
fi

mapfile -t NETCDF_FILES < <(
    find "${RAW_DIR}" -type f -name '*.nc' -print
)

if [[ "${#NETCDF_FILES[@]}" -eq 0 ]]; then
    echo "ERROR: no NetCDF files found under ${RAW_DIR}" >&2
    exit 1
fi

QSUB_VARS="START_DATE=${START_DATE}"
QSUB_VARS+=",END_DATE=${END_DATE}"
QSUB_VARS+=",RAW_DIR=${RAW_DIR}"
QSUB_VARS+=",PROCESSED_DIR=${PROCESSED_DIR}"
QSUB_VARS+=",CHUNKS_TIME=${CHUNKS_TIME}"
QSUB_VARS+=",OVERWRITE=${OVERWRITE}"

echo "Submitting OSI-SAF-450 processing"
echo "  Dates         : ${START_DATE} to ${END_DATE}"
echo "  Raw directory : ${RAW_DIR}"
echo "  Processed dir : ${PROCESSED_DIR}"
echo "  NetCDF files  : ${#NETCDF_FILES[@]}"
echo "  Time chunk    : ${CHUNKS_TIME}"
echo "  Overwrite     : ${OVERWRITE:-0}"

qsub -v "${QSUB_VARS}" ./process_OSI_SAF450.pbs
