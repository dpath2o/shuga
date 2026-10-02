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
  -r  Raw download directory
  -d  Copernicus dataset ID
  -o  Overwrite an existing downloaded file
  -h  Show help
USAGE
}

RAW_DIR="/g/data/gv90/da1339/SeaIce/OSI-SAF-450/raw"
DATASET_ID="OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS"
OVERWRITE=""

while getopts ":b:e:r:d:oh" opt; do
    case "${opt}" in
        b) START_DATE="${OPTARG}" ;;
        e) END_DATE="${OPTARG}" ;;
        r) RAW_DIR="${OPTARG}" ;;
        d) DATASET_ID="${OPTARG}" ;;
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

QSUB_VARS="START_DATE=${START_DATE}"
QSUB_VARS+=",END_DATE=${END_DATE}"
QSUB_VARS+=",RAW_DIR=${RAW_DIR}"
QSUB_VARS+=",DATASET_ID=${DATASET_ID}"
QSUB_VARS+=",OVERWRITE=${OVERWRITE}"

echo "Submitting OSI-SAF-450 download"
echo "  Dates         : ${START_DATE} to ${END_DATE}"
echo "  Raw directory : ${RAW_DIR}"
echo "  Dataset       : ${DATASET_ID}"
echo "  Overwrite     : ${OVERWRITE:-0}"

qsub -v "${QSUB_VARS}" ./download_OSI_SAF450.pbs
