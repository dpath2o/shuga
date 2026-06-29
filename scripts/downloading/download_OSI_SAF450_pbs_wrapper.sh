#!/bin/bash
set -euo pipefail
usage() {
  cat <<USAGE
Usage: $0 -b YYYY-MM-DD -e YYYY-MM-DD [-r RAW_DIR] [-p PROCESSED_DIR] [-d DATASET_ID] [-o]

Submits OSI-SAF-450 download + SIA processing to Gadi copyq.
USAGE
}
RAW_DIR="/g/data/gv90/da1339/SeaIce/OSI-SAF-450/raw"
PROCESSED_DIR="/g/data/gv90/da1339/SeaIce/OSI-SAF-450/processed"
DATASET_ID="OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS"
OVERWRITE=""
while getopts ":b:e:r:p:d:oh" opt; do
  case "$opt" in
    b) START_DATE="$OPTARG" ;;
    e) END_DATE="$OPTARG" ;;
    r) RAW_DIR="$OPTARG" ;;
    p) PROCESSED_DIR="$OPTARG" ;;
    d) DATASET_ID="$OPTARG" ;;
    o) OVERWRITE="1" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
: "${START_DATE:?Missing -b START_DATE}"
: "${END_DATE:?Missing -e END_DATE}"
qsub -v START_DATE="${START_DATE}",END_DATE="${END_DATE}",RAW_DIR="${RAW_DIR}",PROCESSED_DIR="${PROCESSED_DIR}",DATASET_ID="${DATASET_ID}",OVERWRITE="${OVERWRITE}" \
  ./download_OSI_SAF450.pbs
