#!/bin/bash
set -euo pipefail
usage() {
  cat <<USAGE
Usage: $0 -s "SIM1 SIM2" -b YYYY-MM-DD -e YYYY-MM-DD -n NSIDC_SIA_STORE [-o OSISAF_SIA_STORE] [-d OUT_DIR] [-- extra args]
USAGE
}
OSISAF_SIA_STORE="/g/data/gv90/da1339/observations/OSI-SAF-450/processed/OSI-SAF-450_SH_SIA.zarr"
OUT_DIR="/g/data/gv90/da1339/GRAPHICAL/AFIM/SIA_comparisons"
while getopts ":s:b:e:n:o:d:h" opt; do
  case "$opt" in
    s) SIM_NAMES="$OPTARG" ;;
    b) START_DATE="$OPTARG" ;;
    e) END_DATE="$OPTARG" ;;
    n) NSIDC_SIA_STORE="$OPTARG" ;;
    o) OSISAF_SIA_STORE="$OPTARG" ;;
    d) OUT_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))
EXTRA_ARGS="$*"
: "${SIM_NAMES:?Missing -s SIM_NAMES}"
: "${START_DATE:?Missing -b START_DATE}"
: "${END_DATE:?Missing -e END_DATE}"
: "${NSIDC_SIA_STORE:?Missing -n NSIDC_SIA_STORE}"
qsub -v SIM_NAMES="${SIM_NAMES}",START_DATE="${START_DATE}",END_DATE="${END_DATE}",NSIDC_SIA_STORE="${NSIDC_SIA_STORE}",OSISAF_SIA_STORE="${OSISAF_SIA_STORE}",OUT_DIR="${OUT_DIR}",EXTRA_ARGS="${EXTRA_ARGS}" \
  shuga/scripts/comparisons/SIA_timeseries_tables.pbs
