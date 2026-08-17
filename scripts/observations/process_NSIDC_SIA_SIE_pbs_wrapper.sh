#!/usr/bin/env bash
set -euo pipefail
usage(){ echo "Usage: $0 -b YYYY-MM-DD -e YYYY-MM-DD [-H SH|NH] [-r SEAICE_ROOT] [-v NSIDC_VERSION] [-t THRESHOLD] [-C CHUNKS] [-p OUTPUT_STORE] [-o]"; }
SEAICE_ROOT="/g/data/gv90/da1339/SeaIce";
NSIDC_VERSION="G02202_V6";
HEMISPHERE="SH";
THRESHOLD="0.15";
CHUNKS_TIME=31;
OUTPUT_STORE="";
OVERWRITE=""
while getopts ":b:e:H:r:v:t:C:p:oh" opt; do
 case "$opt" in
  b) START_DATE="$OPTARG";;
  e) END_DATE="$OPTARG";;
  H) HEMISPHERE="$OPTARG";;
  r) SEAICE_ROOT="$OPTARG";;
  v) NSIDC_VERSION="$OPTARG";;
  t) THRESHOLD="$OPTARG";;
  C) CHUNKS_TIME="$OPTARG";;
  p) OUTPUT_STORE="$OPTARG";;
  o) OVERWRITE=1;;
  h) usage; exit 0;;
  *) usage; exit 2;;
 esac
done
: "${START_DATE:?Missing -b}"; : "${END_DATE:?Missing -e}"
QSUB_VARS="START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},SEAICE_ROOT=${SEAICE_ROOT},NSIDC_VERSION=${NSIDC_VERSION},THRESHOLD=${THRESHOLD},CHUNKS_TIME=${CHUNKS_TIME},OUTPUT_STORE=${OUTPUT_STORE},OVERWRITE=${OVERWRITE}"
echo "Submitting NSIDC SIA/SIE processing: ${START_DATE} to ${END_DATE} (${HEMISPHERE})"
qsub -v "${QSUB_VARS}" ./process_NSIDC_SIA_SIE.pbs
