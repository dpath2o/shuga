#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./classify_CMEMS_pbs_wrapper.sh -b YYYY-MM-DD -e YYYY-MM-DD [options]

Options:
  -b DATE   start date (required)
  -e DATE   end date (required)
  -H HEM    SH or NH (default SH)
  -m LIST   methods, comma-separated (default raw,binary-days,rolling-mean)
  -s VALUE  ice-speed threshold m/s (default 5e-4)
  -a VALUE  concentration threshold (default 0.15)
  -r PATH   CMEMS root
  -o        overwrite classification stores
  -S        overwrite/rebuild static.zarr
EOF
}

START_DATE=""
END_DATE=""
HEMISPHERE="SH"
METHODS="raw,binary-days,rolling-mean"
ISPD_THRESH="5e-4"
AICE_THRESH="0.15"
ROOT="/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily"
OVERWRITE="false"
OVERWRITE_STATIC="false"

while getopts ":b:e:H:m:s:a:r:oSh" opt; do
  case "$opt" in
    b) START_DATE="$OPTARG" ;;
    e) END_DATE="$OPTARG" ;;
    H) HEMISPHERE="$OPTARG" ;;
    m) METHODS="$OPTARG" ;;
    s) ISPD_THRESH="$OPTARG" ;;
    a) AICE_THRESH="$OPTARG" ;;
    r) ROOT="$OPTARG" ;;
    o) OVERWRITE="true" ;;
    S) OVERWRITE_STATIC="true" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$START_DATE" && -n "$END_DATE" ]] || { usage; exit 2; }

REPO_ROOT="${REPO_ROOT:-$HOME/AFIM/src/mawsons-chest}"
PBS_SCRIPT="${REPO_ROOT}/shuga/scripts/classification/classify_CMEMS.pbs"

METHODS_PBS="${METHODS//,/|}"

qsub -v \
"START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},METHODS=${METHODS_PBS},ISPD_THRESH=${ISPD_THRESH},AICE_THRESH=${AICE_THRESH},ROOT=${ROOT},OVERWRITE=${OVERWRITE},OVERWRITE_STATIC=${OVERWRITE_STATIC},REPO_ROOT=${REPO_ROOT}" \
"$PBS_SCRIPT"
