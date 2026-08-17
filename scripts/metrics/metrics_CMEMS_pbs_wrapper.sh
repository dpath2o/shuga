#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./metrics_CMEMS_pbs_wrapper.sh -b YYYY-MM-DD -e YYYY-MM-DD [options]

Options:
  -b DATE   start date (required)
  -e DATE   end date (required)
  -H HEM    SH or NH (default SH)
  -m METH   classification method (default binary-days)
  -G GROUP  CMEMS metric group (default cmems_core)
  -M NAMES  explicit comma-separated metric names
  -s VALUE  ice-speed threshold m/s (default 5e-4)
  -a VALUE  concentration threshold (default 0.15)
  -r PATH   CMEMS root
  -o        overwrite metrics stores
EOF
}

START_DATE=""
END_DATE=""
HEMISPHERE="SH"
METHOD="binary-days"
METRIC_GROUPS=""
METRIC_NAMES=""
ISPD_THRESH="5e-4"
AICE_THRESH="0.15"
ROOT="/g/data/gv90/da1339/SeaIce/CMEMS/0p083/daily"
OVERWRITE="false"

while getopts ":b:e:H:m:G:M:s:a:r:oh" opt; do
  case "$opt" in
    b) START_DATE="$OPTARG" ;;
    e) END_DATE="$OPTARG" ;;
    H) HEMISPHERE="$OPTARG" ;;
    m) METHOD="$OPTARG" ;;
    G) METRIC_GROUPS="$OPTARG" ;;
    M) METRIC_NAMES="$OPTARG" ;;
    s) ISPD_THRESH="$OPTARG" ;;
    a) AICE_THRESH="$OPTARG" ;;
    r) ROOT="$OPTARG" ;;
    o) OVERWRITE="true" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$START_DATE" && -n "$END_DATE" ]] || { usage; exit 2; }

if [[ -z "$METRIC_GROUPS" && -z "$METRIC_NAMES" ]]; then
    METRIC_GROUPS="cmems_core"
fi

REPO_ROOT="${REPO_ROOT:-$HOME/AFIM/src/mawsons-chest}"
PBS_SCRIPT="${REPO_ROOT}/shuga/scripts/metrics/metrics_CMEMS.pbs"

GROUPS_PBS="${METRIC_GROUPS//,/|}"
NAMES_PBS="${METRIC_NAMES//,/|}"

qsub -v \
"START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},METHOD=${METHOD},METRIC_GROUPS=${GROUPS_PBS},METRIC_NAMES=${NAMES_PBS},ISPD_THRESH=${ISPD_THRESH},AICE_THRESH=${AICE_THRESH},ROOT=${ROOT},OVERWRITE=${OVERWRITE},REPO_ROOT=${REPO_ROOT}" \
"$PBS_SCRIPT"
