#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="$HERE/download_insitu_thickness.pbs"

SOURCES="afiac,stations,asac2500,davis2015,bepsii"
ROOT="/g/data/gv90/da1339/SeaIce/InSitu/raw"
DRY_RUN=false
OVERWRITE=false
BEPSII_LOGSHEETS=false

usage() {
    cat <<EOF
Usage: $0 [options]

  -s SOURCES   comma-separated source aliases
  -r ROOT      output root
  -n           discovery-only / dry run
  -o           overwrite existing files
  -L           include BEPSII logsheets
  -h           help

Aliases:
  afiac
  mawson
  davis
  casey
  stations
  soe
  asac2500
  davis2015
  bepsii
  all
EOF
}

while getopts ":s:r:noLh" opt; do
    case "$opt" in
        s) SOURCES="$OPTARG" ;;
        r) ROOT="$OPTARG" ;;
        n) DRY_RUN=true ;;
        o) OVERWRITE=true ;;
        L) BEPSII_LOGSHEETS=true ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

qsub -v \
SOURCES="$SOURCES",\
ROOT="$ROOT",\
DRY_RUN="$DRY_RUN",\
OVERWRITE="$OVERWRITE",\
BEPSII_LOGSHEETS="$BEPSII_LOGSHEETS" \
"$PBS_SCRIPT"
