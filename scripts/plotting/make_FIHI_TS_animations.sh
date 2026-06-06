#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# make_FIHI_TS_animation.sh
#
# Wrapper script for building stitched FIHI_TS AMJ animations across
# 2 to 4 simulations.
#
# This script calls the Python driver:
#   make_FIHI_TS_animation.py
#
# Usage:
#   ./make_FIHI_TS_animation.sh -r Aus
#   ./make_FIHI_TS_animation.sh -r Aus LD-static-Cs1e-3 LD-static-Cs5e-4
#   ./make_FIHI_TS_animation.sh -r Aus LD-static-Cs1e-3 LD-static-Cs5e-4 LD-blend-base
#   ./make_FIHI_TS_animation.sh -r Aus LD-static-Cs1e-3 LD-static-Cs5e-4 LD-blend-base LD-NIL
#
# Optional:
#   ./make_FIHI_TS_animation.sh -r Aus -n 5
#   ./make_FIHI_TS_animation.sh -r Aus -f 8
#   ./make_FIHI_TS_animation.sh -r Aus -m 04,05,06
#
# Notes:
#   - If no sim_names are provided, defaults are used.
#   - Minimum simulations = 2
#   - Maximum simulations = 4
#   - Default months are AMJ (04,05,06)
###############################################################################

usage() {
  cat <<EOF
Usage:
  $(basename "$0") -r REGION [options] [SIM1 SIM2 [SIM3 SIM4]]

Required:
  -r REGION         Antarctic region (e.g. Aus)

Optional:
  -n NFRAMES        Limit to first N frames (debug/testing)
  -f FPS            Frames per second for MP4 output (default: 6)
  -m MONTHS         Comma-separated months (default: 04,05,06)
  -h                Show this help message

Examples:
  $(basename "$0") -r Aus
  $(basename "$0") -r Aus LD-static-Cs1e-3 LD-static-Cs5e-4
  $(basename "$0") -r Aus -n 3
  $(basename "$0") -r Aus -f 8 LD-static-Cs1e-3 LD-static-Cs5e-4

Environment customisation:
  BASE_DIR          Default: ~/graphical/LD-pub-workspace
  FIELD             Default: FIHI_TS
  METHOD_TAG        Default: binary-days
  FPS               Default: 6
  TITLE_BAND_PX     Default: 70
  POINTSIZE         Default: 42
  TITLE_BG          Default: #eeeeee
  TITLE_FILL        Default: black
  JPEG_QUALITY      Default: 92
EOF
}

REGION=""
NFRAMES=""
FPS="${FPS:-6}"
MONTHS="04,05,06"

while getopts ":r:n:f:m:h" opt; do
  case "$opt" in
    r) REGION="$OPTARG" ;;
    n) NFRAMES="$OPTARG" ;;
    f) FPS="$OPTARG" ;;
    m) MONTHS="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "ERROR: invalid option -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "ERROR: option -$OPTARG requires an argument" >&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "${REGION}" ]]; then
  echo "ERROR: REGION is required." >&2
  usage
  exit 1
fi

SIMS=("$@")
if [[ ${#SIMS[@]} -eq 0 ]]; then
  SIMS=(
    "LD-static-Cs1e-3"
    "LD-static-Cs5e-4"
  )
fi

if [[ ${#SIMS[@]} -lt 2 || ${#SIMS[@]} -gt 4 ]]; then
  echo "ERROR: provide between 2 and 4 simulations." >&2
  echo "Received ${#SIMS[@]}: ${SIMS[*]}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/make_FIHI_TS_animation.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: Python script not found:" >&2
  echo "  ${PY_SCRIPT}" >&2
  exit 1
fi

CMD=(python "${PY_SCRIPT}" --region "${REGION}" --fps "${FPS}" --months "${MONTHS}")

if [[ -n "${NFRAMES}" ]]; then
  CMD+=(--nframes "${NFRAMES}")
fi

CMD+=("${SIMS[@]}")

echo "Calling:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"
