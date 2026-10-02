#!/bin/bash
set -euo pipefail

PBS_SCRIPT="${PBS_SCRIPT:-/home/581/da1339/AFIM/src/mawsons-chest/shuga/scripts/waves/cawcr_regrid.pbs}"

usage() {
  cat <<EOF
Usage:
  $0 YEAR MONTH [options]

Options:
  --project NAME            Default: gv90
  --user NAME               Default: da1339
  --sim-name NAME           Default: LD-waves-exp01
  --hemisphere SH|NH        Default: SH
  --sic-threshold FLOAT     Default: 0.15
  --k-nearest INT           Default: 5
  --idw-power FLOAT         Default: 2.5
  --radius-km FLOAT         Default: 1000.0
  --time-chunk INT          Default: 4
  --ow_nc                   Overwrite output NetCDF
  --ow_wgt                  Overwrite CAWCR->CICE weights
  --ow_sic_wgt              Overwrite NSIDC->CICE weights
  -h, --help                Show this help
EOF
  exit 1
}

[[ $# -ge 2 ]] || usage

YEAR="$1"
MONTH="$(printf "%02d" "$2")"
shift 2

PROJECT="gv90"
RUN_USER="da1339"
SIM_NAME="LD-waves-exp01"
HEMISPHERE="SH"
SIC_THRESHOLD="0.15"
K_NEAREST="5"
IDW_POWER="2.5"
RADIUS_KM="1000.0"
TIME_CHUNK="4"
OW_NC="false"
OW_WGT="false"
OW_SIC_WGT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --user) RUN_USER="$2"; shift 2 ;;
    --sim-name) SIM_NAME="$2"; shift 2 ;;
    --hemisphere) HEMISPHERE="$2"; shift 2 ;;
    --sic-threshold) SIC_THRESHOLD="$2"; shift 2 ;;
    --k-nearest) K_NEAREST="$2"; shift 2 ;;
    --idw-power) IDW_POWER="$2"; shift 2 ;;
    --radius-km) RADIUS_KM="$2"; shift 2 ;;
    --time-chunk) TIME_CHUNK="$2"; shift 2 ;;
    --ow_nc) OW_NC="true"; shift ;;
    --ow_wgt) OW_WGT="true"; shift ;;
    --ow_sic_wgt) OW_SIC_WGT="true"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

echo "Submitting CAWCR regrid for YEAR=${YEAR} MONTH=${MONTH} at $(date)"
echo "  project           = ${PROJECT}"
echo "  user              = ${RUN_USER}"
echo "  sim_name          = ${SIM_NAME}"
echo "  hemisphere        = ${HEMISPHERE}"
echo "  sic_threshold     = ${SIC_THRESHOLD}"
echo "  k_nearest         = ${K_NEAREST}"
echo "  idw_power         = ${IDW_POWER}"
echo "  radius_km         = ${RADIUS_KM}"
echo "  time_chunk        = ${TIME_CHUNK}"
echo "  overwrite nc      = ${OW_NC}"
echo "  overwrite weights = ${OW_WGT}"
echo "  overwrite sic wgt = ${OW_SIC_WGT}"

qsub -N "cawcr_regrid_${YEAR}-${MONTH}" \
  -v YEAR="${YEAR}",MONTH="${MONTH}",PROJECT="${PROJECT}",RUN_USER="${RUN_USER}",SIM_NAME="${SIM_NAME}",HEMISPHERE="${HEMISPHERE}",SIC_THRESHOLD="${SIC_THRESHOLD}",K_NEAREST="${K_NEAREST}",IDW_POWER="${IDW_POWER}",RADIUS_KM="${RADIUS_KM}",TIME_CHUNK="${TIME_CHUNK}",OW_NC="${OW_NC}",OW_WGT="${OW_WGT}",OW_SIC_WGT="${OW_SIC_WGT}" \
  "${PBS_SCRIPT}"

echo "Submitted CAWCR regrid for YEAR=${YEAR} MONTH=${MONTH} at $(date)"
