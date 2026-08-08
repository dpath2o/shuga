#!/usr/bin/env bash
set -euo pipefail

AFIM_ROOT="/g/data/gv90/da1339/afim_output/paper1"
STATIC_STORE="/g/data/gv90/da1339/afim_output/CICE_0p25_Cgrid_coords.zarr"

SIMULATIONS=(
    "AOM2-ERA5"
    "notensnogi"
    "ry93"
    "elps-min"
)

for SIM in "${SIMULATIONS[@]}"; do
    CICE_STORE="${AFIM_ROOT}/${SIM}/zarr/iceh_daily.zarr"

    if [[ ! -d "${CICE_STORE}" ]]; then
        echo "ERROR: CICE store does not exist:"
        echo "       ${CICE_STORE}"
        exit 1
    fi

    echo
    echo "============================================================"
    echo "Submitting corrected SH SIA metrics"
    echo "Simulation : ${SIM}"
    echo "CICE store: ${CICE_STORE}"
    echo "============================================================"

    ./metrics_pbs_wrapper.sh \
        -s "${SIM}" \
        -b 1993-01-01 \
        -e 1999-12-31 \
        -H SH \
        -i SI \
        -g Tb \
        -m raw \
        -M SIA \
        -o \
        --afim-output-root "${AFIM_ROOT}" \
        --cice-store "${CICE_STORE}" \
        --static-store "${STATIC_STORE}"
done
