#!/usr/bin/env bash
set -euo pipefail

SIM="LD-fsnow-sep"
START="1993-01-01"
END="1993-12-31"
METHODS="binary-days,rolling-mean"
GRID="Tc"

PROJECT="${PROJECT:-gv90}"
RUN_USER="${RUN_USER:-da1339}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/metrics.pbs"

submit_metric_job () {
    local ice_type="$1"
    local metric_group="$2"
    local overwrite="$3"
    local dependency="${4:-}"

    local methods_safe="${METHODS//,/|}"
    local groups_safe="${metric_group//,/|}"

    local qsub_vars
    qsub_vars="SIM_NAME=${SIM}"
    qsub_vars+=",START_DATE=${START}"
    qsub_vars+=",END_DATE=${END}"
    qsub_vars+=",HEMISPHERE=SH"
    qsub_vars+=",ICE_TYPE=${ice_type}"
    qsub_vars+=",GRID_TYPE=${GRID}"
    qsub_vars+=",ISPD_THRESH=5e-4"
    qsub_vars+=",METHODS=${methods_safe}"
    qsub_vars+=",BIN_WINDOW=11"
    qsub_vars+=",BIN_MIN_DAYS=9"
    qsub_vars+=",ROLL_WINDOW=15"
    qsub_vars+=",PROJECT=${PROJECT}"
    qsub_vars+=",RUN_USER=${RUN_USER}"
    qsub_vars+=",OVERWRITE=${overwrite}"
    qsub_vars+=",UPDATE_MISSING_ONLY=true"
    qsub_vars+=",REBUILD_ON_INDEX_MISMATCH=false"
    qsub_vars+=",METRIC_GROUPS=${groups_safe}"
    qsub_vars+=",METRIC_NAMES="
    qsub_vars+=",PLOT_FIP=false"
    qsub_vars+=",PLOT_FIA=false"
    qsub_vars+=",PLOT_FIT=false"
    qsub_vars+=",PLOT_SIA=false"
    qsub_vars+=",PLOT_SIT=false"
    qsub_vars+=",PLOT_REGION=total"
    qsub_vars+=",OBS_FIA_VAR=FIA"
    qsub_vars+=",OBS_FIT_VAR=FIT"
    qsub_vars+=",LOG_LEVEL=INFO"

    local job_name="${SIM}_${ice_type}_${metric_group}"

    if [[ -n "${dependency}" ]]; then
        qsub -P "${PROJECT}" \
             -N "${job_name}" \
             -W "depend=afterok:${dependency}" \
             -v "${qsub_vars}" \
             "${PBS_SCRIPT}"
    else
        qsub -P "${PROJECT}" \
             -N "${job_name}" \
             -v "${qsub_vars}" \
             "${PBS_SCRIPT}"
    fi
}

submit_chain_for_ice_type () {
    local ice_type="$1"
    shift
    local metric_groups=("$@")

    local prev_job=""
    local overwrite="true"

    echo
    echo "Submitting dependency chain for ${ice_type}: ${metric_groups[*]}"

    for metric_group in "${metric_groups[@]}"; do
        if [[ -z "${prev_job}" ]]; then
            job_id="$(submit_metric_job "${ice_type}" "${metric_group}" "${overwrite}")"
        else
            job_id="$(submit_metric_job "${ice_type}" "${metric_group}" "false" "${prev_job}")"
        fi

        echo "  ${ice_type} ${metric_group}: ${job_id}"
        prev_job="${job_id}"
        overwrite="false"
    done
}

# FI, PI, and SI chains are independent and may run at the same time.
# Within each chain, PBS dependencies keep writes to the same mets.zarr sequential.

submit_chain_for_ice_type FI \
    fi_core \
    fi_regional \
    fi_summary \
    fi_spec \
    fi_stress \
    fi_diags \
    fi_spatial

submit_chain_for_ice_type PI \
    pi_core \
    pi_regional \
    pi_summary \
    pi_stress \
    pi_diags \
    pi_spatial

submit_chain_for_ice_type SI \
    si_core \
    si_regional \
    si_summary \
    si_stress \
    si_diags \
    si_spatial

echo
echo "All metric dependency chains submitted."
