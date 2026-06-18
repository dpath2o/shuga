#!/usr/bin/env bash
set -euo pipefail

SIM_NAMES=("Cs-high-ktens-high"
           "Cs-high"
           "Cs-high-ktens-mid"
           "Cs-high-eDef"
           "Cs-mid"
           "Cs-low"
           "Cq-high"
           "Cq-mid"
           "Cq-low"
           "Cl-mid"
           "Cl-low"
           "blend-strain-high"
           "blend-strain-mid"
           "blend-starin-low"
           "no-slip-LFI")

START="1995-01-01"
END="2005-12-31"
METHODS="binary-days,rolling-mean"
GRID="Tc"
PROJECT="${PROJECT:-jk72}"
RUN_USER="${RUN_USER:-da1339}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/metrics.pbs"

submit_metric_job () {
    local sim_name="$1"
    local ice_type="$2"
    local metric_group="$3"
    local overwrite="$4"
    local dependency="${5:-}"
    local methods_safe="${METHODS//,/|}"
    local groups_safe="${metric_group//,/|}"
    local job_name
    job_name="${sim_name}_${ice_type}_${metric_group}"
    job_name="${job_name//-/_}"
    job_name="${job_name//,/}"
    local qsub_vars
    qsub_vars="SIM_NAME=${sim_name}"
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

submit_chain_for_sim_ice_type () {
    local sim_name="$1"
    local ice_type="$2"
    shift 2
    local metric_groups=("$@")
    local prev_job=""
    local overwrite="true"
    local job_id=""
    echo
    echo "Submitting dependency chain:"
    echo "  SIM_NAME : ${sim_name}"
    echo "  ICE_TYPE : ${ice_type}"
    echo "  GROUPS   : ${metric_groups[*]}"
    for metric_group in "${metric_groups[@]}"; do
        if [[ -z "${prev_job}" ]]; then
            job_id="$(submit_metric_job "${sim_name}" "${ice_type}" "${metric_group}" "${overwrite}")"
        else
            job_id="$(submit_metric_job "${sim_name}" "${ice_type}" "${metric_group}" "false" "${prev_job}")"
        fi
        echo "  ${sim_name} ${ice_type} ${metric_group}: ${job_id}"
        prev_job="${job_id}"
        overwrite="false"
    done
}

submit_all_chains_for_sim () {
    local sim_name="$1"
    echo
    echo "======================================================================"
    echo "Submitting metric chains for SIM_NAME=${sim_name}"
    echo "======================================================================"
    # FI, PI, and SI chains are independent and may run at the same time.
    # Within each chain, PBS dependencies keep writes to the same mets.zarr sequential.
    submit_chain_for_sim_ice_type "${sim_name}" FI \
        fi_core \
        fi_regional \
        fi_summary \
        fi_spec \
        fi_stress \
        fi_diags \
        fi_spatial
    submit_chain_for_sim_ice_type "${sim_name}" PI \
        pi_core \
        pi_regional \
        pi_summary \
        pi_stress \
        pi_diags \
        pi_spatial
    submit_chain_for_sim_ice_type "${sim_name}" SI \
        si_core \
        si_regional \
        si_summary \
        si_stress \
        si_diags \
        si_spatial
}

for sim_name in "${SIM_NAMES[@]}"; do
    submit_all_chains_for_sim "${sim_name}"
done

echo
echo "All metric dependency chains submitted for ${#SIM_NAMES[@]} simulation(s)."
