#!/usr/bin/env bash
set -euo pipefail

SIM_NAMES=(
    "Cs-high-ktens-high"
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
    "blend-strain-mid"
    "blend-strain-low"
    "blend-strain-high"
    # "no-lateral-drag"
    "no-slip-LFI"
)

START_YEAR="1995"
END_YEAR="2005"

START_DATE="${START_YEAR}-01-01"
END_DATE="${END_YEAR}-12-31"

HEMISPHERE="SH"
ICE_TYPE="FI"        # classify_pbs_wrapper requires FI; it writes FI and PI stores
GRID_TYPE="Tc"
ISPD_THRESH="5e-4"

METHODS="raw,binary-days,rolling-mean"

BIN_WINDOW="11"
BIN_MIN_DAYS="9"
ROLL_WINDOW="15"

ICEH_FREQUENCY="daily"
NETCDF_ENGINE="scipy"

PROJECT="${PROJECT:-gv90}"
RUN_USER="${RUN_USER:-da1339}"

# Toggle these as needed.
OVERWRITE="true"
OVERWRITE_HISTORY="false"
OVERWRITE_STATIC="false"
DELETE_ORIGINAL="true"
DRY_RUN="false"

# Optional roots. Leave empty to let classify_pbs_wrapper.sh / classify.pbs defaults apply.
AFIM_OUTPUT_ROOT=""
CICE_STORE=""
STATIC_STORE=""
CLASSIFICATION_ROOT=""
ARCHIVE_ROOT=""
LOGS_ROOT=""
DAILY_ROOT=""
HOURLY_ROOT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY_WRAPPER="${SCRIPT_DIR}/classify_pbs_wrapper.sh"

if [[ ! -x "${CLASSIFY_WRAPPER}" ]]; then
    echo "ERROR: classify wrapper is not executable or not found:" >&2
    echo "  ${CLASSIFY_WRAPPER}" >&2
    echo "Try: chmod +x ${CLASSIFY_WRAPPER}" >&2
    exit 1
fi

submit_classification_job () {
    local sim_name="$1"

    local cmd=(
        "${CLASSIFY_WRAPPER}"
        -s "${sim_name}"
        -b "${START_DATE}"
        -e "${END_DATE}"
        -H "${HEMISPHERE}"
        -i "${ICE_TYPE}"
        -g "${GRID_TYPE}"
        -t "${ISPD_THRESH}"
        -m "${METHODS}"
        -B "${BIN_WINDOW}"
        -N "${BIN_MIN_DAYS}"
        -R "${ROLL_WINDOW}"
        -P "${PROJECT}"
        -U "${RUN_USER}"
        --iceh-frequency "${ICEH_FREQUENCY}"
        --netcdf-engine "${NETCDF_ENGINE}"
    )

    if [[ "${OVERWRITE}" == "true" ]]; then
        cmd+=( -o )
    fi

    if [[ "${OVERWRITE_HISTORY}" == "true" ]]; then
        cmd+=( --overwrite-history )
    fi

    if [[ "${OVERWRITE_STATIC}" == "true" ]]; then
        cmd+=( --overwrite-static )
    fi

    if [[ "${DELETE_ORIGINAL}" == "true" ]]; then
        cmd+=( --delete-original )
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=( -n )
    fi

    [[ -n "${AFIM_OUTPUT_ROOT}"    ]] && cmd+=( --afim-output-root "${AFIM_OUTPUT_ROOT}" )
    [[ -n "${CICE_STORE}"          ]] && cmd+=( --cice-store "${CICE_STORE}" )
    [[ -n "${STATIC_STORE}"        ]] && cmd+=( --static-store "${STATIC_STORE}" )
    [[ -n "${CLASSIFICATION_ROOT}" ]] && cmd+=( --classification-root "${CLASSIFICATION_ROOT}" )
    [[ -n "${ARCHIVE_ROOT}"        ]] && cmd+=( --archive-root "${ARCHIVE_ROOT}" )
    [[ -n "${LOGS_ROOT}"           ]] && cmd+=( --logs-root "${LOGS_ROOT}" )
    [[ -n "${DAILY_ROOT}"          ]] && cmd+=( --daily-root "${DAILY_ROOT}" )
    [[ -n "${HOURLY_ROOT}"         ]] && cmd+=( --hourly-root "${HOURLY_ROOT}" )

    echo
    echo "======================================================================"
    echo "Submitting classification job"
    echo "  SIM_NAME   : ${sim_name}"
    echo "  START_DATE : ${START_DATE}"
    echo "  END_DATE   : ${END_DATE}"
    echo "  METHODS    : ${METHODS}"
    echo "  DELETE_ORIG: ${DELETE_ORIGINAL}"
    echo "======================================================================"

    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
}

for sim_name in "${SIM_NAMES[@]}"; do
    submit_classification_job "${sim_name}"
done

echo
echo "All classification jobs submitted for ${#SIM_NAMES[@]} simulation(s)."
