#!/bin/bash
set -euo pipefail

SIM_NAME=""
START_DATE="1993-01-01"
END_DATE="1999-12-31"
HEMISPHERE="SH"
ICE_TYPE="FI"
GRID_TYPE="Tc"
ISPD_THRESH="5e-4"
METHODS="binary-days,rolling-mean"
BIN_WINDOW="11"
BIN_MIN_DAYS="9"
ROLL_WINDOW="15"
ICEH_FREQUENCY="daily"
PROJECT="gv90"
RUN_USER="da1339"
OVERWRITE="false"
UPDATE_MISSING_ONLY="true"
REBUILD_ON_INDEX_MISMATCH="false"
METRIC_GROUPS=""
METRIC_NAMES=""
DRY_RUN="false"
PLOT_FIP="false"
PLOT_FIA="false"
PLOT_FIT="false"
PLOT_SIA="false"
PLOT_SIT="false"
PLOT_REGION="total"
OBS_FIA_VAR="FIA"
OBS_FIT_VAR="FIT"
LOG_LEVEL="INFO"

# Optional path overrides. Empty values are resolved after command-line parsing,
# once PROJECT, RUN_USER, SIM_NAME, and ICEH_FREQUENCY are known.
AFIM_OUTPUT_ROOT=""
CICE_STORE=""
STATIC_STORE=""
CLASSIFICATION_ROOT=""
GRAPHICS_ROOT=""
LOGS_ROOT=""
OBS_METRICS_STORE=""
COAST_DISTANCE_VAR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/metrics.pbs"

print_help() {
    cat <<EOF
Usage: $0 -s SIM_NAME [options]

Core options:
  -s, --sim-name NAME             Simulation name (required)
  -b, --start-date DATE           Start date (default: 1993-01-01)
  -e, --end-date DATE             End date (default: 1999-12-31)
  -H, --hemisphere SH|NH          Hemisphere (default: SH)
  -i, --ice-type FI|PI|SI         Ice domain (default: FI)
  -g, --grid-type TYPE            B/C-to-T method (default: Tc)
  -t, --ispd-thresh VALUE         Ice-speed threshold (default: 5e-4)
  -m, --methods CSV               Methods (default: binary-days,rolling-mean)
  -B, --bin-window N              Binary-days window (default: 11)
  -N, --bin-min-days N            Minimum valid days (default: 9)
  -R, --roll-window N             Rolling-mean window (default: 15)
      --iceh-frequency FREQ       daily or hourly (default: daily)

Metric options:
  -G, --metric-groups CSV         Metric groups (default resolves by ice type)
  -M, --metric-names CSV          Explicit metric names
  -u, --update-missing-only       Compute only missing metrics
  -x, --rebuild-on-index-mismatch Back up/rebuild incompatible mets.zarr
  -o, --overwrite                 Overwrite requested metrics store

Path options:
      --afim-output-root DIR      Root containing simulation directories
      --cice-store DIR            Explicit iceh_daily.zarr or iceh_hourly.zarr
      --static-store DIR          Explicit universal CICE static Zarr
      --cice-static DIR           Alias for --static-store
      --classification-root DIR   Explicit classification-product root
      --graphics-root DIR         Explicit graphics output root
      --logs-root DIR             Explicit logs root
      --obs-metrics-store DIR     Explicit observational metrics store
      --coast-distance-var NAME   Optional coast-distance variable

PBS/runtime options:
  -P, --project PROJECT           Gadi project (default: gv90)
  -U, --user USER                 Run owner (default: da1339)
  -L, --log-level LEVEL           DEBUG, INFO, WARNING, or ERROR
  -f, --plot-fip                  Plot FIP
  -a, --plot-fia                  Plot FIA
  -T, --plot-fit                  Plot FIT
  -A, --plot-sia                  Plot SIA
  -S, --plot-sit                  Plot SIT
  -r, --plot-region REGION        total, DML, WIO, EIO, Aus, VOL, AS, BS, WS
  -n, --dry-run                   Print the resolved qsub command
  -h, --help                      Show this help
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "Missing argument for ${option}" >&2
        print_help
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sim-name)
            require_value "$1" "${2:-}"; SIM_NAME="$2"; shift 2 ;;
        -b|--start-date)
            require_value "$1" "${2:-}"; START_DATE="$2"; shift 2 ;;
        -e|--end-date)
            require_value "$1" "${2:-}"; END_DATE="$2"; shift 2 ;;
        -H|--hemisphere)
            require_value "$1" "${2:-}"; HEMISPHERE="$2"; shift 2 ;;
        -i|--ice-type)
            require_value "$1" "${2:-}"; ICE_TYPE="$2"; shift 2 ;;
        -g|--grid-type|--BorC2T-type)
            require_value "$1" "${2:-}"; GRID_TYPE="$2"; shift 2 ;;
        -t|--ispd-thresh)
            require_value "$1" "${2:-}"; ISPD_THRESH="$2"; shift 2 ;;
        -m|--methods)
            require_value "$1" "${2:-}"; METHODS="$2"; shift 2 ;;
        -B|--bin-window)
            require_value "$1" "${2:-}"; BIN_WINDOW="$2"; shift 2 ;;
        -N|--bin-min-days)
            require_value "$1" "${2:-}"; BIN_MIN_DAYS="$2"; shift 2 ;;
        -R|--roll-window)
            require_value "$1" "${2:-}"; ROLL_WINDOW="$2"; shift 2 ;;
        --iceh-frequency)
            require_value "$1" "${2:-}"; ICEH_FREQUENCY="$2"; shift 2 ;;

        -G|--metric-groups)
            require_value "$1" "${2:-}"; METRIC_GROUPS="$2"; shift 2 ;;
        -M|--metric-names)
            require_value "$1" "${2:-}"; METRIC_NAMES="$2"; shift 2 ;;
        -u|--update-missing-only)
            UPDATE_MISSING_ONLY="true"; shift ;;
        -x|--rebuild-on-index-mismatch)
            REBUILD_ON_INDEX_MISMATCH="true"; shift ;;
        -o|--overwrite)
            OVERWRITE="true"; UPDATE_MISSING_ONLY="false"; shift ;;

        --afim-output-root)
            require_value "$1" "${2:-}"; AFIM_OUTPUT_ROOT="$2"; shift 2 ;;
        --cice-store)
            require_value "$1" "${2:-}"; CICE_STORE="$2"; shift 2 ;;
        --static-store|--cice-static|--cice-static-store)
            require_value "$1" "${2:-}"; STATIC_STORE="$2"; shift 2 ;;
        --classification-root)
            require_value "$1" "${2:-}"; CLASSIFICATION_ROOT="$2"; shift 2 ;;
        --graphics-root)
            require_value "$1" "${2:-}"; GRAPHICS_ROOT="$2"; shift 2 ;;
        --logs-root)
            require_value "$1" "${2:-}"; LOGS_ROOT="$2"; shift 2 ;;
        --obs-metrics-store)
            require_value "$1" "${2:-}"; OBS_METRICS_STORE="$2"; shift 2 ;;
        --coast-distance-var)
            require_value "$1" "${2:-}"; COAST_DISTANCE_VAR="$2"; shift 2 ;;

        -P|--project)
            require_value "$1" "${2:-}"; PROJECT="$2"; shift 2 ;;
        -U|--user)
            require_value "$1" "${2:-}"; RUN_USER="$2"; shift 2 ;;
        -L|--log-level)
            require_value "$1" "${2:-}"; LOG_LEVEL="$2"; shift 2 ;;
        -f|--plot-fip)
            PLOT_FIP="true"; shift ;;
        -a|--plot-fia)
            PLOT_FIA="true"; shift ;;
        -T|--plot-fit)
            PLOT_FIT="true"; shift ;;
        -A|--plot-sia)
            PLOT_SIA="true"; shift ;;
        -S|--plot-sit)
            PLOT_SIT="true"; shift ;;
        -r|--plot-region)
            require_value "$1" "${2:-}"; PLOT_REGION="$2"; shift 2 ;;
        -n|--dry-run)
            DRY_RUN="true"; shift ;;
        -h|--help)
            print_help; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            exit 1 ;;
    esac
done

if [[ -z "$SIM_NAME" ]]; then
    echo "Simulation name is required (-s or --sim-name)." >&2
    exit 1
fi

if [[ ! -f "$PBS_SCRIPT" ]]; then
    echo "PBS script not found: $PBS_SCRIPT" >&2
    exit 1
fi

case "$ICEH_FREQUENCY" in
    daily)  CICE_STORE_NAME="iceh_daily.zarr" ;;
    hourly) CICE_STORE_NAME="iceh_hourly.zarr" ;;
    *)
        echo "Unsupported --iceh-frequency '$ICEH_FREQUENCY'; use daily or hourly." >&2
        exit 1 ;;
esac

case "${ICE_TYPE^^}" in
    FI|PI|SI) ;;
    *)
        echo "Unsupported ICE_TYPE=$ICE_TYPE. Use FI, PI, or SI." >&2
        exit 1 ;;
esac

case "${LOG_LEVEL^^}" in
    DEBUG|INFO|WARNING|ERROR) LOG_LEVEL="${LOG_LEVEL^^}" ;;
    *)
        echo "Unsupported LOG_LEVEL=$LOG_LEVEL. Use DEBUG, INFO, WARNING, or ERROR." >&2
        exit 1 ;;
esac

# Resolve paths after PROJECT/USER/SIM/FREQUENCY are known.
if [[ -z "$AFIM_OUTPUT_ROOT" ]]; then
    AFIM_OUTPUT_ROOT="/g/data/${PROJECT}/${RUN_USER}/afim_output"
fi

if [[ -z "$CICE_STORE" ]]; then
    CICE_STORE="${AFIM_OUTPUT_ROOT}/${SIM_NAME}/zarr/${CICE_STORE_NAME}"
fi

if [[ -z "$STATIC_STORE" ]]; then
    STATIC_CANDIDATES=(
        "${AFIM_OUTPUT_ROOT}/CICE_0p25_Cgrid_coords.zarr"
        "/g/data/${PROJECT}/${RUN_USER}/afim_output/CICE_0p25_Cgrid_coords.zarr"
        "${HOME}/AFIM_archive/CICE_0p25_Cgrid_coords.zarr"
    )
    for candidate in "${STATIC_CANDIDATES[@]}"; do
        if [[ -e "$candidate" ]]; then
            STATIC_STORE="$candidate"
            break
        fi
    done
fi

if [[ ! -e "$CICE_STORE" ]]; then
    echo "Resolved CICE store does not exist: $CICE_STORE" >&2
    echo "Supply --cice-store DIR or revise --afim-output-root." >&2
    exit 1
fi

if [[ -z "$STATIC_STORE" || ! -e "$STATIC_STORE" ]]; then
    echo "Could not resolve an existing CICE static store." >&2
    echo "Supply --static-store DIR (or --cice-static DIR)." >&2
    exit 1
fi

if [[ -n "$CLASSIFICATION_ROOT" && ! -e "$CLASSIFICATION_ROOT" ]]; then
    echo "Explicit classification root does not exist: $CLASSIFICATION_ROOT" >&2
    exit 1
fi

# Apply a domain-specific default only when neither explicit names nor groups
# were requested.
if [[ -z "$METRIC_GROUPS" && -z "$METRIC_NAMES" ]]; then
    case "${ICE_TYPE^^}" in
        FI) METRIC_GROUPS="fi_core" ;;
        PI) METRIC_GROUPS="pi_core" ;;
        SI) METRIC_GROUPS="si_core" ;;
    esac
fi

JOB_NAME="${SIM_NAME}_${ICE_TYPE}_${GRID_TYPE}_metrics"

# PBS-safe encoding: commas inside a single qsub -v value are separators.
METHODS_SAFE="${METHODS//,/|}"
METRIC_GROUPS_SAFE="${METRIC_GROUPS//,/|}"
METRIC_NAMES_SAFE="${METRIC_NAMES//,/|}"

QSUB_VARS="SIM_NAME=${SIM_NAME},START_DATE=${START_DATE},END_DATE=${END_DATE},HEMISPHERE=${HEMISPHERE},ICE_TYPE=${ICE_TYPE},GRID_TYPE=${GRID_TYPE},ISPD_THRESH=${ISPD_THRESH},METHODS=${METHODS_SAFE},BIN_WINDOW=${BIN_WINDOW},BIN_MIN_DAYS=${BIN_MIN_DAYS},ROLL_WINDOW=${ROLL_WINDOW},ICEH_FREQUENCY=${ICEH_FREQUENCY},PROJECT=${PROJECT},RUN_USER=${RUN_USER},OVERWRITE=${OVERWRITE},UPDATE_MISSING_ONLY=${UPDATE_MISSING_ONLY},REBUILD_ON_INDEX_MISMATCH=${REBUILD_ON_INDEX_MISMATCH},PLOT_FIP=${PLOT_FIP},PLOT_FIA=${PLOT_FIA},PLOT_FIT=${PLOT_FIT},PLOT_SIA=${PLOT_SIA},PLOT_SIT=${PLOT_SIT},PLOT_REGION=${PLOT_REGION},OBS_FIA_VAR=${OBS_FIA_VAR},OBS_FIT_VAR=${OBS_FIT_VAR},LOG_LEVEL=${LOG_LEVEL},AFIM_OUTPUT_ROOT=${AFIM_OUTPUT_ROOT},CICE_STORE=${CICE_STORE},STATIC_STORE=${STATIC_STORE}"

[[ -n "$METRIC_GROUPS" ]]       && QSUB_VARS+=",METRIC_GROUPS=${METRIC_GROUPS_SAFE}"
[[ -n "$METRIC_NAMES"  ]]       && QSUB_VARS+=",METRIC_NAMES=${METRIC_NAMES_SAFE}"
[[ -n "$CLASSIFICATION_ROOT" ]] && QSUB_VARS+=",CLASSIFICATION_ROOT=${CLASSIFICATION_ROOT}"
[[ -n "$GRAPHICS_ROOT" ]]       && QSUB_VARS+=",GRAPHICS_ROOT=${GRAPHICS_ROOT}"
[[ -n "$LOGS_ROOT" ]]           && QSUB_VARS+=",LOGS_ROOT=${LOGS_ROOT}"
[[ -n "$OBS_METRICS_STORE" ]]   && QSUB_VARS+=",OBS_METRICS_STORE=${OBS_METRICS_STORE}"
[[ -n "$COAST_DISTANCE_VAR" ]]  && QSUB_VARS+=",COAST_DISTANCE_VAR=${COAST_DISTANCE_VAR}"

echo "Resolved AFIM output root : ${AFIM_OUTPUT_ROOT}"
echo "Resolved CICE store       : ${CICE_STORE}"
echo "Resolved CICE static store: ${STATIC_STORE}"
if [[ -n "$CLASSIFICATION_ROOT" ]]; then
    echo "Classification root       : ${CLASSIFICATION_ROOT}"
else
    echo "Classification root       : auto-resolved by ShugaPaths"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] qsub -P ${PROJECT} -N ${JOB_NAME} -v ${QSUB_VARS} ${PBS_SCRIPT}"
    exit 0
fi

qsub -P "${PROJECT}" -N "${JOB_NAME}" -v "${QSUB_VARS}" "${PBS_SCRIPT}"
