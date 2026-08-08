#!/bin/bash
#PBS -N ERA5-nccopy-fast
#PBS -P jk72
#PBS -q normalbw
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l jobfs=20GB
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/gv90+gdata/xp65+scratch/jk72
#PBS -j oe
#PBS -M daniel.atwater@utas.edu.au
#PBS -m abe

set -euo pipefail

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3-26.02

# ---------------------------------------------------------------------
# User controls
# ---------------------------------------------------------------------
ERA5_DIR="${ERA5_DIR:-/g/data/gv90/da1339/afim_input/ERA5/0p25/bilinear/monthly_cice6}"

START_YM="${START_YM:-1999-01}"
END_YM="${END_YM:-2005-12}"

# If files whose time dimension is already fixed-size are skipped.
# This should skip already-converted January file.
SKIP_FIXED_TIME="${SKIP_FIXED_TIME:-1}"

# If print actions but do not modify files.
DRYRUN="${DRYRUN:-0}"

# nccopy mode.
#
# Fastest likely runtime layout for CICE:
#   - NetCDF-4 classic model
#   - no compression
#   - convert unlimited time to fixed-size
#
# This preserves all variables, but rewrites the physical storage layout.
NCCOPY_MODE="${NCCOPY_MODE:--k nc7 -d 0 -u}"

echo "============================================================"
echo "ERA5 monthly nccopy fast-layout conversion"
echo "Host              : $(hostname)"
echo "Date              : $(date)"
echo "PBS job id        : ${PBS_JOBID:-unknown}"
echo "ERA5_DIR          : ${ERA5_DIR}"
echo "START_YM          : ${START_YM}"
echo "END_YM            : ${END_YM}"
echo "SKIP_FIXED_TIME   : ${SKIP_FIXED_TIME}"
echo "DRYRUN            : ${DRYRUN}"
echo "NCCOPY_MODE       : ${NCCOPY_MODE}"
echo "============================================================"

command -v nccopy
command -v ncdump

month_iter() {
python - "$START_YM" "$END_YM" <<'PY'
import sys
from datetime import date

start = sys.argv[1]
end = sys.argv[2]

sy, sm = map(int, start.split("-"))
ey, em = map(int, end.split("-"))

y, m = sy, sm
while (y < ey) or (y == ey and m <= em):
    print(f"{y:04d} {m:02d}")
    m += 1
    if m == 13:
        y += 1
        m = 1
PY
}
SRC_DIR=/g/data/gv90/da1339/afim_input/ERA5/0p25/bilinear/monthly_cice6
DST_DIR=/g/data/gv90/da1339/afim_input/ERA5/0p25/bilinear/monthly_cice6_fast

mkdir -p "${DST_DIR}"

for ym in $(month_iter | sed 's/ /_/g'); do
    year="${ym%_*}"
    mon="${ym#*_}"

    src="${SRC_DIR}/era5_for_cice6_${year}_${mon}.nc"
    dst="${DST_DIR}/era5_for_cice6_${year}_${mon}.nc"
    tmp="${DST_DIR}/.era5_for_cice6_${year}_${mon}.nccopy.tmp.nc"

    echo "------------------------------------------------------------"
    echo "Processing ${year}-${mon}"
    echo "Source: ${src}"
    echo "Dest  : ${dst}"

    [[ -f "${src}" ]] || { echo "WARNING: missing source, skipping"; continue; }

    # Skip only if destination already exists and FORCE=0.
    if [[ -f "${dst}" && "${FORCE:-0}" != "1" ]]; then
        echo "Destination exists; skipping. Set FORCE=1 to overwrite."
        continue
    fi

    rm -f "${tmp}"

    echo "Running: nccopy -k nc7 -d 0 -u ${src} ${tmp}"
    nccopy -k nc7 -d 0 -u "${src}" "${tmp}"

    [[ -s "${tmp}" ]] || { echo "ERROR: tmp missing/empty: ${tmp}" >&2; exit 1; }

    mv -f "${tmp}" "${dst}"

    echo "Installed:"
    ls -lh "${dst}"
done

echo "============================================================"
echo "All requested files processed at $(date)"
echo "============================================================"
