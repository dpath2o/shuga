#!/bin/bash
#PBS -N ERA5-CICE-compress-month
#PBS -P jk72
#PBS -q normal
#PBS -l ncpus=16
#PBS -l mem=100GB
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/gv90+scratch/gv90+gdata/xp65
#PBS -j oe
#PBS -M daniel.atwater@utas.edu.au
#PBS -m abe

set -euo pipefail

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3-26.02
module load nco

FILE="${FILE:?FILE not set}"
DEFLATE="${DEFLATE:-1}"
CHUNKSPEC="${CHUNKSPEC:-time/1}"
KEEP_RAW="${KEEP_RAW:-0}"
indir="$(dirname "${FILE}")"
base="$(basename "${FILE}" .nc)"

raw="${indir}/${base}.raw.nc"
tmp="${indir}/${base}.comp.tmp.nc"
comp="${indir}/${base}.comp.nc"

echo "============================================================"
echo "Compress ERA5 -> CICE monthly forcing"
echo "Date      : $(date)"
echo "Host      : $(hostname)"
echo "Job ID    : ${PBS_JOBID:-unknown}"
echo "Input     : ${FILE}"
echo "Raw backup: ${raw}"
echo "Temp      : ${tmp}"
echo "Compressed: ${comp}"
echo "Deflate   : ${DEFLATE}"
echo "Chunkspec : ${CHUNKSPEC}"
echo "Keep raw  : ${KEEP_RAW}"
echo "============================================================"

[[ -f "${FILE}" ]] || { echo "Input file not found: ${FILE}" >&2; exit 1; }

rm -f "${tmp}" "${comp}"

echo "Input size:"
ls -lh "${FILE}"

nccopy -k nc7 -d "${DEFLATE}" -s -c "${CHUNKSPEC}" -w "${FILE}" "${tmp}"

mv "${tmp}" "${comp}"

echo "Compressed size:"
ls -lh "${comp}"

echo "Inspecting compressed file:"
ncdump -k "${comp}"
ncdump -h -s "${comp}" | head -80

if [[ "${KEEP_RAW}" == "1" ]]; then
  mv "${FILE}" "${raw}"
  mv "${comp}" "${FILE}"
  echo "Raw file moved to: ${raw}"
  echo "Compressed file installed as: ${FILE}"
else
  mv "${comp}" "${FILE}"
  echo "Compressed file installed as: ${FILE}"
  echo "Raw file overwritten."
fi

echo "Final size:"
ls -lh "${FILE}"

echo "Finished at $(date)"
