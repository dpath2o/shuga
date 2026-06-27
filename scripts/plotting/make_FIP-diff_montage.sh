#!/usr/bin/env bash
set -euo pipefail

REGION="DML"
BASE="${HOME}/graphical/LD-pub-workspace"
D_SRC="FIP_diff/${REGION}/"
D_OUT="${HOME}/graphical/LD-pub-workspace/FIP_diff"
mkdir -p "${D_OUT}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Prefer ImageMagick 7 syntax, fall back to ImageMagick 6.
if command -v magick >/dev/null 2>&1; then
    IM=(magick)
    MONTAGE=(magick montage)
else
    IM=(convert)
    MONTAGE=(montage)
fi

make_row () {
    local sim="$1"
    local label="$2"
    local tag="$3"
    local src="${BASE}/${sim}/${D_SRC}/2000-04-01_2005-12-30_${sim}_${REGION}.png"
    local dst="${TMP}/${tag}.png"
    if [[ ! -f "${src}" ]]; then
        echo "Missing source image: ${src}" >&2
        exit 1
    fi
    "${IM[@]}" "${src}" \
        -fuzz 1% -trim +repage \
        -bordercolor white -border 20 \
        -gravity north -background white -splice 0x72 \
        -gravity northwest -fill black -pointsize 36 \
        -annotate +24+22 "${label}" \
        "${dst}"
}

# ---------------------------------------------------------------------
# Figure 2: form-function comparison
# ---------------------------------------------------------------------

make_row "Cs-high"              "(a) Cs-high"              "fig2_a_Cs-high"
make_row "Cq-high"              "(b) Cq-high"              "fig2_b_Cq-high"
make_row "Cl-mid"               "(c) CL-mid"               "fig2_c_Cl-mid"
make_row "blend-strain-high"    "(d) blend-strain-high"    "fig2_d_blend-strain-high"

"${MONTAGE[@]}" \
    "${TMP}/fig2_a_Cs-high.png" \
    "${TMP}/fig2_b_Cq-high.png" \
    "${TMP}/fig2_c_Cl-mid.png" \
    "${TMP}/fig2_d_blend-strain-high.png" \
    -tile 1x -geometry +0+30 -background white \
    "${D_OUT}/${REGION}_FIP-diff_2000-2005.png"

echo "Wrote:"
echo "  ${D_OUT}/${REGION}_FIP-diff_2000-2005.png"
