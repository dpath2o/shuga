#!/usr/bin/env bash
set -euo pipefail

BASE="${HOME}/graphical/LD-pub-workspace"
SRC_REL="FIST-FIP_and_strain-invariant/Aus/2000-01-01_2005-12-31.png"
OUT_DIR="${HOME}/graphical/LD-pub-workspace/FIST-FIP_and_strain-invariant"
mkdir -p "${OUT_DIR}"

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
    local src="${BASE}/${sim}/${SRC_REL}"
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
# Figure 1: no-slip-LFI versus Cs-high-ktens-high
# ---------------------------------------------------------------------

make_row "no-slip-LFI"          "(a) no-slip-LFI"          "fig1_a_no-slip-LFI"
make_row "Cs-high-ktens-high"   "(b) Cs-high-ktens-high"   "fig1_b_Cs-high-ktens-high"

"${MONTAGE[@]}" \
    "${TMP}/fig1_a_no-slip-LFI.png" \
    "${TMP}/fig1_b_Cs-high-ktens-high.png" \
    -tile 1x -geometry +0+30 -background white \
    "${OUT_DIR}/Aus_FIST-FIP_strain_no-slip-LFI_vs_Cs-high-ktens-high_2000-2005.png"

# ---------------------------------------------------------------------
# Figure 2: form-function comparison
# ---------------------------------------------------------------------

make_row "Cs-high-ktens-high"   "(a) Cs-high-ktens-high"   "fig2_a_Cs-high-ktens-high"
make_row "Cq-high"              "(b) Cq-high"              "fig2_b_Cq-high"
make_row "Cl-mid"               "(c) CL-mid"               "fig2_c_Cl-mid"
make_row "blend-strain-high"    "(d) blend-strain-high"    "fig2_d_blend-strain-high"

"${MONTAGE[@]}" \
    "${TMP}/fig2_a_Cs-high-ktens-high.png" \
    "${TMP}/fig2_b_Cq-high.png" \
    "${TMP}/fig2_c_Cl-mid.png" \
    "${TMP}/fig2_d_blend-strain-high.png" \
    -tile 1x -geometry +0+30 -background white \
    "${OUT_DIR}/Aus_FIST-FIP_strain_form-comparison_2000-2005.png"

echo "Wrote:"
echo "  ${OUT_DIR}/Aus_FIST-FIP_strain_no-slip-LFI_vs_Cs-high-ktens-high_2000-2005.png"
echo "  ${OUT_DIR}/Aus_FIST-FIP_strain_form-comparison_2000-2005.png"
