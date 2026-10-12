#!/usr/bin/env bash
# Evaluate a patch on LIBERO (needs the policy server running).
# The patch overlays on the wrist camera at the bottom-center by default
# (auto-anchored to the patch size: 32px→192,96 / 64px→160,80).
#   OPENPI_ROOT  path to a cloned openpi repo
# Usage:
#   ./scripts/eval.sh                                   # clean baseline (no patch)
#   ./scripts/eval.sh patches/drift_pi0_32px.npy libero_spatial drift_pi0
set -e
: "${OPENPI_ROOT:?set OPENPI_ROOT to your cloned openpi repo}"
DRIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="${1:-}"
SUITE="${2:-libero_spatial}"
EXP="${3:-drift}"
source "$OPENPI_ROOT/examples/libero/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$OPENPI_ROOT/third_party/libero"
export MUJOCO_GL=egl
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
PATCH_ARGS=""
if [ -n "$PATCH" ]; then PATCH_ARGS="--patch-path $PATCH"; fi
python "$DRIFT_DIR/patch_eval_libero.py" \
    --task-suite-name "$SUITE" --num-trials-per-task 10 --exp-name "$EXP" \
    $PATCH_ARGS "${@:4}"
