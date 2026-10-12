#!/usr/bin/env bash
# Collect a LIBERO observation pool for training (needs the policy server running).
#   OPENPI_ROOT   path to a cloned openpi repo (LIBERO venv at $OPENPI_ROOT/examples/libero/.venv)
# Usage: ./scripts/collect_obs.sh [out.npy]
set -e
: "${OPENPI_ROOT:?set OPENPI_ROOT to your cloned openpi repo}"
DRIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-collected_obs_libero.npy}"
source "$OPENPI_ROOT/examples/libero/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$OPENPI_ROOT/third_party/libero"
export MUJOCO_GL=egl
# On some headless GPU nodes you may also need:
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
python "$DRIFT_DIR/collect_libero_obs.py" --out "$DRIFT_DIR/$OUT"
