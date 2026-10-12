#!/usr/bin/env bash
# Train the DRIFT patch (first-step velocity-divergence attack).
#   OPENPI_ROOT     path to a cloned openpi repo
#   PI0_CHECKPOINT  checkpoint dir
#   CONFIG          openpi config (default: pi0_libero; use pi05_libero for pi0.5)
# Usage: ./scripts/attack.sh [obs_pool.npy] [out_patch.npy]
set -e
: "${OPENPI_ROOT:?set OPENPI_ROOT to your cloned openpi repo}"
: "${PI0_CHECKPOINT:?set PI0_CHECKPOINT to the checkpoint dir}"
DRIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-pi0_libero}"
OBS="${1:-$DRIFT_DIR/collected_obs_libero.npy}"
OUT="${2:-$DRIFT_DIR/patches/drift_patch.npy}"
cd "$DRIFT_DIR"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --project "$OPENPI_ROOT" python attack_patch.py \
    --train-config "$CONFIG" --checkpoint-dir "$PI0_CHECKPOINT" \
    --libero-obs-path "$OBS" \
    --patch-size 32 --pgd-steps 500 --top-k-steps 1 \
    --out "$OUT"
