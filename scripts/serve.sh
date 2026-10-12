#!/usr/bin/env bash
# Start the pi0 / pi0.5 policy server (openpi).
#   OPENPI_ROOT     path to a cloned openpi repo
#   PI0_CHECKPOINT  checkpoint dir (pi0_libero or pi05_libero)
#   CONFIG          openpi config name (default: pi0_libero)
#   PORT            server port (default: 8000)
set -e
: "${OPENPI_ROOT:?set OPENPI_ROOT to your cloned openpi repo}"
: "${PI0_CHECKPOINT:?set PI0_CHECKPOINT to the checkpoint dir}"
CONFIG="${CONFIG:-pi0_libero}"
PORT="${PORT:-8000}"
cd "$OPENPI_ROOT"
uv run scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config="$CONFIG" --policy.dir="$PI0_CHECKPOINT"
