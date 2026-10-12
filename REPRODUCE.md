# Reproducing DRIFT

DRIFT attacks off-the-shelf flow-matching VLAs (π0 / π0.5). This repo contains the
patch-training and LIBERO-evaluation code; the victim models and simulator are
external dependencies.

## 1. Setup

**openpi** (π0 / π0.5, JAX). Clone and set up per its README, then:
```bash
export OPENPI_ROOT=/path/to/openpi          # cloned openpi repo
export PI0_CHECKPOINT=/path/to/pi0_libero   # π0 checkpoint (use pi05_libero for π0.5)
```
The LIBERO evaluation uses openpi's LIBERO venv at `$OPENPI_ROOT/examples/libero/.venv`
and the LIBERO package on `PYTHONPATH` (`$OPENPI_ROOT/third_party/libero`).

**This repo:**
```bash
pip install -r requirements.txt
```

## 2. Pipeline

```bash
# 1) start the policy server (Terminal A)
./scripts/serve.sh

# 2) collect a LIBERO observation pool for training (Terminal B)
./scripts/collect_obs.sh collected_obs_libero.npy

# 3) train the DRIFT patch (first-step attack; Terminal B, JAX env)
./scripts/attack.sh collected_obs_libero.npy patches/drift_patch.npy

# 4) evaluate on a LIBERO suite (server must be running)
./scripts/eval.sh patches/drift_patch.npy libero_spatial drift_spatial
```
`eval.sh` prints per-task and overall success rates and writes `results/libero/<exp>/results.json`.
Attack Success Rate (ASR) = (clean TSR − attacked TSR) / clean TSR; run step 4 once with no
patch (`./scripts/eval.sh`) for the clean baseline.

## 3. Released patches

Pre-trained highest-ASR DRIFT patches (evaluate directly, no training needed). The
overlay anchor (wrist bottom-center) is derived automatically from the patch size:
- `patches/drift_pi0_32px.npy`  — π0, 32×32 wrist patch
- `patches/drift_pi05_64px.npy` — π0.5, 64×64 wrist patch

```bash
# π0
./scripts/eval.sh patches/drift_pi0_32px.npy libero_spatial drift_pi0
# π0.5 (serve the pi0.5 checkpoint first)
CONFIG=pi05_libero PI0_CHECKPOINT=/path/to/pi05_libero ./scripts/serve.sh   # Terminal A
./scripts/eval.sh patches/drift_pi05_64px.npy libero_spatial drift_pi05
```

## 4. Key ablation — "less is more"

Attacking only the first denoising step is strongest and cheapest. Reproduce by sweeping
`--top-k-steps` in `attack_patch.py` (via `scripts/attack.sh`, editing the flag):
- `--top-k-steps 1` → first-step-only DRIFT (recommended)
- `--top-k-steps 3` / `--top-k-steps 5` → attacking a wider early window (weaker)

## 5. Stealthy patch (optional)

`attack_patch.py` supports a natural-looking patch via `--init-image <img> --epsilon 0.06
--tv-weight 1e-3` (L∞ budget around the init image + total-variation smoothing).
