#!/usr/bin/env python3
"""
DRIFT: adversarial patch attack on flow-matching VLAs (pi0 / pi0.5).

Optimizes a universal patch that maximizes the divergence between the patched and
clean denoising velocity fields over the first K' denoising steps (uniform weight).

Environment:
    OPENPI_ROOT     path to a cloned `openpi` repo (we insert $OPENPI_ROOT/src on sys.path)
    PI0_CHECKPOINT  default checkpoint dir if --checkpoint-dir is not given

Usage (run inside the openpi JAX env):
    OPENPI_ROOT=/path/to/openpi PI0_CHECKPOINT=/path/to/pi0_libero \\
    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python attack_patch.py \\
        --train-config pi0_libero \\
        --libero-obs-path collected_obs_libero.npy \\
        --patch-size 32 --pgd-steps 500 --top-k-steps 1 \\
        --out patches/drift_pi0_32px.npy
"""

import argparse
import csv
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

# openpi is an external dependency: point OPENPI_ROOT at a cloned openpi repo.
_OPENPI_ROOT = os.environ.get("OPENPI_ROOT")
if not _OPENPI_ROOT:
    sys.exit("ERROR: set OPENPI_ROOT to your cloned openpi repo (we add $OPENPI_ROOT/src to sys.path).")
sys.path.insert(0, str(pathlib.Path(_OPENPI_ROOT) / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # allow `import utils` from any cwd

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from utils.attack import (
    preprocess_obs_batch,
    apply_patch,
    prefix_kv_cache,
    suffix_velocity,
    WRIST_KEY,
)

DEFAULT_TRAIN_CONFIG = "pi0_libero"


# ---------------------------------------------------------------------------
# LIBERO obs loader (pool collected by collect_libero_obs.py)
# ---------------------------------------------------------------------------

class LIBERODataLoader:
    """Random-batch sampler over a pool of LIBERO observations (matches the eval
    domain). Same interface used by the PGD loop each step."""

    def __init__(self, obs_path, input_transform, batch_size, seed=0):
        data = np.load(obs_path, allow_pickle=True).item()
        self._images  = data["observation/image"]          # [N, 224, 224, 3] uint8
        self._wrists  = data["observation/wrist_image"]     # [N, 224, 224, 3] uint8
        self._states  = data["observation/state"]           # [N, state_dim] float32
        self._prompts = data.get("prompts", data.get("prompt", "pick up the object"))
        self._N       = len(self._images)
        self._B       = batch_size
        self._transform = input_transform
        self._rng     = np.random.default_rng(seed)
        print(f"  LIBERO obs pool: N={self._N}  batch={self._B}")

    def next_batch(self):
        idx = self._rng.integers(0, self._N, size=self._B)
        obs_data = {
            "observation/image":       self._images[idx],
            "observation/wrist_image": self._wrists[idx],
            "observation/state":       self._states[idx],
            "prompts": ([self._prompts[i] for i in idx]
                        if isinstance(self._prompts, list)
                        else [self._prompts] * self._B),
        }
        return preprocess_obs_batch(self._transform, obs_data, self._B)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_patch_preview(obs_clean, patch_01, out_path, pos_y=None, pos_x=None,
                       cam_key="base_0_rgb"):
    from PIL import Image

    if cam_key not in obs_clean.images:
        cam_key = "base_0_rgb"
    img_np   = np.array(obs_clean.images[cam_key][0])
    H, W     = img_np.shape[:2]
    ph, pw   = patch_01.shape[:2]

    img_u8   = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    patch_u8 = (np.array(patch_01) * 255).clip(0, 255).astype(np.uint8)

    cy = pos_y if pos_y is not None else (H - ph) // 2
    cx = pos_x if pos_x is not None else (W - pw) // 2
    patched = img_u8.copy()
    patched[cy:cy + ph, cx:cx + pw] = patch_u8

    patch_big = np.array(
        Image.fromarray(patch_u8).resize((int(pw * H / ph), H), Image.NEAREST)
    )
    canvas = np.concatenate([img_u8, patched, patch_big], axis=1)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(str(out_path))
    print(f"  Preview saved → {out_path}")


# ---------------------------------------------------------------------------
# Stealth helpers: image init + smoothness / naturalness priors
# ---------------------------------------------------------------------------

def load_init_image(path, ph, pw):
    """Load an init image, resize to (ph, pw), return float32 [ph, pw, 3] in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((pw, ph), Image.LANCZOS)
    return jnp.asarray(np.asarray(img, dtype=np.float32) / 255.0)


def _total_variation(p):
    """Anisotropic total variation of a [H, W, 3] patch (smoothness prior)."""
    dh = jnp.mean(jnp.abs(p[1:, :, :] - p[:-1, :, :]))
    dw = jnp.mean(jnp.abs(p[:, 1:, :] - p[:, :-1, :]))
    return dh + dw


def _l2_to_base(patch, base):
    """Mean squared deviation from the base image (naturalness prior)."""
    return jnp.mean((patch - base) ** 2)


# ---------------------------------------------------------------------------
# DRIFT loss: velocity-field divergence over the first K' steps
# ---------------------------------------------------------------------------

def build_drift_loss(graphdef, B, action_horizon, action_dim, num_steps,
                     pos_y, pos_x, wrist_pos_y, wrist_pos_x,
                     step_weights, tv_weight=0.0, l2_weight=0.0, base=None,
                     attack_base=True):
    """Returns loss_fn(patch, state, rng, obs_clean) -> (loss, divergence).

    Gradient flows only into `patch_01` (argnums=0). `divergence` is the pure
    velocity-divergence term (no TV/L2), returned as has_aux for logging.
    """
    dt = -1.0 / num_steps

    def loss_fn(patch_01, state, rng_key, obs_clean):
        m = nnx.merge(graphdef, state)
        k_noise, _ = jax.random.split(rng_key)
        noise = jax.random.normal(k_noise, (B, action_horizon, action_dim))

        obs_clean_pp = _model.preprocess_observation(None, obs_clean, train=False)
        obs_p = apply_patch(obs_clean, patch_01, pos_y, pos_x,
                            wrist_pos_y, wrist_pos_x, attack_base=attack_base)
        obs_p_pp = _model.preprocess_observation(None, obs_p, train=False)

        pmask_c, kv_c = prefix_kv_cache(m, obs_clean_pp)
        pmask_a, kv_a = prefix_kv_cache(m, obs_p_pp)

        x_clean = noise
        x_adv   = noise
        total   = jnp.zeros(())
        for step in range(num_steps):
            t     = 1.0 + step * dt
            t_arr = jnp.full((B,), t)
            w     = step_weights[step]
            v_c = suffix_velocity(m, obs_clean_pp, x_clean, t_arr, pmask_c, kv_c)
            v_a = suffix_velocity(m, obs_p_pp,     x_adv,   t_arr, pmask_a, kv_a)
            total += w * jnp.mean(jnp.sum((v_a - v_c) ** 2, axis=-1))
            x_clean = x_clean + dt * v_c   # gradient flows only through the current-step velocity
            x_adv   = x_adv   + dt * v_a

        loss = -total + tv_weight * _total_variation(patch_01)
        if l2_weight > 0.0 and base is not None:
            loss = loss + l2_weight * _l2_to_base(patch_01, base)
        return loss, total

    return loss_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DRIFT adversarial patch attack on flow-matching VLAs.")
    # optimization
    p.add_argument("--out",         default="patches/drift_patch.npy")
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--patch-size",  type=int,   default=32)
    p.add_argument("--pgd-steps",   type=int,   default=500)
    p.add_argument("--alpha",       type=float, default=0.01, help="PGD step size.")
    p.add_argument("--seed",        type=int,   default=0)
    # denoising-step weighting (the DRIFT less-is-more knob)
    p.add_argument("--num-steps",   type=int,   default=10, help="Denoising steps per PGD iter (K).")
    p.add_argument("--top-k-steps", type=int,   default=1,
                   help="Attack the first K' denoising steps with uniform weight. "
                        "Default 1 = first-step-only DRIFT (recommended); 3/5 for the less-is-more ablation.")
    # placement
    p.add_argument("--pos-y",       type=int,   default=0, help="Patch top-left y in the agent image.")
    p.add_argument("--pos-x",       type=int,   default=0, help="Patch top-left x in the agent image.")
    p.add_argument("--wrist-pos-y", type=int,   default=None, help="Patch top-left y in the wrist image (default: bottom-center).")
    p.add_argument("--wrist-pos-x", type=int,   default=None, help="Patch top-left x in the wrist image (default: bottom-center).")
    p.add_argument("--attack-base", action="store_true", help="Also patch the base/agentview camera (default: wrist-only, the DRIFT config).")
    # stealth (optional)
    p.add_argument("--init-image",  default=None, help="Init patch from an image (default: gray 0.5).")
    p.add_argument("--init-noise",  action="store_true", help="Init patch from random noise (seeded).")
    p.add_argument("--epsilon",     type=float, default=None, help="L-inf budget around the init image ([0,1] scale). Off by default.")
    p.add_argument("--tv-weight",   type=float, default=0.0, help="Total-variation smoothness weight.")
    p.add_argument("--l2-weight",   type=float, default=0.0, help="L2-to-base naturalness weight (needs --init-image).")
    p.add_argument("--grayscale",   action="store_true", help="Constrain the patch to grayscale (R=G=B).")
    # model + data
    p.add_argument("--train-config",   default=DEFAULT_TRAIN_CONFIG, help="openpi train config (pi0_libero | pi05_libero).")
    p.add_argument("--checkpoint-dir", default=None, help="Checkpoint dir (default: $PI0_CHECKPOINT).")
    p.add_argument("--libero-obs-path", required=True, help="LIBERO obs pool .npy from collect_libero_obs.py.")
    args = p.parse_args()

    # ---- model ----
    checkpoint_dir = args.checkpoint_dir or os.environ.get("PI0_CHECKPOINT")
    if not checkpoint_dir:
        sys.exit("ERROR: pass --checkpoint-dir or set PI0_CHECKPOINT (path to the pi0/pi05 checkpoint).")
    print(f"Loading model: config={args.train_config}  ckpt={checkpoint_dir}")
    train_config = _config.get_config(args.train_config)
    policy       = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    model        = policy._model
    graphdef, model_state = nnx.split(model)
    print(f"  action_horizon={model.action_horizon}  action_dim={model.action_dim}")

    # ---- data ----
    attack_base = args.attack_base   # default: wrist-only (DRIFT config)
    print("Dataset: LIBERO obs pool (per-step random sampling)")
    loader    = LIBERODataLoader(args.libero_obs_path, policy._input_transform,
                                 batch_size=args.batch_size, seed=args.seed)
    obs_clean = loader.next_batch()
    B = obs_clean.state.shape[0]
    print(f"  Batch: {B}  image: {obs_clean.images['base_0_rgb'].shape}")

    rng   = jax.random.key(args.seed)
    ph = pw = args.patch_size
    IMG_H = IMG_W = 224
    wrist_pos_y = args.wrist_pos_y if args.wrist_pos_y is not None else IMG_H - ph
    wrist_pos_x = args.wrist_pos_x if args.wrist_pos_x is not None else (IMG_W - pw) // 2

    # ---- step weighting ----
    assert 0 < args.top_k_steps <= args.num_steps, f"--top-k-steps must be in [1, {args.num_steps}]"
    step_weights = [1.0 if k < args.top_k_steps else 0.0 for k in range(args.num_steps)]
    weight_desc  = f"top-{args.top_k_steps}-steps (uniform)"

    # ---- patch init (base = center of the epsilon-ball) ----
    if args.init_image:
        base = load_init_image(args.init_image, ph, pw); init_str = f"image({args.init_image})"
    elif args.init_noise:
        base = jax.random.uniform(jax.random.key(args.seed), (ph, pw, 3), dtype=jnp.float32); init_str = f"noise(seed={args.seed})"
    else:
        base = jnp.full((ph, pw, 3), 0.5, dtype=jnp.float32); init_str = "gray(0.5)"
    if args.grayscale:
        base = jnp.broadcast_to(base.mean(-1, keepdims=True), base.shape); init_str += "+grayscale"
    patch = base
    eps_str = f"eps={args.epsilon}" if args.epsilon is not None else "eps=inf(none)"
    print(f"  Stealth: init={init_str}  {eps_str}  tv_weight={args.tv_weight}  l2_weight={args.l2_weight}")

    cameras = "wrist-only" if not attack_base else "base+wrist"
    if attack_base:
        loc = f"base@(y={args.pos_y},x={args.pos_x}) + wrist@(y={wrist_pos_y},x={wrist_pos_x})"
    else:
        loc = f"wrist@(y={wrist_pos_y},x={wrist_pos_x})"
    print(f"\nDRIFT PGD: {args.pgd_steps} steps  alpha={args.alpha}  K={args.num_steps}  {weight_desc}")
    print(f"  Patch {ph}x{pw}px  fixed {loc}  cameras={cameras}")
    print("  (First step JIT-compiles — may take a few minutes)")

    loss_fn = build_drift_loss(
        graphdef, B, model.action_horizon, model.action_dim, args.num_steps,
        pos_y=args.pos_y, pos_x=args.pos_x, wrist_pos_y=wrist_pos_y, wrist_pos_x=wrist_pos_x,
        step_weights=step_weights, tv_weight=args.tv_weight, l2_weight=args.l2_weight,
        base=base if args.l2_weight > 0.0 else None, attack_base=attack_base,
    )
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))

    out_path     = pathlib.Path(args.out)
    preview_path = out_path.parent / (out_path.stem + "_preview.png")
    if attack_base:
        save_patch_preview(obs_clean, patch, preview_path, args.pos_y, args.pos_x, cam_key="base_0_rgb")
    else:
        save_patch_preview(obs_clean, patch, preview_path, wrist_pos_y, wrist_pos_x, cam_key=WRIST_KEY)

    log_records = []
    for step in range(args.pgd_steps):
        if step > 0:
            obs_clean = loader.next_batch()
        rng, step_key = jax.random.split(rng)
        (loss, total), grad = grad_fn(patch, model_state, step_key, obs_clean)
        g = grad.mean(axis=-1, keepdims=True) if args.grayscale else grad
        patch = jnp.clip(patch - args.alpha * jnp.sign(g), 0.0, 1.0)
        if args.epsilon is not None:
            patch = jnp.clip(base + jnp.clip(patch - base, -args.epsilon, args.epsilon), 0.0, 1.0)

        divergence = float(total)
        log_records.append((step, divergence))
        if step % 20 == 0 or step == args.pgd_steps - 1:
            print(f"  [{step:4d}/{args.pgd_steps}]  divergence = {divergence:.6f}", flush=True)

    # ---- save ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch_uint8 = (np.array(patch) * 255).astype(np.uint8)
    np.save(str(out_path), patch_uint8)
    from PIL import Image
    Image.fromarray(patch_uint8).save(str(out_path.with_suffix(".png")))
    print(f"\nPatch saved → {out_path}  +  {out_path.with_suffix('.png')}")

    log_path = out_path.with_suffix(".csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "divergence"]); w.writerows(log_records)
    print(f"Divergence log saved → {log_path}")


if __name__ == "__main__":
    main()
