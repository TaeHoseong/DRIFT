"""Shared utilities for the DRIFT adversarial patch attack on flow-matching VLAs."""

import os
import pathlib
import sys

import einops
import jax.numpy as jnp
import numpy as np

# openpi is an external dependency: point OPENPI_ROOT at a cloned openpi repo.
_OPENPI_ROOT = os.environ.get("OPENPI_ROOT")
if _OPENPI_ROOT:
    sys.path.insert(0, str(pathlib.Path(_OPENPI_ROOT) / "src"))

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


def _stack(values):
    return np.stack([np.asarray(v) for v in values])


def preprocess_obs_batch(transform, obs_data: dict, batch_size: int) -> _model.Observation:
    n = min(batch_size, len(obs_data["observation/image"]))
    # Support both "prompts" (list, multi-task) and "prompt" (str, legacy).
    prompts = obs_data.get("prompts") or [str(obs_data["prompt"])] * n
    processed = []
    for i in range(n):
        raw = {
            "observation/image":       obs_data["observation/image"][i],
            "observation/wrist_image": obs_data["observation/wrist_image"][i],
            "observation/state":       obs_data["observation/state"][i],
            "prompt":                  prompts[i],
        }
        processed.append(transform(raw))

    batched = {}
    for key in processed[0]:
        val0 = processed[0][key]
        if isinstance(val0, dict):
            batched[key] = {k: _stack([p[key][k] for p in processed]) for k in val0}
        elif isinstance(val0, (np.ndarray, np.generic)):
            batched[key] = _stack([p[key] for p in processed])

    obs_np = _model.Observation.from_dict(batched)
    return _model.Observation(
        images={k: jnp.asarray(v) for k, v in obs_np.images.items()},
        image_masks={k: jnp.asarray(v) for k, v in obs_np.image_masks.items()},
        state=jnp.asarray(obs_np.state),
        tokenized_prompt=(
            jnp.asarray(obs_np.tokenized_prompt)
            if obs_np.tokenized_prompt is not None else None
        ),
        tokenized_prompt_mask=(
            jnp.asarray(obs_np.tokenized_prompt_mask)
            if obs_np.tokenized_prompt_mask is not None else None
        ),
    )


WRIST_KEY = "left_wrist_0_rgb"


def apply_patch(
    obs: _model.Observation,
    patch_01: jnp.ndarray,
    pos_y: int,
    pos_x: int,
    wrist_pos_y=None,
    wrist_pos_x=None,
    attack_base: bool = True,
) -> _model.Observation:
    """Overwrite the patch region in the agent view (and optionally the wrist view)
    with patch_01 in [0,1]."""
    ph, pw = patch_01.shape[:2]
    B = obs.images["base_0_rgb"].shape[0]
    patch_norm = patch_01 * 2.0 - 1.0
    patch_tiled = jnp.broadcast_to(patch_norm[None], (B, ph, pw, 3))

    images = dict(obs.images)
    if attack_base:
        images["base_0_rgb"] = images["base_0_rgb"].at[
            :, pos_y:pos_y + ph, pos_x:pos_x + pw, :
        ].set(patch_tiled)

    if wrist_pos_y is not None and WRIST_KEY in images:
        images[WRIST_KEY] = images[WRIST_KEY].at[
            :, wrist_pos_y:wrist_pos_y + ph, wrist_pos_x:wrist_pos_x + pw, :
        ].set(patch_tiled)

    return _model.Observation(
        images=images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask,
    )


def prefix_kv_cache(m, obs):
    """Embed prefix tokens and return (prefix_mask, kv_cache)."""
    prefix_tokens, prefix_mask, prefix_ar_mask = m.embed_prefix(obs)
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = m.PaliGemma.llm(
        [prefix_tokens, None], mask=attn_mask, positions=positions
    )
    return prefix_mask, kv_cache


def suffix_velocity(m, obs, x_t, t_arr, prefix_mask, kv_cache):
    """Single denoising step via precomputed KV cache → velocity [B, ah, ad]."""
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = m.embed_suffix(obs, x_t, t_arr)
    prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    suffix_attn = make_attn_mask(suffix_mask, suffix_ar_mask)
    full_mask   = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
    positions   = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1) - 1
    )
    (_, suffix_out), _ = m.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    return m.action_out_proj(suffix_out[:, -m.action_horizon:])
