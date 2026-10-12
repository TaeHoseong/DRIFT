"""Adversarial patch loading utility."""

import logging
import pathlib

import numpy as np


def load_patch(patch_path: str) -> np.ndarray:
    """Load a patch from a .npy or image file. Returns uint8 H×W×3."""
    p = pathlib.Path(patch_path)
    if p.suffix == ".npy":
        patch = np.load(p)
    else:
        from PIL import Image
        patch = np.array(Image.open(p).convert("RGB"))

    if patch.dtype != np.uint8:
        patch = (np.clip(patch, 0.0, 1.0) * 255).astype(np.uint8) if patch.max() <= 1.0 \
                else patch.astype(np.uint8)

    logging.info("Loaded patch from %s  shape=%s", patch_path, patch.shape)
    return patch
