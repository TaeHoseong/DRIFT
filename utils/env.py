"""LIBERO environment creation and observation utilities."""

import math
import pathlib

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def make_env(task, resolution: int, seed: int):
    """Create an OffScreenRenderEnv for the given LIBERO task. Returns (env, task_desc)."""
    task_desc = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task_desc


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a (x, y, z, w) quaternion to an axis-angle vector."""
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
