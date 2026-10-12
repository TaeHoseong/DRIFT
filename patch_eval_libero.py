#!/usr/bin/env python3
"""
DRIFT patch evaluation on the LIBERO benchmark for flow-matching VLAs.

The patch is applied as a digital overlay on the camera image(s) before inference,
exactly matching the training setup. By default it is placed on the **wrist camera
at the bottom-center** (the DRIFT configuration); the base/agentview camera can be
patched additionally with --base.

Requires a running policy server (openpi `serve_policy.py`) and the LIBERO package
on PYTHONPATH.

Usage:
    export PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/third_party/libero

    # clean baseline (no patch)
    python patch_eval_libero.py --task-suite-name libero_spatial --exp-name baseline

    # DRIFT patch (wrist bottom-center overlay is the default)
    python patch_eval_libero.py \
        --task-suite-name libero_spatial \
        --patch-path patches/drift_pi0_32px.npy \
        --exp-name drift_spatial
"""

import collections
import dataclasses
import datetime
import json
import logging
import pathlib
import sys
from typing import Optional

import imageio
import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Allow `import utils` regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from utils.patch import load_patch
from utils.env import make_env, quat2axisangle

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256  # native LIBERO camera resolution


@dataclasses.dataclass
class Args:
    # model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    seed: int = 7

    # patch (digital overlay)
    patch_path: Optional[str] = None
    """Path to the patch (.npy or image). None = clean baseline (no patch)."""
    wrist_pos_y: Optional[int] = None
    """Patch top-left y in the wrist image. Default: bottom-center for the patch size."""
    wrist_pos_x: Optional[int] = None
    """Patch top-left x in the wrist image. Default: bottom-center for the patch size."""
    no_wrist: bool = False
    """Skip the wrist camera (only meaningful together with --base)."""
    base: bool = False
    """Also overlay the patch on the base/agentview camera (default: wrist only)."""
    base_pos_y: int = 0
    """Patch top-left y in the base image (with --base)."""
    base_pos_x: int = 0
    """Patch top-left x in the base image (with --base)."""

    # output
    exp_name: str = ""
    """Experiment name. Videos/results saved under results/libero/{exp_name}/."""
    results_dir: Optional[str] = None
    """Override results root directory. Default: results/libero/"""


def eval_libero_with_patch(args: Args) -> None:
    np.random.seed(args.seed)

    use_patch = args.patch_path is not None
    exp_dir = args.exp_name if args.exp_name else ("drift" if use_patch else "baseline")
    if args.results_dir:
        video_dir = pathlib.Path(args.results_dir) / exp_dir
    else:
        video_dir = pathlib.Path(__file__).parent / "results" / "libero" / exp_dir

    patch_overlay = None
    if use_patch:
        patch_overlay = load_patch(args.patch_path)          # uint8 [ph, pw, 3]
        ph, pw = patch_overlay.shape[:2]
        H = W = args.resize_size
        # default anchors: wrist bottom-center, base top-left
        wy = args.wrist_pos_y if args.wrist_pos_y is not None else H - ph
        wx = args.wrist_pos_x if args.wrist_pos_x is not None else (W - pw) // 2
        by, bx = args.base_pos_y, args.base_pos_x
        cams = ("base+wrist" if (args.base and not args.no_wrist) else
                ("base" if args.base else "wrist"))
        logging.info("Overlay patch %dx%d on %s  (wrist@%d,%d)", ph, pw, cams, wy, wx)
    logging.info("Mode: %s  →  %s", "overlay" if use_patch else "baseline", video_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[args.task_suite_name]()
    num_tasks      = task_suite.n_tasks
    logging.info("Task suite: %s  (%d tasks)", args.task_suite_name, num_tasks)

    max_steps_map = {
        "libero_spatial": 220, "libero_object": 280,
        "libero_goal":    300, "libero_10":     520, "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 300)

    video_dir.mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_eps = total_ok = 0
    task_results = []

    for task_id in tqdm.tqdm(range(num_tasks)):
        task       = task_suite.get_task(task_id)
        inits      = task_suite.get_task_init_states(task_id)
        env, task_desc = make_env(task, ENV_RES, args.seed)

        task_eps = task_ok = 0
        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(inits[ep_idx])

            action_plan = collections.deque()
            t    = 0
            done = False
            wrist_frames = []
            agent_frames = []

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    agent_model = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(agent_img, args.resize_size, args.resize_size)
                    )
                    wrist_model = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if patch_overlay is not None:
                        if args.base:
                            agent_model = agent_model.copy()
                            agent_model[by:by+ph, bx:bx+pw] = patch_overlay
                        if not args.no_wrist:
                            wrist_model = wrist_model.copy()
                            wrist_model[wy:wy+ph, wx:wx+pw] = patch_overlay

                    wrist_frames.append(wrist_model)
                    agent_frames.append(agent_model)

                    if not action_plan:
                        obs_dict = {
                            "observation/image":       agent_model,
                            "observation/wrist_image": wrist_model,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_desc),
                        }
                        chunk = client.infer(obs_dict)["actions"]
                        assert len(chunk) >= args.replan_steps
                        action_plan.extend(chunk[: args.replan_steps])

                    obs, _, done, _ = env.step(action_plan.popleft().tolist())
                    if done:
                        task_ok += 1
                        total_ok += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Episode error: %s", e)
                    break

            task_eps += 1
            total_eps += 1

            suffix   = "success" if done else "failure"
            tag      = task_desc.replace(" ", "_")[:40]
            out_path = video_dir / f"ep{ep_idx:03d}_{tag}_{suffix}.mp4"
            if wrist_frames:
                imageio.mimwrite(str(out_path), [np.asarray(f) for f in wrist_frames], fps=10)
            if agent_frames:
                agent_path = out_path.with_name(out_path.stem + "_agentview.mp4")
                imageio.mimwrite(str(agent_path), [np.asarray(f) for f in agent_frames], fps=10)

        sr = task_ok / max(task_eps, 1)
        logging.info("Task '%s'  %d/%d  (%.1f%%)", task_desc, task_ok, task_eps, sr * 100)
        task_results.append({
            "task_id":     task_id,
            "description": task_desc,
            "success":     task_ok,
            "total":       task_eps,
            "rate":        round(sr, 4),
        })
        env.close()

    overall_rate = total_ok / max(total_eps, 1)
    logging.info("\nFinal  %d/%d  (%.1f%%)", total_ok, total_eps, overall_rate * 100)

    results = {
        "exp_name":   args.exp_name,
        "task_suite": args.task_suite_name,
        "patch_path": args.patch_path,
        "timestamp":  datetime.datetime.now().isoformat(),
        "overall":    {"success": total_ok, "total": total_eps, "rate": round(overall_rate, 4)},
        "tasks":      task_results,
    }
    results_path = video_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("Results saved → %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero_with_patch(tyro.cli(Args))
