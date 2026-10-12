#!/usr/bin/env python3
"""
Collect a pool of LIBERO observations for training the DRIFT patch.

Rolls out the (clean) policy across LIBERO tasks and stores the per-replan-step
observations. Requires a running policy server (openpi `serve_policy.py`) and the
LIBERO package on PYTHONPATH.

Usage:
    # Terminal 1: start the policy server (see scripts/serve.sh)

    # Terminal 2: collect observations
    export PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/third_party/libero
    MUJOCO_GL=egl python collect_libero_obs.py --out collected_obs_libero.npy

Output: collected_obs_libero.npy (dict)
    "observation/image":       [N, 224, 224, 3] uint8
    "observation/wrist_image": [N, 224, 224, 3] uint8
    "observation/state":       [N, 8]           float32
    "prompts":                 list[str]  len=N
"""

import argparse
import collections
import pathlib
import sys

import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.env import make_env, quat2axisangle

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES      = 256
RESIZE_SIZE  = 224
REPLAN_STEPS = 5

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
}

ALL_SUITES = list(MAX_STEPS.keys())


def collect_episode(env, task_suite, task_id, task_desc, client, max_steps,
                    num_steps_wait, seed):
    """Run one episode -> list of obs_dicts (one per replan step)."""
    np.random.seed(seed)
    env.reset()
    inits = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(inits[seed % len(inits)])

    obs_list    = []
    action_plan = collections.deque()
    t = 0

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        agent_224 = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(agent_img, RESIZE_SIZE, RESIZE_SIZE))
        wrist_224 = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))

        if not action_plan:
            obs_dict = {
                "observation/image":       agent_224,
                "observation/wrist_image": wrist_224,
                "observation/state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )).astype(np.float32),
                "prompt": str(task_desc),
            }
            obs_list.append({k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in obs_dict.items()})
            chunk = client.infer(obs_dict)["actions"]
            action_plan.extend(chunk[:REPLAN_STEPS])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action)
        t += 1
        if done:
            break

    return obs_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",           default="0.0.0.0")
    parser.add_argument("--port",           type=int, default=8000)
    parser.add_argument("--suites",         nargs="+", default=ALL_SUITES,
                        help="LIBERO suites to collect from (default: all 4).")
    parser.add_argument("--n-episodes",     type=int, default=3,
                        help="Episodes per task per suite (default: 3).")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--out",            default="collected_obs_libero.npy")
    args = parser.parse_args()

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Connected to policy server @ {args.host}:{args.port}")

    benchmark_dict = benchmark.get_benchmark_dict()

    all_images, all_wrists, all_states, all_prompts = [], [], [], []
    total = 0

    for suite_name in args.suites:
        max_steps  = MAX_STEPS[suite_name]
        task_suite = benchmark_dict[suite_name]()
        n_tasks    = task_suite.n_tasks
        print(f"\n[{suite_name}]  tasks={n_tasks}  episodes/task={args.n_episodes}")

        for task_id in range(n_tasks):
            task      = task_suite.get_task(task_id)
            env, task_desc = make_env(task, ENV_RES, seed=0)
            env._task_suite = task_suite
            env._task_id    = task_id
            print(f"  task {task_id:02d}: {task_desc[:60]}")

            for ep in range(args.n_episodes):
                obs_list = collect_episode(
                    env, task_suite, task_id, task_desc, client,
                    max_steps, args.num_steps_wait, seed=ep)
                n = len(obs_list)
                print(f"    ep{ep}: {n} frames")

                for o in obs_list:
                    all_images.append(o["observation/image"])
                    all_wrists.append(o["observation/wrist_image"])
                    all_states.append(o["observation/state"])
                    all_prompts.append(o["prompt"])
                total += n

            env.close()

    print(f"\nTotal frames collected: {total}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "observation/image":       np.stack(all_images,  axis=0),
        "observation/wrist_image": np.stack(all_wrists,  axis=0),
        "observation/state":       np.stack(all_states,  axis=0),
        "prompts":                 all_prompts,
    }
    np.save(str(out_path), data)
    print(f"Saved → {out_path}")
    print(f"  image shape: {data['observation/image'].shape}")
    print(f"  state shape: {data['observation/state'].shape}")


if __name__ == "__main__":
    main()
