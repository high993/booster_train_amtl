# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play an exported TorchScript policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an exported TorchScript policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--policy",
    "--checkpoint",
    dest="policy",
    type=str,
    required=True,
    help="Path to an exported TorchScript policy (.pt).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401


def _load_torchscript_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
    """Load an exported TorchScript policy."""
    if not policy_path.endswith(".pt"):
        raise ValueError(
            f"Unsupported exported policy format: {policy_path}. "
            "This script currently supports TorchScript '.pt' files only."
        )
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    return policy


def _extract_policy_obs(obs) -> torch.Tensor:
    """Extract the actor observation tensor expected by the exported policy."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, torch.Tensor):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        pass
    raise TypeError(f"Unsupported observation type for exported policy playback: {type(obs)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with an exported TorchScript policy."""
    policy_path = retrieve_file_path(args_cli.policy)
    log_dir = os.path.dirname(policy_path)

    # keep env setup consistent with the trained policy.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_exported"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during exported-policy playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for Isaac Lab / RSL-RL observation and action handling
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading exported TorchScript policy from: {policy_path}")
    policy = _load_torchscript_policy(policy_path, device=env.unwrapped.device)
    is_recurrent = hasattr(policy, "reset")

    if is_recurrent and env.num_envs != 1:
        raise ValueError(
            "The exported recurrent TorchScript policy only supports a single environment. "
            "Run play_exported.py with --num_envs 1."
        )

    if args_cli.headless and not args_cli.video:
        print("[INFO] Headless mode without video is enabled. Running playback without rendering.")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.reset()
    obs = _extract_policy_obs(obs)
    if is_recurrent:
        policy.reset()

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            if isinstance(actions, tuple):
                actions = actions[0]
            obs, _, dones, _ = env.step(actions)
            obs = _extract_policy_obs(obs)
            if is_recurrent and torch.any(dones):
                policy.reset()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
