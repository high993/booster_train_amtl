# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint trained with the local AMTL RSL-RL components."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time
from pathlib import Path

AMTL_REPO_ROOT = Path(__file__).resolve().parents[2]
AMTL_SOURCE_PATH = AMTL_REPO_ROOT / "source" / "booster_train"
sys.path.insert(0, str(AMTL_SOURCE_PATH))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RL agent trained with the local AMTL RSL-RL pipeline.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from amtl.on_policy_runner import OnPolicyRunner
from amtl.reward_manager import RewardManager as AMTLRewardManager

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.envs.manager_based_rl_env as manager_based_rl_env
import isaaclab.managers as isaaclab_managers
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401

# Make manager-based environments construct the local AMTL reward manager.
isaaclab_managers.RewardManager = AMTLRewardManager
manager_based_rl_env.RewardManager = AMTLRewardManager


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with an AMTL-trained RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_amtl"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = ppo_runner.alg.policy

    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    export_policy_as_jit(
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename=f"{agent_cfg.experiment_name}_{run_name}.pt",
    )
    export_policy_as_onnx(
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename=f"{agent_cfg.experiment_name}_{run_name}.onnx",
    )

    if args_cli.headless and not args_cli.video:
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
