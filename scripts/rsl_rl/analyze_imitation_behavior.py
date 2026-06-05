"""Offline diagnostics for motion-imitation checkpoints.

This script evaluates one or more saved AMTL/PPO checkpoints and reports whether the
policy is tracking the reference motion or exploiting a standing/stability local optimum.
It does not modify training behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

AMTL_REPO_ROOT = Path(__file__).resolve().parents[2]
AMTL_SOURCE_PATH = AMTL_REPO_ROOT / "source" / "booster_train"
sys.path.insert(0, str(AMTL_SOURCE_PATH))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Analyze imitation behavior for AMTL checkpoints.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to use for evaluation.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--analysis-checkpoint", type=str, default=None, help="Optional explicit checkpoint path.")
parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing model_*.pt checkpoints.")
parser.add_argument("--checkpoint-start", type=int, default=350, help="Inclusive checkpoint iteration start.")
parser.add_argument("--checkpoint-end", type=int, default=450, help="Inclusive checkpoint iteration end.")
parser.add_argument("--checkpoint-step", type=int, default=25, help="Checkpoint iteration step.")
parser.add_argument("--episodes-per-checkpoint", type=int, default=48, help="Completed episodes to collect.")
parser.add_argument("--max-steps", type=int, default=6000, help="Maximum environment steps per checkpoint.")
parser.add_argument(
    "--standing-speed-threshold",
    type=float,
    default=0.2,
    help="Robot root speed threshold below which the robot is treated as standing.",
)
parser.add_argument(
    "--reference-speed-threshold",
    type=float,
    default=0.5,
    help="Reference root speed threshold above which the reference is treated as moving.",
)
parser.add_argument(
    "--reference-contact-height-margin",
    type=float,
    default=0.035,
    help="Added margin above each foot's minimum reference height for heuristic contact detection.",
)
parser.add_argument(
    "--reference-contact-speed-threshold",
    type=float,
    default=0.75,
    help="Maximum reference foot speed for heuristic contact detection.",
)
parser.add_argument(
    "--termination-history-length",
    type=int,
    default=25,
    help="Number of steps before termination to retain for drift-vs-sudden analysis.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write CSV/JSON/Markdown outputs. Defaults to <run-dir>/imitation_diagnostics.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from amtl.on_policy_runner import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
import isaaclab.envs.manager_based_rl_env as manager_based_rl_env
import isaaclab.managers as isaaclab_managers
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from amtl.reward_manager import RewardManager as AMTLRewardManager

import booster_train.tasks  # noqa: F401

isaaclab_managers.RewardManager = AMTLRewardManager
manager_based_rl_env.RewardManager = AMTLRewardManager


SELECTED_PHASE_TERMS = {
    "body_pos": "motion_body_pos",
    "body_ori": "motion_body_ori",
    "body_lin_vel": "motion_body_lin_vel",
    "body_ang_vel": "motion_body_ang_vel",
    "foot_pos": "motion_foot_pos",
    "foot_ori": "motion_foot_ori",
    "trunk_pos": "motion_trunk_pos",
    "trunk_ori": "motion_trunk_ori",
    "global_anchor_pos": "motion_global_anchor_pos",
    "global_anchor_ori": "motion_global_anchor_ori",
}

EE_BODY_NAMES = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]
EE_BODY_POS_THRESHOLD = 0.25


@dataclass
class FootContactHeuristic:
    height_threshold: float
    speed_threshold: float


def _extract_observations(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:
        return tensor
    return tensor.squeeze(-1)


def _tensor_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return float("nan")
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _resolve_checkpoints(run_dir: str | None, checkpoint: str | None, start: int, end: int, step: int) -> list[tuple[int, str]]:
    if checkpoint:
        checkpoint_path = os.path.abspath(checkpoint)
        stem = Path(checkpoint_path).stem
        try:
            iteration = int(stem.split("_")[-1])
        except ValueError:
            iteration = -1
        return [(iteration, checkpoint_path)]
    if run_dir is None:
        raise ValueError("Either --checkpoint or --run-dir must be provided.")

    checkpoints: list[tuple[int, str]] = []
    for iteration in range(start, end + 1, step):
        path = os.path.join(run_dir, f"model_{iteration}.pt")
        if os.path.exists(path):
            checkpoints.append((iteration, path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir} for iterations {start}-{end}.")
    return checkpoints


def _derive_reference_contact_heuristics(command, height_margin: float, speed_threshold: float) -> dict[str, FootContactHeuristic]:
    heuristics: dict[str, FootContactHeuristic] = {}
    for foot_name in ["left_foot_link", "right_foot_link"]:
        foot_index = command.cfg.body_names.index(foot_name)
        foot_heights = command.motion.body_pos_w[:, foot_index, 2]
        height_threshold = float(torch.min(foot_heights).item() + height_margin)
        heuristics[foot_name] = FootContactHeuristic(
            height_threshold=height_threshold,
            speed_threshold=speed_threshold,
        )
    return heuristics


def _compute_reference_contact(command, foot_name: str, heuristic: FootContactHeuristic) -> torch.Tensor:
    foot_index = command.cfg.body_names.index(foot_name)
    foot_height = command.body_pos_w[:, foot_index, 2]
    foot_speed = torch.norm(command.body_lin_vel_w[:, foot_index], dim=-1)
    return (foot_height <= heuristic.height_threshold) & (foot_speed <= heuristic.speed_threshold)


def _compute_robot_contact(contact_sensor, foot_body_index: int) -> torch.Tensor:
    net_forces = contact_sensor.data.net_forces_w[:, foot_body_index]
    force_norm = torch.norm(net_forces, dim=-1)
    return force_norm > contact_sensor.cfg.force_threshold


def _compute_ee_body_pos_metric(command) -> torch.Tensor:
    body_indexes = [command.cfg.body_names.index(name) for name in EE_BODY_NAMES]
    z_error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.max(z_error, dim=-1).values


def _format_phase_bin(bin_index: int) -> str:
    start = bin_index * 10
    end = start + 10
    return f"{start:02d}-{end:02d}%"


class DiagnosticsAccumulator:
    def __init__(
        self,
        reward_term_names: list[str],
        phase_term_names: dict[str, str],
        termination_history_length: int,
        standing_speed_threshold: float,
        reference_speed_threshold: float,
    ) -> None:
        self.reward_term_names = reward_term_names
        self.reward_index = {name: idx for idx, name in enumerate(reward_term_names)}
        self.phase_term_names = phase_term_names
        self.termination_history_length = termination_history_length
        self.standing_speed_threshold = standing_speed_threshold
        self.reference_speed_threshold = reference_speed_threshold

        self.phase_bin_counts = np.zeros(10, dtype=np.int64)
        self.phase_bin_terminations = np.zeros(10, dtype=np.int64)
        self.phase_bin_reward_sums = {
            alias: np.zeros(10, dtype=np.float64) for alias in phase_term_names
        }

        self.root_speed_ratio_samples: list[float] = []
        self.root_speed_ratio_moving_samples: list[float] = []
        self.root_position_errors: list[float] = []
        self.root_orientation_errors: list[float] = []
        self.root_linear_velocity_errors: list[float] = []
        self.root_angular_velocity_errors: list[float] = []
        self.robot_root_speeds: list[float] = []
        self.reference_root_speeds: list[float] = []
        self.total_step_samples = 0

        self.contact_counts = {
            "left_foot_link": {"match": 0, "mismatch": 0, "robot_contact": 0, "reference_contact": 0},
            "right_foot_link": {"match": 0, "mismatch": 0, "robot_contact": 0, "reference_contact": 0},
        }
        self.phase_bin_contact_mismatches = {
            "left_foot_link": np.zeros(10, dtype=np.int64),
            "right_foot_link": np.zeros(10, dtype=np.int64),
            "any_mismatch": np.zeros(10, dtype=np.int64),
        }
        self.contact_reward_summary = {
            "all_match": {"reward_total": 0.0, "step_count": 0},
            "any_mismatch": {"reward_total": 0.0, "step_count": 0},
            "left_foot_link": {
                "match_reward_total": 0.0,
                "match_step_count": 0,
                "mismatch_reward_total": 0.0,
                "mismatch_step_count": 0,
            },
            "right_foot_link": {
                "match_reward_total": 0.0,
                "match_step_count": 0,
                "mismatch_reward_total": 0.0,
                "mismatch_step_count": 0,
            },
        }
        self.contact_mismatch_samples = {
            "left_foot_link": [],
            "right_foot_link": [],
            "any_mismatch": [],
        }

        self.standing_reward_total = 0.0
        self.standing_reward_breakdown = defaultdict(float)
        self.standing_step_count = 0

        self.weighted_reward_sums = defaultdict(float)
        self.raw_reward_sums = defaultdict(float)
        self.total_positive_weighted_reward = 0.0

        self.termination_metric_samples: list[float] = []
        self.termination_metric_episode_end: list[float] = []
        self.termination_histories: list[list[float]] = []

        self.completed_episode_returns: list[float] = []
        self.completed_episode_lengths: list[int] = []

        self._episode_returns: dict[int, float] = {}
        self._episode_lengths: dict[int, int] = {}
        self._episode_termination_history: dict[int, list[float]] = {}
        self._episode_contact_history: dict[int, list[dict[str, float | bool | int]]] = {}
        self.pre_failure_contact_histories: list[list[dict[str, float | bool | int]]] = []

    def observe_pre_step(
        self,
        env_ids: torch.Tensor,
        phase_bins: torch.Tensor,
        ee_metric: torch.Tensor,
        root_pos_error: torch.Tensor,
        root_ori_error: torch.Tensor,
        root_lin_vel_error: torch.Tensor,
        root_ang_vel_error: torch.Tensor,
        robot_root_speed: torch.Tensor,
        reference_root_speed: torch.Tensor,
        robot_contacts: dict[str, torch.Tensor],
        reference_contacts: dict[str, torch.Tensor],
        contact_mismatches: dict[str, torch.Tensor],
        any_contact_mismatch: torch.Tensor,
    ) -> None:
        env_ids_np = env_ids.cpu().numpy()
        phase_bins_np = phase_bins.cpu().numpy()
        for env_id, phase_bin in zip(env_ids_np, phase_bins_np, strict=True):
            self.phase_bin_counts[int(phase_bin)] += 1
            history = self._episode_termination_history.setdefault(int(env_id), [])
            history.append(float(ee_metric[env_id].item()))
            if len(history) > self.termination_history_length:
                del history[0]

            self._episode_returns.setdefault(int(env_id), 0.0)
            self._episode_lengths.setdefault(int(env_id), 0)
            self._episode_contact_history.setdefault(int(env_id), [])

        self.termination_metric_samples.extend(ee_metric[env_ids].cpu().tolist())
        self.root_position_errors.extend(root_pos_error[env_ids].cpu().tolist())
        self.root_orientation_errors.extend(root_ori_error[env_ids].cpu().tolist())
        self.root_linear_velocity_errors.extend(root_lin_vel_error[env_ids].cpu().tolist())
        self.root_angular_velocity_errors.extend(root_ang_vel_error[env_ids].cpu().tolist())

        robot_root_speed_cpu = robot_root_speed[env_ids].cpu().numpy()
        reference_root_speed_cpu = reference_root_speed[env_ids].cpu().numpy()
        self.total_step_samples += len(env_ids_np)
        self.robot_root_speeds.extend(robot_root_speed_cpu.tolist())
        self.reference_root_speeds.extend(reference_root_speed_cpu.tolist())
        ratios = robot_root_speed_cpu / (reference_root_speed_cpu + 1.0e-8)
        self.root_speed_ratio_samples.extend(ratios.tolist())
        moving_mask = reference_root_speed_cpu > self.reference_speed_threshold
        if np.any(moving_mask):
            self.root_speed_ratio_moving_samples.extend(ratios[moving_mask].tolist())

        for foot_name in robot_contacts:
            robot_contact = robot_contacts[foot_name][env_ids].cpu().numpy().astype(bool)
            reference_contact = reference_contacts[foot_name][env_ids].cpu().numpy().astype(bool)
            mismatch = contact_mismatches[foot_name][env_ids].cpu().numpy().astype(bool)
            counts = self.contact_counts[foot_name]
            counts["match"] += int(np.sum(robot_contact == reference_contact))
            counts["mismatch"] += int(np.sum(robot_contact != reference_contact))
            counts["robot_contact"] += int(np.sum(robot_contact))
            counts["reference_contact"] += int(np.sum(reference_contact))
            self.contact_mismatch_samples[foot_name].extend(mismatch.astype(np.float64).tolist())
            for phase_bin, mismatch_value in zip(phase_bins_np, mismatch, strict=True):
                self.phase_bin_contact_mismatches[foot_name][int(phase_bin)] += int(mismatch_value)

        any_mismatch_np = any_contact_mismatch[env_ids].cpu().numpy().astype(bool)
        self.contact_mismatch_samples["any_mismatch"].extend(any_mismatch_np.astype(np.float64).tolist())
        for env_id, phase_bin, mismatch_value in zip(env_ids_np, phase_bins_np, any_mismatch_np, strict=True):
            self.phase_bin_contact_mismatches["any_mismatch"][int(phase_bin)] += int(mismatch_value)
            history = self._episode_contact_history.setdefault(int(env_id), [])
            history.append(
                {
                    "phase_bin": int(phase_bin),
                    "ee_metric": float(ee_metric[int(env_id)].item()),
                    "left_mismatch": bool(contact_mismatches["left_foot_link"][int(env_id)].item()),
                    "right_mismatch": bool(contact_mismatches["right_foot_link"][int(env_id)].item()),
                    "any_mismatch": bool(mismatch_value),
                }
            )
            if len(history) > self.termination_history_length:
                del history[0]

    def observe_post_step(
        self,
        env_ids: torch.Tensor,
        phase_bins: torch.Tensor,
        step_reward_raw: torch.Tensor,
        step_reward_weighted_dt: torch.Tensor,
        dones: torch.Tensor,
        robot_root_speed: torch.Tensor,
        reference_root_speed: torch.Tensor,
        contact_mismatches: dict[str, torch.Tensor],
        any_contact_mismatch: torch.Tensor,
    ) -> None:
        env_ids_np = env_ids.cpu().numpy()
        phase_bins_np = phase_bins.cpu().numpy()
        dones_np = dones[env_ids].cpu().numpy().astype(bool)

        for alias, term_name in self.phase_term_names.items():
            term_idx = self.reward_index[term_name]
            values_np = step_reward_raw[env_ids, term_idx].cpu().numpy()
            for phase_bin, value in zip(phase_bins_np, values_np, strict=True):
                self.phase_bin_reward_sums[alias][int(phase_bin)] += float(value)

        for env_id, done, phase_bin in zip(env_ids_np, dones_np, phase_bins_np, strict=True):
            reward_value = float(step_reward_weighted_dt[env_id].sum().item())
            self._episode_returns[int(env_id)] = self._episode_returns.get(int(env_id), 0.0) + reward_value
            self._episode_lengths[int(env_id)] = self._episode_lengths.get(int(env_id), 0) + 1
            if done:
                self.phase_bin_terminations[int(phase_bin)] += 1
                self.completed_episode_returns.append(self._episode_returns.get(int(env_id), 0.0))
                self.completed_episode_lengths.append(self._episode_lengths.get(int(env_id), 0))
                history = list(self._episode_termination_history.get(int(env_id), []))
                if history:
                    self.termination_histories.append(history)
                    self.termination_metric_episode_end.append(history[-1])
                contact_history = list(self._episode_contact_history.get(int(env_id), []))
                if contact_history:
                    self.pre_failure_contact_histories.append(contact_history)
                self._episode_returns[int(env_id)] = 0.0
                self._episode_lengths[int(env_id)] = 0
                self._episode_termination_history[int(env_id)] = []
                self._episode_contact_history[int(env_id)] = []

        robot_root_speed_cpu = robot_root_speed[env_ids].cpu().numpy()
        reference_root_speed_cpu = reference_root_speed[env_ids].cpu().numpy()
        standing_mask = (
            (robot_root_speed_cpu < self.standing_speed_threshold)
            & (reference_root_speed_cpu > self.reference_speed_threshold)
        )
        if np.any(standing_mask):
            standing_env_ids = env_ids_np[standing_mask]
            self.standing_step_count += int(np.sum(standing_mask))
            self.standing_reward_total += float(step_reward_weighted_dt[standing_env_ids].sum().item())
            for term_name, term_idx in self.reward_index.items():
                term_weighted_sum = float(step_reward_weighted_dt[standing_env_ids, term_idx].sum().item())
                self.standing_reward_breakdown[term_name] += term_weighted_sum

        scalar_rewards_np = step_reward_weighted_dt[env_ids].sum(dim=-1).cpu().numpy()
        any_mismatch_np = any_contact_mismatch[env_ids].cpu().numpy().astype(bool)
        all_match_np = ~any_mismatch_np
        self.contact_reward_summary["all_match"]["reward_total"] += float(np.sum(scalar_rewards_np[all_match_np]))
        self.contact_reward_summary["all_match"]["step_count"] += int(np.sum(all_match_np))
        self.contact_reward_summary["any_mismatch"]["reward_total"] += float(np.sum(scalar_rewards_np[any_mismatch_np]))
        self.contact_reward_summary["any_mismatch"]["step_count"] += int(np.sum(any_mismatch_np))
        for foot_name in ["left_foot_link", "right_foot_link"]:
            mismatch_np = contact_mismatches[foot_name][env_ids].cpu().numpy().astype(bool)
            match_np = ~mismatch_np
            foot_summary = self.contact_reward_summary[foot_name]
            foot_summary["match_reward_total"] += float(np.sum(scalar_rewards_np[match_np]))
            foot_summary["match_step_count"] += int(np.sum(match_np))
            foot_summary["mismatch_reward_total"] += float(np.sum(scalar_rewards_np[mismatch_np]))
            foot_summary["mismatch_step_count"] += int(np.sum(mismatch_np))

        for term_name, term_idx in self.reward_index.items():
            self.weighted_reward_sums[term_name] += float(step_reward_weighted_dt[env_ids, term_idx].sum().item())
            self.raw_reward_sums[term_name] += float(step_reward_raw[env_ids, term_idx].sum().item())
            positive_values = step_reward_weighted_dt[env_ids, term_idx]
            self.total_positive_weighted_reward += float(torch.clamp(positive_values, min=0.0).sum().item())

    def build_summary(self, checkpoint_iteration: int, checkpoint_path: str) -> dict:
        phase_rows = []
        for bin_index in range(10):
            count = int(self.phase_bin_counts[bin_index])
            termination_rate = float(self.phase_bin_terminations[bin_index] / count) if count > 0 else float("nan")
            row = {
                "checkpoint_iteration": checkpoint_iteration,
                "checkpoint_path": checkpoint_path,
                "phase_bin": _format_phase_bin(bin_index),
                "sample_count": count,
                "termination_rate": termination_rate,
                "left_contact_mismatch_rate": float(self.phase_bin_contact_mismatches["left_foot_link"][bin_index] / count)
                if count > 0
                else float("nan"),
                "right_contact_mismatch_rate": float(self.phase_bin_contact_mismatches["right_foot_link"][bin_index] / count)
                if count > 0
                else float("nan"),
                "any_contact_mismatch_rate": float(self.phase_bin_contact_mismatches["any_mismatch"][bin_index] / count)
                if count > 0
                else float("nan"),
            }
            for alias, values in self.phase_bin_reward_sums.items():
                row[alias] = float(values[bin_index] / count) if count > 0 else float("nan")
            phase_rows.append(row)

        ratio_values = self.root_speed_ratio_moving_samples or self.root_speed_ratio_samples
        root_ratio_hist_counts, root_ratio_hist_edges = np.histogram(
            np.asarray(ratio_values, dtype=np.float64),
            bins=20,
            range=(0.0, max(2.0, float(np.nanpercentile(ratio_values, 99)) if ratio_values else 2.0)),
        )

        reward_contributions = []
        positive_denominator = self.total_positive_weighted_reward if self.total_positive_weighted_reward > 0 else float("nan")
        for term_name in self.reward_term_names:
            weighted_value = self.weighted_reward_sums[term_name]
            contribution = weighted_value / positive_denominator if positive_denominator == positive_denominator else float("nan")
            reward_contributions.append(
                {
                    "objective_name": term_name,
                    "weighted_reward_sum": weighted_value,
                    "raw_reward_sum": self.raw_reward_sums[term_name],
                    "positive_contribution": contribution,
                }
            )
        reward_contributions.sort(key=lambda item: abs(item["weighted_reward_sum"]), reverse=True)

        standing_breakdown = sorted(
            (
                {
                    "objective_name": name,
                    "weighted_reward_sum": value,
                }
                for name, value in self.standing_reward_breakdown.items()
            ),
            key=lambda item: abs(item["weighted_reward_sum"]),
            reverse=True,
        )

        termination_trend = []
        if self.termination_histories:
            max_len = max(len(history) for history in self.termination_histories)
            for offset in range(max_len):
                samples = []
                for history in self.termination_histories:
                    idx = len(history) - max_len + offset
                    if idx >= 0:
                        samples.append(history[idx])
                if samples:
                    termination_trend.append({"offset_from_end": offset - max_len + 1, "mean_metric": float(np.mean(samples))})

        foot_contact_summary = {}
        for foot_name, counts in self.contact_counts.items():
            total = counts["match"] + counts["mismatch"]
            foot_rewards = self.contact_reward_summary[foot_name]
            foot_contact_summary[foot_name] = {
                "match_rate": counts["match"] / total if total > 0 else float("nan"),
                "mismatch_rate": counts["mismatch"] / total if total > 0 else float("nan"),
                "robot_contact_fraction": counts["robot_contact"] / total if total > 0 else float("nan"),
                "reference_contact_fraction": counts["reference_contact"] / total if total > 0 else float("nan"),
                "mean_reward_when_match": foot_rewards["match_reward_total"] / foot_rewards["match_step_count"]
                if foot_rewards["match_step_count"] > 0
                else float("nan"),
                "mean_reward_when_mismatch": foot_rewards["mismatch_reward_total"] / foot_rewards["mismatch_step_count"]
                if foot_rewards["mismatch_step_count"] > 0
                else float("nan"),
            }

        pre_failure_contact_trend = []
        if self.pre_failure_contact_histories:
            max_len = max(len(history) for history in self.pre_failure_contact_histories)
            for offset in range(max_len):
                left_samples = []
                right_samples = []
                any_samples = []
                ee_samples = []
                phase_samples = []
                for history in self.pre_failure_contact_histories:
                    idx = len(history) - max_len + offset
                    if idx >= 0:
                        row = history[idx]
                        left_samples.append(float(row["left_mismatch"]))
                        right_samples.append(float(row["right_mismatch"]))
                        any_samples.append(float(row["any_mismatch"]))
                        ee_samples.append(float(row["ee_metric"]))
                        phase_samples.append(float(row["phase_bin"]))
                if left_samples:
                    pre_failure_contact_trend.append(
                        {
                            "offset_from_end": offset - max_len + 1,
                            "left_mismatch_rate": float(np.mean(left_samples)),
                            "right_mismatch_rate": float(np.mean(right_samples)),
                            "any_mismatch_rate": float(np.mean(any_samples)),
                            "mean_ee_metric": float(np.mean(ee_samples)),
                            "mean_phase_bin": float(np.mean(phase_samples)),
                        }
                    )

        early_pre_failure_rows = [row for row in pre_failure_contact_trend if -10 <= row["offset_from_end"] <= -6]
        left_early_mismatch = _safe_mean([row["left_mismatch_rate"] for row in early_pre_failure_rows])
        right_early_mismatch = _safe_mean([row["right_mismatch_rate"] for row in early_pre_failure_rows])
        if math.isnan(left_early_mismatch) or math.isnan(right_early_mismatch):
            first_failing_foot = "unknown"
        elif left_early_mismatch > right_early_mismatch:
            first_failing_foot = "left_foot_link"
        elif right_early_mismatch > left_early_mismatch:
            first_failing_foot = "right_foot_link"
        else:
            first_failing_foot = "tied"

        any_mismatch_mean = self.contact_reward_summary["any_mismatch"]["reward_total"] / self.contact_reward_summary["any_mismatch"]["step_count"] if self.contact_reward_summary["any_mismatch"]["step_count"] > 0 else float("nan")
        all_match_mean = self.contact_reward_summary["all_match"]["reward_total"] / self.contact_reward_summary["all_match"]["step_count"] if self.contact_reward_summary["all_match"]["step_count"] > 0 else float("nan")

        return {
            "checkpoint_iteration": checkpoint_iteration,
            "checkpoint_path": checkpoint_path,
            "phase_rows": phase_rows,
            "root_motion": {
                "mean_root_speed_ratio": float(np.mean(self.root_speed_ratio_samples)) if self.root_speed_ratio_samples else float("nan"),
                "median_root_speed_ratio": float(np.median(self.root_speed_ratio_samples)) if self.root_speed_ratio_samples else float("nan"),
                "mean_root_speed_ratio_when_reference_moving": float(np.mean(self.root_speed_ratio_moving_samples))
                if self.root_speed_ratio_moving_samples
                else float("nan"),
                "median_root_speed_ratio_when_reference_moving": float(np.median(self.root_speed_ratio_moving_samples))
                if self.root_speed_ratio_moving_samples
                else float("nan"),
                "root_speed_ratio_hist_edges": root_ratio_hist_edges.tolist(),
                "root_speed_ratio_hist_counts": root_ratio_hist_counts.tolist(),
                "mean_root_position_error": float(np.mean(self.root_position_errors)) if self.root_position_errors else float("nan"),
                "mean_root_orientation_error": float(np.mean(self.root_orientation_errors)) if self.root_orientation_errors else float("nan"),
                "mean_root_linear_velocity_error": float(np.mean(self.root_linear_velocity_errors)) if self.root_linear_velocity_errors else float("nan"),
                "mean_root_angular_velocity_error": float(np.mean(self.root_angular_velocity_errors)) if self.root_angular_velocity_errors else float("nan"),
                "mean_robot_root_speed": float(np.mean(self.robot_root_speeds)) if self.robot_root_speeds else float("nan"),
                "mean_reference_root_speed": float(np.mean(self.reference_root_speeds)) if self.reference_root_speeds else float("nan"),
                "total_step_samples": self.total_step_samples,
            },
            "contact_timing": foot_contact_summary,
            "contact_mismatch": {
                "mean_reward_all_match": all_match_mean,
                "mean_reward_any_mismatch": any_mismatch_mean,
                "reward_delta_mismatch_minus_match": any_mismatch_mean - all_match_mean
                if not math.isnan(any_mismatch_mean) and not math.isnan(all_match_mean)
                else float("nan"),
                "phase_rows": [
                    {
                        "phase_bin": row["phase_bin"],
                        "left_contact_mismatch_rate": row["left_contact_mismatch_rate"],
                        "right_contact_mismatch_rate": row["right_contact_mismatch_rate"],
                        "any_contact_mismatch_rate": row["any_contact_mismatch_rate"],
                        "termination_rate": row["termination_rate"],
                    }
                    for row in phase_rows
                ],
                "correlation_any_mismatch_vs_ee_metric": _safe_correlation(
                    self.contact_mismatch_samples["any_mismatch"], self.termination_metric_samples
                ),
                "correlation_left_mismatch_vs_ee_metric": _safe_correlation(
                    self.contact_mismatch_samples["left_foot_link"], self.termination_metric_samples
                ),
                "correlation_right_mismatch_vs_ee_metric": _safe_correlation(
                    self.contact_mismatch_samples["right_foot_link"], self.termination_metric_samples
                ),
                "phase_correlation_any_mismatch_vs_termination": _safe_correlation(
                    [row["any_contact_mismatch_rate"] for row in phase_rows],
                    [row["termination_rate"] for row in phase_rows],
                ),
                "pre_failure_trend": pre_failure_contact_trend,
                "first_failing_foot": first_failing_foot,
                "largest_mismatch_phase": max(
                    phase_rows,
                    key=lambda row: row["any_contact_mismatch_rate"]
                    if not math.isnan(row["any_contact_mismatch_rate"])
                    else -1.0,
                )["phase_bin"],
            },
            "standing_local_optimum": {
                "standing_step_count": self.standing_step_count,
                "standing_reward_total": self.standing_reward_total,
                "standing_reward_breakdown": standing_breakdown,
            },
            "termination": {
                "metric_name": "max_abs_ee_body_z_error",
                "metric_threshold": EE_BODY_POS_THRESHOLD,
                "metric_body_names": EE_BODY_NAMES,
                "mean_metric": float(np.mean(self.termination_metric_samples)) if self.termination_metric_samples else float("nan"),
                "p95_metric": _tensor_percentile(self.termination_metric_samples, 95),
                "p99_metric": _tensor_percentile(self.termination_metric_samples, 99),
                "mean_metric_at_episode_end": float(np.mean(self.termination_metric_episode_end)) if self.termination_metric_episode_end else float("nan"),
                "termination_history_mean": termination_trend,
            },
            "reward_contributions": reward_contributions,
            "episode_summary": {
                "mean_episode_reward": float(np.mean(self.completed_episode_returns)) if self.completed_episode_returns else float("nan"),
                "mean_episode_length": float(np.mean(self.completed_episode_lengths)) if self.completed_episode_lengths else float("nan"),
                "completed_episodes": len(self.completed_episode_returns),
            },
        }


def _evaluate_checkpoint(env, policy, checkpoint_iteration: int, checkpoint_path: str, args) -> dict:
    obs = _extract_observations(env.reset()[0])
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("motion")
    reward_manager = base_env.reward_manager
    contact_sensor = base_env.scene.sensors["contact_forces"]
    foot_body_indexes = {
        "left_foot_link": contact_sensor.body_names.index("left_foot_link"),
        "right_foot_link": contact_sensor.body_names.index("right_foot_link"),
    }
    reference_contact_heuristics = _derive_reference_contact_heuristics(
        command,
        height_margin=args.reference_contact_height_margin,
        speed_threshold=args.reference_contact_speed_threshold,
    )

    diagnostics = DiagnosticsAccumulator(
        reward_term_names=reward_manager.active_terms,
        phase_term_names=SELECTED_PHASE_TERMS,
        termination_history_length=args.termination_history_length,
        standing_speed_threshold=args.standing_speed_threshold,
        reference_speed_threshold=args.reference_speed_threshold,
    )

    completed_episodes = 0
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)

    for _ in range(args.max_steps):
        normalized_phase = command.time_steps.float() / max(command.motion.time_step_total - 1, 1)
        phase_bins = torch.clamp((normalized_phase * 10.0).long(), min=0, max=9)

        ee_metric = _compute_ee_body_pos_metric(command)
        root_pos_error = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
        root_ori_error = command.metrics["error_anchor_rot"].clone()
        root_lin_vel_error = torch.norm(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w, dim=-1)
        root_ang_vel_error = torch.norm(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w, dim=-1)
        robot_root_speed = torch.norm(command.robot_anchor_lin_vel_w, dim=-1)
        reference_root_speed = torch.norm(command.anchor_lin_vel_w, dim=-1)

        robot_contacts = {
            foot_name: _compute_robot_contact(contact_sensor, foot_index)
            for foot_name, foot_index in foot_body_indexes.items()
        }
        reference_contacts = {
            foot_name: _compute_reference_contact(command, foot_name, heuristic)
            for foot_name, heuristic in reference_contact_heuristics.items()
        }

        diagnostics.observe_pre_step(
            env_ids=env_ids,
            phase_bins=phase_bins,
            ee_metric=ee_metric,
            root_pos_error=root_pos_error,
            root_ori_error=root_ori_error,
            root_lin_vel_error=root_lin_vel_error,
            root_ang_vel_error=root_ang_vel_error,
            robot_root_speed=robot_root_speed,
            reference_root_speed=reference_root_speed,
            robot_contacts=robot_contacts,
            reference_contacts=reference_contacts,
            contact_mismatches={foot_name: robot_contacts[foot_name] != reference_contacts[foot_name] for foot_name in robot_contacts},
            any_contact_mismatch=torch.logical_or(
                robot_contacts["left_foot_link"] != reference_contacts["left_foot_link"],
                robot_contacts["right_foot_link"] != reference_contacts["right_foot_link"],
            ),
        )

        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            obs = _extract_observations(obs)

        step_reward_raw = reward_manager.step_reward_raw.clone()
        step_reward_weighted_dt = reward_manager.step_reward_weighted_dt.clone()
        dones_flat = _flatten_tensor(dones).to(dtype=torch.bool)

        diagnostics.observe_post_step(
            env_ids=env_ids,
            phase_bins=phase_bins,
            step_reward_raw=step_reward_raw,
            step_reward_weighted_dt=step_reward_weighted_dt,
            dones=dones_flat,
            robot_root_speed=robot_root_speed,
            reference_root_speed=reference_root_speed,
            contact_mismatches={foot_name: robot_contacts[foot_name] != reference_contacts[foot_name] for foot_name in robot_contacts},
            any_contact_mismatch=torch.logical_or(
                robot_contacts["left_foot_link"] != reference_contacts["left_foot_link"],
                robot_contacts["right_foot_link"] != reference_contacts["right_foot_link"],
            ),
        )

        completed_episodes += int(dones_flat.sum().item())
        if completed_episodes >= args.episodes_per_checkpoint:
            break

    summary = diagnostics.build_summary(checkpoint_iteration=checkpoint_iteration, checkpoint_path=checkpoint_path)
    summary["reference_contact_heuristic"] = {
        foot_name: {
            "height_threshold": heuristic.height_threshold,
            "speed_threshold": heuristic.speed_threshold,
        }
        for foot_name, heuristic in reference_contact_heuristics.items()
    }
    return summary


def _write_phase_rows(output_dir: str, checkpoint_summaries: list[dict]) -> str:
    path = os.path.join(output_dir, "phase_conditioned_tracking.csv")
    fieldnames = [
        "checkpoint_iteration",
        "checkpoint_path",
        "phase_bin",
        "sample_count",
        "termination_rate",
        "left_contact_mismatch_rate",
        "right_contact_mismatch_rate",
        "any_contact_mismatch_rate",
        *SELECTED_PHASE_TERMS.keys(),
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in checkpoint_summaries:
            writer.writerows(summary["phase_rows"])
    return path


def _write_checkpoint_summary(output_dir: str, checkpoint_summaries: list[dict]) -> str:
    path = os.path.join(output_dir, "checkpoint_summary.csv")
    fieldnames = [
        "checkpoint_iteration",
        "checkpoint_path",
        "mean_episode_reward",
        "mean_episode_length",
        "mean_root_speed_ratio",
        "mean_root_speed_ratio_when_reference_moving",
        "left_contact_match_rate",
        "right_contact_match_rate",
        "left_contact_mismatch_rate",
        "right_contact_mismatch_rate",
        "any_contact_mismatch_rate",
        "mean_reward_all_match",
        "mean_reward_any_mismatch",
        "contact_mismatch_ee_corr",
        "standing_step_count",
        "standing_reward_total",
        "termination_mean_metric",
        "termination_p95_metric",
        "termination_p99_metric",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in checkpoint_summaries:
            writer.writerow(
                {
                    "checkpoint_iteration": summary["checkpoint_iteration"],
                    "checkpoint_path": summary["checkpoint_path"],
                    "mean_episode_reward": summary["episode_summary"]["mean_episode_reward"],
                    "mean_episode_length": summary["episode_summary"]["mean_episode_length"],
                    "mean_root_speed_ratio": summary["root_motion"]["mean_root_speed_ratio"],
                    "mean_root_speed_ratio_when_reference_moving": summary["root_motion"][
                        "mean_root_speed_ratio_when_reference_moving"
                    ],
                    "left_contact_match_rate": summary["contact_timing"]["left_foot_link"]["match_rate"],
                    "right_contact_match_rate": summary["contact_timing"]["right_foot_link"]["match_rate"],
                    "left_contact_mismatch_rate": summary["contact_timing"]["left_foot_link"]["mismatch_rate"],
                    "right_contact_mismatch_rate": summary["contact_timing"]["right_foot_link"]["mismatch_rate"],
                    "any_contact_mismatch_rate": _safe_mean(
                        [row["any_contact_mismatch_rate"] for row in summary["contact_mismatch"]["phase_rows"]]
                    ),
                    "mean_reward_all_match": summary["contact_mismatch"]["mean_reward_all_match"],
                    "mean_reward_any_mismatch": summary["contact_mismatch"]["mean_reward_any_mismatch"],
                    "contact_mismatch_ee_corr": summary["contact_mismatch"]["correlation_any_mismatch_vs_ee_metric"],
                    "standing_step_count": summary["standing_local_optimum"]["standing_step_count"],
                    "standing_reward_total": summary["standing_local_optimum"]["standing_reward_total"],
                    "termination_mean_metric": summary["termination"]["mean_metric"],
                    "termination_p95_metric": summary["termination"]["p95_metric"],
                    "termination_p99_metric": summary["termination"]["p99_metric"],
                }
            )
    return path


def _aggregate_window_report(checkpoint_summaries: list[dict]) -> dict:
    best_summary = max(checkpoint_summaries, key=lambda item: item["episode_summary"]["mean_episode_reward"])
    phase_aggregate = defaultdict(list)
    for summary in checkpoint_summaries:
        for row in summary["phase_rows"]:
            phase_aggregate[row["phase_bin"]].append(row)

    aggregated_phase_rows = []
    for phase_bin in [_format_phase_bin(i) for i in range(10)]:
        rows = phase_aggregate[phase_bin]
        agg_row = {"phase_bin": phase_bin}
        if rows:
            agg_row["termination_rate"] = float(np.mean([row["termination_rate"] for row in rows]))
            for alias in SELECTED_PHASE_TERMS:
                agg_row[alias] = float(np.mean([row[alias] for row in rows]))
        else:
            agg_row["termination_rate"] = float("nan")
            for alias in SELECTED_PHASE_TERMS:
                agg_row[alias] = float("nan")
        aggregated_phase_rows.append(agg_row)

    termination_metrics = []
    termination_ends = []
    for summary in checkpoint_summaries:
        termination_metrics.append(summary["termination"]["mean_metric"])
        termination_ends.append(summary["termination"]["mean_metric_at_episode_end"])

    reward_contributions = defaultdict(list)
    for summary in checkpoint_summaries:
        for row in summary["reward_contributions"]:
            reward_contributions[row["objective_name"]].append(row)
    aggregated_contributions = []
    for objective_name, rows in reward_contributions.items():
        aggregated_contributions.append(
            {
                "objective_name": objective_name,
                "weighted_reward_sum": float(np.mean([row["weighted_reward_sum"] for row in rows])),
                "raw_reward_sum": float(np.mean([row["raw_reward_sum"] for row in rows])),
                "positive_contribution": float(np.mean([row["positive_contribution"] for row in rows])),
            }
        )
    aggregated_contributions.sort(key=lambda item: abs(item["weighted_reward_sum"]), reverse=True)

    standing_breakdown = defaultdict(list)
    for summary in checkpoint_summaries:
        for row in summary["standing_local_optimum"]["standing_reward_breakdown"]:
            standing_breakdown[row["objective_name"]].append(row["weighted_reward_sum"])
    aggregated_standing_breakdown = sorted(
        (
            {
                "objective_name": objective_name,
                "weighted_reward_sum": float(np.mean(values)),
            }
            for objective_name, values in standing_breakdown.items()
        ),
        key=lambda item: abs(item["weighted_reward_sum"]),
        reverse=True,
    )

    left_match = float(np.mean([summary["contact_timing"]["left_foot_link"]["match_rate"] for summary in checkpoint_summaries]))
    right_match = float(np.mean([summary["contact_timing"]["right_foot_link"]["match_rate"] for summary in checkpoint_summaries]))
    left_mismatch = float(np.mean([summary["contact_timing"]["left_foot_link"]["mismatch_rate"] for summary in checkpoint_summaries]))
    right_mismatch = float(np.mean([summary["contact_timing"]["right_foot_link"]["mismatch_rate"] for summary in checkpoint_summaries]))
    mean_root_speed_ratio = float(np.mean([summary["root_motion"]["mean_root_speed_ratio"] for summary in checkpoint_summaries]))
    mean_root_speed_ratio_when_reference_moving = float(
        np.mean([summary["root_motion"]["mean_root_speed_ratio_when_reference_moving"] for summary in checkpoint_summaries])
    )
    contact_mismatch_phase_rows = defaultdict(list)
    for summary in checkpoint_summaries:
        for row in summary["contact_mismatch"]["phase_rows"]:
            contact_mismatch_phase_rows[row["phase_bin"]].append(row)
    aggregated_contact_mismatch_phase_rows = []
    for phase_bin in [_format_phase_bin(i) for i in range(10)]:
        rows = contact_mismatch_phase_rows[phase_bin]
        aggregated_contact_mismatch_phase_rows.append(
            {
                "phase_bin": phase_bin,
                "left_contact_mismatch_rate": float(np.mean([row["left_contact_mismatch_rate"] for row in rows])) if rows else float("nan"),
                "right_contact_mismatch_rate": float(np.mean([row["right_contact_mismatch_rate"] for row in rows])) if rows else float("nan"),
                "any_contact_mismatch_rate": float(np.mean([row["any_contact_mismatch_rate"] for row in rows])) if rows else float("nan"),
                "termination_rate": float(np.mean([row["termination_rate"] for row in rows])) if rows else float("nan"),
            }
        )
    pre_failure_rows = defaultdict(list)
    for summary in checkpoint_summaries:
        for row in summary["contact_mismatch"]["pre_failure_trend"]:
            pre_failure_rows[row["offset_from_end"]].append(row)
    aggregated_pre_failure_trend = []
    for offset in sorted(pre_failure_rows):
        rows = pre_failure_rows[offset]
        aggregated_pre_failure_trend.append(
            {
                "offset_from_end": offset,
                "left_mismatch_rate": float(np.mean([row["left_mismatch_rate"] for row in rows])),
                "right_mismatch_rate": float(np.mean([row["right_mismatch_rate"] for row in rows])),
                "any_mismatch_rate": float(np.mean([row["any_mismatch_rate"] for row in rows])),
                "mean_ee_metric": float(np.mean([row["mean_ee_metric"] for row in rows])),
            }
        )
    mean_reward_all_match = float(np.mean([summary["contact_mismatch"]["mean_reward_all_match"] for summary in checkpoint_summaries]))
    mean_reward_any_mismatch = float(np.mean([summary["contact_mismatch"]["mean_reward_any_mismatch"] for summary in checkpoint_summaries]))
    first_failing_votes = [summary["contact_mismatch"]["first_failing_foot"] for summary in checkpoint_summaries]
    if first_failing_votes:
        first_failing_foot = max(set(first_failing_votes), key=first_failing_votes.count)
    else:
        first_failing_foot = "unknown"

    return {
        "best_checkpoint_iteration": best_summary["checkpoint_iteration"],
        "best_checkpoint_path": best_summary["checkpoint_path"],
        "best_checkpoint_mean_reward": best_summary["episode_summary"]["mean_episode_reward"],
        "best_checkpoint_mean_episode_length": best_summary["episode_summary"]["mean_episode_length"],
        "aggregated_phase_rows": aggregated_phase_rows,
        "mean_root_speed_ratio": mean_root_speed_ratio,
        "mean_root_speed_ratio_when_reference_moving": mean_root_speed_ratio_when_reference_moving,
        "left_contact_match_rate": left_match,
        "right_contact_match_rate": right_match,
        "left_contact_mismatch_rate": left_mismatch,
        "right_contact_mismatch_rate": right_mismatch,
        "contact_mismatch_phase_rows": aggregated_contact_mismatch_phase_rows,
        "pre_failure_contact_trend": aggregated_pre_failure_trend,
        "mean_reward_all_match": mean_reward_all_match,
        "mean_reward_any_mismatch": mean_reward_any_mismatch,
        "contact_mismatch_ee_corr": float(
            np.mean([summary["contact_mismatch"]["correlation_any_mismatch_vs_ee_metric"] for summary in checkpoint_summaries])
        ),
        "contact_mismatch_phase_termination_corr": float(
            np.mean([summary["contact_mismatch"]["phase_correlation_any_mismatch_vs_termination"] for summary in checkpoint_summaries])
        ),
        "first_failing_foot": first_failing_foot,
        "termination_mean_metric": float(np.mean(termination_metrics)),
        "termination_mean_metric_at_episode_end": float(np.mean(termination_ends)),
        "reward_contributions": aggregated_contributions,
        "standing_reward_breakdown": aggregated_standing_breakdown,
    }


def _build_final_answers(window_report: dict) -> dict[str, str]:
    phase_rows = window_report["aggregated_phase_rows"]
    worst_phase = max(phase_rows, key=lambda row: row["termination_rate"])
    mean_root_speed_ratio = window_report["mean_root_speed_ratio_when_reference_moving"]
    left_match = window_report["left_contact_match_rate"]
    right_match = window_report["right_contact_match_rate"]
    leading_contribution = window_report["reward_contributions"][0]["objective_name"] if window_report["reward_contributions"] else "unknown"
    standing_positive = [row for row in window_report["standing_reward_breakdown"] if row["weighted_reward_sum"] > 0][:4]
    standing_objectives = ", ".join(row["objective_name"] for row in standing_positive) if standing_positive else "none"

    if mean_root_speed_ratio < 0.6:
        root_translation_answer = "Root translation is under-tracked; the robot moves substantially less than the reference."
    else:
        root_translation_answer = "Root translation is being tracked to a meaningful degree."

    gait_match = statistics.fmean([left_match, right_match])
    if gait_match < 0.65:
        gait_answer = "Gait timing is weakly tracked; foot contact timing mismatches are frequent."
    else:
        gait_answer = "Gait timing is being tracked reasonably well."

    if window_report["standing_reward_breakdown"] and standing_positive:
        standing_answer = (
            "Yes. There are standing-like periods where the robot still collects positive reward, "
            f"mainly from {standing_objectives}."
        )
    else:
        standing_answer = "No strong standing-local-optimum signature was detected in the evaluated checkpoints."

    pre_failure_trend = window_report["pre_failure_contact_trend"]
    early_rows = [row for row in pre_failure_trend if -10 <= row["offset_from_end"] <= -6]
    late_rows = [row for row in pre_failure_trend if -5 <= row["offset_from_end"] <= -1]
    early_any = _safe_mean([row["any_mismatch_rate"] for row in early_rows])
    late_any = _safe_mean([row["any_mismatch_rate"] for row in late_rows])
    if math.isnan(early_any) or math.isnan(late_any):
        mismatch_rise_answer = "Insufficient pre-failure history to determine whether contact mismatch rises before failure."
    elif late_any > early_any + 0.05:
        mismatch_rise_answer = "Yes. Contact mismatch rises in the last few steps before failure."
    else:
        mismatch_rise_answer = "No strong late pre-failure rise in contact mismatch was detected."

    largest_mismatch_phase = max(
        window_report["contact_mismatch_phase_rows"],
        key=lambda row: row["any_contact_mismatch_rate"] if not math.isnan(row["any_contact_mismatch_rate"]) else -1.0,
    )["phase_bin"]

    return {
        "is_policy_truly_imitating": (
            "Partially. The policy matches some pose/trunk objectives, but not enough to conclude full motion imitation."
        ),
        "is_policy_exploiting_standing_local_optimum": standing_answer,
        "is_root_translation_tracked": root_translation_answer,
        "is_gait_timing_tracked": gait_answer,
        "single_most_likely_reason_for_fall": (
            "The policy drifts into the phase region with the highest ee_body_pos termination rate, "
            f"especially around {worst_phase['phase_bin']} of the motion."
        ),
        "does_contact_mismatch_rise_before_failure": mismatch_rise_answer,
        "which_foot_fails_first": f"The earliest elevated pre-failure mismatch is on `{window_report['first_failing_foot']}`.",
        "which_phase_has_largest_contact_mismatch": f"The largest contact mismatch occurs around `{largest_mismatch_phase}`.",
        "most_direct_reward_modification": (
            f"The most direct reward-side fix would be to strengthen objectives that are weak in the failing phase region, "
            f"especially whichever of the translation/contact-sensitive terms is underperforming relative to {leading_contribution}."
        ),
    }


def _write_report(output_dir: str, window_report: dict, answers: dict[str, str], checkpoint_summaries: list[dict]) -> str:
    report_path = os.path.join(output_dir, "imitation_behavior_report.md")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("# Imitation Behavior Diagnostics\n\n")
        report.write(f"Best checkpoint in evaluated window: `{window_report['best_checkpoint_iteration']}`\n\n")
        report.write("## Final Answers\n\n")
        for key, value in answers.items():
            title = key.replace("_", " ").capitalize()
            report.write(f"- **{title}:** {value}\n")
        report.write("\n## Root Motion\n\n")
        report.write(f"- Mean root speed ratio: `{window_report['mean_root_speed_ratio']:.4f}`\n")
        report.write(
            f"- Mean root speed ratio when reference is moving: "
            f"`{window_report['mean_root_speed_ratio_when_reference_moving']:.4f}`\n"
        )
        report.write(f"- Left foot contact match rate: `{window_report['left_contact_match_rate']:.4f}`\n")
        report.write(f"- Right foot contact match rate: `{window_report['right_contact_match_rate']:.4f}`\n")
        report.write(
            f"- Mean ee_body_pos metric across window: `{window_report['termination_mean_metric']:.4f}`"
            f" with mean end-of-episode metric `{window_report['termination_mean_metric_at_episode_end']:.4f}`\n"
        )
        report.write(
            f"- Mean reward when both feet match contact timing: `{window_report['mean_reward_all_match']:.4f}`\n"
        )
        report.write(
            f"- Mean reward when any foot mismatches contact timing: `{window_report['mean_reward_any_mismatch']:.4f}`\n"
        )
        report.write(
            f"- Correlation between any contact mismatch and ee_body_pos metric: "
            f"`{window_report['contact_mismatch_ee_corr']:.4f}`\n"
        )
        report.write("\n## Phase-Conditioned Tracking\n\n")
        for row in window_report["aggregated_phase_rows"]:
            report.write(
                f"- `{row['phase_bin']}`: termination `{row['termination_rate']:.4f}`, "
                f"trunk_pos `{row['trunk_pos']:.4f}`, trunk_ori `{row['trunk_ori']:.4f}`, "
                f"foot_pos `{row['foot_pos']:.4f}`, body_pos `{row['body_pos']:.4f}`\n"
            )
        report.write("\n## Contact Mismatch\n\n")
        report.write(
            f"- Left foot mismatch rate: `{window_report['left_contact_mismatch_rate']:.4f}`\n"
        )
        report.write(
            f"- Right foot mismatch rate: `{window_report['right_contact_mismatch_rate']:.4f}`\n"
        )
        report.write(
            f"- Phase correlation between any contact mismatch and termination: "
            f"`{window_report['contact_mismatch_phase_termination_corr']:.4f}`\n"
        )
        report.write(
            f"- Earliest failing foot heuristic: `{window_report['first_failing_foot']}`\n"
        )
        for row in window_report["contact_mismatch_phase_rows"]:
            report.write(
                f"- `{row['phase_bin']}`: left_mismatch `{row['left_contact_mismatch_rate']:.4f}`, "
                f"right_mismatch `{row['right_contact_mismatch_rate']:.4f}`, "
                f"any_mismatch `{row['any_contact_mismatch_rate']:.4f}`, termination `{row['termination_rate']:.4f}`\n"
            )
        if window_report["pre_failure_contact_trend"]:
            report.write("\n## Pre-Failure Contact Trend\n\n")
            for row in window_report["pre_failure_contact_trend"][-10:]:
                report.write(
                    f"- `t{row['offset_from_end']}`: left `{row['left_mismatch_rate']:.4f}`, "
                    f"right `{row['right_mismatch_rate']:.4f}`, any `{row['any_mismatch_rate']:.4f}`, "
                    f"ee_metric `{row['mean_ee_metric']:.4f}`\n"
                )
        report.write("\n## Standing-Like Period Reward Breakdown\n\n")
        for row in window_report["standing_reward_breakdown"][:8]:
            report.write(f"- `{row['objective_name']}`: `{row['weighted_reward_sum']:.6f}`\n")
        report.write("\n## Reward Contributions\n\n")
        for row in window_report["reward_contributions"][:12]:
            report.write(
                f"- `{row['objective_name']}`: weighted `{row['weighted_reward_sum']:.6f}`, "
                f"positive-share `{row['positive_contribution']:.4f}`\n"
            )
        report.write("\n## Evaluated Checkpoints\n\n")
        for summary in checkpoint_summaries:
            report.write(
                f"- `{summary['checkpoint_iteration']}`: mean_reward `{summary['episode_summary']['mean_episode_reward']:.4f}`, "
                f"mean_length `{summary['episode_summary']['mean_episode_length']:.4f}`\n"
            )
    return report_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    run_dir = os.path.abspath(args_cli.run_dir) if args_cli.run_dir else None
    checkpoints = _resolve_checkpoints(
        run_dir=run_dir,
        checkpoint=args_cli.analysis_checkpoint,
        start=args_cli.checkpoint_start,
        end=args_cli.checkpoint_end,
        step=args_cli.checkpoint_step,
    )

    output_dir = (
        os.path.abspath(args_cli.output_dir)
        if args_cli.output_dir
        else os.path.join(run_dir or os.path.dirname(checkpoints[0][1]), "imitation_diagnostics")
    )
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    checkpoint_summaries = []
    for checkpoint_iteration, checkpoint_path in checkpoints:
        print(f"[INFO] Evaluating checkpoint {checkpoint_iteration} from {checkpoint_path}")
        ppo_runner.load(retrieve_file_path(checkpoint_path))
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        summary = _evaluate_checkpoint(env, policy, checkpoint_iteration, checkpoint_path, args_cli)
        checkpoint_summaries.append(summary)

    phase_csv_path = _write_phase_rows(output_dir, checkpoint_summaries)
    checkpoint_csv_path = _write_checkpoint_summary(output_dir, checkpoint_summaries)

    window_report = _aggregate_window_report(checkpoint_summaries)
    final_answers = _build_final_answers(window_report)
    json_path = os.path.join(output_dir, "imitation_behavior_summary.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "evaluated_checkpoints": checkpoint_summaries,
                "window_report": window_report,
                "final_answers": final_answers,
                "ee_body_pos_formula": "max(abs(command.body_pos_relative_w[..., z] - command.robot_body_pos_w[..., z])) over left/right hand and foot",
                "ee_body_pos_threshold": EE_BODY_POS_THRESHOLD,
                "ee_body_pos_body_names": EE_BODY_NAMES,
                "reference_contact_note": "Reference foot contact is heuristic: low foot height plus low foot speed because the motion file has no contact labels.",
            },
            json_file,
            indent=2,
        )
    report_path = _write_report(output_dir, window_report, final_answers, checkpoint_summaries)

    print("[INFO] Diagnostics written to:")
    print_dict(
        {
            "output_dir": output_dir,
            "phase_csv": phase_csv_path,
            "checkpoint_csv": checkpoint_csv_path,
            "summary_json": json_path,
            "report_md": report_path,
        },
        nesting=2,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
