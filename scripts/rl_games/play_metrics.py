# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games, with extra metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games (with metrics).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--samples", type=int, default=1000, help="Number of metric samples to collect, then stop.")
parser.add_argument(
    "--metrics_out",
    type=str,
    default=None,
    help="Optional CSV path for per-step metrics (yaw_rate, drop_rate, avg_net_yaw_wrapped, avg_turns).",
)
parser.add_argument(
    "--runs",
    type=int,
    default=None,
    help="Optional number of completed episodes (across envs) to collect before stopping.",
)
parser.add_argument(
    "--drop_tip_thresh",
    type=float,
    default=0.003,
    help="Drop detection threshold (m) for screwdriver tip height above env origin.",
)
parser.add_argument(
    "--log_fall_envs",
    action="store_true",
    help="When set, print environment indices that experience a fall event.",
)
parser.add_argument(
    "--fall_angle_deg",
    type=float,
    default=19.0,
    help="Angle threshold (deg) for fall detection based on |pitch| or |roll|.",
)
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
import math
import os
import random
import time
import torch
import csv

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import tactile_tasks.tasks  # noqa: F401
from isaaclab.utils.math import matrix_from_quat, quat_apply
try:
    # Optional: for tip offset
    from tactile_tasks.tactile_tasks.tasks.manager_based.tactile_tasks.screwdriver import ScrewdriverCfg  # type: ignore
except Exception:  # pragma: no cover
    ScrewdriverCfg = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent with metrics."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    base_env = env.unwrapped

    # metrics (windowed averages)
    log_interval = 100  # steps
    rot_avg_accum = 0.0
    drop_avg_accum = 0.0
    window_steps = 0
    # per-step metric buffers for collating
    yaw_rate_series = []  # list[float]
    drop_rate_series = []  # list[float]
    net_yaw_series = []  # list[float], avg across envs at each step (wrapped)
    net_yaw_unwrapped_series = []  # list[float], mean of yaw_cumulative (unwrapped)
    mean_signed_yaw_rate_series = []  # list[float], per-step mean signed yaw rate
    net_yaw_from_mean_series = []  # list[float], integrated mean signed yaw rate
    # integrated signed yaw per-env (radians)
    yaw_cumulative = None
    # total completed episodes across envs (for --runs)
    run_count = 0
    # fall event tracking
    prev_dropped_mask = None  # torch.BoolTensor per env
    falls_per_step = []  # list[int] number of new falls at each step
    # episode-based net yaw tracking (displacement from start per trial)
    init_quat = None  # torch.Tensor (N,4)
    net_yaw_trials = []  # list[float]
    # per-env fall counters per trial
    current_trial_fallen = None  # torch.BoolTensor (N,)
    fallen_trials = None  # torch.IntTensor (N,)
    total_trials_env = None  # torch.IntTensor (N,)

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0  # used when recording video
    global_step = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # compute pre-step relative yaw from starting orientation
            try:
                screwdriver_pre = base_env.scene["screwdriver"]
                quat_pre = screwdriver_pre.data.root_quat_w  # (N,4)
                if init_quat is None or init_quat.shape != quat_pre.shape:
                    init_quat = quat_pre.clone()
                R_pre = matrix_from_quat(quat_pre)
                R_init = matrix_from_quat(init_quat)
                R_rel_pre = torch.bmm(R_pre, R_init.transpose(-2, -1))
                yaw_rel_pre = torch.atan2(R_rel_pre[:, 1, 0], R_rel_pre[:, 0, 0])  # (N,)
            except Exception:
                yaw_rel_pre = None
            # initialize per-env fall tracking buffers once
            if current_trial_fallen is None:
                num_envs_local = base_env.num_envs
                device_local = base_env.device
                current_trial_fallen = torch.zeros(num_envs_local, dtype=torch.bool, device=device_local)
                fallen_trials = torch.zeros(num_envs_local, dtype=torch.int32, device=device_local)
                total_trials_env = torch.zeros(num_envs_local, dtype=torch.int32, device=device_local)
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            # track completed runs (episodes). In this play loop, dones is a 1D mask per env.
            # Count how many envs ended at this step; accumulate towards --runs if provided.
            # Note: extras may contain per-episode logs; here we just count terminations.
            # If asymmetric truncations/terminations matter, both are already in dones.
            completed_this_step = int(dones.sum().item()) if hasattr(dones, "sum") else 0
            # capture trial net yaw for done envs using pre-step yaw and update fall counters
            if completed_this_step:
                done_ids_tensor = torch.nonzero(dones, as_tuple=False).flatten()
                if yaw_rel_pre is not None and done_ids_tensor.numel() > 0:
                    for idx in done_ids_tensor.tolist():
                        net_yaw_trials.append(float(torch.abs(yaw_rel_pre[idx]).item()))
                # update per-env fall counters
                if total_trials_env is not None and done_ids_tensor.numel() > 0:
                    total_trials_env[done_ids_tensor] += 1
                    fallen_trials[done_ids_tensor] += current_trial_fallen[done_ids_tensor].to(torch.int32)
                    current_trial_fallen[done_ids_tensor] = False

            # ---- metrics: screwdriver rotation and drop-rate ----
            try:
                screwdriver = base_env.scene["screwdriver"]
                # yaw angular speed (local z)
                ang_w = screwdriver.data.root_ang_vel_w  # (N, 3)
                quat_w = screwdriver.data.root_quat_w    # (N, 4) wxyz
                R = matrix_from_quat(quat_w)             # (N, 3, 3)
                ang_local = torch.bmm(R.transpose(-2, -1), ang_w.unsqueeze(-1)).squeeze(-1)
                yaw_signed = ang_local[:, 2]            # (N,)
                yaw_speed = yaw_signed.abs()            # (N,)
                step_yaw_signed_mean = float(yaw_signed.mean().item())

                # integrate net yaw per env
                if yaw_cumulative is None or yaw_cumulative.shape[0] != yaw_signed.shape[0]:
                    yaw_cumulative = torch.zeros_like(yaw_signed)
                yaw_cumulative = yaw_cumulative + yaw_signed * dt

                # fall detection based on pitch/roll > threshold (upright deviation)
                pos_w = screwdriver.data.root_pos_w      # (N, 3)
                quat_w = screwdriver.data.root_quat_w    # (N, 4)
                R = matrix_from_quat(quat_w)             # (N, 3, 3)
                # local z-axis of screwdriver in world
                z_axis = R[:, :, 2]                      # (N, 3)
                cos_theta = torch.clamp(z_axis[:, 2], -1.0, 1.0)
                threshold_cos = math.cos(math.radians(args_cli.fall_angle_deg))
                dropped = cos_theta < threshold_cos
                # detect new fall events (transition false->true)
                if prev_dropped_mask is None or prev_dropped_mask.shape[0] != dropped.shape[0]:
                    prev_dropped_mask = torch.zeros_like(dropped, dtype=torch.bool)
                new_falls_mask = (~prev_dropped_mask) & dropped
                falls_count = int(new_falls_mask.sum().item())
                if falls_count and args_cli.log_fall_envs:
                    env_ids = torch.nonzero(new_falls_mask, as_tuple=False).flatten().tolist()
                    print(f"[FALL] step={global_step} envs={env_ids}")
                prev_dropped_mask = dropped.clone()
                falls_per_step.append(falls_count)
                # mark envs that have fallen in this trial
                if current_trial_fallen is not None:
                    current_trial_fallen |= dropped

                step_yaw_mean = float(yaw_speed.mean().item())
                step_drop_mean = float(dropped.float().mean().item())

                rot_avg_accum += step_yaw_mean
                drop_avg_accum += step_drop_mean
                window_steps += 1

                # record series
                yaw_rate_series.append(step_yaw_mean)
                drop_rate_series.append(step_drop_mean)
                # wrapped avg net yaw across envs (to [-pi, pi]) and avg turns
                avg_net_yaw = float(yaw_cumulative.mean().item())
                net_yaw_unwrapped_series.append(avg_net_yaw)
                # wrap to [-pi, pi]
                pi = math.pi
                wrapped = ((avg_net_yaw + pi) % (2 * pi)) - pi
                net_yaw_series.append(wrapped)
                # signed mean yaw rate and its integral over time
                mean_signed_yaw_rate_series.append(step_yaw_signed_mean)
                prev_ny = net_yaw_from_mean_series[-1] if net_yaw_from_mean_series else 0.0
                net_yaw_from_mean_series.append(prev_ny + step_yaw_signed_mean * dt)

                if window_steps >= log_interval:
                    print(
                        f"[METRICS] step={global_step:06d} avg|yaw_rate|={rot_avg_accum / window_steps:.3f} rad/s, "
                        f"drop_rate={drop_avg_accum / window_steps:.3f}, "
                        f"avg_net_yaw_wrapped={wrapped:.3f} rad, "
                        f"avg_turns={avg_net_yaw / (2 * math.pi):.3f}, "
                        f"mean_signed_yaw_rate={step_yaw_signed_mean:.3f} rad/s, "
                        f"net_yaw_from_mean={net_yaw_from_mean_series[-1]:.3f} rad"
                    )
                    rot_avg_accum = 0.0
                    drop_avg_accum = 0.0
                    window_steps = 0

                # Stop after enough samples collected
                if len(yaw_rate_series) >= args_cli.samples:
                    break
            except Exception:
                # Keep play robust even if task doesn't include the screwdriver
                pass
            # If we are counting runs, break when reaching the target
            if args_cli.runs is not None and args_cli.runs > 0:
                run_count += completed_this_step
                if run_count >= args_cli.runs:
                    break
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        global_step += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Collate and print summary; optionally save to CSV
    total = len(yaw_rate_series)
    if total > 0:
        yaw_mean = sum(yaw_rate_series) / total
        drop_mean = sum(drop_rate_series) / total
        final_avg_net_yaw_wrapped = net_yaw_series[-1] if net_yaw_series else 0.0
        final_avg_turns = final_avg_net_yaw_wrapped / (2 * math.pi)
        total_falls = sum(falls_per_step) if falls_per_step else 0
        final_net_yaw_from_mean = net_yaw_from_mean_series[-1] if net_yaw_from_mean_series else 0.0
        final_avg_net_yaw_unwrapped = net_yaw_unwrapped_series[-1] if net_yaw_unwrapped_series else 0.0
        consistency_err = final_avg_net_yaw_unwrapped - final_net_yaw_from_mean
        # per-trial displacement stats
        trials_count = len(net_yaw_trials)
        trials_mean = sum(net_yaw_trials)/trials_count if trials_count else 0.0
        trials_max = max(net_yaw_trials) if trials_count else 0.0
        # overall fall fraction across all envs and trials
        total_trials_sum = int(total_trials_env.sum().item()) if total_trials_env is not None else 0
        fallen_trials_sum = int(fallen_trials.sum().item()) if fallen_trials is not None else 0
        fall_fraction = (fallen_trials_sum / total_trials_sum) if total_trials_sum > 0 else 0.0
        print(f"[SUMMARY] mean_trial_net_yaw={trials_mean:.4f} rad, fall_fraction={fall_fraction:.3f} ({fallen_trials_sum}/{total_trials_sum})")
        if args_cli.metrics_out:
            with open(args_cli.metrics_out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "yaw_rate_abs",
                    "drop_rate",
                    "mean_signed_yaw_rate",
                    "avg_net_yaw_unwrapped",
                    "avg_net_yaw_wrapped",
                    "avg_turns",
                    "net_yaw_from_mean",
                    "falls_this_step",
                ])
                for i in range(total):
                    y = yaw_rate_series[i]
                    d = drop_rate_series[i]
                    y_signed = mean_signed_yaw_rate_series[i]
                    nyu = net_yaw_unwrapped_series[i]
                    nyw = net_yaw_series[i]
                    turns = nyw / (2 * math.pi)
                    nym = net_yaw_from_mean_series[i]
                    fcount = falls_per_step[i] if i < len(falls_per_step) else 0
                    writer.writerow([i, y, d, y_signed, nyu, nyw, turns, nym, fcount])
                # Also write per-trial displacements
                writer.writerow([])
                writer.writerow(["trial_index", "net_yaw_displacement_rad"])
                for i, val in enumerate(net_yaw_trials):
                    writer.writerow([i, val])
            print(f"[INFO] Wrote metrics CSV to: {args_cli.metrics_out}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
