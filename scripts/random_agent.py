# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tactile_tasks.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    
    # Get robot reference for joint monitoring
    robot = env.unwrapped.scene["robot"]
    prev_joint_pos = robot.data.joint_pos.clone()
    
    # simulate environment
    step_count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            
            # Print action stats every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}: actions mean={actions.mean():.4f}, std={actions.std():.4f}, min={actions.min():.4f}, max={actions.max():.4f}")
                print(f"Non-zero actions: {(actions != 0).sum()}/{actions.numel()}")
                print(f"Action shape: {actions.shape}, Action space: {env.action_space}")
                # Show first few action values
                if actions.numel() <= 20:
                    print(f"All actions: {actions.flatten()}")
                else:
                    print(f"First 10 actions: {actions.flatten()[:10]}")
            
            # apply actions
            env.step(actions)
            
            # Check joint movement
            curr_joint_pos = robot.data.joint_pos
            joint_delta = (curr_joint_pos - prev_joint_pos).abs().mean()
            if step_count % 100 == 0:
                print(f"Mean joint movement: {joint_delta:.6f}")
            prev_joint_pos = curr_joint_pos.clone()
            
            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
