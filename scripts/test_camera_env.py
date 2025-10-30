#!/usr/bin/env python3

"""
Test script for the tiled camera environment with point cloud extraction.

This script demonstrates how to use the tiled camera to extract point cloud data
from the tactile tasks environment.
"""

import torch
import numpy as np
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# import after launching the app
import gymnasium as gym
from tactile_tasks.tasks.manager_based.tactile_tasks.hand_env_cfg import TestCameraEnvCfg


def test_camera_environment():
    """Test the camera environment and point cloud extraction."""
    
    # Create the environment
    env_cfg = TestCameraEnvCfg()
    env_cfg.scene.num_envs = 4  # Use fewer environments for testing
    env_cfg.scene.env_spacing = 2.0  # Reduce spacing for better visibility
    
    # Create the environment
    env = gym.make("Isaac-Template-TactileTasks-v0", cfg=env_cfg)
    
    print("Environment created successfully!")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset the environment
    obs, _ = env.reset()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Print observation shapes
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Test point cloud extraction
    if 'point_cloud' in obs:
        point_cloud = obs['point_cloud']
        print(f"\nPoint cloud shape: {point_cloud.shape}")
        print(f"Point cloud data type: {point_cloud.dtype}")
        
        # Check for valid points (non-zero points)
        for env_idx in range(min(4, env.num_envs)):
            env_point_cloud = point_cloud[env_idx]
            valid_points = env_point_cloud[torch.any(env_point_cloud != 0, dim=1)]
            print(f"Environment {env_idx}: {len(valid_points)} valid points")
            
            if len(valid_points) > 0:
                print(f"  Point cloud range - X: [{valid_points[:, 0].min():.3f}, {valid_points[:, 0].max():.3f}]")
                print(f"  Point cloud range - Y: [{valid_points[:, 1].min():.3f}, {valid_points[:, 1].max():.3f}]")
                print(f"  Point cloud range - Z: [{valid_points[:, 2].min():.3f}, {valid_points[:, 2].max():.3f}]")
                print(f"  Color range - R: [{valid_points[:, 3].min():.3f}, {valid_points[:, 3].max():.3f}]")
                print(f"  Color range - G: [{valid_points[:, 4].min():.3f}, {valid_points[:, 4].max():.3f}]")
                print(f"  Color range - B: [{valid_points[:, 5].min():.3f}, {valid_points[:, 5].max():.3f}]")
    
    # Test camera RGB data
    if 'camera_rgb' in obs:
        rgb_data = obs['camera_rgb']
        print(f"\nCamera RGB shape: {rgb_data.shape}")
        print(f"Camera RGB data type: {rgb_data.dtype}")
        print(f"Camera RGB range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
    
    # Test camera depth data
    if 'camera_depth' in obs:
        depth_data = obs['camera_depth']
        print(f"\nCamera depth shape: {depth_data.shape}")
        print(f"Camera depth data type: {depth_data.dtype}")
        print(f"Camera depth range: [{depth_data.min():.3f}, {depth_data.max():.3f}]")
    
    # Run a few steps
    print("\nRunning simulation steps...")
    for step in range(5):
        # Random actions
        actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
        obs, rewards, dones, truncated, info = env.step(actions)
        
        print(f"Step {step + 1}:")
        print(f"  Rewards: {rewards[:4].cpu().numpy()}")  # Show first 4 environments
        print(f"  Dones: {dones[:4].cpu().numpy()}")
        
        # Check point cloud data again
        if 'point_cloud' in obs:
            point_cloud = obs['point_cloud']
            for env_idx in range(min(2, env.num_envs)):
                env_point_cloud = point_cloud[env_idx]
                valid_points = env_point_cloud[torch.any(env_point_cloud != 0, dim=1)]
                print(f"  Env {env_idx}: {len(valid_points)} valid points")
    
    print("\nTest completed successfully!")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    try:
        test_camera_environment()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the simulation
        simulation_app.close()
