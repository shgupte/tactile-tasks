import gymnasium as gym
import os
from hand_env_cfg import TestEnvCfg

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(script_dir, "hand_env_ppo_config.yaml")

gym.register(
  id="HandManipulation-v0",
  entry_point="isaaclab.envs:ManagerBasedRLEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": "hand_env_cfg:TestEnvCfg",
    "rl_games_cfg_entry_point": yaml_config_path,  # points to your YAML config file
  },
)