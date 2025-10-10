 # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv



from hand_env_cfg import TestEnvCfg, ScrewdriverCfg, ScrewdriverCurriculumEnvCfg
from pxr import Usd, Sdf, UsdGeom, UsdPhysics, PhysxSchema, Gf
from isaacsim.core.utils.stage import get_current_stage



def create_spherical_pivot_at_tip(stage, env_index, screwdriver_prim_path, tip_world_pos, tip_offset_local):
    """Create a spherical joint that anchors the screwdriver tip to the world."""
    joint_path = f"/World/envs/env_{env_index}/ScrewdriverTipPivot"
    joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)

    # Body0 = world (empty target means world frame)
    joint.CreateBody0Rel().SetTargets([])
    # Body1 = screwdriver rigid body
    joint.CreateBody1Rel().SetTargets([Sdf.Path(screwdriver_prim_path)])

    # Set local frames so the joint anchor is at the screwdriver tip
    tip_world_pos_f = tuple(float(x) for x in (tip_world_pos.tolist() if hasattr(tip_world_pos, "tolist") else tip_world_pos))
    tip_offset_local_f = tuple(float(x) for x in (tip_offset_local.tolist() if hasattr(tip_offset_local, "tolist") else tip_offset_local))
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*tip_world_pos_f))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*tip_offset_local_f))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    return joint


def setup_screwdriver_constraints(scene):
    """Set up spherical joints to anchor each screwdriver tip to the world."""
    stage = get_current_stage()
    screwdriver = scene["screwdriver"]
    
    # Get tip offset from screwdriver config
    screwdriver_cfg = ScrewdriverCfg()
    tip_offset_local = screwdriver_cfg.tip_offset_local
    
    # Create joints for each environment
    for env_i in range(scene.num_envs):
        # Get world pose of screwdriver root in env_i
        root_pose = screwdriver.data.root_state_w[env_i, :7]  # [x,y,z, qw,qx,qy,qz]
        pos = root_pose[:3].cpu().numpy()
        qw, qx, qy, qz = root_pose[3:].cpu().numpy()
        

        # Compute tip world position = root_pose ⊕ tip_offset_local
        from isaaclab.utils.math import quat_apply
        tip_offset = torch.tensor(tip_offset_local, dtype=screwdriver.data.root_state_w.dtype)
        quat = torch.tensor([qw, qx, qy, qz], dtype=screwdriver.data.root_state_w.dtype)
        tip_world = (torch.tensor(pos, dtype=screwdriver.data.root_state_w.dtype) + 
                    quat_apply(quat, tip_offset)).cpu().numpy()

        # Obtain the prim path from the PhysX view for this instance
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        create_spherical_pivot_at_tip(stage, env_i, screwdriver_prim_path, tip_world, tip_offset_local)
        
    print(f"[INFO]: Created spherical joints for {scene.num_envs} screwdriver tips")



def main():
    """Main function."""
    # create environment configuration
    env_cfg = TestEnvCfg()  # Use TestEnvCfg for viewer-enabled testing
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Ensure gravity is enabled for the simulation
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    
    # Ensure viewer is properly configured for testing
    if hasattr(env_cfg, 'viewer'):
        env_cfg.viewer.eye = (2.0, 2.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "world"
    
    # Print configuration being loaded
    print("\n" + "="*80)
    print("HAND ENVIRONMENT CONFIGURATION")
    print("="*80)
    print("Robot configuration joint positions:")
    for joint_name, pos in env_cfg.scene.robot.init_state.joint_pos.items():
        if joint_name != ".*":  # Skip the default pattern
            print(f"  {joint_name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
    print("="*80 + "\n")
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Initial reset so PhysX views are valid and defaults are applied
    env.reset()
    
    # Apply configured hand joint positions immediately after reset
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Print actual robot joint configuration after environment setup
    print("\n" + "="*80)
    print("LOADED ROBOT JOINT CONFIGURATION")
    print("="*80)
    print(f"Number of joints: {robot.num_joints}")
    print("Default joint positions loaded in environment:")
    for i, name in enumerate(robot.joint_names):
        if i < robot.data.default_joint_pos.shape[1]:
            pos = robot.data.default_joint_pos[0, i].item()
            print(f"  {name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
    print("="*80 + "\n")

    # simulate physics
    
    count = 0
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                
                # Manually set joint positions after reset
                robot = env.scene["robot"]
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                
                print("-" * 80)
                print("[INFO]: Resetting environment and applying configured joint positions...")
                
                # Print joint positions after reset
                # robot = env.scene["robot"]
                print(f"[RESET] Joint positions after reset:")
                current_joint_pos = robot.data.joint_pos[0]  # First environment
                for i, name in enumerate(robot.joint_names):
                    if "allegro_hand" in name and i < current_joint_pos.shape[0]:
                        pos = current_joint_pos[i].item()
                        print(f"  {name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
                        
            # sample random actions
            joint_efforts = torch.zeros_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            count += 1
            
            

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()