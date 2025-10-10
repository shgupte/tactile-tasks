#!/usr/bin/env python3
"""
Test script to verify the screwdriver scene with table and tip constraints.
This script can be run to test the implementation.
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test screwdriver scene with table and tip constraints.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from arm_allegro import AllegroCfg
from screwdriver import ScrewdriverCfg

# Import USD and PhysX APIs for joint creation
from pxr import Usd, UsdPhysics, Gf, Sdf
from isaacsim.core.utils.stage import get_current_stage


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for testing the screwdriver scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.04),  # LxWxH (m)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,  # Static table
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, 
                dynamic_friction=1.0, 
                restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),  # Half height above z=0
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # screwdriver
    screwdriver: AssetBaseCfg = ScrewdriverCfg(prim_path="{ENV_REGEX_NS}/Screwdriver")
    
    # articulation
    robot: ArticulationCfg = AllegroCfg(prim_path="{ENV_REGEX_NS}/Robot")


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
    print("\n" + "="*80)
    print("TESTING SCREWDRIVER SCENE WITH TABLE AND TIP CONSTRAINTS")
    print("="*80)
    
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    # Design scene
    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Set up screwdriver constraints (joints to anchor tips to world)
    setup_screwdriver_constraints(scene)
    
    print("[INFO]: Setup complete! The screwdriver should now be constrained to rotate about its tip.")
    print("[INFO]: You can interact with the scene in Isaac Sim to verify the constraints work.")
    print("[INFO]: Press Ctrl+C to exit.")
    
    # Simple simulation loop to keep the scene running
    try:
        while simulation_app.is_running():
            sim.step()
    except KeyboardInterrupt:
        print("\n[INFO]: Exiting simulation...")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
