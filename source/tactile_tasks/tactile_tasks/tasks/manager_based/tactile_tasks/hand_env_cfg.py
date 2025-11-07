# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omni.usd import get_context
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

from omni.isaac.dynamic_control import _dynamic_control as dc  # 5.0.0: use underscored module, then acquire
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.sim.utils import bind_physics_material

from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils import configclass
from pxr import Usd, Sdf, UsdGeom, UsdPhysics, PhysxSchema, Gf
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import quat_apply, matrix_from_quat
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensorCfg
import os
import glob
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .arm_allegro import AllegroCfg
from .screwdriver import ScrewdriverCfg

# Scene definition
# from hand_scene import AllegroSceneCfg
@configclass
class AllegroSceneCfg(InteractiveSceneCfg):
    """Configuration for testing the screwdriver scene."""

    # Allow per-environment USD differences so we can swap geometry references
    replicate_physics = False
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    
    # # For if we want there to be a surface other than the world under the screwdriver
    # table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.6, 0.6, 0.01),  # LxWxH (m)
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             disable_gravity=True,  # Static table
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.005,
    #             rest_offset=0.0,
    #         ),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.0, 
    #             dynamic_friction=0.0, 
    #             restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.005),  # Half height above z=0
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )
    
    # screwdriver
    screwdriver: AssetBaseCfg = ScrewdriverCfg(prim_path="{ENV_REGEX_NS}/Screwdriver")
    
    # articulation
    robot: ArticulationCfg = AllegroCfg(prim_path="{ENV_REGEX_NS}/Robot")
    
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_aftc_base_link$",
        # prim_path="{ENV_REGEX_NS}/Robot/.*ee$",
    
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )
    # screwdriver_joint_path = "{ENV_REGEX_NS}/Joints/screwdriver_tip_joint"


def add_spherical_joint_at_tip(stage: Usd.Stage,
                               joint_prim_path: str,
                               parent_body: str,
                               child_body: str,
                               tip_in_parent: Gf.Vec3d,
                               tip_in_child: Gf.Vec3d):
    # Define the joint prim and mark it as PhysXSphericalJoint
    joint_prim = stage.DefinePrim(Sdf.Path(joint_prim_path), "PhysicsJoint")
    # Bind USD Physics joint API
    usd_joint = UsdPhysics.Joint(joint_prim)
    usd_joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_body)])
    usd_joint.CreateBody1Rel().SetTargets([Sdf.Path(child_body)])

    # Set local frames so both origins coincide at the tip
    usd_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(tip_in_parent))
    usd_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # identity

    # Child frame (on screwdriver): tip in the child's local coordinates
    usd_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(tip_in_child))
    usd_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # identity

    # Tag as PhysX spherical joint (ball joint)
    PhysxSchema.PhysxPhysicsSphericalJointAPI.Apply(joint_prim)
    # Leave swing/twist limits off for full 3-DoF rotation




class AllegroScene(InteractiveScene):
    def __init__(self, cfg: AllegroSceneCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def _setup(self):
        # Spawn assets per cfg (ground, light, screwdriver, robot)
        super()._setup()

        # After screwdriver is spawned but before cloning envs, author the joint
        stage = self.stage
        


def setup_screwdriver_tip_pivots(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> None:
    """
    Connects screwdriver tip to a fixed point in world space using a D6 joint with damping.
    This is MUCH simpler than using three revolute joints.
    """
    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    screwdriver_cfg = ScrewdriverCfg()
    tip_offset_local = screwdriver_cfg.tip_offset_local
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))

    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        base = f"/World/envs/env_{env_i}"

        # Compute tip world position
        root_pose = screwdriver.data.root_state_w[env_i, :7]
        pos = root_pose[:3].cpu().numpy()
        qw, qx, qy, qz = root_pose[3:].cpu().numpy()
        q = Gf.Quatd(float(qw), float(qx), float(qy), float(qz))
        tip_off = Gf.Vec3d(*[float(v) for v in tip_offset_local])
        tip_world = Gf.Vec3d(*(pos.tolist())) + q.Transform(tip_off)

        # Create or get the D6 joint
        joint_path = f"{base}/TipSphericalJoint"
        if not stage.GetPrimAtPath(joint_path).IsValid():
            joint = UsdPhysics.Joint.Define(stage, joint_path)
        else:
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        
        jp = stage.GetPrimAtPath(joint_path)
        
        # Body0 is world (don't set it), Body1 is screwdriver
        joint.CreateBody1Rel().SetTargets([Sdf.Path(screwdriver_prim_path)])
        
        # Set the joint position at the tip in world space
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in tip_world]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        # On the screwdriver, attach at the tip offset
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in tip_offset_local]))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        # Apply PhysX API to access advanced features
        PhysxSchema.PhysxJointAPI.Apply(jp)
        
        # Lock all translations (X, Y, Z) - no movement
        for axis in ["transX", "transY", "transZ"]:
            limit = UsdPhysics.LimitAPI.Apply(jp, axis)
            limit.CreateLowAttr().Set(0.0)
            limit.CreateHighAttr().Set(0.0)
        
        # Leave rotations free but add damping
        axes = ["rotZ"]
        for axis in axes:
            drive = UsdPhysics.DriveAPI.Apply(jp, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(0.0)
            # Randomize damping slightly
            damping = 0.1
            drive.CreateDampingAttr().Set(damping)
            drive.CreateMaxForceAttr().Set(0.001)
        # axes = ["rotY"]
        # for axis in axes:
        #     drive = UsdPhysics.DriveAPI.Apply(jp, axis)
        #     drive.CreateTypeAttr().Set("force")
        #     drive.CreateStiffnessAttr().Set(0.0)
        #     # Randomize damping slightly
        #     damping = 0.1
        #     drive.CreateDampingAttr().Set(damping)
        #     drive.CreateMaxForceAttr().Set(0.01)
        


def recursively_uninstance_prim(prim):
    if prim.IsInstance():
        prim.SetInstanceable(False)
    for child in prim.GetChildren():
        recursively_uninstance_prim(child)

def apply_screwdriver_friction(env, env_ids, static_friction, dynamic_friction, restitution):
    screwdriver = env.scene["screwdriver"]
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    stage = get_context().get_stage()

    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        material_path = f"{screwdriver_prim_path}/PhysicsMaterial"

        screwdriver_prim = stage.GetPrimAtPath(screwdriver_prim_path)
        recursively_uninstance_prim(screwdriver_prim)

        mat_cfg = RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        mat_cfg.func(material_path, mat_cfg)


        success = bind_physics_material(screwdriver_prim_path, material_path)
        if not success:
            pass


def add_screwdriver_rotation_markers(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> None:
    """Add a small red visual marker to each screwdriver to make rotation visible.

    The marker is a `UsdGeom.Sphere` child prim under the screwdriver root, placed with a small
    lateral offset so that spinning is obvious in the viewer. The prim is visual-only and does not
    participate in physics since no physics API is applied to it.
    """
    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]

    # Choose a small offset from the local origin so that the marker traces a circle when rotating.
    # Adjust if your screwdriver geometry needs a different radius.
    marker_local_offset = Gf.Vec3f(0.04, 0.0, 0.03)
    marker_radius = 0.004

    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        # Ensure we can author children under the screwdriver prim
        screw_prim = stage.GetPrimAtPath(screwdriver_prim_path)
        if screw_prim.IsInstanceable():
            screw_prim.SetInstanceable(False)
        marker_prim_path = f"{screwdriver_prim_path}/RotationMarker"

        marker_prim = stage.GetPrimAtPath(marker_prim_path)
        if not marker_prim.IsValid():
            # Create a new sphere marker
            marker = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_prim_path))
            marker.CreateRadiusAttr(marker_radius)

            # Color it red using displayColor on the gprim
            gprim = UsdGeom.Gprim(marker.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

            # Place it with a local translation relative to the screwdriver
            xformable = UsdGeom.Xformable(marker.GetPrim())
            xformable.AddTranslateOp().Set(marker_local_offset)
        else:
            # If it already exists, ensure basic attributes are set (idempotent)
            marker = UsdGeom.Sphere(stage.GetPrimAtPath(marker_prim_path))
            if not marker.GetRadiusAttr():
                marker.CreateRadiusAttr(marker_radius)
            else:
                marker.GetRadiusAttr().Set(marker_radius)

            gprim = UsdGeom.Gprim(marker.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

            xformable = UsdGeom.Xformable(marker.GetPrim())
            # Add or update translate op
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpName().startswith("xformOp:translate"):
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            translate_op.Set(marker_local_offset)


def _discover_random_screwdriver_usds() -> list[str]:
    """Return all screwdriver USDs from the attached random set on disk.

    Searches: .../usd_files/object/random_screwdrivers/**/screwdriver.usd
    """
    base_dir = "/home/armlab/Documents/Github/tactile-tasks/tactile_tasks/source/tactile_tasks/assets/usd/screwdriver"
    pattern = os.path.join(base_dir, "screwdriver_fric*.usd")
    return sorted(glob.glob(pattern))


def randomize_screwdriver_geometry_prestartup(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
    usd_paths: list[str] | None = None,
) -> None:
    """Swap each env's screwdriver USD reference before play starts (root-level properties persist).

    Operates on USD references only (safe at prestartup). Requires replicate_physics == False.
    """
    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    # Resolve per-env prims from regex prim path
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if not usd_paths:
        usd_paths = _discover_random_screwdriver_usds()
    if not usd_paths:
        return

    env_indices = list(range(len(prim_paths))) if env_ids is None else env_ids.tolist()
    with Sdf.ChangeBlock():
        for env_i in env_indices:
            prim_path = prim_paths[env_i]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            refs = prim.GetReferences()
            choice_idx = (env_i * 131 + (env.cfg.seed or 0)) % len(usd_paths)
            usd_path = usd_paths[choice_idx]
            refs.ClearReferences()
            refs.AddReference(usd_path)

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Right now I will only be moving finger joints
    hand_joint_pos = mdp.RelativeJointPositionActionCfg(asset_name="robot", 
                                                # use_default_offset=True,
                                                joint_names=[
                                                "allegro_hand_hitosashi_finger_finger_joint_0",
                                                "allegro_hand_hitosashi_finger_finger_joint_1",
                                                "allegro_hand_hitosashi_finger_finger_joint_2",
                                                "allegro_hand_hitosashi_finger_finger_joint_3",
                                                "allegro_hand_naka_finger_finger_joint_4",
                                                "allegro_hand_naka_finger_finger_joint_5",
                                                "allegro_hand_naka_finger_finger_joint_6",
                                                "allegro_hand_naka_finger_finger_joint_7",
                                                "allegro_hand_kusuri_finger_finger_joint_8",
                                                "allegro_hand_kusuri_finger_finger_joint_9",
                                                "allegro_hand_kusuri_finger_finger_joint_10",
                                                "allegro_hand_kusuri_finger_finger_joint_11",
                                                "allegro_hand_oya_finger_joint_12",
                                                "allegro_hand_oya_finger_joint_13",
                                                "allegro_hand_oya_finger_joint_14",
                                                "allegro_hand_oya_finger_joint_15"],
                                           scale=1.0,  # Reduced from 1.0 for stability
                                           preserve_order=True,
                                        #    clip={"allegro_hand_hitosashi_finger_finger_joint_0": (-2.0, 2.0),
                                        #          "allegro_hand_hitosashi_finger_finger_joint_1": (-2.0, 2.0),
                                        #          "allegro_hand_hitosashi_finger_finger_joint_2": (-2.0, 2.0),
                                        #          "allegro_hand_hitosashi_finger_finger_joint_3": (-2.0, 2.0),
                                        #          "allegro_hand_naka_finger_finger_joint_4": (-2.0, 2.0),
                                        #          "allegro_hand_naka_finger_finger_joint_5": (-2.0, 2.0),
                                        #          "allegro_hand_naka_finger_finger_joint_6": (-2.0, 2.0),
                                        #          "allegro_hand_naka_finger_finger_joint_7": (-2.0, 2.0),
                                        #          "allegro_hand_kusuri_finger_finger_joint_8": (-2.0, 2.0),
                                        #          "allegro_hand_kusuri_finger_finger_joint_9": (-2.0, 2.0),
                                        #          "allegro_hand_kusuri_finger_finger_joint_10": (-2.0, 2.0),
                                        #          "allegro_hand_kusuri_finger_finger_joint_11": (-2.0, 2.0),
                                        #          "allegro_hand_oya_finger_joint_12": (-2.0, 2.0),
                                        #          "allegro_hand_oya_finger_joint_13": (-2.0, 2.0),
                                        #          "allegro_hand_oya_finger_joint_14": (-2.0, 2.0),
                                        #          "allegro_hand_oya_finger_joint_15": (-2.0, 2.0)}
                                           clip={"allegro_hand_hitosashi_finger_finger_joint_0": (-5.0, 5.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_1": (-5.0, 5.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_2": (-5.0, 5.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_3": (-5.0, 5.0),
                                                 "allegro_hand_naka_finger_finger_joint_4": (-5.0, 5.0),
                                                 "allegro_hand_naka_finger_finger_joint_5": (-5.0, 5.0),
                                                 "allegro_hand_naka_finger_finger_joint_6": (-5.0, 5.0),
                                                 "allegro_hand_naka_finger_finger_joint_7": (-5.0, 5.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_8": (-5.0, 5.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_9": (-5.0, 5.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_10": (-5.0, 5.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_11": (-5.0, 5.0),
                                                 "allegro_hand_oya_finger_joint_12": (-5.0, 5.0),
                                                 "allegro_hand_oya_finger_joint_13": (-5.0, 5.0),
                                                 "allegro_hand_oya_finger_joint_14": (-5.0, 5.0),
                                                 "allegro_hand_oya_finger_joint_15": (-5.0, 5.0)}
                                        
    )
    
    
def joint_pos_in_order(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # Get the actual values from the
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids]


def screwdriver_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
    """Screwdriver pose (position + quaternion) in the environment frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    # Get position relative to environment origin
    pos = asset.data.root_pos_w - env.scene.env_origins
    # Get quaternion (w, x, y, z)
    quat = asset.data.root_quat_w
    # Concatenate position and quaternion: [x, y, z, w, x, y, z]
    return torch.cat([pos, quat], dim=-1)


def screwdriver_orientation_z_axis(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
    """Z-axis direction of the screwdriver in world frame (for upright reward)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get rotation matrix from quaternion
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    # Extract z-axis (third column of rotation matrix)
    z_axis = rot_matrix[:, :, 2]  # Shape: (num_envs, 3)
    return z_axis


def screwdriver_angular_velocity_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
    """Angular velocity around screwdriver's z-axis (for rotation reward)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get angular velocity in world frame
    ang_vel_w = asset.data.root_ang_vel_w
    # Get rotation matrix to transform to local frame
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    # Transform angular velocity to local frame
    ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    # Return only z-component (yaw rotation)
    return ang_vel_local[:, 2:3]  # Shape: (num_envs, 1)


# Curriculum tracking variables some- this will be managed by the environment
CURRENT_CURRICULUM_STAGE = 0  # 0: upright focus, 1: rotation focus (legacy global)
CURRICULUM_CHECK_COUNTER = 0  # Track how many times we've checked
CURRICULUM_CHECK_INTERVAL = 5  # Check every 20 epochs


def _get_curriculum_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env curriculum stage tensor, creating it if missing.

    0: upright focus, 1: rotation focus.
    """
    if not hasattr(env, "curriculum_stage"):
        env.curriculum_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.curriculum_stage


def init_curriculum_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> None:
    """Initialize per-env curriculum stages to 0. If env_ids provided, only reset those."""
    stage = _get_curriculum_stage(env)
    if env_ids is None:
        stage.zero_()
    else:
        stage[env_ids] = 0



def screwdriver_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
    """Reward for keeping screwdriver upright (z-axis aligned with world z-axis).
    
    Returns 0 reward for any deviation greater than 20 degrees from upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get rotation matrix from quaternion
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    # Extract z-axis (third column of rotation matrix)
    z_axis = rot_matrix[:, :, 2]  # Shape: (num_envs, 3)
    # Reward is dot product with world z-axis [0, 0, 1]
    upright_alignment = z_axis[:, 2]  # Just the z-component
    
    # cos(20°) ≈ 0.94 - this is our threshold
    threshold_cos_20_deg = math.cos(math.radians(20.0))
    
    # Zero reward if deviation > 20 degrees (alignment < cos(20°))
    upright_reward = torch.where(
        upright_alignment >= threshold_cos_20_deg,
        ((upright_alignment - threshold_cos_20_deg) / (1.0 - threshold_cos_20_deg)) ** 2,  # Normalize to [0,1] and square
        torch.zeros_like(upright_alignment)  # Zero reward for > 20 degree deviation
    )
    
    return upright_reward


def screwdriver_upright_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
    """Angle deviation from upright.

    Returns the angle between the screwdriver's local z-axis and the world z-axis.
    Output is in radians by default; set ``degrees=True`` for degrees.
    Shape: (num_envs, 1).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    # Local z-axis of screwdriver in world frame
    z_axis = rot_matrix[:, :, 2]  # (num_envs, 3)
    # Dot with world z-axis [0, 0, 1] -> just the z component; clamp for numerical safety
    cos_theta = torch.clamp(z_axis[:, 2], -1.0, 1.0)
    angle = torch.acos(cos_theta)
    if degrees:
        angle = angle * (180.0 / math.pi)
    return angle.unsqueeze(-1)


def screwdriver_tilt_exceeds(
    env: ManagerBasedRLEnv,
    threshold_deg: float = 25.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
) -> torch.Tensor:
    """Return per-env boolean indicating tilt greater than threshold degrees.

    Computes the angle between the screwdriver local z-axis and the world z-axis.
    Returns True when the tilt angle is strictly greater than ``threshold_deg``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    # cos(theta) is the z-component of the local z-axis in world frame
    cos_theta = torch.clamp(rot_matrix[:, :, 2][:, 2], -1.0, 1.0)
    threshold_cos = math.cos(math.radians(threshold_deg))
    return cos_theta < threshold_cos

# def screwdriver_signed_yaw_velocity_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
#     *,
#     degrees: bool = False,
#     vmax: float | None = None,
# ) -> torch.Tensor:
#     """Reward negative (clockwise) yaw velocity, penalize positive (counter-clockwise).
    
#     - Positive reward for negative yaw_vel (clockwise rotation)
#     - Negative reward for positive yaw_vel (counter-clockwise rotation)
#     - Symmetric: reward magnitude equals penalty magnitude
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
    
#     yaw_vel = screwdriver_yaw_velocity(env, asset_cfg=asset_cfg, degrees=degrees).squeeze(-1)
#     # Normalize and clip instead of using an explicit gain
#     # if vmax is not None and vmax > 0:
#     #     norm = yaw_vel / vmax
#     # else:
#     #     norm = yaw_vel
#     base = -torch.clamp(yaw_vel, -4.0, 4.0)
#     # Gate by curriculum stage: Stage 0 = off; Stage 1+ = on
#     stage = 1#_get_curriculum_stage(env)
#     return base


# def screwdriver_yaw_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
#     """Return yaw angular velocity of the screwdriver in local frame.

#     - Units: rad/s by default; set ``degrees=True`` for deg/s.
#     - Shape: (num_envs, 1)
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
#     ang_vel_w = asset.data.root_ang_vel_w
#     quat = asset.data.root_quat_w
#     rot_matrix = matrix_from_quat(quat)
#     ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
#     yaw_vel = ang_vel_local[:, 2:3]
#     if degrees:
#         yaw_vel = yaw_vel * (180.0 / math.pi)
#     return yaw_vel

def screwdriver_signed_yaw_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
    *,
    degrees: bool = False,
    vmax: float | None = None,
) -> torch.Tensor:
    """Reward negative (clockwise) yaw velocity, penalize positive (counter-clockwise).
    
    Only rewards rotation when screwdriver is reasonably upright (within 20 degrees).
    This prevents wobbling motions from being falsely interpreted as yaw rotation.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get angular velocity in local frame
    ang_vel_w = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    yaw_vel = ang_vel_local[:, 2]  # Rotation around screwdriver's own Z-axis
    
    # Base reward for clockwise rotation (negative yaw velocity)
    base_reward = -torch.clamp(yaw_vel, -5.0, 5.0)
    
    # Get upright alignment to gate the reward
    z_axis = rot_matrix[:, :, 2]  # Screwdriver's local Z-axis in world frame
    upright_alignment = z_axis[:, 2]  # Dot product with world Z-axis [0,0,1]
    
    # Only reward when reasonably upright (within ~20 degrees)
    # cos(20°) ≈ 0.94
    threshold_cos_20_deg = math.cos(math.radians(20.0))
    upright_mask = (upright_alignment >= threshold_cos_20_deg).float()
    
    # Gate the reward: zero if tilted, full if upright
    return base_reward * upright_mask


def screwdriver_yaw_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
    """Return per-timestep yaw angle change about local z.

    - Units: radians per step by default; set ``degrees=True`` for degrees per step.
    - Shape: (num_envs, 1)
    """
    yaw_vel = screwdriver_yaw_velocity(env, asset_cfg=asset_cfg, degrees=degrees)
    dt = env.physics_dt
    return yaw_vel * dt




def screwdriver_stability_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
    """L2 norm squared of local x/y angular velocity.

    - Units: (rad/s)^2 by default; set ``degrees=True`` for (deg/s)^2.
    - Returns per-env scalar (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    vel_xy = ang_vel_local[:, :2]
    if degrees:
        vel_xy = vel_xy * (180.0 / math.pi)
    return -torch.sum(vel_xy * vel_xy, dim=1)


# Effort and energy penalties

def torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   joint_names: list[str] | None = None, weight: float = 1e-2) -> torch.Tensor:
    """Penalize squared actuator torques per env: -weight * sum(tau^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    else:
        joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    tau = asset.data.applied_torque[:, joint_ids]
    base = -weight * torch.sum(tau * tau, dim=1)
    # Curriculum-independent: no stage gating
    return base

def energy_penalty_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                       joint_names: list[str] | None = None, scale: float = 5e-2) -> torch.Tensor:
    """Penalize per-step mechanical energy via |tau * qd| integrated over dt.

    Returns approximately -scale * energy[J] per step per env.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    else:
        joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    tau = asset.data.applied_torque[:, joint_ids]
    qd = asset.data.joint_vel[:, joint_ids]
    power_abs = torch.sum(torch.abs(tau * qd), dim=1)  # |W|
    dt = env.physics_dt
    base = -scale * power_abs * dt
    # Curriculum-independent: no stage gating
    return base

def add_action_noise(env, std: float = 0.02):
    a = env.action_manager.action
    env.action_manager.action = torch.clamp(a + std * torch.randn_like(a), -1.0, 1.0)


def finger_joint_deviation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), degrees: bool = False) -> torch.Tensor:
    """Penalty for finger joints deviating from their initial positions to encourage finger gaiting.

    Computes the L2 norm squared of joint deviations per env.
    - Units: rad^2 by default; set ``degrees=True`` for deg^2.
    """
    stage = 1#_get_curriculum_stage(env)
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get current joint positions
    joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    
    # Reference joint positions from articulation defaults
    initial_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    
    # Calculate L2 deviation from initial positions with unit selection
    delta = current_joint_pos - initial_joint_pos
    if degrees:
        delta = delta * (180.0 / math.pi)
    joint_deviation = torch.sum(delta * delta, dim=1)
    
    # Convert to penalty (negative reward). Increase penalty strength in stage 3.
    # Right now coefficients are arbitrary
    
    deviation_penalty = joint_deviation
    
    #mask = (stage >= 1).float()
    # Do I need the exp?
    return -deviation_penalty 

# Should probably not be using this
# UPDATE: I DO NOT USE THIS ANYMORE
def curriculum_weighted_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Combined reward that adapts based on curriculum stage."""
    global CURRENT_CURRICULUM_STAGE
    
    # Get individual reward components
    upright_rew = screwdriver_upright_reward(env)
    rotation_rew = screwdriver_signed_yaw_velocity_reward(env)
    stability_rew = screwdriver_stability_reward(env)
    finger_penalty = finger_joint_deviation_penalty(env, SceneEntityCfg("robot", joint_names=[
        "allegro_hand_hitosashi_finger_finger_joint_0",
        "allegro_hand_hitosashi_finger_finger_joint_1",
        "allegro_hand_hitosashi_finger_finger_joint_2",
        "allegro_hand_hitosashi_finger_finger_joint_3",
        "allegro_hand_naka_finger_finger_joint_4",
        "allegro_hand_naka_finger_finger_joint_5",
        "allegro_hand_naka_finger_finger_joint_6",
        "allegro_hand_naka_finger_finger_joint_7",
        "allegro_hand_kusuri_finger_finger_joint_8",
        "allegro_hand_kusuri_finger_finger_joint_9",
        "allegro_hand_kusuri_finger_finger_joint_10",
        "allegro_hand_kusuri_finger_finger_joint_11",
        "allegro_hand_oya_finger_joint_12",
        "allegro_hand_oya_finger_joint_13",
        "allegro_hand_oya_finger_joint_14",
        "allegro_hand_oya_finger_joint_15"
    ]))
    
    # Stage 0: Focus purely on keeping upright - no velocity concerns
    if CURRENT_CURRICULUM_STAGE == 0:
        total_reward = (
            1.0 * upright_rew +      # Primary: keep upright
            0.2 * stability_rew +    # Secondary: stability
            0.05 * finger_penalty    # Small penalty to encourage finger movement
        )
    # Stage 1: Focus on rotation while maintaining upright
    else:
        total_reward = (
            0.3 * upright_rew +      # Still important: stay upright
            1.0 * rotation_rew +     # Primary: achieve rotation
            0.2 * stability_rew +    # Secondary: stability
            0.1 * finger_penalty     # Higher penalty to encourage finger gaiting for rotation
        )
    
    return total_reward



def curriculum_reward_weights(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> torch.Tensor:
    """Curriculum function that adjusts reward weights based on performance."""
    global CURRENT_CURRICULUM_STAGE
    
    # Stage 0: Focus purely on upright stability - no rotation concerns
    if CURRENT_CURRICULUM_STAGE == 0:
        upright_weight = 1.0
        torque_weight = 0.0
        energy_weight = 0.0
        rotation_weight = 0.0  # No rotation reward in Stage 0
        stability_weight = 0.2  # Some stability to encourage smooth control
        deviation_weight = 0.0
    # Stage 1: Balance upright and rotation with increased stability
    else:
        upright_weight = 0.3   
        torque_weight = 0.3
        energy_weight = 0.3
        rotation_weight = 1.0
        stability_weight = 0.4  # Increased stability reward in Stage 1
        deviation_weight = 0.2
    
    return torch.tensor([upright_weight, rotation_weight, stability_weight], device=env.device)


def advance_curriculum_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> None:
    """Advance per-env curriculum stages based on performance."""
    global CURRICULUM_CHECK_COUNTER, CURRICULUM_CHECK_INTERVAL
    
    CURRICULUM_CHECK_COUNTER += 1
    if CURRICULUM_CHECK_COUNTER < CURRICULUM_CHECK_INTERVAL:
        return
    CURRICULUM_CHECK_COUNTER = 0

    stage = _get_curriculum_stage(env)
    upright_rew = screwdriver_upright_reward(env)          # (num_envs,)
    stability_rew = screwdriver_stability_reward(env)      # (num_envs,)

    # Log average curriculum stage and reward statistics
    avg_stage = torch.mean(stage.float()).item()
    avg_upright = torch.mean(upright_rew).item()
    avg_stability = torch.mean(stability_rew).item()
    stage_0_count = int((stage == 0).sum().item())
    stage_1_count = int((stage == 1).sum().item())
    stage_2_count = int((stage == 2).sum().item())
    stage_3_count = int((stage == 3).sum().item())
    
    # print(f"[CURRICULUM] Avg Stage: {avg_stage:.2f} | Stages: 0:{stage_0_count} 1:{stage_1_count} 2:{stage_2_count} 3:{stage_3_count}")
    # print(f"[CURRICULUM] Avg Rewards - Upright: {avg_upright:.3f}, Stability: {avg_stability:.3f}")

    # Conditions for promoting to Stage 1
    promote_0_to_1 = (stage == 0) & (upright_rew > 0.95) & (stability_rew > 0.9)
    if torch.any(promote_0_to_1):
        num = int(promote_0_to_1.sum().item())
        stage[promote_0_to_1] = 1
        # print(f"[CURRICULUM] Promoted {num} env(s) from Stage 0 → 1")

# DONT REALLY NEED THIS ANYMORE
def log_training_progress(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> None:
    """Log current training progress metrics without advancing curriculum."""
    global CURRENT_CURRICULUM_STAGE
    
    # Get current performance metrics
    upright_rew = screwdriver_upright_reward(env)
    stability_rew = screwdriver_stability_reward(env)
    
    avg_upright = torch.mean(upright_rew).item()
    avg_stability = torch.mean(stability_rew).item()
    
    # Get rotational velocity metrics
    asset = env.scene["screwdriver"]
    ang_vel_w = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    yaw_vel = ang_vel_local[:, 2]  # z-component (yaw rotation)
    
    avg_rot_vel = torch.mean(torch.abs(yaw_vel)).item()  # Average absolute rotational velocity
    max_rot_vel = torch.max(torch.abs(yaw_vel)).item()   # Maximum absolute rotational velocity
    
    # Always print average upright percentage on reset (across all envs)
    asset_all: RigidObject = env.scene[SceneEntityCfg("screwdriver").name]
    rot_matrix_all = matrix_from_quat(asset_all.data.root_quat_w)
    cos_theta_all = torch.clamp(rot_matrix_all[:, :, 2][:, 2], -1.0, 1.0)
    threshold_cos = math.cos(math.radians(19.0))
    upright_percent = float((cos_theta_all >= threshold_cos).float().mean().item() * 100.0)
    # print(f"[UPRIGHT] Upright={upright_percent:.1f}%")
    
    # Retain periodic detailed log
    if hasattr(log_training_progress, 'call_count'):
        log_training_progress.call_count += 1
    else:
        log_training_progress.call_count = 0
    if log_training_progress.call_count % 100 == 0:
        pass# print(f"[PROGRESS] Stage {CURRENT_CURRICULUM_STAGE}: Upright={avg_upright:.3f}, Stability={avg_stability:.3f}, RotVel={avg_rot_vel:.3f} (max={max_rot_vel:.3f})")


def reset_curriculum_stage(env: ManagerBasedRLEnv, stage: int = 0) -> None:
    """Reset curriculum stage to specified value."""
    global CURRENT_CURRICULUM_STAGE
    CURRENT_CURRICULUM_STAGE = stage
    # print(f"🔄 CURRICULUM RESET: Stage {stage}")



def log_contact_sensor_sample(env, env_ids=None) -> None:
        # Access the sensor by its config name ("contact_forces" in AllegroSceneCfg)
        sensor = env.scene["contact_forces"]

        # Try common fields; fall back to introspection so you can see what's available
        data = sensor.data
        if hasattr(data, "net_forces_w"):
            f = data.net_forces_w          # shape: (num_envs, num_bodies, 3)
        elif hasattr(data, "forces_w"):
            f = data.forces_w              # shape: (num_envs, num_bodies, 3)
        else:
            # print("Contact sensor fields:", [k for k in dir(data) if not k.startswith("_")])
            return

        # Example: print max contact force magnitude per env (first few)
        # import torch_
        f_norm = torch.linalg.norm(f, dim=-1)  # (num_envs, num_bodies)
        max_per_env = f_norm.max(dim=1).values
        # print("[CONTACT] max |F| per env (first 8):", max_per_env[:8].tolist())


def contact_forces_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation function for contact forces from contact sensors.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        
    Returns:
        Contact forces tensor with shape (num_envs, num_bodies * 3) flattened.
    """
    # Access the contact sensor by its config name
    sensor = env.scene[sensor_cfg.name]
    data = sensor.data
    
    # Get contact forces - try different possible attribute names
    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w  # shape: (num_envs, num_bodies, 3)
    elif hasattr(data, "forces_w"):
        forces = data.forces_w      # shape: (num_envs, num_bodies, 3)
    else:
        # Fallback: return zeros if sensor data not available
        num_envs = env.num_envs
        num_bodies = 4  # Assuming 4 finger tips based on the sensor config
        forces = torch.zeros((num_envs, num_bodies, 3), device=env.device)
    
    # Flatten the forces tensor: (num_envs, num_bodies, 3) -> (num_envs, num_bodies * 3)
    return forces.flatten(start_dim=1)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=joint_pos_in_order, noise=None, 
                        params={"asset_cfg": SceneEntityCfg("robot",
                        joint_names=["allegro_hand_hitosashi_finger_finger_joint_0",
                                        "allegro_hand_hitosashi_finger_finger_joint_1",
                                        "allegro_hand_hitosashi_finger_finger_joint_2",
                                        "allegro_hand_hitosashi_finger_finger_joint_3",
                                        "allegro_hand_naka_finger_finger_joint_4",
                                        "allegro_hand_naka_finger_finger_joint_5",
                                        "allegro_hand_naka_finger_finger_joint_6",
                                        "allegro_hand_naka_finger_finger_joint_7",
                                        "allegro_hand_kusuri_finger_finger_joint_8",
                                        "allegro_hand_kusuri_finger_finger_joint_9",
                                        "allegro_hand_kusuri_finger_finger_joint_10",
                                        "allegro_hand_kusuri_finger_finger_joint_11",
                                        "allegro_hand_oya_finger_joint_12",
                                        "allegro_hand_oya_finger_joint_13",
                                        "allegro_hand_oya_finger_joint_14",
                                        "allegro_hand_oya_finger_joint_15"])})

        # Screwdriver pose (position + quaternion)
        screwdriver_pose = ObsTerm(
            func=screwdriver_pose,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        # Screwdriver z-axis orientation (for upright tracking)
        screwdriver_orientation_z = ObsTerm(
            func=screwdriver_orientation_z_axis,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        # Screwdriver angular velocity around z-axis (for rotation tracking)
        screwdriver_angular_velocity_z = ObsTerm(
            func=screwdriver_angular_velocity_z,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ContactObservationCfg:
    """Observation specifications for the MDP with contact sensor data."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group including contact forces."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=joint_pos_in_order, noise=None, 
                        params={"asset_cfg": SceneEntityCfg("robot",
                        joint_names=["allegro_hand_hitosashi_finger_finger_joint_0",
                                        "allegro_hand_hitosashi_finger_finger_joint_1",
                                        "allegro_hand_hitosashi_finger_finger_joint_2",
                                        "allegro_hand_hitosashi_finger_finger_joint_3",
                                        "allegro_hand_naka_finger_finger_joint_4",
                                        "allegro_hand_naka_finger_finger_joint_5",
                                        "allegro_hand_naka_finger_finger_joint_6",
                                        "allegro_hand_naka_finger_finger_joint_7",
                                        "allegro_hand_kusuri_finger_finger_joint_8",
                                        "allegro_hand_kusuri_finger_finger_joint_9",
                                        "allegro_hand_kusuri_finger_finger_joint_10",
                                        "allegro_hand_kusuri_finger_finger_joint_11",
                                        "allegro_hand_oya_finger_joint_12",
                                        "allegro_hand_oya_finger_joint_13",
                                        "allegro_hand_oya_finger_joint_14",
                                        "allegro_hand_oya_finger_joint_15"])})

        # Screwdriver pose (position + quaternion)
        screwdriver_pose = ObsTerm(
            func=screwdriver_pose,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        # Screwdriver z-axis orientation (for upright tracking)
        screwdriver_orientation_z = ObsTerm(
            func=screwdriver_orientation_z_axis,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        # Screwdriver angular velocity around z-axis (for rotation tracking)
        screwdriver_angular_velocity_z = ObsTerm(
            func=screwdriver_angular_velocity_z,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        # Contact forces from finger tips
        contact_forces = ObsTerm(
            func=contact_forces_obs,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
def randomize_screwdriver_mass(env, env_ids=None, asset_cfg=SceneEntityCfg("screwdriver"),
                               mass_range=(0.05, 0.20)):
    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    for env_i in env_indices:
        prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        prim = stage.GetPrimAtPath(prim_path)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        # sample mass per env (kg)
        m = float(torch.empty(1, device=env.device).uniform_(mass_range[0], mass_range[1]).cpu())
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr:
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(m)


@configclass
class CurriculumCfg:
    """Curriculum learning configuration using Isaac Lab's built-in curriculum manager."""
    
    # Need to figure out how this works in IsaacLab
    pass


@configclass
class EventCfg:
    """Configuration for events."""
    
    # Reset hand root pose to fixed position
    reset_hand_root_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    
    # inside EventCfg in hand_env_cfg.py
    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),  # no offset from defaults
            "velocity_range": (0.0, 0.0),
        },
    )
    
    add_action_noise = EventTerm(func=add_action_noise, mode="pre_physics_step", params={"std": 0.02})
    
    # apply_screwdriver_friction = EventTerm(
    #     func=apply_screwdriver_friction,
    #     mode="reset",
    #     params = {"static_friction" : 10.0, "dynamic_friction" : 10.0, "restitution" : 0.0}
    # )
    
    # # Prestartup: per-env geometry selection (safe USD edits before play)
    # randomize_screwdriver_usd = EventTerm(
    #     func=randomize_screwdriver_geometry_prestartup,
    #     mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("screwdriver")},
    # )

    # # Prestartup: randomize physics material friction for screwdriver colliders
    # randomize_screwdriver_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("screwdriver"),
    #         "static_friction_range": (0.4, 1.2),
    #         "dynamic_friction_range": (0.3, 1.0),
    #         "restitution_range": (0.0, 0.05),
    #         "num_buckets": 8,
    #         "make_consistent": True,
    #     },
    # )

    # Reset screwdriver positions (this adds env_origins offset)
    reset_screwdriver_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("screwdriver"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    
    create_tip_pivots = EventTerm(
        func=setup_screwdriver_tip_pivots,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )
    
    # Add visual markers on the screwdriver so rotation is easy to see
    create_rotation_markers = EventTerm(
        func=add_screwdriver_rotation_markers,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    # Ensure markers are visible immediately at startup (not only after first reset)
    create_rotation_markers_startup = EventTerm(
        func=add_screwdriver_rotation_markers,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )
    
    # Log training progress every 100 resets
    progress_log = EventTerm(
        func=log_training_progress,
        mode="reset",
    )
    
    # Curriculum advancement - checks every 5 seconds
    # Not sure if this is a good way to check for curriculum advancement
    curriculum_advancement = EventTerm(
        func=advance_curriculum_stage,
        mode="interval",
        interval_range_s=(5.0, 5.0),  # Check every 5 seconds
    )
    
    # Log contact sensor readings every 0.2 seconds
    contact_sensor_log = EventTerm(
        func=log_contact_sensor_sample,
        mode="interval",
        interval_range_s=(0.2, 0.2),  # Log every 0.2s
    )

    # randomize_screwdriver_mass_event = EventTerm(
    #     func=randomize_screwdriver_mass,
    #     mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("screwdriver"), "mass_range": (0.05, 0.20)},
    # )





# Curriculum weights are now handled by per-env masking in reward functionss
# Base weights: upright=1.0, stable=8.0, deviation=0.3
# Angle weights: 30°=3.0 (stage 1), 60°=6.0 (stage 2), 90°=10.0 (stage 3)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    
    # I'm not sure if I wnat the terminating and alive rewards; the paper does not mention them
    
    # (1) Constant running reward (keep small so it doesn't dominate)
    # alive = RewTerm(func=mdp.is_alive, weight=0.1)
    
    # (2) Failure penalty (comparable scale with other terms)
    # terminating = RewTerm(func=mdp.is_terminated, weight=0.0)
    
    ## Penalizes deviation from upright; paper only punishes unwanted angular velocitioes
    # screwdriver_upright = RewTerm(
    #     func=screwdriver_upright_reward,
    #     weight=1.0,  # Always important
    #     params={"asset_cfg": SceneEntityCfg("screwdriver")},
    # )
    
    # (4) Rotation velocity reward - partial reward for positive rotation
    screwdriver_rotation = RewTerm(
        func=screwdriver_signed_yaw_velocity_reward,
        # Map yaw velocity to [-1, 1] via clipping with vmax
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("screwdriver"),
            "vmax": 4.0,    # normalize & clip to [-1,1]
        },
    )
    
    # (5) Stability reward: minimize unwanted angular velocities
    screwdriver_stability = RewTerm(
        func=screwdriver_stability_reward,
        # stability reward returns exp(-||w_xy||^2) ∈ (0, 1], keep weight ~1
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    
    
    # (6) Finger joint deviation penalty to encourage finger gaiting (masked by stage)
    finger_deviation_penalty = RewTerm(
        func=finger_joint_deviation_penalty,
        weight=8.0,  # Small penalty to encourage movement without overwhelming other rewards
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "allegro_hand_hitosashi_finger_finger_joint_0",
            "allegro_hand_hitosashi_finger_finger_joint_1",
            "allegro_hand_hitosashi_finger_finger_joint_2",
            "allegro_hand_hitosashi_finger_finger_joint_3",
            "allegro_hand_naka_finger_finger_joint_4",
            "allegro_hand_naka_finger_finger_joint_5",
            "allegro_hand_naka_finger_finger_joint_6",
            "allegro_hand_naka_finger_finger_joint_7",
            "allegro_hand_kusuri_finger_finger_joint_8",
            "allegro_hand_kusuri_finger_finger_joint_9",
            "allegro_hand_kusuri_finger_finger_joint_10",
            "allegro_hand_kusuri_finger_finger_joint_11",
            "allegro_hand_oya_finger_joint_12",
            "allegro_hand_oya_finger_joint_13",
            "allegro_hand_oya_finger_joint_14",
            "allegro_hand_oya_finger_joint_15"
        ])},
    )

    # (7) Effort and energy regularization
    torque_mag_penalty = RewTerm(
        func=torque_penalty,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": None, "weight": 1e-3},
    )
    energy_penalty = RewTerm(
        func=energy_penalty_abs,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": None, "scale": 5e-4},
    )

    
    # (8) Curriculum-based combined reward (alternative to individual rewards)
    # curriculum_reward = RewTerm(
    #     func=curriculum_weighted_reward,
    #     weight=1.0,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Screwdriver tilt exceeds threshold (terminate when > 25 degrees from upright)
    screwdriver_tilt_limit = DoneTerm(
        func=screwdriver_tilt_exceeds,
        params={"threshold_deg": 20.0, "asset_cfg": SceneEntityCfg("screwdriver")},
    )

    # (3) Success termination: reached current target angle per-env



##
# Environment configuration
##



@configclass
class EmptyRewardsCfg:
    """Empty rewards configuration."""
    pass

@configclass
class EmptyTerminationsCfg:
    """Empty terminations configuration."""
    pass

@configclass
class EmptyEventsCfg:
    """Empty events configuration."""
    pass

@configclass
class TestEnvCfg(ManagerBasedRLEnvCfg):
    """Test configuration with viewer enabled."""
    
    # scene: AllegroSceneCfg = AllegroSceneCfg(num_envs=1, env_spacing=4.0, clone_in_fabric=True)
    scene = AllegroSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        # Enable viewer with proper configuration
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)  # Look at the center
        self.viewer.origin_type = "world"  # Fixed world camera
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_position_iteration_count = 16  # Default: 4
        self.sim.physx.solver_velocity_iteration_count = 4   # Default: 1
        
@configclass
class TestContactEnvCfg(ManagerBasedRLEnvCfg):
    """Test configuration with viewer enabled."""
    
    # scene: AllegroSceneCfg = AllegroSceneCfg(num_envs=1, env_spacing=4.0, clone_in_fabric=True)
    scene = AllegroSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ContactObservationCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        # Enable viewer with proper configuration
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)  # Look at the center
        self.viewer.origin_type = "world"  # Fixed world camera
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

@configclass
class ScrewdriverCurriculumEnvCfg(ManagerBasedRLEnvCfg):
    """Full training configuration with curriculum learning."""
    
    scene = AllegroSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5  # Longer episodes for training
        # Disable viewer for training
        self.viewer.eye = None
        self.viewer.origin_type = "world"
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation



    