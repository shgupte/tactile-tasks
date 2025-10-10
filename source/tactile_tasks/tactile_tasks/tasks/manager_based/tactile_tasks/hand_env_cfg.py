# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils import configclass
from pxr import Usd, Sdf, UsdGeom, UsdPhysics, PhysxSchema, Gf
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import quat_apply, matrix_from_quat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .arm_allegro import AllegroCfg
from .screwdriver import ScrewdriverCfg, apply_screwdriver_friction

# Scene definition
# from hand_scene import AllegroSceneCfg
@configclass
class AllegroSceneCfg(InteractiveSceneCfg):
    """Configuration for testing the screwdriver scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
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

    # Child frame (on screwdriver): tip in the child’s local coordinates
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
    """Create a spherical joint at the screwdriver tip for each env, called once during initialization."""
    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    screwdriver_cfg = ScrewdriverCfg()
    tip_offset_local = screwdriver_cfg.tip_offset_local
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    
    for env_i in env_indices:
        # Use per-environment state, even if prim is shared
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        joint_path = f"/World/envs/env_{env_i}/ScrewdriverTipPivot"
        
        # print(f"Env {env_i}: Creating joint at {joint_path}, Screwdriver state index = {env_i}")
        
        # Compute tip world position using per-environment state
        root_pose = screwdriver.data.root_state_w[env_i, :7]
        pos = root_pose[:3].cpu().numpy()
        qw, qx, qy, qz = root_pose[3:].cpu().numpy()
        tip_offset = torch.tensor(tip_offset_local, dtype=torch.float64)
        quat = torch.tensor([qw, qx, qy, qz], dtype=torch.float64)
        quat = quat / torch.norm(quat)
        tip_world = (torch.tensor(pos, dtype=torch.float64) + quat_apply(quat, tip_offset)).cpu().numpy()
        
         # print(f"Env {env_i}: Root pos = {pos}, Quat = {[qw, qx, qy, qz]}, Tip world = {tip_world}")
        
        # Create or update joint
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
            # print(f"Env {env_i}: Defined new joint at {joint_path}")
        else:
            joint = UsdPhysics.SphericalJoint(joint_prim)
            # print(f"Env {env_i}: Updating existing joint at {joint_path}")
        joint_prim = stage.GetPrimAtPath(joint_path)

        
        # Set joint targets (body0 = world, body1 = screwdriver)
        joint.CreateBody0Rel().SetTargets([])  # World frame
        joint.CreateBody1Rel().SetTargets([Sdf.Path(screwdriver_prim_path)])
        
        # Set anchor points
        tip_world_pos_f = tuple(float(x) for x in tip_world)
        tip_offset_local_f = tuple(float(x) for x in tip_offset_local)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3d(*tip_world_pos_f))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3d(*tip_offset_local_f))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                
        physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        # I wanted to add some joint friction but IsaacSim was not happy about it
        physx_joint_api.CreateJointFrictionAttr().Set(0.0)

        
        # print(f"Env {env_i}: Applied PhysX joint API with minimal friction")

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
    marker_local_offset = Gf.Vec3f(0.02, 0.0, 0.0)
    marker_radius = 0.004

    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
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
                                           scale=1.0, 
                                           preserve_order=True,
                                           clip={"allegro_hand_hitosashi_finger_finger_joint_0": (-2.0, 2.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_1": (-2.0, 2.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_2": (-2.0, 2.0),
                                                 "allegro_hand_hitosashi_finger_finger_joint_3": (-2.0, 2.0),
                                                 "allegro_hand_naka_finger_finger_joint_4": (-2.0, 2.0),
                                                 "allegro_hand_naka_finger_finger_joint_5": (-2.0, 2.0),
                                                 "allegro_hand_naka_finger_finger_joint_6": (-2.0, 2.0),
                                                 "allegro_hand_naka_finger_finger_joint_7": (-2.0, 2.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_8": (-2.0, 2.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_9": (-2.0, 2.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_10": (-2.0, 2.0),
                                                 "allegro_hand_kusuri_finger_finger_joint_11": (-2.0, 2.0),
                                                 "allegro_hand_oya_finger_joint_12": (-2.0, 2.0),
                                                 "allegro_hand_oya_finger_joint_13": (-2.0, 2.0),
                                                 "allegro_hand_oya_finger_joint_14": (-2.0, 2.0),
                                                 "allegro_hand_oya_finger_joint_15": (-2.0, 2.0)}
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

#------------------Ignore, since we will be using velocity reward instead------------------

# def _get_initial_screwdriver_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Return per-env initial screwdriver quaternion tensor (wxyz), creating it if missing."""
#     if not hasattr(env, "screwdriver_init_quat_w"):
#         asset: RigidObject = env.scene["screwdriver"]
#         env.screwdriver_init_quat_w = asset.data.root_quat_w.clone()
#     return env.screwdriver_init_quat_w



# def _get_cumulative_rotation(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Get or create the per-env cumulative positive rotation tensor."""
#     if not hasattr(env, 'cumulative_positive_rotation'):
#         env.cumulative_positive_rotation = torch.zeros(env.num_envs, device=env.device)
#     return env.cumulative_positive_rotation


# def init_initial_screwdriver_orientation(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> None:
#     """Capture current screwdriver orientation as the per-env initial orientation (at reset)."""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     init_quat = _get_initial_screwdriver_quat(env)
#     init_quat[env_ids] = asset.data.root_quat_w[env_ids]
    
#     # Also reset cumulative rotation for these environments
#     cumulative_rot = _get_cumulative_rotation(env)
#     cumulative_rot[env_ids] = 0.0


# def reset_cumulative_rotation(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
#     """Reset cumulative rotation for specified environments."""
#     cumulative_rot = _get_cumulative_rotation(env)
#     cumulative_rot[env_ids] = 0.0

# ------------------Ignore, since we will be using velocity reward instead------------------


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

# ------------------Ignore, since we will be using velocity reward instead------------------
# def screwdriver_rotation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     """Reward for cumulative positive rotation around the screwdriver's z-axis."""
#     asset: RigidObject = env.scene[asset_cfg.name]
    
#     # Get angular velocity in the screwdriver's local frame
#     ang_vel_w = asset.data.root_ang_vel_w
#     quat = asset.data.root_quat_w
#     rot_matrix = matrix_from_quat(quat)
#     ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    
#     # Z-axis velocity (yaw) is what we care about
#     yaw_vel = ang_vel_local[:, 2]
    
#     # Get time step (assuming dt is available)
#     dt = env.physics_dt
    
#     # Update cumulative rotation with only positive contributions
#     cumulative_rot = _get_cumulative_rotation(env)
#     positive_rotation = torch.clamp(yaw_vel * dt, min=0.0)  # Only positive rotation
#     cumulative_rot += positive_rotation
    
#     # Reward based on cumulative rotation with diminishing returns
#     # Use a logarithmic scale so early rotation gets good rewards, but it doesn't explode
#     rotation_reward = torch.log(1.0 + cumulative_rot * 10.0)  # Scale factor of 10.0
    
#     # Normalize to reasonable range (0 to ~2.3 for 1 radian of rotation)
#     rotation_reward = torch.clamp(rotation_reward, max=2.3)
    
#     stage = _get_curriculum_stage(env)
#     # Active during stages 1-2; stage 3 uses signed velocity reward instead
#     mask = ((stage == 1) | (stage == 2)).float()
#     return rotation_reward * mask
# ------------------Ignore, since we will be using velocity reward instead------------------

def screwdriver_signed_yaw_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
    *,
    degrees: bool = False,
    gain: float = 1.0,
    vmax: float | None = None,
) -> torch.Tensor:
    """
    - Units: rad/s by default; set ``degrees=True`` for deg/s.
    - Optional clipping: set ``vmax`` to clamp velocity to ``[-vmax, vmax]`` before scaling.
    - Final reward = ``gain * yaw_velocity`` masked to stage 3.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    yaw_vel = screwdriver_yaw_velocity(env, asset_cfg=asset_cfg, degrees=degrees).squeeze(-1)
    if vmax is not None:
        yaw_vel = torch.clamp(yaw_vel, -vmax, vmax)
    reward = gain * yaw_vel

    # Is this practical?
    return torch.exp(reward)


def screwdriver_yaw_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
    """Return yaw angular velocity of the screwdriver in local frame.

    - Units: rad/s by default; set ``degrees=True`` for deg/s.
    - Shape: (num_envs, 1)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    ang_vel_local = torch.bmm(rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)).squeeze(-1)
    yaw_vel = ang_vel_local[:, 2:3]
    if degrees:
        yaw_vel = yaw_vel * (180.0 / math.pi)
    return yaw_vel


def screwdriver_yaw_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False) -> torch.Tensor:
    """Return per-timestep yaw angle change about local z.

    - Units: radians per step by default; set ``degrees=True`` for degrees per step.
    - Shape: (num_envs, 1)
    """
    yaw_vel = screwdriver_yaw_velocity(env, asset_cfg=asset_cfg, degrees=degrees)
    dt = env.physics_dt
    return yaw_vel * dt


# ------------------Ignore, since we will be using velocity reward instead------------------
# def _relative_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Compute relative yaw (radians) of screwdriver from its initial orientation."""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     cur_q = asset.data.root_quat_w  # (N, 4)
#     init_q = _get_initial_screwdriver_quat(env)  # (N, 4)
#     cur_R = matrix_from_quat(cur_q)      # (N, 3, 3)
#     init_R = matrix_from_quat(init_q)    # (N, 3, 3)
#     R_rel = torch.bmm(init_R.transpose(-2, -1), cur_R)
#     rel_r00 = R_rel[:, 0, 0]
#     rel_r10 = R_rel[:, 1, 0]
#     return torch.atan2(rel_r10, rel_r00)
# ------------------Ignore, since we will be using velocity reward instead------------------

# ------------------Ignore, since we will be using velocity reward instead------------------            
# def _angle_target_reward(yaw_rel: torch.Tensor, target_rad: float, sharpness: float = 4.0) -> torch.Tensor:
#     """Return exp(-k * angle_error^2) with angle wraparound."""
#     error = torch.abs(yaw_rel - target_rad)
#     error = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
#     return torch.exp(-sharpness * (error ** 2))


# def screwdriver_reach_30deg_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     yaw_rel = _relative_yaw(env, asset_cfg)
#     reward = _angle_target_reward(yaw_rel, math.radians(30.0), sharpness=6.0)
#     stage = _get_curriculum_stage(env)
#     return reward * (stage == 1).float()


# def screwdriver_reach_60deg_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     yaw_rel = _relative_yaw(env, asset_cfg)
#     reward = _angle_target_reward(yaw_rel, math.radians(60.0), sharpness=6.0)
#     stage = _get_curriculum_stage(env)
#     return reward * (stage == 2).float()


# def screwdriver_reach_90deg_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     yaw_rel = _relative_yaw(env, asset_cfg)
#     reward = _angle_target_reward(yaw_rel, math.radians(90.0), sharpness=6.0)
#     stage = _get_curriculum_stage(env)
#     return reward * (stage == 3).float()
# ------------------Ignore, since we will be using velocity reward instead------------------

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
    return torch.exp(torch.sum(vel_xy * vel_xy, dim=1))


# removed torque penalty


def finger_joint_deviation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), degrees: bool = False) -> torch.Tensor:
    """Penalty for finger joints deviating from their initial positions to encourage finger gaiting.

    Computes the L2 norm squared of joint deviations per env.
    - Units: rad^2 by default; set ``degrees=True`` for deg^2.
    """
    stage = _get_curriculum_stage(env)
    
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
    
    mask = (stage >= 1).float()
    # Do I need the exp?
    return torch.exp(deviation_penalty * mask)

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


# def screwdriver_fell_over(env: ManagerBasedRLEnv, threshold_angle: float = 30.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     """Terminate per-env if screwdriver falls over beyond its per-env threshold angle."""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # Get rotation matrix from quaternion
#     quat = asset.data.root_quat_w
#     rot_matrix = matrix_from_quat(quat)
#     # Extract z-axis (third column of rotation matrix)
#     z_axis = rot_matrix[:, :, 2]  # Shape: (num_envs, 3)
#     # Check angle with world z-axis
#     z_component = z_axis[:, 2]  # Dot product with [0, 0, 1]
    
#     # Per-env thresholds from curriculum (degrees) -> cos
#     # per_env_thresh_deg = curriculum_termination_threshold(env)  # (num_envs,)
#     per_env_thresh_cos = torch.cos(torch.deg2rad(per_env_thresh_deg))
#     # Terminate per env
#     fell_over = z_component < per_env_thresh_cos
#     return fell_over

# #------------------Ignore, since we will be using velocity reward instead------------------
# def curriculum_rotation_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
#     """Per-env target rotation velocity based on curriculum stage.

#     All stages use 0.0 now (no velocity target in final stage).
#     """
#     stage = _get_curriculum_stage(env)
#     target = torch.zeros_like(stage, dtype=torch.float32)
#     return target.to(env.device)

# #------------------Ignore, since we will be using velocity reward instead------------------
# def curriculum_termination_threshold(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
#     """Per-env termination threshold based on curriculum stage (degrees)."""
#     stage = _get_curriculum_stage(env)
#     th = torch.where(
#         stage == 0,
#         torch.full_like(stage, 45, dtype=torch.float32),
#         torch.full_like(stage, 30, dtype=torch.float32)
#     )
#     return th.to(env.device)

# #------------------Ignore, since we will be using velocity reward instead------------------
def curriculum_reward_weights(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> torch.Tensor:
    """Curriculum function that adjusts reward weights based on performance."""
    global CURRENT_CURRICULUM_STAGE
    
    # Stage 0: Focus purely on upright stability - no rotation concerns
    if CURRENT_CURRICULUM_STAGE == 0:
        upright_weight = 1.0
        rotation_weight = 0.0  # No rotation reward in Stage 0
        stability_weight = 0.2  # Some stability to encourage smooth control
        deviation_weight = 0.0
    # Stage 1: Balance upright and rotation with increased stability
    else:
        upright_weight = 0.3
        rotation_weight = 1.0
        stability_weight = 0.4  # Increased stability reward in Stage 1
        deviation_weight = 0.0
    
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
    
    print(f"[CURRICULUM] Avg Stage: {avg_stage:.2f} | Stages: 0:{stage_0_count} 1:{stage_1_count} 2:{stage_2_count} 3:{stage_3_count}")
    print(f"[CURRICULUM] Avg Rewards - Upright: {avg_upright:.3f}, Stability: {avg_stability:.3f}")

    # Conditions for promoting to Stage 1
    promote_0_to_1 = (stage == 0) & (upright_rew > 0.95) & (stability_rew > 0.9)
    if torch.any(promote_0_to_1):
        num = int(promote_0_to_1.sum().item())
        stage[promote_0_to_1] = 1
        print(f"[CURRICULUM] Promoted {num} env(s) from Stage 0 → 1")

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
    
    # Log progress every 100 resets
    if hasattr(log_training_progress, 'call_count'):
        log_training_progress.call_count += 1
    else:
        log_training_progress.call_count = 0
    
    if log_training_progress.call_count % 100 == 0:
        print(f"[PROGRESS] Stage {CURRENT_CURRICULUM_STAGE}: Upright={avg_upright:.3f}, Stability={avg_stability:.3f}, RotVel={avg_rot_vel:.3f} (max={max_rot_vel:.3f})")


def reset_curriculum_stage(env: ManagerBasedRLEnv, stage: int = 0) -> None:
    """Reset curriculum stage to specified value."""
    global CURRENT_CURRICULUM_STAGE
    CURRENT_CURRICULUM_STAGE = stage
    print(f"🔄 CURRICULUM RESET: Stage {stage}")

# #------------------Ignore, since we will be using velocity reward instead------------------
# def screwdriver_reached_target_angle(env: ManagerBasedRLEnv, tol_deg: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")) -> torch.Tensor:
#     """Per-env success termination when current stage's target angle is reached.

#     Stages:
#       - 1 → 30° target
#       - 2 → 60° target
#       - 3 → 90° target
#     Stage 0 never triggers success.
#     """
#     stage = _get_curriculum_stage(env)
#     yaw_rel = _relative_yaw(env, asset_cfg)  # radians

#     # Compute per-env target angle in radians; stage 3 has no angle target (velocity-focused)
#     target_deg = torch.zeros_like(stage, dtype=torch.float32)
#     target_deg = torch.where(stage == 1, torch.full_like(target_deg, 30.0), target_deg)
#     target_deg = torch.where(stage == 2, torch.full_like(target_deg, 60.0), target_deg)
#     target_rad = torch.deg2rad(target_deg)

#     # Angle error with wrap to [-pi, pi]
#     error = torch.abs(yaw_rel - target_rad)
#     error = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
#     tol_rad = math.radians(tol_deg)

#     # Only allow angle-based success for stages 1..2
#     active = (stage == 1) | (stage == 2)
#     reached = (error <= tol_rad) & active
#     return reached

# #------------------Ignore, since we will be using velocity reward instead------------------
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
    
    # apply_screwdriver_friction = EventTerm(
    #     func=apply_screwdriver_friction,
    #     mode="reset",
    #     params = {"static_friction" : 1.0, "dynamic_friction" : 0.8, "restitution" : 0.0}
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


    


# Curriculum weights are now handled by per-env masking in reward functionss
# Base weights: upright=1.0, stable=8.0, deviation=0.3
# Angle weights: 30°=3.0 (stage 1), 60°=6.0 (stage 2), 90°=10.0 (stage 3)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    
    # I'm not sure if I wnat the terminating and alive rewards; the paper does not mention them
    
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    ## Penalizes deviation from upright; paper only punishes unwanted angular velocitioes
    # screwdriver_upright = RewTerm(
    #     func=screwdriver_upright_reward,
    #     weight=1.0,  # Always important
    #     params={"asset_cfg": SceneEntityCfg("screwdriver")},
    # )
    
    # (4) Rotation velocity reward - partial reward for positive rotation
    screwdriver_rotation = RewTerm(
        func=screwdriver_signed_yaw_velocity_reward,
        weight=200.0,  # Moderate weight for encouraging rotation
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )
    
    # (5) Stability reward: minimize unwanted angular velocities
    screwdriver_stability = RewTerm(
        func=screwdriver_stability_reward,
        weight=8.0,  # Always important for smooth control
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    
    
    # (6) Finger joint deviation penalty to encourage finger gaiting (masked by stage)
    finger_deviation_penalty = RewTerm(
        func=finger_joint_deviation_penalty,
        weight=0.3,  # Small penalty to encourage movement without overwhelming other rewards
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
    
    # (2) Screwdriver fell over (beyond threshold angle)
    # screwdriver_fell_over = DoneTerm(
    #     func=screwdriver_fell_over,
    #     params={"threshold_angle": 20.0, "asset_cfg": SceneEntityCfg("screwdriver")},
    # )

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
        self.viewer.eye = (2.0, 2.0, 2.0)
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
