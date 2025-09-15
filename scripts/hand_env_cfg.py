# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv

from arm_allegro import AllegroCfg
from screwdriver import ScrewdriverCfg
# Scene definition
# from hand_scene import AllegroSceneCfg




##
# MDP settings
##@configclass
@configclass
class AllegroSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # screwdriver
    screwdriver: AssetBaseCfg = ScrewdriverCfg(prim_path="{ENV_REGEX_NS}/Screwdriver")
    
    # articulation
    robot: ArticulationCfg = AllegroCfg(prim_path="{ENV_REGEX_NS}/Robot")



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

@configclass
class ObservationsCfg:
    
    
    """Observation specifications for the MDP."""
    
    
        
    # Right now I will only be using joint positions and screwdriver position

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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    pass

    # # reset
    # reset_cart_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass

    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    pass
    # """Termination terms for the MDP."""

    # # (1) Time out
    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


##
# Environment configuration
##


# @configclass
# class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
#     """Configuration for the cartpole environment."""

#     # Scene settings
#     scene: AllegroSceneCfg = AllegroSceneCfg(num_envs=16, env_spacing=4.0, clone_in_fabric=True)
#     # Basic settings
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     events: EventCfg = EventCfg()
#     # MDP settings
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()

#     # Post initialization
#     def __post_init__(self) -> None:
#         """Post initialization."""
#         # general settings
#         self.decimation = 2
#         self.episode_length_s = 5
#         # viewer settings
#         self.viewer.eye = (8.0, 0.0, 5.0)
#         # simulation settings
#         self.sim.dt = 1 / 120
#         self.sim.render_interval = self.decimation


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
    
    scene: AllegroSceneCfg = AllegroSceneCfg(num_envs=1, env_spacing=4.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        # Enable viewer
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)  # Look at the center
        self.viewer.origin_type = "env"  # Camera follows environment
        self.viewer.env_index = 0  # Show first environment
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
