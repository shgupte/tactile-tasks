
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import pathlib

CURRENT_DIR = str(pathlib.Path(__file__).resolve().parent)

@configclass
class AllegroCfg(ArticulationCfg):
    spawn = sim_utils.UsdFileCfg(
            usd_path = f'/home/armlab/Documents/Github/tactile-tasks/tactile_tasks/source/tactile_tasks/assets/usd/arm_allegro.usd',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=True,
            fix_root_link=True,
            ),
        )
    init_state = ArticulationCfg.InitialStateCfg(
            # Place the hand closer to the screwdriver for proper grasp alignment
            # Screwdriver initial pos is set to (-0.02, 0.05, 0.075)
            # Lower the hand to be closer to the screwdriver shaft
            pos=(-0.01, -0.03, 0.25),
            # Flip hand upside down with -20° X-axis rotation, then add -20° Y-axis rotation
            # -20° Y rotation ⊗ (180° Y ⊗ -20° X) = (-0.173648, 0.984807, -0.173648, 0.0)
            rot=( .707107, 0.0, 0.707107, 0.0),
            joint_pos={
            # Gentle power grasp: fingers lightly curled around screwdriver shaft
            # Index (hitosashi) - light curl for gentle grip
            # "allegro_hand_hitosashi_finger_finger_joint_0": 0.1,
            # "allegro_hand_hitosashi_finger_finger_joint_1": 0.6,
            # "allegro_hand_hitosashi_finger_finger_joint_2": 0.6,
            # "allegro_hand_hitosashi_finger_finger_joint_3": 1.1,
            # "allegro_hand_naka_finger_finger_joint_4": 0.3,
            # "allegro_hand_naka_finger_finger_joint_5": 0.5,
            # "allegro_hand_naka_finger_finger_joint_6": 0.9,
            # "allegro_hand_naka_finger_finger_joint_7": 1.0,
            # "allegro_hand_kusuri_finger_finger_joint_8": 0.468,
            # "allegro_hand_kusuri_finger_finger_joint_9": 0.85,
            # "allegro_hand_kusuri_finger_finger_joint_10": 0.8,
            # "allegro_hand_kusuri_finger_finger_joint_11": 0.8,
            # "allegro_hand_oya_finger_joint_12": 1.2,
            # "allegro_hand_oya_finger_joint_13": 0.3,
            # "allegro_hand_oya_finger_joint_14": 0.3,
            # "allegro_hand_oya_finger_joint_15": 1.0,
            
            "allegro_hand_hitosashi_finger_finger_joint_0": 0.8,
            "allegro_hand_hitosashi_finger_finger_joint_1": 0.6,
            "allegro_hand_hitosashi_finger_finger_joint_2": 0.6,
            "allegro_hand_hitosashi_finger_finger_joint_3": 1.1,
            "allegro_hand_naka_finger_finger_joint_4": 0.3,
            "allegro_hand_naka_finger_finger_joint_5": 0.5,
            "allegro_hand_naka_finger_finger_joint_6": 0.9,
            "allegro_hand_naka_finger_finger_joint_7": 1.0,
            "allegro_hand_kusuri_finger_finger_joint_8": 0.468,
            "allegro_hand_kusuri_finger_finger_joint_9": 0.85,
            "allegro_hand_kusuri_finger_finger_joint_10": 0.8,
            "allegro_hand_kusuri_finger_finger_joint_11": 0.8,
            "allegro_hand_oya_finger_joint_12": 1.2,
            "allegro_hand_oya_finger_joint_13": 0.3,
            "allegro_hand_oya_finger_joint_14": 0.3,
            "allegro_hand_oya_finger_joint_15": 1.0,
            },
        )
    actuators = {
            "allegro_joints": ImplicitActuatorCfg(
                joint_names_expr = [".*"],
                effort_limit = 400.0,
                velocity_limit = 100.0,
                stiffness = 3.0,
                damping = 3.0,
                friction = 1.0,
                ),
        }
