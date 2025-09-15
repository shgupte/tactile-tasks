import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import pathlib

CURRENT_DIR = str(pathlib.Path(__file__).resolve().parent)

@configclass
class ScrewdriverCfg(RigidObjectCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path=f'{CURRENT_DIR}/../source/tactile_tasks/assets/usd/screwdriver/screwdriver.usd',
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Position above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
    )
