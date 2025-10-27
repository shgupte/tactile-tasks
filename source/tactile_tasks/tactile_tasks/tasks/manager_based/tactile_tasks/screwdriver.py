import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import pathlib
from isaaclab.sim.spawners import materials
from isaaclab.sim.utils import bind_physics_material
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg


CURRENT_DIR = str(pathlib.Path(__file__).resolve().parent)
# Package root: .../tactile_tasks/source/tactile_tasks/tactile_tasks
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

@configclass
class ScrewdriverCfg(RigidObjectCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path='/home/shgupte/omniverse/tactile-tasks/source/tactile_tasks/assets/usd/screwdriver/screwdriver_fric.usd',
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Enable gravity so it rests on table
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        )
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.015),  # Position above table
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    
    # Tip offset in local frame (meters) - adjust based on your screwdriver USD
    tip_offset_local = (0.0, 0.0, -0.045)


