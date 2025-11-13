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
        #usd_path='/home/shgupte/omniverse/tactile-tasks/source/tactile_tasks/assets/usd/screwdriver/screwdriver_fric.usd',
        usd_path = '/home/armlab/Documents/Github/tactile-tasks/tactile_tasks/source/tactile_tasks/assets/usd/screwdriver/high_fric.usd',
        # usd_path='/home/armlab/Documents/Github/tactile-tasks/tactile_tasks/source/tactile_tasks/assets/sd_root/usd_files/object/random_screwdrivers/random_screwdriver_0/screwdriver.usd',
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Enable gravity so it rests on table
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
            enable_gyroscopic_forces=True,
        )
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.015),
        # pos=(0.0, 0.0, 0.155),  # Position above table
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    
    # Tip offset in local frame (meters) - adjust based on your screwdriver USD
    tip_offset_local = (0.0, 0.0, -0.045) 
    # tip_offset_local = (0.0, 0.0, -0.13) # use ths one for random screwdrivers




def apply_screwdriver_friction(env, static_friction=1.0, dynamic_friction=0.8, restitution=0.0):
    screwdriver = env.scene["screwdriver"]
    # Loop over per-env prim paths if you have multiple envs
    for prim_path in screwdriver.root_physx_view.prim_paths:
        material_path = f"{prim_path}/PhysicsMaterial"
        mat_cfg = materials.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        # Create the material prim
        mat_cfg.func(material_path, mat_cfg)
        # Bind to the screwdriver prim (applies to its colliders)
        bind_physics_material(prim_path, material_path)
