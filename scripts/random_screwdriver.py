
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
# parser.add_argument("input", type=str, help="The path to the input mesh file.")
# parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument("--asset-dir", type=str, help="The path to the asset directory for output files.")
parser.add_argument("--handle-radius", type=float, help="Specify the handle radius (overrides random generation).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # run headless for batch mesh conversion

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from mesh_utils import generate_short_screwdriver, generate_random_screwdriver, generate_random_regular_screwdriver, generate_screwdriver_with_custom_handle_radius



"""Rest everything follows."""

import carb
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.dict import print_dict
import json
import random

def main():
    # Export STL
    # random.seed(1234)
    random.seed(0)
    # determine asset directory: CLI arg > env var > default
    if args_cli.asset_dir:
        asset_dir = os.path.abspath(args_cli.asset_dir)
    else:
        asset_dir = os.environ.get("SCREWDRIVER_ASSET_DIR",
                                   os.path.abspath(os.path.join(os.getcwd(), "assets")))
    
    print(f"Using asset directory: {asset_dir}")
    
    # Check if custom handle radius is provided
    if args_cli.handle_radius is not None:
        print(f"Using custom handle radius: {args_cli.handle_radius}")
    else:
        print("Using random handle radius")
    
    for i in range(0, 30):
        # Generate screwdriver with custom handle radius if provided, otherwise use random
        if False:
            screwdriver_mesh, info = generate_screwdriver_with_custom_handle_radius(args_cli.handle_radius)
        else:
            # screwdriver_mesh, info = generate_random_regular_screwdriver()
            screwdriver_mesh, info = generate_short_screwdriver()
        output_dir = os.path.join(asset_dir, f"usd_files/object/random_screwdrivers/random_screwdriver_{i}/stl/")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "random_screwdriver.stl")
        screwdriver_mesh.export(mesh_path)
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.abspath(mesh_path)

        # create destination path (USD output without extension)
        dest_path = os.path.join(asset_dir, f"usd_files/object/random_screwdrivers/random_screwdriver_{i}/screwdriver")
        # dest_path = "random_screwdriver/usd/random_screwdriver"
        os.makedirs(dest_path, exist_ok=True)
        if not os.path.isabs(dest_path):
            dest_path = os.path.abspath(dest_path)
        
        # write info into a text file, info is a dictionary
        txt_path = os.path.join(dest_path, "mesh_converter_config.txt")
        with open(txt_path, "w") as f:
            json.dump(info, f)


        # Mass properties
        mass_props = schemas_cfg.MassPropertiesCfg(mass=0.15)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()

        # Collision properties
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)

        # Create Mesh converter config
        mesh_converter_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=mesh_path,
            force_usd_conversion=True,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            make_instanceable=False,
            collision_approximation='convexDecomposition',
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input Mesh file: {mesh_path}")
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        # print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

    # # Determine if there is a GUI to update:
    # # acquire settings interface
    # carb_settings_iface = carb.settings.get_settings()
    # # read flag for whether a local GUI is enabled
    # local_gui = carb_settings_iface.get("/app/window/enabled")
    # # read flag for whether livestreaming GUI is enabled
    # livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # # Simulate scene (if not headless)
    # if local_gui or livestream_gui:
    #     # Open the stage with USD
    #     stage_utils.open_stage(mesh_converter.usd_path)
    #     # Reinitialize the simulation
    #     app = omni.kit.app.get_app_interface()
    #     # Run simulation
    #     with contextlib.suppress(KeyboardInterrupt):
    #         while app.is_running():
    #             # perform step
    #             app.update()

if __name__ == "__main__":
    main()