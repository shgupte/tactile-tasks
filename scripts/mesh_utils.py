
import trimesh
import random
import numpy as np 
from shapely.geometry import Polygon

def generate_random_screwdriver():
    # Random parameters
    handle_size_x = random.uniform(0.02, 0.06)
    handle_size_y = random.uniform(0.02, 0.06)
    handle_size_z = random.uniform(0.06, 0.12)
    low = np.array([-handle_size_x / 2, -handle_size_y / 2, -handle_size_z / 2])
    high = np.array([handle_size_x / 2, handle_size_y / 2, handle_size_z / 2])
    points = np.random.uniform(low, high, (100, 3))
    lowest_height = np.min(points[:, 2]) * 0.9

    handle = trimesh.convex.convex_hull(trimesh.PointCloud(points))

    shaft_radius = random.uniform(0.005, 0.005)
    # shaft_height = random.uniform(0.09, 0.11)
    shaft_height = random.uniform(0.1, 0.1)

    # Create shaft (cylinder)
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_height, sections=32
    )
    shaft.apply_translation([0 ,0, lowest_height - shaft_height / 2 ])

    # Combine all parts
    screwdriver = trimesh.util.concatenate([handle, shaft]) 
    info = {"handle_size_x": handle_size_x,
            "handle_size_y": handle_size_y,
            "handle_size_z": handle_size_z,
            "shaft_radius": shaft_radius,
            "shaft_height": shaft_height,
            "total_height": handle_size_z + shaft_height,
            "center_height": handle_size_z / 2 + shaft_height / 2}
    
    return screwdriver, info


def generate_random_regular_screwdriver(size_range="regular"):
    # Random parameters
    if size_range == "regular":
        handle_radius = random.uniform(0.01, 0.03)
        handle_height = random.uniform(0.07, 0.14)
        shaft_radius = random.uniform(0.004, 0.008)
        shaft_height = random.uniform(0.08, 0.10)
    elif size_range == "wide":
        handle_radius = random.uniform(0.01, 0.04)
        handle_height = random.uniform(0.04, 0.14)
        shaft_radius = random.uniform(0.004, 0.008)
        shaft_height = random.uniform(0.04, 0.10)    
    # Create handle (cylinder)
    handle = trimesh.creation.cylinder(
        radius=handle_radius, height=handle_height, sections=32
    )
    # handle.apply_translation([0, 0, handle_height / 2])

    # Create shaft (cylinder)
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_height, sections=32
    )
    z_trans = handle_height / 2 + shaft_height / 2
    # z_trans = min(0.05 + shaft_height / 2, handle_height / 2 + shaft_height / 2)
    shaft.apply_translation([0, 0, -z_trans])

    # Combine all parts
    screwdriver = trimesh.util.concatenate([handle, shaft])
    # screwdriver.show()
    center_height = z_trans + shaft_height / 2
    info = {"handle_radius": handle_radius,
        "handle_height": handle_height,
        "shaft_radius": shaft_radius,
        "shaft_height": shaft_height,
        "total_height": handle_height + shaft_height, # total height computation is wrong
        "center_height": center_height}
    return screwdriver, info

def generate_short_screwdriver():
    # Random parameters
    # param for corl
    handle_radius = random.uniform(0.01, 0.03)
    handle_height = random.uniform(0.1, 0.1)
    shaft_radius = random.uniform(0.005, 0.005)
    shaft_height = random.uniform(0.05, 0.05)

    shaft_radius = random.uniform(0.005, 0.005)
    shaft_height = random.uniform(0.1, 0.1)
    
    # temporary param for actual screwdriver
    # handle_radius = random.uniform(0.016, 0.016)
    # handle_height = random.uniform(0.115, 0.115)

    # shaft_radius = random.uniform(0.0025, 0.0025)
    # shaft_height = random.uniform(0.1, 0.1)
    # Create handle (cylinder)
    handle = trimesh.creation.cylinder(
        radius=handle_radius, height=handle_height, sections=32
    )
    # handle.apply_translation([0, 0, handle_height / 2])

    # Create shaft (cylinder)
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_height, sections=32
    )
    z_trans = handle_height / 2 + shaft_height / 2
    # z_trans = min(0.05 + shaft_height / 2, handle_height / 2 + shaft_height / 2)
    shaft.apply_translation([0, 0, -z_trans])

    # Combine all parts
    screwdriver = trimesh.util.concatenate([handle, shaft])
    # screwdriver.show()
    center_height = z_trans + shaft_height / 2
    info = {"handle_radius": handle_radius,
            "handle_height": handle_height,
            "shaft_radius": shaft_radius,
            "shaft_height": shaft_height,
            "total_height": handle_height + shaft_height,
            "center_height": center_height}
    return screwdriver, info

def generate_hex_nut(outer_radius=None, height=None, hole_radius=None):
    """
    Create a simple hex nut (hexagonal prism with cylindrical hole).
    
    Args:
        outer_radius: Distance from center to flat side of the hex
        height: Height of the nut (z-axis)
        hole_radius: Radius of the center hole
    
    Returns:
        trimesh.Trimesh object of the nut
    """
    if outer_radius is None:
        outer_radius = random.uniform(0.03, 0.04)
    if hole_radius is None:
        hole_radius = random.uniform(0.01, 0.025)
    if height is None:
        height = random.uniform(0.03, 0.05)
    # Create hexagon 2D vertices
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = [(outer_radius * np.cos(a), outer_radius * np.sin(a)) for a in angles]

    # Create the hole (inner circle)
    hole_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    inner = [(hole_radius * np.cos(a), hole_radius * np.sin(a)) for a in hole_angles]

    # Use shapely to define polygon with hole
    polygon = Polygon(shell=outer, holes=[inner])

    # Extrude polygon into a 3D mesh
    hex_nut = trimesh.creation.extrude_polygon(polygon, height=height)

    return hex_nut

def generate_screwdriver_with_custom_handle_radius(handle_radius):
    """
    Generate a screwdriver with a custom handle radius.
    Other parameters (handle height, shaft radius, shaft height) remain random.
    
    Args:
        handle_radius: The radius of the handle (in meters)
    
    Returns:
        tuple: (screwdriver_mesh, info_dict)
    """
    # Use random parameters for everything except handle radius
    handle_height = random.uniform(0.02, 0.04)
    shaft_radius = random.uniform(0.003, 0.007)
    shaft_height = random.uniform(0.02, 0.04)
    
    # Create handle (cylinder) with custom radius
    handle = trimesh.creation.cylinder(
        radius=handle_radius, height=handle_height, sections=32
    )

    # Create shaft (cylinder)
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_height, sections=32
    )
    z_trans = handle_height / 2 + shaft_height / 2
    shaft.apply_translation([0, 0, -z_trans])

    # Combine all parts
    screwdriver = trimesh.util.concatenate([handle, shaft])
    center_height = z_trans + shaft_height / 2
    info = {"handle_radius": handle_radius,
            "handle_height": handle_height,
            "shaft_radius": shaft_radius,
            "shaft_height": shaft_height,
            "total_height": handle_height + shaft_height,
            "center_height": center_height}
    return screwdriver, info

if __name__ == "__main__":
    screwdriver_mesh = generate_short_screwdriver()
    screwdriver_mesh.show()