# Tiled Camera Integration for Point Cloud Extraction

This document explains how to use the tiled camera functionality that has been added to the tactile tasks environment for extracting point cloud data.

## Overview

The tiled camera integration provides the ability to extract 3D point cloud data from the simulation environment, which can be used for various applications including:

- Visual perception for robotic manipulation
- 3D scene understanding
- Point cloud-based machine learning models (e.g., PointNet)
- Tactile feedback simulation

## Features

### 1. Tiled Camera Configuration

The tiled camera is configured in the `AllegroSceneCfg` class with the following parameters:

```python
tiled_camera: TiledCameraCfg = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    update_period=0.0,
    data_types=["rgb", "distance_to_image_plane"],
    width=640,
    height=480,
    horizontal_fov=60.0,
    vertical_fov=45.0,
    horizontal_aperture=20.955,
    vertical_aperture=15.955,
    clipping_range=(0.1, 1000.0),
    position=(0.0, 0.0, 0.5),  # Position above the scene
    orientation=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
    debug_vis=True,
)
```

### 2. Observation Functions

Three main observation functions are provided:

#### `tiled_camera_rgb_obs(env, sensor_cfg)`
- Extracts RGB image data from the tiled camera
- Returns flattened RGB tensor: `(num_envs, height * width * 3)`

#### `tiled_camera_depth_obs(env, sensor_cfg)`
- Extracts depth data from the tiled camera
- Returns flattened depth tensor: `(num_envs, height * width)`

#### `extract_point_cloud_from_camera(env, sensor_cfg)`
- Combines RGB and depth data to create 3D point clouds
- Returns point cloud tensor: `(num_envs, num_points, 6)` where each point has `[x, y, z, r, g, b]`
- Automatically filters invalid points and handles variable point cloud sizes

### 3. Environment Configurations

#### `TestCameraEnvCfg`
A test configuration that includes all camera observations:

```python
@configclass
class TestCameraEnvCfg(ManagerBasedRLEnvCfg):
    """Test configuration with tiled camera for point cloud extraction."""
    
    scene = AllegroSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = CameraObservationCfg()  # Includes camera data
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
```

## Usage Examples

### 1. Basic Environment Usage

```python
import gymnasium as gym
from tactile_tasks.tasks.manager_based.tactile_tasks.hand_env_cfg import TestCameraEnvCfg

# Create environment with camera
env_cfg = TestCameraEnvCfg()
env = gym.make("Isaac-Template-TactileTasks-v0", cfg=env_cfg)

# Reset and get observations
obs, _ = env.reset()

# Access camera data
rgb_data = obs['camera_rgb']  # Shape: (num_envs, height * width * 3)
depth_data = obs['camera_depth']  # Shape: (num_envs, height * width)
point_cloud = obs['point_cloud']  # Shape: (num_envs, num_points, 6)
```

### 2. Point Cloud Processing

```python
# Process point cloud data
for env_idx in range(env.num_envs):
    env_point_cloud = point_cloud[env_idx]
    
    # Filter valid points (non-zero points)
    valid_points = env_point_cloud[torch.any(env_point_cloud != 0, dim=1)]
    
    # Extract 3D coordinates and colors
    positions = valid_points[:, :3]  # (num_points, 3)
    colors = valid_points[:, 3:]     # (num_points, 3)
    
    print(f"Environment {env_idx}: {len(valid_points)} valid points")
```

### 3. Integration with PointNet

```python
from tactile_tasks.tasks.manager_based.tactile_tasks.pointnet import PointNet

# Create PointNet model
pointnet = PointNet(k=3).to(device)

# Process point cloud data
points_3d = point_cloud[:, :, :3].transpose(1, 2)  # (num_envs, 3, num_points)
features = pointnet(points_3d)  # (num_envs, feature_dim)
```

## Test Scripts

### 1. Basic Camera Test
Run the basic camera environment test:

```bash
cd /home/armlab/Documents/Github/tactile-tasks/tactile_tasks
python scripts/test_camera_env.py
```

This script will:
- Create the camera environment
- Extract and display point cloud data
- Show RGB and depth data statistics
- Run a few simulation steps

### 2. PointNet Integration Test
Run the PointNet integration test:

```bash
cd /home/armlab/Documents/Github/tactile-tasks/tactile_tasks
python scripts/pointnet_example.py
```

This script will:
- Demonstrate point cloud processing techniques
- Test PointNet integration with real camera data
- Show various point cloud filtering and normalization methods

## Configuration Options

### Camera Parameters

You can modify the camera configuration in `hand_env_cfg.py`:

- `width`, `height`: Image resolution
- `horizontal_fov`, `vertical_fov`: Field of view in degrees
- `position`: Camera position in world coordinates
- `orientation`: Camera orientation as quaternion
- `clipping_range`: Near and far clipping planes

### Point Cloud Processing

The point cloud extraction function includes several configurable parameters:

- Depth filtering: Points with depth < 0.1 or > 1000.0 are filtered out
- Point cloud padding: Variable-sized point clouds are padded to the same size
- Color normalization: RGB values are expected to be in [0, 1] range

## Performance Considerations

1. **Memory Usage**: Point cloud data can be memory-intensive, especially with high-resolution cameras and many environments
2. **Processing Time**: Point cloud extraction involves coordinate transformations and filtering
3. **Camera Resolution**: Higher resolution provides more detailed point clouds but increases computational cost

## Troubleshooting

### Common Issues

1. **No Point Cloud Data**: Ensure the camera is positioned correctly and objects are within the clipping range
2. **Empty Point Clouds**: Check that the depth data is valid and objects are visible to the camera
3. **Import Errors**: Make sure Isaac Lab is properly installed and the environment is activated

### Debug Options

Enable debug visualization by setting `debug_vis=True` in the camera configuration. This will show the camera frustum in the simulation viewer.

## Future Enhancements

Potential improvements to the camera integration:

1. **Multiple Camera Views**: Add support for multiple cameras at different angles
2. **Point Cloud Compression**: Implement efficient point cloud compression for large datasets
3. **Real-time Processing**: Optimize point cloud extraction for real-time applications
4. **Advanced Filtering**: Add more sophisticated point cloud filtering and segmentation
5. **Integration with RL**: Develop reward functions based on point cloud features

## Dependencies

- Isaac Lab
- PyTorch
- NumPy
- The PointNet implementation in `pointnet.py`

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [Tiled Rendering in Isaac Lab](https://isaac-sim.github.io/IsaacLab/v1.3.0/source/features/tiled_rendering.html)
