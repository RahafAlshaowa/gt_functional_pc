import argparse

import torch
from isaaclab.app import AppLauncher
import os
import tempfile
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import numpy as np

from mesh_utils import classify_multiple_points, generate_kdtrees, load_pc_and_split
from utils import timeit_decorator


class Visualizer:
    def __init__(self, visualize_pc):
        if not visualize_pc:
            return
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.first = True
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        view_ctl = self.vis.get_view_control()
        view_ctl.set_zoom(0.5)

    def update(self, points):
        if not self.vis:
            return
        if not isinstance(points, o3d.geometry.PointCloud):
            self.pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        else:
            self.pcd.points = points.points
            self.pcd.colors = points.colors

        if self.first:
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            self.first = False
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            self.vis.reset_view_point(True)


parser = argparse.ArgumentParser(
    description="Demo app for showing the ground truth functional affordances."
)
parser.add_argument(
    "--mesh_path",
    type=str,
    help="Path to the mesh file. The mesh file should be in usd format, or a format that can be converted to usd.",
)
parser.add_argument(
    "--gt_pc_path",
    type=str,
    help="Path to the ground truth point cloud file. The point cloud file should be in ply format.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=0.001,
    help="Scale of the object. Defaults to 0.001.",
)
parser.add_argument(
    "--visualize_pc",
    action="store_true",
    help="Visualize the point cloud using open3d.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sensors import Camera, FrameTransformerCfg
from isaaclab.sim import SimulationCfg, SimulationContext, configclass
import isaaclab.sim as sim_utils
from mesh_convert import convert_mesh
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors.camera import CameraCfg
from scipy.spatial.transform import Rotation as R
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils.math import (
    unproject_depth,
    subtract_frame_transforms,
    transform_points,
)

# from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData


# CAM_POS = np.array([1.8, -2.5, 2.33])
# CAM_ROT = R.from_quat([-0.1423, 0.0799, 0.90374, 0.39126])

CAM_POS = np.array([0.0, 0.0, 3.5])
CAM_ROT = R.from_euler("xyz", [180, 0, 0], degrees=True)
pc_vis = Visualizer(args_cli.visualize_pc)

COLORS = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]


def get_object_usd():
    """Returns the path to the model usd file. If the model is provided as a different format, convert it to usd."""
    # print(args_cli.mesh_path)
    mesh_path = args_cli.mesh_path
    if mesh_path.endswith(".usd"):
        return mesh_path
    else:
        mesh_name = os.path.basename(mesh_path)
        mesh_name = mesh_name.split(".")[0]
        temp_dir = tempfile.mkdtemp()
        usd_path = os.path.join(temp_dir, f"{mesh_name}.usd")
        print("Converting mesh to usd...")
        # Convert mesh to usd
        convert_mesh(
            input=mesh_path,
            output=usd_path,
            mass=1.0,
            collision_approximation="convexDecomposition",
            make_instanceable=True,
        )
        return usd_path


@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.0
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(10, 10), semantic_tags=[("class", "ground"), ("color", "white")]
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    obj = RigidObjectCfg(
        prim_path="/World/Object",
        spawn=UsdFileCfg(
            usd_path=get_object_usd(),
            scale=(args_cli.scale, args_cli.scale, args_cli.scale),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
            semantic_tags=[("class", "object"), ("color", "red")],
        ),
    )
    cam = CameraCfg(
        prim_path="/World/Camera",
        height=480,
        width=720,
        data_types=[
            "rgb",
            "distance_to_camera",
            "instance_segmentation_fast",
        ],
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(CAM_POS.tolist()),
            rot=tuple(CAM_ROT.as_quat()[[3, 0, 1, 2]].tolist()),
            convention="ros",
        ),
        debug_vis=True,
    )


def object_trajectory(t):
    t = torch.tensor(t)
    x = 0.5 * torch.sin(t)
    y = 0.5 * torch.cos(t)
    z = 0.5 * torch.sin(t) + 1

    roll = np.pi * torch.cos(2 * t)
    pitch = np.pi * torch.sin(0.5 * t + 0.5)
    yaw = np.pi * torch.sin(t)

    quaternion = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()

    return torch.tensor(
        [x, y, z, quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    )


def generate_pc(camera: Camera):
    img_rgb = camera.data.output["rgb"]
    dist_to_cam = camera.data.output["distance_to_camera"]
    instance_seg = camera.data.output["instance_segmentation_fast"]

    instanec_seg_info = camera.data.info[0]["instance_segmentation_fast"]
    id_object = -1
    for id, info in instanec_seg_info["idToSemantics"].items():
        if info["class"] == "object":
            id_object = id
            break
    mask = instance_seg == id_object
    dist_to_cam = dist_to_cam * mask
    dist_to_cam[torch.isnan(dist_to_cam)] = 0.0

    points = unproject_depth(
        depth=dist_to_cam,
        intrinsics=camera.data.intrinsic_matrices,
        is_ortho=False,
    )
    points = points[points[..., 2] > 0]
    return points


def get_camera_to_world_transform(camera: Camera):
    pos = camera.data.pos_w
    quat = camera.data.quat_w_ros
    return pos, quat


def get_gt_pc():
    pc_dir = os.path.dirname(args_cli.gt_pc_path)
    pc_name = os.path.basename(args_cli.gt_pc_path)
    pcs = load_pc_and_split(pc_dir, pc_name)

    for pc in pcs:
        # pc.points = torch.tensor(pc.points) * args_cli.scale
        pc.points = o3d.utility.Vector3dVector(
            torch.tensor(np.array(pc.points)) * args_cli.scale
        )

    pc_trees = generate_kdtrees(pcs)
    return pc_trees


def get_object_to_world_transform(object: RigidObject):
    pos = object.data.root_pos_w
    quat = object.data.root_quat_w

    return pos, quat


def get_world_to_object_transform(object: RigidObject):
    pos = object.data.root_pos_w
    quat = object.data.root_quat_w

    pos, quat = subtract_frame_transforms(pos, quat)
    return pos, quat


@timeit_decorator
def calculate_classified_pc(scene, gt_trees):
    points_o_c = generate_pc(scene["cam"])  # observed points in camera frame

    pos_c_w, quat_c_w = get_camera_to_world_transform(scene["cam"])
    points_o_w = transform_points(  # observed points in world frame
        points_o_c,
        pos_c_w,
        quat_c_w,
    )

    pos_w_o, quat_w_o = get_world_to_object_transform(scene["obj"])
    points_o_o = transform_points(  # observed points in object frame
        points_o_w,
        pos_w_o,
        quat_w_o,
    )

    query_classes = classify_multiple_points(points_o_o.cpu().numpy(), gt_trees)
    query_colors = np.array([COLORS[query_class] for query_class in query_classes])

    pc_classified = o3d.geometry.PointCloud()
    pc_classified.points = o3d.utility.Vector3dVector(points_o_w.cpu().numpy())
    pc_classified.colors = o3d.utility.Vector3dVector(query_colors)

    return pc_classified


def main():
    gt_trees = get_gt_pc()

    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    sim_dt = sim.get_physics_dt()

    scene_cfg = MySceneCfg()
    scene = InteractiveScene(cfg=scene_cfg)

    sim.reset()

    print("Setup Complete. Running simulation...")
    print("Got args: ", args_cli)

    step = 0
    while simulation_app.is_running():

        object_state = scene["obj"].data.default_root_state.clone()
        object_state[..., :7] = object_trajectory(step * sim_dt)
        scene["obj"].write_root_pose_to_sim(object_state[:, :7])

        step += 1
        sim.step()
        scene.update(sim_dt)

        pc_classified = calculate_classified_pc(scene, gt_trees)
        pc_vis.update(pc_classified)


if __name__ == "__main__":
    main()
    simulation_app.close()
