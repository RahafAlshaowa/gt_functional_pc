import argparse

import torch
from isaaclab.app import AppLauncher
import os
import tempfile


parser = argparse.ArgumentParser(
    description="Demo app for showing the ground truth functional affordances."
)
parser.add_argument(
    "--mesh_path",
    type=str,
    help="Path to the mesh file. The mesh file should be in usd format, or a format that can be converted to usd.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=0.001,
    help="Scale of the object. Defaults to 0.001.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext, configclass
import isaaclab.sim as sim_utils
from mesh_convert import convert_mesh
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors.camera import CameraCfg
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene

# from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData


CAM_POS = np.array([1.955, -1.29826, 0.64681])
CAM_ROT = R.from_quat([-0.1423, 0.0799, 0.90374, 0.39126])


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
        update_period=0.05,
        height=720,
        width=480,
        colorize_semantic_segmentation=True,
        semantic_filter="class: object | ground",
        data_types=[
            "rgb",
            "distance_to_camera",
            "instance_id_segmentation_fast",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(CAM_POS.tolist()),
            rot=tuple(CAM_ROT.as_quat()[[3, 0, 1, 2]].tolist()),
            convention="world",
        ),
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


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    sim_dt = sim.get_physics_dt()

    # scene = design_scene()
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
        img_rgb = scene["cam"].data.output["rgb"]
        dist_to_cam = scene["cam"].data.output["distance_to_camera"]
        instance_id = scene["cam"].data.output["instance_id_segmentation_fast"]

        print(scene["obj"].__dict__)
        print(scene["obj"].data.root_state_w)


if __name__ == "__main__":
    main()
    simulation_app.close()
