"""Adopted from IsaacLab's convert_mesh.py"""

import os

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def convert_mesh(input, output, mass, collision_approximation, make_instanceable):
    # check valid file path
    mesh_path = input
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.abspath(mesh_path)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid mesh file path: {mesh_path}")

    # create destination path
    dest_path = output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Mass properties
    if mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=collision_approximation != "none"
    )

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=make_instanceable,
        collision_approximation=collision_approximation,
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
