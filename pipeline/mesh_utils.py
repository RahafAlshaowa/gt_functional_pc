import os
import open3d as o3d
import numpy as np
import pathlib


def load_pc_and_split(
    model_path,
    labeled_pc_name="point_cloud_labeled.ply",
):
    """
    Load the mesh and the labeled point cloud from the model directory.
    Split the point cloud into separate point clouds based on the color (for functional/non-functional split).
    """
    path_pc_str = os.path.join(model_path, labeled_pc_name)
    path_pc = pathlib.Path(path_pc_str)
    pc = o3d.io.read_point_cloud(path_pc)

    colors = np.unique(np.array(pc.colors), axis=0)
    pointclouds = []
    for color in colors:
        ind = np.where(np.all(pc.colors == color, axis=1))[0]
        pc_temp = o3d.geometry.PointCloud()
        pc_temp.points = o3d.utility.Vector3dVector(np.array(pc.points)[ind])
        pc_temp.colors = o3d.utility.Vector3dVector(np.array(pc.colors)[ind])
        pointclouds.append(pc_temp)

    return pointclouds


def load_mesh(
    model_path,
    mesh_name="object_convex_decomposition.obj",
):
    path_mesh_str = os.path.join(model_path, mesh_name)
    path_mesh = pathlib.Path(path_mesh_str)
    mesh = o3d.io.read_triangle_mesh(path_mesh)
    return mesh


def classify_point(point, pc_trees):
    distances = []
    for tree in pc_trees:
        [_, idx, dis] = tree.search_knn_vector_3d(point, 1)
        distances.append(dis[0])
    return np.argmin(distances)


def classify_multiple_points(points, pc_trees):
    classes = []
    for point in points:
        classes.append(classify_point(point, pc_trees))
    return classes


def generate_kdtrees(pointclouds):
    pc_trees = []
    for pc in pointclouds:
        pc_tree = o3d.geometry.KDTreeFlann(pc)
        pc_trees.append(pc_tree)
    return pc_trees


if __name__ == "__main__":
    model_path = "./dataset/fixed_joint_pliers/model_0/"

    pcs = load_pc_and_split(model_path)
    pc_trees = generate_kdtrees(pcs)
    mesh = load_mesh(model_path)

    query = mesh.sample_points_uniformly(number_of_points=1000)

    query_points = np.array(query.points)

    query_classes = classify_multiple_points(query_points, pc_trees)
    colors = np.random.rand(len(pcs), 3)
    query_colors = np.array([colors[query_class] for query_class in query_classes])

    query_labeled = o3d.geometry.PointCloud()
    query_labeled.points = o3d.utility.Vector3dVector(query_points)
    query_labeled.colors = o3d.utility.Vector3dVector(query_colors)

    print("Visualizing the labeled point cloud")
    print("Num of classes: ", len(pcs))
    print("Num of points: ", len(query_points))
    o3d.visualization.draw_geometries([query_labeled])
