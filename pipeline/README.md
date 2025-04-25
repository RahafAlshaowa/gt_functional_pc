- First install isaaclab (tested on binary installation [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html))

- Install extra python dependencies: `pip install -r requirements.txt`

- Download and extract dataset from [here](https://disk.yandex.ru/d/wn96YnqAKPJ_Zw)

- Run the script as follows, using any object/model from the dataset instead of fixed_joint_pliers/model_0

```bash
python demo.py  --mesh_path dataset/fixed_joint_pliers/model_0/object_convex_decomposition.obj --gt_pc_path dataset/fixed_joint_pliers/model_0/point_cloud_labeled.ply --device cuda --scale 0.001 --enable_cameras --visualize_pc

```
