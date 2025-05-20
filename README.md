# Functional Classification of Point Clouds
## Overview 
This repository contains code to generate functional classification of observed point clouds in IsaacLab. This means classifying each point in the point cloud into either `functional` or `non-functional` areas.

## Intallation 
- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html).
- Clone this repo:
``` bash
git clone https://github.com/BE2R-Lab-RND-AI-Grasping/gt_functional_pc.git
cd gt_functional_pc
```
- Install extra python dependencies: `python -m pip install -r requirements.txt`

- Download and extract dataset from [here](https://disk.yandex.ru/d/wn96YnqAKPJ_Zw)

## Usage
- Run the script as follows, using any object/model from the dataset instead of fixed_joint_pliers/model_0

```bash
python demo.py  --mesh_path dataset/fixed_joint_pliers/model_0/object_convex_decomposition.obj --gt_pc_path dataset/fixed_joint_pliers/model_0/point_cloud_labeled.ply --device cuda --scale 0.001 --enable_cameras --visualize_pc

```

## Examples
Functional areas are labeled in red, and non-functional areas in blue. 

https://github.com/user-attachments/assets/8e492152-cf91-4958-adc4-e582ee31cd52

https://github.com/user-attachments/assets/29211f05-9774-411c-bcb4-06525fa23f4f



