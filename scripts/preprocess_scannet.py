import argparse
import glob
import json
import os

import numpy as np
import torch
import tqdm
from PIL import Image
from plyfile import PlyData, PlyElement
import math

# def save_arrays_to_ply(input_scene, output_file="output.ply"):
#     """
#     Saves arrays (points, colors, normals) to a .ply file using Open3D.

#     Args:
#         points (numpy.ndarray): Nx3 array of 3D points.
#         colors (numpy.ndarray): Nx3 array of RGB colors (optional, values in [0, 1]).
#         normals (numpy.ndarray): Nx3 array of normals (optional).
#         output_file (str): Path to the output .ply file.
#     """
#     scene_name = input_scene.split('/')[-1]
#     try:
#         arrays = torch.load(f"./scannet/scannet_3d/train/{scene_name}_vh_clean_2.pth")
#         point_array = arrays[0]
#         color_array = arrays[1]
#         # Create a PointCloud object
#         point_cloud = o3d.geometry.PointCloud()

#         # Set points
#         point_cloud.points = o3d.utility.Vector3dVector(point_array)
#         point_cloud.normals = o3d.utility.Vector3dVector(np.zeros_like(point_array))

#         # Set colors if provided
#         color_array += 1
#         color_array /= 2
#         point_cloud.colors = o3d.utility.Vector3dVector(color_array)

#         o3d.io.write_point_cloud(f"{input_scene}/points3d.ply", point_cloud)
#         print(f"Saved point cloud of {input_scene}")
#     except:
#         None
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def find_scannet_3d_file(scannet_root, scene_name):
    for split in ("train", "val", "test"):
        candidate = os.path.join(scannet_root, "scannet_3d", split, f"{scene_name}_vh_clean_2.pth")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"3D file not found for {scene_name} under {scannet_root}/scannet_3d")


def save_arrays_to_ply(input_scene, scannet_root, output_file=None):
    scene_name = input_scene.split('/')[-1]
    arrays = torch.load(find_scannet_3d_file(scannet_root, scene_name), weights_only=False)

    points = np.asarray(arrays[0], dtype=np.float32)          # (N, 3)
    colors = np.asarray(arrays[1], dtype=np.float32)          # (N, 3) in [-1,1]
    labels = np.asarray(arrays[2])                            # (N,)

    # color: [-1,1] -> [0,255] uint8
    colors01 = (colors + 1.0) * 0.5
    colors01 = np.clip(colors01, 0.0, 1.0)
    colors255 = (colors01 * 255.0 + 0.5).astype(np.uint8)

    # labels dtype 결정:
    # - 정수 라벨이면 int32 추천
    # - 현재 예시가 13., 0. 처럼 float면 일단 int로 캐스팅해도 됨(원하면 float로 유지 가능)
    labels_i = labels.astype(np.int32)

    n = points.shape[0]
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    verts = np.empty(n, dtype=vertex_dtype)

    verts['x'], verts['y'], verts['z'] = points[:, 0], points[:, 1], points[:, 2]
    verts['nx'], verts['ny'], verts['nz'] = 0.0, 0.0, 0.0
    verts['red'], verts['green'], verts['blue'] = colors255[:, 0], colors255[:, 1], colors255[:, 2]
    if output_file is None:
        output_file = f"{input_scene}/points3d.ply"
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(output_file)
    print(f"Saved point3d: {output_file}")

    labeled_verts = np.empty(n, dtype=vertex_dtype + [('label', 'i4')])
    for field in verts.dtype.names:
        labeled_verts[field] = verts[field]
    labeled_verts['label'] = labels_i

    labeled_output_file = f"{os.path.splitext(output_file)[0]}_with_label.ply"
    el = PlyElement.describe(labeled_verts, 'vertex')
    PlyData([el], text=False).write(labeled_output_file)
    print(f"Saved with label: {labeled_output_file}")

def save_pose_to_json(input_scene, output_file="output.ply"):
    """
    Saves arrays (points, colors, normals) to a .ply file using Open3D.

    Args:
        points (numpy.ndarray): Nx3 array of 3D points.
        colors (numpy.ndarray): Nx3 array of RGB colors (optional, values in [0, 1]).
        normals (numpy.ndarray): Nx3 array of normals (optional).
        output_file (str): Path to the output .ply file.
    """
    # json_file = {
    #     "w": 1296,
    #     "h": 968,
    #     "fl_x": 1170.187988,
    #     "fl_y": 1170.187988,
    #     "cx": 647.75,
    #     "cy": 483.75,
    #     "frames":[]
    # }
    json_file = {
        "w": 320,
        "h": 240,
        "fl_x": 2.889353024999999775e+02,
        "fl_y": 2.889353024999999775e+02,
        "cx": 1.595000000000000000e+02,
        "cy": 1.195000000000000000e+02,
        "frames":[]
    }

    frame_dirs = sorted(glob.glob(f"{input_scene}/color/*.jpg"))
    if len(frame_dirs) == 0:
        frame_dirs = sorted(glob.glob(f"{input_scene}/color/*.png"))
        if len(frame_dirs) == 0:
            print("there's no frames available!")
            exit()

    for frame in frame_dirs:
        frame_num = frame.split('/')[-1].split('.')[0]
        png_path = os.path.splitext(frame)[0] + ".png"

        # ensure a .png exists (gaussian-splatting Blender loader expects .png)
        if not os.path.exists(png_path):
            Image.open(frame).save(png_path)
            os.remove(frame)

        with open(os.path.join(input_scene,"pose",frame_num+".txt"), "r", encoding='utf-8') as file:
            pose = np.loadtxt(file, dtype=float)

        if np.sum(np.isnan(pose))>0 or np.sum(np.isinf(pose))>0:
            continue

        file_path = os.path.join(os.path.basename(os.path.dirname(frame)), frame_num)
        pose[:, 1] *= -1.0
        pose[:, 2] *= -1.0

        frame_dict = {
            "file_path": file_path,
            "transform_matrix": pose.tolist(),
        }

        json_file['frames'].append(frame_dict)
        json_file['camera_angle_x'] = focal2fov(json_file['fl_x'], json_file['w'])

    with open(os.path.join(input_scene,"transforms_train.json"), "w") as file:
        json.dump(json_file, file, indent=4)
    with open(os.path.join(input_scene,"transforms_test.json"), "w") as file:
        json.dump(json_file, file, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_root", type=str, default="./scannet")
    parser.add_argument("scenes", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_root = os.path.join(args.scannet_root, "scannet_2d")

    if args.scenes:
        scene_dirs = [os.path.join(scene_root, scene_name) for scene_name in args.scenes]
    else:
        scene_dirs = glob.glob(os.path.join(scene_root, "scene*"))

    for scene in tqdm.tqdm(scene_dirs):
        if not os.path.isdir(scene):
            print(f"Skip missing scene dir: {scene}")
            continue
        save_arrays_to_ply(scene, args.scannet_root)
        save_pose_to_json(scene)
        print()


if __name__ == "__main__":
    main()
