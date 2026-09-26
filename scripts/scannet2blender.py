import os
import json
import numpy as np

def load_transform_matrix(file_path):
    """
    Load the transform matrix from a text file.
    """
    return np.loadtxt(file_path)

def process_directory(directory_path):
    """
    Process each directory and create a JSON file with the transform matrices.
    """
    color_dir = os.path.join(directory_path, "color")
    pose_dir = os.path.join(directory_path, "pose")
    intrinsic_dir = os.path.join(directory_path, "intrinsic")
    has_intrinsics = os.path.isdir(intrinsic_dir)

    # Check if both directories exist
    if not os.path.isdir(color_dir) or not os.path.isdir(pose_dir):
        return

    # scannet
    # transform_data = {
    #         'w': 1296,
    #         'h': 968,
    #         'fl_x': 1170.187988,
    #         'fl_y': 1170.187988,
    #         'cx': 647.75,
    #         'cy': 483.75,
    #         # 'aabb_scale': 2,
    #         'frames': [],
    #     }
    # # scannet
    transform_data = {
            "w": 320,
            "h": 240,
            "fl_x": 2.889353024999999775e+02,
            "fl_y": 2.889353024999999775e+02,
            "cx": 1.595000000000000000e+02,
            "cy": 1.195000000000000000e+02,
            # 'aabb_scale': 2,
            'frames': [],
        }
    # Collect all image names and sort them
    img_names = [entry.name for entry in os.scandir(color_dir) if entry.is_file() and entry.name.endswith(".jpg")]
    img_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Iterate over the color images
    for img_name in img_names:
        pose_file_path = os.path.join(pose_dir, f"{os.path.splitext(img_name)[0]}.txt")

        if not os.path.isfile(pose_file_path):
            continue

        transform_matrix = load_transform_matrix(pose_file_path)

        # note: colmap --> blender
        transform_matrix[:3, 1:3] *= -1
        transform_matrix = transform_matrix.tolist()

        frame_data = {
            "file_path": os.path.join("color", os.path.splitext(img_name)[0]),
            "transform_matrix": transform_matrix
        }

        if has_intrinsics:
            intrinsic_file_path = os.path.join(intrinsic_dir, f"{os.path.splitext(img_name)[0]}.txt")
            if os.path.isfile(intrinsic_file_path):
                intrinsic_info = load_transform_matrix(intrinsic_file_path)
                frame_data.update({
                    'fl_x': float(intrinsic_info[0][0]),
                    'fl_y': float(intrinsic_info[1][1]),
                    'cx':  float(intrinsic_info[0][2]),
                    'cy': float(intrinsic_info[1][2])
                })

        transform_data["frames"].append(frame_data)

    return transform_data

# Directory containing the scenes
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scannet", "scannet_2d"))

# Process each scene directory and create JSON files
for scene_dir in os.listdir(base_directory):
    # if scene_dir != "scene0000_00":
    #     continue

    scene_path = os.path.join(base_directory, scene_dir)
    if os.path.isdir(scene_path):
        # Process the directory and get the transform data
        transform_data = process_directory(scene_path)

        print(scene_path)

        # Create the JSON file
        if transform_data:
            json_file_path = os.path.join(scene_path, "transforms_train.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(transform_data, json_file, indent=4)
