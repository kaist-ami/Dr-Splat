#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    # orig_w, orig_h = cam_info.image.size
    orig_w, orig_h = cam_info.width, cam_info.height
    image = Image.open(cam_info.image_path)

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # TODO 变更条件
            if orig_h > 1080:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)
    scale_x = resolution[0] / orig_w
    scale_y = resolution[1] / orig_h

    gt_sam_mask = None
    if getattr(cam_info, "sam_mask", None) is not None:
        step = int(max(args.resolution, 1))
        gt_sam_mask = torch.from_numpy(cam_info.sam_mask[:, ::step, ::step])
        if resized_image_rgb.shape[1] != gt_sam_mask.shape[1] or resized_image_rgb.shape[2] != gt_sam_mask.shape[2]:
            resolution = (gt_sam_mask.shape[2], gt_sam_mask.shape[1])
            resized_image_rgb = PILtoTorch(image, resolution)
            scale_x = resolution[0] / orig_w
            scale_y = resolution[1] / orig_h

    mask_feat = None
    if getattr(cam_info, "mask_feat", None) is not None:
        mask_feat = torch.from_numpy(cam_info.mask_feat)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  cx=getattr(cam_info, "cx", orig_w / 2) * scale_x,
                  cy=getattr(cam_info, "cy", orig_h / 2) * scale_y,
                  depth=getattr(cam_info, "depth", None),
                  gt_sam_mask=gt_sam_mask, gt_mask_feat=mask_feat,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
