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

import torch
import math
from ashawkey_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.opengs_utlis import *

def render_other(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, iteration,
            scaling_modifier = 1.0, override_color = None, visible_mask = None, mask_num=0,
            cluster_idx=None,       # per-point cluster id (coarse-level)
            leaf_cluster_idx=None,  # per-point cluster id (fine-level)
            rescale=True,           # re-scale (for enhance ins_feat)
            origin_feat=False,      # origin ins_feat (not quantized)
            render_feat_map=True,   # render image-level feat map
            render_color=True,      # render rgb image
            render_cluster=False,   # render cluster, stage 2.2
            better_vis=False,       # filter some points
            selected_root_id=None,  # coarse-level cluster id
            selected_leaf_id=None,  # fine-level cluster id (possibly more than one)
            pre_mask=None,
            seg_rgb=True,          # render cluster rgb, not feat
            post_process=False,     # post
            filter_idx=None,
            root_num=64, leaf_num=10):  # k1, k2
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if render_color:
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = None, None, None, None

    rendered_ins_feat, silhouette = None, None
    cluster_occur = None
    rendered_clusters, rendered_cluster_silhouettes = None, None

    _colors_precomp1 = None
    _colors_precomp2 = None
    rendered_leaf_clusters = []
    rendered_leaf_cluster_silhouettes = []

    if filter_idx == None:
        rendered_leaf_cluster, _, _, leaf_cluster_silhouette = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,                          # rgb or feat
            colors_precomp = _colors_precomp1,   # rgb or feat
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_leaf_cluster, _, _, leaf_cluster_silhouette = rasterizer(
            means3D = means3D[filter_idx],
            means2D = means2D[filter_idx],
            shs = shs[filter_idx],                          # rgb or feat
            colors_precomp = _colors_precomp1,   # rgb or feat
            opacities = opacity[filter_idx],
            scales = (scales)[filter_idx],
            rotations = rotations[filter_idx],
            cov3D_precomp = cov3D_precomp)

    rendered_leaf_clusters.append(rendered_leaf_cluster)
    rendered_leaf_cluster_silhouettes.append(leaf_cluster_silhouette)

    # rendered_leaf_clusters = None
    # rendered_leaf_cluster_silhouettes =  None
    occured_leaf_id = None

    return {"render": rendered_image,
            "alpha": rendered_alpha,
            "depth": rendered_depth,    # not used
            "silhouette": silhouette,
            "ins_feat": rendered_ins_feat,          # image-level feat map
            "cluster_imgs": rendered_clusters,      # coarse cluster feat map/image
            "cluster_silhouettes": rendered_cluster_silhouettes,    # coarse cluster mask
            "leaf_clusters_imgs": rendered_leaf_clusters,           # fine cluster feat map/image
            "leaf_cluster_silhouettes": rendered_leaf_cluster_silhouettes,  # fine cluster mask
            "occured_leaf_id": occured_leaf_id,     # fine cluster
            "cluster_occur": cluster_occur,         # coarse cluster
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
