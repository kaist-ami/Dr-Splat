
import torch
from scene import Scene
import os
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"] = "24"
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from plyfile import PlyData

from torch.distributions import MultivariateNormal
import numpy as np

color_map = torch.Tensor([
    [ 0,  0, 0],
    [ 14,  68,  63],
    [107,  90, 246],
    [237,  84, 183],
    [221, 101, 102],
    [156, 109, 108],
    [177, 243, 139],
    [230,  44,  12],
    [198, 214,  75],
    [224, 218, 245],
    [108, 144, 125],
    [ 68,  36, 106],
    [215, 113,  47],
    [249, 244, 150],
    [157, 210,  87],
    [191, 159, 158],
    [176, 241, 103],
    [125,  76,   5],
    [ 68, 165,  15],
    [ 96, 191, 227],
    [ 64, 198,  25],
    [155,   5, 212],
    [ 92,  92,  52],
    [177,  98,  65],
    [225, 170,  36],
    [199,  42,  45],
    [232,  92, 106],
    [ 67, 217, 110],
    [ 37,  50, 209],
    [ 25, 146,  70],
    [ 46,  45,  45],
    [156, 217,  14],
    [212, 209, 168],
    [ 27, 175,  35],
    [  3, 223,  47],
    [171,  39, 150],
    [ 47,  99,  94],
    [  4,   0, 124],
    [ 96, 168, 201],
    [ 23, 166, 117],
    [ 66,  23, 108]
]).type(torch.uint8)


nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

# ScanNet 20 classes
scannet19_dict = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 14: "desk", 16: "curtain",
    24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub", # 39: "otherfurniture"
}
def custom_normal(x, cov, points):
    inv_cov = torch.linalg.inv(cov)
    denom = torch.sqrt((2*torch.pi)**3 * torch.linalg.det(cov))
    for i in range(len(points)):
        p = points[i]
        alpha = torch.exp((-1/2) * (((x-p).unsqueeze(1) @ inv_cov) @ (x-p).unsqueeze(-1)).squeeze()) / denom

    return alpha

def cov6d_to_3x3(cov3D):
    """
    Convert a batch of 6D covariance vectors to (3, 3) covariance matrices.

    Parameters:
        cov3D (torch.Tensor): Tensor of shape (N, 6), where each row represents
                              the upper triangular elements of a 3x3 covariance matrix.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) representing the full 3x3 covariance matrices.
    """
    N = cov3D.shape[0]

    # Initialize an empty tensor for the (3,3) covariance matrices
    cov_matrices = torch.zeros((N, 3, 3), dtype=cov3D.dtype, device=cov3D.device)

    # Populate the upper triangular and diagonal elements
    cov_matrices[:, 0, 0] = cov3D[:, 0]  # σ11
    cov_matrices[:, 0, 1] = cov3D[:, 1]  # σ12
    cov_matrices[:, 0, 2] = cov3D[:, 2]  # σ13
    cov_matrices[:, 1, 1] = cov3D[:, 3]  # σ22
    cov_matrices[:, 1, 2] = cov3D[:, 4]  # σ23
    cov_matrices[:, 2, 2] = cov3D[:, 5]  # σ33

    # Use symmetry to fill in the lower triangular elements
    cov_matrices[:, 1, 0] = cov_matrices[:, 0, 1]  # σ21 = σ12
    cov_matrices[:, 2, 0] = cov_matrices[:, 0, 2]  # σ31 = σ13
    cov_matrices[:, 2, 1] = cov_matrices[:, 1, 2]  # σ32 = σ23

    return cov_matrices

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

def compute_cov_3d(scale, rotation, mod=1):
    """
    Compute the 3D covariance matrices for a batch of scales and rotations.

    Parameters:
        scale (torch.Tensor): Tensor of shape (N, 3) containing scaling factors for each batch.
        mod (float): Modulation factor (applied uniformly across the batch).
        rotation (torch.Tensor): Tensor of shape (N, 4) containing quaternion rotations for each batch.

    Returns:
        torch.Tensor: Tensor of shape (N, 6) containing the upper triangular elements of the covariance matrix for each batch.
    """
    N = scale.shape[0]

    # Create scaling matrices (N, 3, 3)
    S = torch.zeros((N, 3, 3), dtype=scale.dtype, device=scale.device)
    S[:, 0, 0] = mod * scale[:, 0]
    S[:, 1, 1] = mod * scale[:, 1]
    S[:, 2, 2] = mod * scale[:, 2]

    # Unpack the quaternion rotation components
    r, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    # Compute rotation matrices R (N, 3, 3) from quaternion
    R = torch.zeros((N, 3, 3), dtype=rotation.dtype, device=rotation.device)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    # Compute M = S * R (batch matrix multiplication)
    M = torch.bmm(S, R)

    # Compute covariance matrix Sigma = M.T * M (batch matrix multiplication)
    Sigma = torch.bmm(M.transpose(1, 2), M)

    # Extract the upper triangular elements of Sigma for each batch
    cov3D = torch.zeros((N, 6), dtype=Sigma.dtype, device=Sigma.device)
    cov3D[:, 0] = Sigma[:, 0, 0]  # Sigma[0, 0]
    cov3D[:, 1] = Sigma[:, 0, 1]  # Sigma[0, 1]
    cov3D[:, 2] = Sigma[:, 0, 2]  # Sigma[0, 2]
    cov3D[:, 3] = Sigma[:, 1, 1]  # Sigma[1, 1]
    cov3D[:, 4] = Sigma[:, 1, 2]  # Sigma[1, 2]
    cov3D[:, 5] = Sigma[:, 2, 2]  # Sigma[2, 2]

    return cov3D, Sigma

import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Extract the coordinates and labels of the points. The labels are from 1 to 40 for the NYU40 dataset, with 0 being invalid.
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels


if __name__ == "__main__":

    output_dir = 'labeled_gaussians'

    iteration = 30000

    model_miou = 0
    model_macc = 0


    eps = 0.0001

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)

    # args.model_path = os.path.join('output', scan_name)
    # args.model_path = os.path.join(gaussian_scene_dir, scan_name)
    # args.source_path = os.path.join(scannet_dir, scan_name)
    scan_name = args.model_path.split('/')[-1]
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(model, gaussians, shuffle=False)
    checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    # gaussians.restore_eval(model_params, args)
    # import pdb; pdb.set_trace()
    # gaussians.restore_eval(model_params)
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)

    # load gt labels
    gt_file_path = f"{args.source_path}/points3d_with_label.ply"
    points, labels = read_labels_from_ply(gt_file_path)
    points = torch.Tensor(points)
    points = points.type(torch.float64)
    labels = torch.tensor(labels.astype(np.int32))

    eps = 1e-6

    gs_positions = gaussians.get_xyz.clone().detach().cpu()
    gs_scales = gaussians.get_scaling.clone().detach().cpu()
    gs_rotation = gaussians.get_rotation.clone().detach().cpu()

    gs_positions = gs_positions.type(torch.float64)
    gs_scales = gs_scales.type(torch.float64)
    gs_rotation = gs_rotation.type(torch.float64)

    gs_scales += eps
    _, cov = compute_cov_3d(gs_scales, gs_rotation)
    inv_cov = torch.linalg.inv(cov)

    valid_gaussians = torch.linalg.cholesky_ex(cov).info.eq(0)

    denom = torch.sqrt((2*torch.pi)**3 * torch.linalg.det(cov[valid_gaussians]))
    allocate = torch.zeros((len(gs_positions), torch.max(labels)+1))
    alpha = torch.zeros(len(gs_positions), dtype=torch.float64)
    for i in tqdm(range(len(points))):
        x = gs_positions[valid_gaussians]
        p = points[i]
        label = labels[i]
        if label == 255:
            label = -1
        alpha = torch.exp((-1/2) * (((x-p).unsqueeze(1) @ inv_cov[valid_gaussians]) @ (x-p).unsqueeze(-1)).squeeze())
        allocate[valid_gaussians,label+1] += alpha[valid_gaussians]
        alpha[:] = 0

    gaussain_labels_valid = torch.argmax(allocate[valid_gaussians], dim=1)
    gaussian_labels = torch.zeros(len(gs_positions), dtype=torch.long)
    gaussian_labels[valid_gaussians] = gaussain_labels_valid

    gaussian_color_map = color_map[gaussian_labels]/255
    gaussians._features_dc = gaussians._features_dc.detach()
    gaussians._features_rest = gaussians._features_rest.detach()
    gaussians._opacity = gaussians._opacity.detach()
    gaussians._opacity[~valid_gaussians] = -10

    offset = 0.5
    SH_C0 = 0.28209479177387814
    gaussians._features_dc[:,0,:] = (gaussian_color_map - offset) / SH_C0
    gaussians._features_rest[:,:,:] = 0


    save_path = os.path.join(output_dir, scan_name)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'labels.npy'), gaussian_labels.numpy())
    # point_cloud_save_path = os.path.join(output_dir, scan_name, 'viewer')
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, 'point_cloud.ply'))
