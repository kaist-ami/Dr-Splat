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

import os
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
import faiss
import gc

from autoencoder.model import Autoencoder


def update_voting_mat(result_dict, language_feature_mask, gt_language_feature, contribution, ids, args):
    # Select only locations where Mask is True
    mask_idx = language_feature_mask.squeeze(0).nonzero(as_tuple=True)
    
    # Get the ID and contributions of the gaussians who contributed from that location
    contrib = contribution[mask_idx]  # shape: [N, 100]
    ray_ids = ids[mask_idx]  # shape: [N, 100]
    gt_feats = gt_language_feature[:, mask_idx[0], mask_idx[1]]  # shape: [3, N]
    
    _, indices = torch.topk(contrib, args.topk, dim=1)
    ray_ids = torch.gather(ray_ids, 1, indices)
    
    # Filter only valid contributions (non-1 IDs and non-0 contributions)
    valid_mask = (ray_ids != -1)
    ray_ids = ray_ids[valid_mask].view(-1)  # shape: [M] (valid Gaussian ID)
    gt_feats = gt_feats.T.unsqueeze(1).repeat(1, args.topk, 1)[valid_mask]  # shape: [M, 3]

    unique_ids = torch.unique(ray_ids)
    
    for uid in unique_ids:
        mask = ray_ids == uid
        if uid.item() not in result_dict:
            result_dict[uid.item()] = [gt_feats[mask]]
        else:
            result_dict[uid.item()].append(gt_feats[mask])

    return result_dict


def compute_average(features):
    averaged_tensor = features.mean(dim=0).unsqueeze(0)  # 평균 계산
    averaged_tensor = averaged_tensor / (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
    return averaged_tensor


def majority_voting(gaussians, scene, pipe, background, dataset, args):
    lf_path = "/" + os.path.join(*dataset.lf_path.split('/')[:-1], "language_features")
    if args.use_pq:
        voting_mat = -1 * torch.ones((gaussians._opacity.shape[0], 17), dtype=torch.uint8, device="cuda")
    else:
        voting_mat = -1 * torch.zeros((gaussians._opacity.shape[0], 3), dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    
    from collections import defaultdict
    result_dict = defaultdict(list)
    
    #### code edit ####
    num_masks_array = torch.zeros(len(viewpoint_stack), dtype=torch.int)
    for i in range(len(viewpoint_stack)):
        language_feature_name = os.path.join(lf_path, viewpoint_stack[i].image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        num_masks_array[i] = feature_map.shape[0]

    num_masks = torch.sum(num_masks_array)
    num_gaussians = len(gaussians.get_opacity)
    features_array = torch.zeros((num_masks,512))
    allocate_array = torch.zeros((num_gaussians, num_masks), dtype=torch.float32)
    offset = 0
    for i in tqdm(range(len(viewpoint_stack))):
        viewpoint_cam = viewpoint_stack[i]
        language_feature_name = os.path.join(lf_path, viewpoint_cam.image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        features_array[offset:offset+num_masks_array[i]] = feature_map
        
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        ids, contribution = (
            render_pkg['per_pixel_gaussian_ids'].detach(),
            render_pkg['per_pixel_gaussian_contributions'].detach(),
        )
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).type(torch.int64)[dataset.feature_level].unsqueeze(0)
        seg_map_bool = seg_map != -1
        seg_map += offset

        mask_idx = seg_map_bool.squeeze(0).nonzero(as_tuple=True)
        
        # Get the ID and contributions of the gaussians who contributed from that location
        contrib = contribution[mask_idx]  # shape: [N, 100]
        ray_ids = ids[mask_idx]  # shape: [N, 100]

        gt_segmentations = seg_map[0, mask_idx[0], mask_idx[1]]
        gt_segmentations = gt_segmentations.repeat(args.topk,1).T.reshape(-1)

        weights, indices = torch.topk(contrib, args.topk, dim=1)
        ray_ids = torch.gather(ray_ids, 1, indices)

        weights = weights.reshape(-1)
        ray_ids = ray_ids.reshape(-1)
        valid_mask = (ray_ids != -1)
        ray_ids = ray_ids[valid_mask]
        weights = weights[valid_mask]
        gt_segmentations = gt_segmentations[valid_mask.cpu()]

        weights = weights.cpu()
        ray_ids = ray_ids.cpu().type(torch.int64)
        valid_mask = valid_mask.cpu()
        weight_sum = torch.zeros(num_gaussians)
        allocate_array.index_put_((ray_ids, gt_segmentations), weights, accumulate=True)

        offset += num_masks_array[i]


    features_array /= (features_array.norm(dim=-1, keepdim=True) + 1e-9)

    if args.use_pq:
        index = faiss.read_index(args.pq_index)

        weight_sum = torch.sum(allocate_array, 1)
        threshold = 1e-4
        weight_sum_over_zero = weight_sum>0
        weight_sum_under_threshold = weight_sum<threshold
        reweight_index = weight_sum_over_zero * weight_sum_under_threshold
        allocate_array[reweight_index][allocate_array[reweight_index]>0] = 1


        averaged_tensor = torch.matmul(allocate_array.type(torch.float32) ,features_array)
        averaged_tensor /= (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
        invalid_gaussians = torch.sum(averaged_tensor,1) == 0


        if args.faiss_add: index.add(averaged_tensor.cpu().numpy())
        averaged_tensor = index.sa_encode(averaged_tensor.cpu().numpy())
        averaged_tensor = torch.ByteTensor(averaged_tensor).to("cuda")
        averaged_tensor[invalid_gaussians,:] = -1
        

    return averaged_tensor


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    maj_feat = majority_voting(gaussians, scene, pipe, background, dataset, args)
    
    gaussians._language_feature = maj_feat
    
    iteration = 0

    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)

    if (iteration in checkpoint_iterations):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--name_extra", type=str, default = None)
    parser.add_argument("--mode", type=str, default = "mean")
    parser.add_argument("--topk", type=int, default = 1)
    
    parser.add_argument("--use_pq", action="store_true")
    parser.add_argument("--pq_index", type=str, default=None)
    
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--faiss_add", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    index = faiss.read_index(args.pq_index)


    try:
        args.modelpath = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.coarsecode_size()+index.code_size}"
    except :
        args.model_path = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.code_size}"

    if args.use_pq:
        if args.pq_index is None:
            raise ValueError("PQ index file is not provided.")
        lp._language_features_name = "language_features_pq"

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")

