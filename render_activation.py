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
import numpy as np
import torch
from scene import Scene
import os

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path

from evaluation.openclip_encoder import OpenCLIPNetwork

import time

import faiss

from evaluation import colormaps

import cv2

COLORMAP_OPTIONS = colormaps.ColormapOptions(
    colormap="turbo",
    normalize=True,
    colormap_min=-1.0,
    colormap_max=1.0,
)

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, label, clip_model, img_label):
    render_path = os.path.join(model_path, name, f"renders_colormap_{img_label}")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        rendering = output["render"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))


def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, label, clip_model, index, img_save_label):
    with torch.no_grad():
        start_time = time.time()
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')

        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        features = gaussians._language_feature.clone()
        zero_mask = torch.all(features == -1, dim=-1)

        leaf_lang_feat = torch.from_numpy(index.sa_decode(features[~zero_mask].cpu().numpy())).to("cuda")
        activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
        _activation_features = clip_model.get_activation(leaf_lang_feat, label)
        activation_features[~zero_mask] = _activation_features

        thr = args.threshold
        
        activation_threshold = torch.where(activation_features.squeeze() > thr)[0]

        features_colormap = colormaps.apply_colormap(activation_features, colormap_options=COLORMAP_OPTIONS)
        features_colormap = (features_colormap.unsqueeze(1) - 0.5) / 0.28209479177387814
        gaussians._features_dc[activation_threshold] = features_colormap[activation_threshold]
        gaussians._features_rest[activation_threshold] = torch.zeros_like(gaussians._features_rest)[activation_threshold].cuda()


        end_time = time.time()
        print(f'Running time : {end_time - start_time}')

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, label, clip_model, img_save_label)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, label, clip_model, img_save_label)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--semantic_model", default='dino', type=str)
    parser.add_argument("--pq_index", type=str, default=None)
    parser.add_argument("--img_save_label", type=str, default=None)
    parser.add_argument("--img_label", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    
    img_labels = [args.img_label]

    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(img_labels)


    index = faiss.read_index(args.pq_index)

    negative_text_features = torch.from_numpy(np.load('assets/text_negative.npy')).to(torch.float32)  # [num_text, 512]

    for label in range(len(img_labels)):
        text_feat = clip_model.encode_text(img_labels[label], device=device).float()
        render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test, args, label, clip_model, index, args.img_save_label)