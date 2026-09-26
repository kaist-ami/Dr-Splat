
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
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_eval import render_other
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import json
from utils.opengs_utlis import get_SAM_mask_and_feat

from autoencoder.model import Autoencoder
import faiss
import open_clip

np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)

class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)
        self.device = device

    def encode_text(self, text):
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in text]).to(self.device)
            embeds = self.model.encode_text(tok_phrases)
        return embeds / embeds.norm(dim=-1, keepdim=True)

def convert_codes(codes, nbits):
    """
    Convert multiple byte codes into integers based on `nbits`.
    Args:
        codes: A PyTorch tensor of shape (N, M), where N is the number of codes, and M is the number of bytes per code.
        nbits: Number of bits to split each byte string into.
    Returns:
        A PyTorch tensor of shape (N, M * 8 // nbits) with integer representations of the split binary strings.
    """
    # Ensure codes are uint8 (for bit-level operations)
    codes = codes.to(torch.uint8)

    # Expand each byte into 8 individual bits
    bits = torch.stack([(codes >> i) & 1 for i in range(7, -1, -1)], dim=-1)  # Shape: (N, M, 8)

    # Reshape bits into chunks of `nbits`
    bits_flat = bits.reshape(codes.size(0), -1)  # Flatten to (N, M * 8)
    split_size = bits_flat.size(1) // nbits  # Number of resulting integers per code row
    bits_split = bits_flat.unfold(1, nbits, nbits)  # Create rolling windows of size `nbits`

    # Convert each chunk of `nbits` into an integer
    powers_of_two = 2 ** torch.arange(nbits - 1, -1, -1, device=codes.device, dtype=torch.int32)
    result = (bits_split * powers_of_two).sum(dim=-1)  # Shape: (N, M * 8 // nbits)

    return result

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_name, model_params, args):

    if len(model_params) == 13 :
        (active_sh_degree,
        _xyz,
        _features_dc,
        _features_rest,
        _scaling,
        _rotation,
        _opacity,
        _language_feature,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale) = model_params
    elif len(model_params) == 12: # 这是一个不训练feature保存的ckpt
        (active_sh_degree,
        _xyz,
        _features_dc,
        _features_rest,
        _scaling,
        _rotation,
        _opacity,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale) = model_params

    # FIXME: decoder for making it back into 512-dim feature
    if args.use_3dim:
        device = torch.device("cuda")
        checkpoint = torch.load(args.ae_ckpt_path, map_location=device)
        model = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        channel = 3
        with torch.no_grad():
            _language_feature_clip = model.decode(_language_feature)

    if args.use_pq:
        index = faiss.read_index(args.pq_index)
        zero_mask = torch.all(_language_feature == -1, dim=-1)
        index.add_sa_codes(_language_feature[~zero_mask].cpu().numpy())

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)

    # text feature
    with open('assets/text_features.json', 'r') as f:
        data_loaded = json.load(f)

    negative_text_features = np.load("assets/text_negative.npy")
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]

    threshold = args.threshold

    scene_texts = {
        "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
                'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
                'pour-over vessel', 'knife', 'yellow desk'],
        "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
        "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
        "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple',
                'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                'bag of cookies']
    }

    # note: query text
    target_text = scene_texts[scene_name]
    query_text_feats = torch.zeros(len(target_text), 512).cuda()

    for i, text in enumerate(target_text):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats[i] = feat

    for t_i, text_feat in enumerate(query_text_feats):
        print(f"rendering the {t_i+1}-th query of {len(target_text)} texts: {target_text[t_i]}")
        text_feat = F.normalize(text_feat.unsqueeze(0), dim=1, p=2)

        if args.use_3dim:
            leaf_lang_feat = F.normalize(_language_feature_clip, dim=1, p=2)
            cosine_similarity = torch.matmul(text_feat, leaf_lang_feat.transpose(0, 1))
            # threshold = torch.quantile(cosine_similarity, 0.99, interpolation='nearest')
            mask = (cosine_similarity > threshold)[0]

        elif args.use_pq:
            # import pdb; pdb.set_trace()
             # FIXME: only activation with reconstruction
            # leaf_lang_feat = index.sa_decode(_language_feature[~zero_mask].cpu().numpy())
            # cosine_similarity = torch.matmul(text_feat, torch.from_numpy(leaf_lang_feat).to("cuda").transpose(0, 1))
            # threshold = torch.quantile(cosine_similarity, 0.99, interpolation='nearest')
            # mask = torch.zeros(_language_feature.shape[0], dtype=torch.bool, device="cuda")
            # mask[~zero_mask] = (cosine_similarity >= threshold).squeeze(-1)
            # mask = mask.to("cuda")
            # print(torch.sum(1*mask))

             # FIXME: only activation with search
            # dist, ids = index.search(text_feat.cpu().numpy(), index.ntotal)

            # sorted_indices = np.argsort(ids, axis=1)

            # sorted_ids = np.take_along_axis(ids, sorted_indices, axis=1)
            # sorted_dist = np.take_along_axis(dist, sorted_indices, axis=1)

            # positive_vals = torch.from_numpy(sorted_dist).transpose(1,0).to("cuda")

            # threshold = positive_vals.max() * 0.9
            # mask = torch.zeros(_language_feature.shape[0], dtype=torch.bool, device="cuda")
            # mask[~zero_mask] = (positive_vals >= threshold).squeeze(-1)
            # mask = mask.to("cuda")
            # print(torch.sum(1*mask))

            # # FIXME: lerf style
            # import pdb; pdb.set_trace()
            # precomputed_table = torch.load("/home/kjs/project/iclr25/lerf/precomputed_table_64_8.pt")
            # concat text_feature with negative_text_features
            index.nprobe = index.nlist
            text_feature = torch.cat([text_feat, torch.from_numpy(negative_text_features).to("cuda")], dim=0)
            text_feature_code = index.sa_encode(text_feature.cpu().numpy())

            testset_1_enc = torch.tensor(_language_feature[~zero_mask].cpu().numpy(), dtype=torch.int64, device='cuda')  # (500000, pq_m)
            testset_2_enc = torch.tensor(text_feature_code, dtype=torch.int64, device='cuda')  # (500000, pq_m)

            testset_1_enc = convert_codes(testset_1_enc, index.pq.nbits)
            testset_2_enc = convert_codes(testset_2_enc, index.pq.nbits)

            # inner_products = precomputed_table[
            #     torch.arange(index.pq.M, device='cuda').view(-1, 1),
            #     testset_1_enc.T,
            #     testset_2_enc.T,
            # ]
            # norms = precomputed_table[
            #     torch.arange(index.pq.M, device='cuda').view(-1, 1),
            #     testset_1_enc.T,
            #     testset_1_enc.T,
            # ]

            # norms_target = precomputed_table[
            #     torch.arange(index.pq.M, device='cuda').view(-1, 1),
            #     testset_2_enc.T,
            #     testset_2_enc.T,
            # ]
            # cos_sim_approx = inner_products.sum(dim=0)
            # norm_approx = norms.sum(dim=0)
            # norm_approx_target = norms_target.sum(dim=0)

            # cos_sim_approx = cos_sim_approx / (torch.sqrt(norm_approx) * torch.sqrt(norm_approx_target))

            # positive_vals = cos_sim_approx[:text_feat.shape[0]]
            # negative_vals = cos_sim_approx[text_feat.shape[0]:]
            # repeated_pos = positive_vals.repeat(1, len(negative_text_features))

            # sims = torch.stack((repeated_pos, negative_vals), dim=-1)
            # softmax = torch.softmax(10 * sims, dim=-1)

            # best_id = softmax[..., 0].argmin(dim=1)
            # score = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negative_text_features), 2))[:, 0, 0]

            # # threshold = torch.quantile(score, 0.99, interpolation='nearest')
            # mask = torch.zeros(_language_feature.shape[0], dtype=torch.bool, device="cuda")
            # mask[~zero_mask] = (score >= args.threshold)
            # print(torch.sum(1*mask))

            dist, ids = index.search(text_feat.cpu().numpy(), index.ntotal)
            dist_negative, ids_negative = index.search(negative_text_features, index.ntotal)
            sorted_indices = np.argsort(ids, axis=1)
            sorted_indices_negative = np.argsort(ids_negative, axis=1)

            sorted_ids_negative = np.take_along_axis(ids_negative, sorted_indices_negative, axis=1)
            sorted_dist_negative = np.take_along_axis(dist_negative, sorted_indices_negative, axis=1)

            sorted_ids = np.take_along_axis(ids, sorted_indices, axis=1)
            sorted_dist = np.take_along_axis(dist, sorted_indices, axis=1)

            positive_vals = torch.from_numpy(sorted_dist).transpose(1,0).to("cuda")
            negative_vals = torch.from_numpy(sorted_dist_negative).transpose(1,0).to("cuda")
            repeated_pos = positive_vals.repeat(1, len(negative_text_features))

            sims = torch.stack((repeated_pos, negative_vals), dim=-1)
            softmax = torch.softmax(10 * sims, dim=-1)

            best_id = softmax[..., 0].argmin(dim=1)
            score = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negative_text_features), 2))[:, 0, 0]

            # threshold = torch.quantile(score, 0.99, interpolation='nearest')
            mask = torch.zeros(_language_feature.shape[0], dtype=torch.bool, device="cuda")
            mask[~zero_mask] = (score >= threshold)
            print(torch.sum(1*mask))

        # render
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # note: evaluation frame
            scene_gt_frames = {
                "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
                "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
                "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
                "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
            }
            candidate_frames = scene_gt_frames[scene_name]

            if  view.image_name not in candidate_frames:
                continue
            # import pdb; pdb.set_trace()
            render_pkg = render_other(view, gaussians, pipeline, background, iteration, rescale=False)
            # RGB
            rendering = render_pkg["render"]
            gt = view.original_image[0:3, :, :]

            # ins_feat
            # rendered_ins_feat = render_pkg["ins_feat"]
            # import pdb; pdb.set_trace()
            gt_sam_mask = view.original_sam_mask.cuda()    # [4, H, W]

            # RGB
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            # NOTE get SAM id, mask bool, mask_feat, invalid pix
            mask_id, mask_bool, mask_feat, invalid_pix = \
                get_SAM_mask_and_feat(gt_sam_mask, level=3, original_mask_feat=view.original_mask_feat)

            # sam mask
            mask_color_rand = colors_defined[mask_id.detach().cpu().type(torch.int64)].type(torch.float64)
            mask_color_rand = mask_color_rand.permute(2, 0, 1)
            torchvision.utils.save_image(mask_color_rand/255.0, os.path.join(gt_sam_mask_path, '{0:05d}'.format(idx) + ".png"))

            try:
                # render target object
                render_pkg = render_other(view, gaussians, pipeline, background, iteration,
                                    rescale=False,                #)  # wherther to re-scale the gaussian scale
                                    filter_idx=mask,
                                    render_feat_map=False,
                                    render_cluster=False,
                                    better_vis=True,
                                    seg_rgb=True,
                                    post_process=True,)
                rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
                rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]

                render_cluster_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster")
                render_cluster_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster_silhouette")
                makedirs(render_cluster_path, exist_ok=True)
                makedirs(render_cluster_silhouette_path, exist_ok=True)
                for i, img in enumerate(rendered_cluster_imgs):
                    # save object RGB
                    torchvision.utils.save_image(img[:3,:,:], os.path.join(render_cluster_path, \
                        view.image_name + f"_{target_text[t_i]}.png"))
                    # save object mask
                    cluster_silhouette = rendered_leaf_cluster_silhouettes[i] > 0.7
                    torchvision.utils.save_image(cluster_silhouette.to(torch.float32), os.path.join(render_cluster_silhouette_path, \
                        view.image_name + f"_{target_text[t_i]}.png"))
            except:
                continue

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                scene_name: str, args : None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # from plyfile import PlyData, PlyElement
        # checkpoint = os.path.join(args.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply")
        # plydata = PlyData.read(checkpoint)
        # if args.load_ply:
        #     import pdb; pdb.set_trace()
        #     checkpoint = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        #     if not os.path.exists(checkpoint):
        #         print("Model file not found")
        #         exit()
        #     name = os.path.join("point_cloud", f"iteration_{iteration}")
        #     gaussians.load_ply(checkpoint)
        # else :
        checkpoint = os.path.join(args.model_path, f'chkpnt{str(args.checkpoint_iter)}.pth')
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)

        gaussians.restore(model_params, args, mode='test')
        # name = "train" if not skip_train else "test"

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "text2obj", scene.loaded_iter, scene.getTrainCameras(),
                        gaussians, pipeline, background, scene_name, model_params, args)
        if not skip_test:
             render_set(dataset.model_path, "text2obj", scene.loaded_iter, scene.getTestCameras(),
                        gaussians, pipeline, background, scene_name, model_params, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--checkpoint_iter", default=30000, type=int)

    # This is for 3dim case
    parser.add_argument("--use_3dim", action="store_true")
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
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
    # This is for pq case
    parser.add_argument("--use_pq", action="store_true")
    parser.add_argument("--pq_index", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.use_3dim:
        args.dataset_name = args.source_path.split("/")[-1]
        args.ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.dataset_name, "best_ckpt.pth")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scene_name, args)
