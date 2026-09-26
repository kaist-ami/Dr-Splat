import os
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import numpy as np
import torch
import json

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
from autoencoder.model import Autoencoder
import faiss
import sys

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

scannet_19_to_nyu = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36], dtype=torch.int32)

import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def write_ply(vertex_data, output_path):
    vertices = []
    for vertex in vertex_data:
        r = (vertex['ins_feat_r'] + 1)/2 * 255
        g = (vertex['ins_feat_g'] + 1)/2 * 255
        b = (vertex['ins_feat_b'] + 1)/2 * 255
        new_vertex = (vertex['x'], vertex['y'], vertex['z'], r, g, b)
        vertices.append(new_vertex)

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.array(vertices, dtype=vertex_dtype)

    el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([el], text=True).write(output_path)

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Extract the coordinates and labels of the points. The labels are from 1 to 40 for the NYU40 dataset, with 0 being invalid.
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

def calculate_metrics(gt, pred, total_classes):
    gt = gt.cpu()
    pred = pred.cpu()
    pred[gt == 0] = 0

    ious = torch.zeros(total_classes)

    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
        correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        total[cls] = torch.sum(gt == cls).item()

    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    # Only consider the categories that exist in the current scene
    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]  # ignore 0

    # miou
    mean_iou = ious[valid_gt_classes].mean().item()

    # acc
    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else float('nan')

    class_accuracy = correct / total
    # mAcc.
    mean_class_accuracy = class_accuracy[valid_gt_classes].mean().item()

    return ious, mean_iou, accuracy, mean_class_accuracy

def calculate_localization(gt, similarity, weight, total_classes, invalids):
    gt = gt.cpu()
    similarity = similarity.cpu()
    weight = weight.cpu()
    # pred_bool = torch.zeros(similarity.shape, dtype=bool)
    pred_bool = similarity
    gt_bool = torch.zeros(similarity.shape, dtype=bool)
    gt_bool[range(gt_bool.shape[0]),gt] = True
    # pred[gt == 0] = 0


    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)

    intersection = pred_bool * gt_bool
    union = pred_bool + gt_bool
    # weight=torch.ones_like(weight)
    if weight==None:
        ious = torch.zeros(total_classes)
        intersection_sum = torch.sum(intersection,0)
        union_sum = torch.sum(union,0)
    else:
        ious = torch.zeros(total_classes, dtype=torch.float64)
        intersection_sum = torch.sum(weight.unsqueeze(-1) * intersection,0)
        union_sum = torch.sum(weight.unsqueeze(-1) * union,0)

    valid_union = union_sum != 0
    ious[valid_union] = intersection_sum[valid_union] / union_sum[valid_union]


    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]
    valid_ious = ious[valid_gt_classes]
    mean_iou = torch.mean(valid_ious)

    iou_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50]
    # localization_result = {"activation":threshold}
    localization_result = []
    for iou_threshold in iou_thresholds:
        result = torch.sum(valid_ious > iou_threshold) / len(valid_ious)
        localization_result.append(result.item())

    return ious, mean_iou, localization_result

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=True)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument('--hooking_method', type=str, default="weight")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--use_pq", action="store_true")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--pq_index", type=str, default="./ckpts/pq_index.faiss")
    args = get_combined_args(parser)

    scene_list = [args.scene_name]
    iteration = 30000


    pq_index = args.pq_index
    index = faiss.read_index(pq_index)

    # activation_thresholds = [0.239]
    # activation_thresholds = activation_thresholds = [0.555, 0.556, 0.557, 0.558, 0.559, 0.560, 0.561, 0.562, 0.563, 0.564, 0.565,0.566,0.567,0.568,0.569]
    # activation_thresholds = [0.562]
    # activation_thresholds = [0.555, 0.556, 0.557, 0.558, 0.559, 0.560, 0.561, 0.562, 0.563, 0.564, 0.565,0.566,0.567,0.568,0.569]
    # activation_thresholds = [0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74]
    # activation_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # activation_thresholds = [0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84]
    # activation_thresholds = [0.237]
    # for threshold in activation_thresholds:
    model_miou = 0
    model_localization = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    model_valid_percent = 0
    model_valid_weight_percent = 0
    for scan_name in scene_list:
        cosine_similarity_list = []
        # for feature_level in range(1,4):
        for feature_level in range(1):
            # args.model_path = "/".join(args.model_path.split('/')[:-1])
            # args.model_path = os.path.join(args.model_path, scan_name)
            # args.model_path = os.path.join(args.model_path, f'{scan_name}_1_wopq_topk40')


            # args.model_path = os.path.join('output', scan_name)
            # args.model_path = args.model_path + "_" + str(feature_level) + "_pq_topk" + str(args.topk) + f"_{args.hooking_method}_" + str(index.coarse_code_size()+index.code_size)
            # args.model_path = args.model_path + "_" + str(feature_level) + "_pq_topk" + str(args.topk) + "_argmax_" + str(index.coarse_code_size()+index.code_size)
            model = model_params.extract(args)
            # (1) GT ply    change!
            gt_file_path = f"./labeled_gaussians/{scan_name}/labels.npy"
            # gt_file_path = f"labeled_gaussians/{scan_name}/labels.npy"
            # points, labels = read_labels_from_ply(gt_file_path)
            labels = np.load(gt_file_path)

            # (2) note: 19 & 15 & 10 classes
            # Given the category ID that needs to be queried (relative to the original NYU40), obtain the corresponding category name.
            target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]   # 19
            # target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]   # 15
            # target_id = [1,2,4,5,6,7,8,9,10,33] # 10

            target_dict = {key: nyu40_dict[key] for key in target_id}
            target_names = list(target_dict.values())

            # (3) update gt label
            # Obtained new point cloud labels, taking 19 categories as an example, where updated_labels are labels 0, 1-19.
            target_id_mapping = {value: index + 1 for index, value in enumerate(target_id)}
            updated_labels = np.zeros_like(labels)
            for original_value, new_value in target_id_mapping.items():
                updated_labels[labels == original_value] = new_value
            updated_gt_labels = torch.from_numpy(updated_labels.astype(np.int64)).cuda()


            gaussians = GaussianModel(model.sh_degree)
            # scene = Scene(model, gaussians, shuffle=False)
            checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
            gaussians.restore(model_params, args, mode='test')


            # (4) load gaussian ply file
            # model_path = f"output/{scan_name}/"
            # model_path = f"output_nonfix3/{scan_name}/"
            ply_path = os.path.join(args.model_path, f"point_cloud/iteration_0/point_cloud.ply")
            ply_data = PlyData.read(ply_path)
            vertex_data = ply_data['vertex'].data
            # NOTE Filter out points based on their opacity values.
            ignored_pts = sigmoid(vertex_data["opacity"]) < 0.1
            # ignored_pts = sigmoid(vertex_data["opacity"]) < 0.05
            # ignored_pts = sigmoid(vertex_data["opacity"]) < 0
            updated_gt_labels[ignored_pts] = 0


            # (6) load query text feat.
            with open('assets/text_features.json', 'r') as f:
                data_loaded = json.load(f)
            all_texts = list(data_loaded.keys())
            text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]

            query_text_feats = torch.zeros(len(target_names), 512).cuda()
            for i, text in enumerate(target_names):
                feat = text_features[all_texts.index(text)].unsqueeze(0)
                query_text_feats[i] = feat

            # get reconstructed clip feature
            # pq_index = "/home/kjs/project/iclr25/lerf/trained_index_ver3_1018.faiss"

            ##### encode decode #####
            # features_array_pq = index.sa_encode(language_features)
            language_features = gaussians._language_feature.clone()

            # #### reconstruction ####
            # clip_features = np.zeros((language_features.shape[0], 512))
            # language_features = language_features.cpu().numpy()
            # clip_features[invalid_index.cpu() == False] = index.sa_decode(language_features[invalid_index.cpu() == False])
            # clip_features = torch.Tensor(clip_features).cuda()
            # # (7) Calculate the cosine similarity and return the ID of the category with the highest value.
            # query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
            # clip_features = F.normalize(clip_features, dim=1, p=2)
            # # clip_features[invalid_index] = 0
            # cosine_similarity = torch.matmul(query_text_feats, clip_features.transpose(0, 1))
            # cosine_similarity_list.append(cosine_similarity)

            if args.use_pq:
                check_valid = torch.sum(language_features, 1)
                # invalid_index = check_valid == 255 * (index.coarse_code_size()+index.code_size)
                invalid_index = check_valid == 255 * (index.code_size)
                # #### search ####
                query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
                zero_mask = torch.all(language_features == -1, dim=-1)
                index = faiss.read_index(pq_index)
                index.add_sa_codes(language_features[~zero_mask].cpu().numpy())
                index.nprobe = index.nlist
                dist, ids = index.search(query_text_feats.cpu().numpy(), index.ntotal)
                sorted_indices = np.argsort(ids, axis=1)
                sorted_ids = np.take_along_axis(ids, sorted_indices, axis=1)
                sorted_dist = np.take_along_axis(dist, sorted_indices, axis=1)
                cosine_similarity = np.zeros((sorted_dist.shape[0], len(zero_mask)))
                cosine_similarity[:,~zero_mask.cpu().numpy()] = sorted_dist
                cosine_similarity_list.append(torch.tensor(cosine_similarity))

            else:
                # valid_mask = torch.all(language_features == -1, dim=-1)
                valid_mask = ~torch.all(language_features == -1, dim=-1)
                invalid_index = ~valid_mask
                zero_mask = invalid_index

                valid_features = language_features[valid_mask].to(query_text_feats.device).float()
                query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
                valid_features = F.normalize(valid_features, dim=1, p=2)

                cosine_sim = torch.matmul(query_text_feats, valid_features.T)
                full_cosine = torch.zeros((len(target_names), len(language_features))).cuda()
                full_cosine[:, valid_mask] = cosine_sim
                cosine_similarity_list.append(full_cosine)

            pred_bool = torch.zeros((language_features.shape[0], len(query_text_feats)+1), dtype=torch.bool)

            for i in range(len(query_text_feats)):
                if args.use_pq:
                    negative_text_features = torch.from_numpy(np.load('assets/text_negative.npy')).to(torch.float32)  # [num_text, 512]
                    negative_text_features = negative_text_features.cpu().numpy()

                    # # FIXME: lerf style
                    dist, ids = index.search(query_text_feats[i:i+1].cpu().numpy(), index.ntotal)
                    dist_negative, ids_negative = index.search(negative_text_features, index.ntotal)

                    sorted_indices = np.argsort(ids, axis=1)
                    sorted_indices_negative = np.argsort(ids_negative, axis=1)

                    sorted_ids_negative = np.take_along_axis(ids_negative, sorted_indices_negative, axis=1)
                    sorted_dist_negative = np.take_along_axis(dist_negative, sorted_indices_negative, axis=1)

                    sorted_ids = np.take_along_axis(ids, sorted_indices, axis=1)
                    sorted_dist = np.take_along_axis(dist, sorted_indices, axis=1)

                    positive_vals = torch.from_numpy(sorted_dist).transpose(1,0).to("cuda") # (num_gaussian, 4)
                    negative_vals = torch.from_numpy(sorted_dist_negative).transpose(1,0).to("cuda")
                    repeated_pos = positive_vals.repeat(1, len(negative_text_features))

                    sims = torch.stack((repeated_pos, negative_vals), dim=-1)
                    softmax = torch.softmax(10 * sims, dim=-1)

                    best_id = softmax[..., 0].argmin(dim=1)
                    score = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negative_text_features), 2))[:, 0, 0]

                    # threshold = threshold
                    # import pdb; pdb.set_trace()
                    mask = torch.zeros(language_features.shape[0], dtype=torch.bool, device="cuda")
                    mask[~zero_mask] = (score >= (score.max() * 0.8))

                    pred_bool[:,i+1] = mask
                else:
                    negative_text_features = torch.from_numpy(np.load('assets/text_negative.npy')).to(torch.float32)
                    negative_text_features = F.normalize(negative_text_features, dim=1, p=2).to("cuda")
                    # Positive similarity (각 점에 대해)
                    pos_sim = torch.matmul(query_text_feats[i:i+1], valid_features.T).transpose(0,1)
                    repeated_pos = pos_sim.repeat(1, len(negative_text_features)).cuda()

                    # Negative similarity
                    neg_sim = torch.matmul(negative_text_features, valid_features.T).transpose(0,1).cuda() # [num_neg, num_valid_feats]

                    # Stack and apply softmax
                    sims = torch.stack((repeated_pos, neg_sim), dim=-1).cuda()  # [num_neg, num_valid_feats, 2]
                    softmax = torch.softmax(10 * sims, dim=-1)

                    best_id = softmax[..., 0].argmin(dim=1)
                    score = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negative_text_features), 2))[:, 0, 0]

                    # threshold 기반 filtering
                    mask = torch.zeros(language_features.shape[0], dtype=torch.bool, device="cuda")
                    mask[valid_mask] = (score >= threshold)

                    pred_bool[:, i+1] = mask

            cosine_similarity_final = torch.cat(cosine_similarity_list).T
            cosine_similarity_dummy = torch.zeros((cosine_similarity_final.shape[0],1))
            cosine_similarity_final = torch.hstack((cosine_similarity_dummy.cuda(), cosine_similarity_final.cuda()))
            cosine_similarity_final[invalid_index] = 0

            # leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
            # cosine_similarity = torch.mm(query_text_feats, leaf_lang_feat.t())   # [cls_num, cluster_num]

            # get weight for each Gaussian
            scales = gaussians.get_scaling.type(torch.float64)
            opacities = gaussians.get_opacity.type(torch.float64)
            volume = scales[:,0] * scales[:,1] * scales[:,2]
            weight = volume * opacities.squeeze()

            # weight = torch.ones_like(weight)

            # origin_label = scannet_19_to_nyu[max_id]
            # origin_label[invalid_index] = 0

            np.save(os.path.join(args.model_path, 'labels_localization.npy'), pred_bool.numpy())
            ious, mean_iou, localization_result = calculate_localization(updated_gt_labels, pred_bool, weight, len(target_names)+1, invalid_index)

            # ious, mean_iou, accuracy, mean_acc = calculate_metrics(updated_gt_labels, pred_pts_cls_id, total_classes=len(target_names)+1)
            print(f"Scene: {scan_name}, mIoU: {mean_iou:.4f}, local.: {localization_result[0],localization_result[1],localization_result[2],localization_result[3],localization_result[4], localization_result[5]}")

            model_miou += mean_iou
            model_localization += torch.tensor(localization_result)
            model_valid_percent += (torch.sum(~zero_mask) / len(zero_mask)).item()
            model_valid_weight_percent += (torch.sum(weight[~zero_mask]) / torch.sum(weight)).item()


        model_miou /= len(scene_list)
        model_localization /= len(scene_list)

        model_valid_percent /= len(scene_list)
        model_valid_weight_percent /= len(scene_list)

        print(f"Total results, mIoU: {model_miou:.4f}, localization.: {model_localization[0],model_localization[1],model_localization[2],model_localization[3],model_localization[4], model_localization[5]}")

        save_results = {}
        save_results["model_path"] = args.model_path
        # save_results["pq_dim"] = index.coarse_code_size()+index.code_size
        save_results["pq_dim"] = index.code_size
        save_results["pq_index"] = pq_index
        save_results["topk"] = args.topk
        # save_results["activation_threshold"] = threshold
        save_results["mIoU"] = model_miou.item()
        save_results["localization"] = model_localization.tolist()
        save_results["valid_percent"] = model_valid_percent
        save_results["valid_weight"] = model_valid_weight_percent
        save_dir = os.path.join("result/localization_negative", args.hooking_method)
        os.makedirs(save_dir, exist_ok=True)
        import json
        with open(os.path.join(save_dir, args.model_path.split('/')[-1]), 'w') as json_file:
            json.dump(save_results, json_file, indent=4)
