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

def calculate_metrics(gt, pred, weight, total_classes):
    gt = gt.cpu()
    pred = pred.cpu()
    pred[gt == 0] = 0
    weight = weight.cpu()

    ious = torch.zeros(total_classes)

    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)


    if weight == None:
        for cls in range(1, total_classes):
            intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
            union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
            correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
            total[cls] = torch.sum(gt == cls).item()
    else:
        for cls in range(1, total_classes):
            intersection[cls] = torch.sum(weight * ((gt == cls) & (pred == cls))).item()
            union[cls] = torch.sum(weight * ((gt == cls) | (pred == cls))).item()
            correct[cls] = torch.sum(weight * ((gt == cls) & (pred == cls))).item()
            total[cls] = torch.sum(weight * (gt == cls)).item()

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

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=True)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--use_pq", action="store_true")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--pq_index", type=str, default="./ckpts/pq_index.faiss")
    args = get_combined_args(parser)

    scene_list = [args.scene_name]
    iteration = 30000

    model_miou = 0
    model_macc = 0

    pq_index = args.pq_index
    index = faiss.read_index(pq_index)

    # Given the category ID that needs to be queried (relative to the original NYU40), obtain the corresponding category name.
    target_id_19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]   # 19
    target_id_15 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]   # 15
    target_id_10 = [1, 2, 4, 5, 6, 7, 8, 9, 10, 33] # 10
    # target_ids = [target_id_10, target_id_15, target_id_19]
    target_ids = [target_id_19]

    for target_id in target_ids:
        for scan_name in scene_list:
            cosine_similarity_list = []
            # for feature_level in range(1,4):
            for feature_level in range(1):
                print(args.model_path)
                # args.model_path = os.path.join('output', scan_name)
                # args.model_path = "/".join(args.model_path.split('/')[:-1])
                # args.model_path = os.path.join(args.model_path, scan_name)
                # args.model_path = os.path.join('output_opengs', scan_name)
                # args.model_path = args.model_path + "_" + str(feature_level) + "_pq_topk" + str(args.topk) + "_weight_" + str(index.coarse_code_size()+index.code_size)
                # args.model_path = args.model_path + "_" + str(feature_level) + "_pq_topk" + str(args.topk) + "_argmax_" + str(index.coarse_code_size()+index.code_size)
                model = model_params.extract(args)
                # (1) GT ply change!
                # gt_file_path = f"data/scannet/{scan_name}/{scan_name}_vh_clean_2.labels.ply"
                # points, labels = read_labels_from_ply(gt_file_path)
                # gt_file_path = f"labeled_gaussians/{scan_name}/labels.npy"
                # points, labels = read_labels_from_ply(gt_file_path)
                # gt_file_path = f"labeled_gaussians_no_denom_fix/{scan_name}/labels.npy"
                gt_file_path = f"./labeled_gaussians/{scan_name}/labels.npy"
                labels = np.load(gt_file_path)

                # (2) note: 19 & 15 & 10 classes
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
                # args.model_path = f"output_nonfix3/{scan_name}_1_pq_topk45_weight_130/"
                # model_path = f"output_opengs/{scan_name}/"
                # ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
                ply_path = os.path.join(args.model_path, f"point_cloud/iteration_0/point_cloud.ply")
                ply_data = PlyData.read(ply_path)
                vertex_data = ply_data['vertex'].data
                # NOTE Filter out points based on their opacity values.

                ignored_pts = sigmoid(vertex_data["opacity"]) < 0.1
                updated_gt_labels[ignored_pts] = 0

                print(updated_gt_labels.shape)  #### dahye

                # # (5) load cluster language file
                # mapping_file = os.path.join(model_path, "cluster_lang.npz")
                # # load the saved codebook(leaf id) and instance-level language feature
                # # 'leaf_feat', 'leaf_acore', 'occu_count', 'leaf_ind'
                # saved_data = np.load(mapping_file)
                # leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=k1*k2, 512]
                # leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=k1*k2]
                # leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=k1*k2]
                # leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts]
                # leaf_lang_feat[leaf_occu_count < 2] *= 0.0
                # leaf_ind = leaf_ind.clamp(max=319)  # 64*5=320

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
                print(language_features.shape) #### dahye
                if args.use_pq:
                    check_valid = torch.sum(language_features, 1)
                    # invalid_index = check_valid == 255 * (index.coarse_code_size()+index.code_size)
                    invalid_index = check_valid == 255 * (index.code_size)

                    # ##### recon #####
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

                    #### search ####
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

                    cosine_similarity_final = torch.cat(cosine_similarity_list)

                    # leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
                    # cosine_similarity = torch.mm(query_text_feats, leaf_lang_feat.t())   # [cls_num, cluster_num]
                    max_id = torch.argmax(cosine_similarity_final, dim=0) # [cluster_num]
                    max_id = max_id % 19
                    pred_pts_cls_id = max_id + 1
                    pred_pts_cls_id[invalid_index] = 0

                # else:
                #     ### dahye
                #     # check_valid = torch.sum(language_features, 1)
                #     # invalid_index = check_valid == 255 * (index.code_size)

                #     valid_mask = ~torch.all(language_features == -1, dim=-1)  # 유효한 feature만 사용
                #     valid_features = language_features[valid_mask].to(query_text_feats.device).float()

                #     # 정규화
                #     query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
                #     valid_features = F.normalize(valid_features, dim=1, p=2)

                #     # 유사도 계산
                #     cosine_sim = torch.matmul(query_text_feats, valid_features.T)  # [num_texts, num_valid_feats]

                #     # 전체 shape 맞추기
                #     full_cosine = torch.zeros((len(target_names), len(language_features))).to(cosine_sim.device)
                #     full_cosine[:, valid_mask] = cosine_sim
                #     cosine_similarity_list.append(full_cosine)


                #     cosine_similarity_final = torch.cat(cosine_similarity_list)

                #     # leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
                #     # cosine_similarity = torch.mm(query_text_feats, leaf_lang_feat.t())   # [cls_num, cluster_num]
                #     max_id = torch.argmax(cosine_similarity_final, dim=0) # [cluster_num]
                #     max_id = max_id % 19
                #     pred_pts_cls_id = max_id
                #     pred_pts_cls_id[~valid_mask] = 0

            # get weight for each Gaussian
            scales = gaussians.get_scaling.type(torch.float64)
            opacities = gaussians.get_opacity.type(torch.float64)
            volume = scales[:,0] * scales[:,1] * scales[:,2]
            weight = volume * opacities.squeeze()

            weight = torch.ones_like(weight)

            # origin_label = scannet_19_to_nyu[max_id]
            # origin_label[invalid_index] = 0
            # np.save(os.path.join(args.model_path, 'labels.npy'), origin_label.numpy())
            # import pdb; pdb.set_trace()
            ious, mean_iou, accuracy, mean_acc = calculate_metrics(updated_gt_labels, pred_pts_cls_id, weight, total_classes=len(target_names)+1)
            print(f"Scene: {scan_name}, mIoU: {mean_iou:.4f}, mAcc.: {mean_acc:.4f}")

            model_miou += mean_iou
            model_macc += mean_acc

        model_miou /= len(scene_list)
        model_macc /= len(scene_list)

        print(f"Total result, mIoU: {model_miou:.4f}, mAcc.: {model_macc:.4f}")

        save_results = {}
        save_results["model_path"] = args.model_path
        # save_results["pq_dim"] = index.coarse_code_size()+index.code_size
        save_results["pq_dim"] = index.code_size
        save_results["pq_index"] = pq_index
        save_results["topk"] = args.topk
        save_results["mIoU"] = model_miou
        save_results["mAcc"] = model_macc

        save_dir = "result_nonfix3_3dseg"
        os.makedirs(os.path.join(save_dir, "labels_" + str(len(target_id))), exist_ok=True)
        import json
        with open(os.path.join(save_dir, "labels_" + str(len(target_id)), args.model_path.split('/')[-1]), 'w') as json_file:
            json.dump(save_results, json_file, indent=4)

    gaussian_color_map = color_map[pred_pts_cls_id]/255
    gaussians._features_dc = gaussians._features_dc.detach()
    gaussians._features_rest = gaussians._features_rest.detach()
    gaussians._opacity = gaussians._opacity.detach()
    # gaussians._opacity[~invalid_index] = -10

    offset = 0.5
    SH_C0 = 0.28209479177387814
    gaussians._features_dc[:,0,:] = (gaussian_color_map - offset) / SH_C0
    gaussians._features_rest[:,:,:] = 0
    gaussians.save_ply(os.path.join(save_dir, 'point_cloud.ply'))
