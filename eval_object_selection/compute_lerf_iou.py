import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import logging
from typing import Dict, Union
from pathlib import Path
from collections import defaultdict
import glob
import json

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    if not gt_json_paths:
        return {}, None, img_paths

    gt_ann = {}
    # import pdb; pdb.set_trace()
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)

        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask

            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def evalute(gt_base, pred_base, scene_name, log_base, exp_name):
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    frame_names = scene_gt_frames[scene_name]
    _, hw, _ = eval_gt_lerfdata(gt_base, Path(gt_base))
    if hw is None:
        raise FileNotFoundError(
            f"No GT annotation json files were found under '{gt_base}'. "
            "Expected files matching 'frame_*.json'."
        )

    ious = []
    for frame in frame_names:
        print("frame:", frame)
        gt_folder = os.path.join(gt_base, "gt", frame)
        # import pdb; pdb.set_trace()
        file_names = [f for f in os.listdir(gt_folder) if f.endswith('.jpg')]
        for file_name in file_names:
            base_name = os.path.splitext(file_name)[0]
            gt_obj_path = os.path.join(gt_folder, file_name)
            pred_obj_path = os.path.join(pred_base, frame + "_" + base_name + '.png')
            if not os.path.exists(pred_obj_path):
                print(f"Missing pred file for {file_name}, skipping...")
                print(f"IoU for {file_name}: 0")
                ious.append(0.0)
                continue
            mask_gt = load_image_as_binary(gt_obj_path)
            mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
            iou = calculate_iou(mask_gt, mask_pred)
            ious.append(iou)
            print(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}")

    # Acc.
    total_count = len(ious)
    if total_count == 0:
        raise RuntimeError(
            f"No IoU entries were computed for scene '{scene_name}'. "
            f"Check GT output under '{gt_base}/gt' and predictions under '{pred_base}'."
        )
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    # mIoU
    average_iou = np.mean(ious)
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Acc@0.25: {count_iou_025/total_count:.4f}")
    print(f"Acc@0.5: {count_iou_05/total_count:.4f}")

    import json
    with open(os.path.join(log_base, f"{exp_name}_iou.json"), "a") as f:
        json.dump({"scene_name": scene_name,
                   "average_iou": f"{average_iou*100:.4f}",
                   "acc_025": f"{count_iou_025/total_count*100:.4f}",
                   "acc_05": f"{count_iou_05/total_count*100:.4f}"}, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser("Compute LeRF IoU")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    parser.add_argument("--path_gt", type=str, help="Path to the ground truth masks")
    parser.add_argument("--path_pred", type=str, help="Path to the predicted masks")
    parser.add_argument("--log", type=str, default="./logs", help="Path to the log file")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    args = parser.parse_args()
    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")

    # TODO: change
    # path_gt = "./data/label/waldo_kitchen/gt"
    # # renders_cluster_silhouette is the predicted mask
    # path_pred = "output/xxxxxxxx-x/text2obj/ours_70000/renders_cluster_silhouette"
    evalute(args.path_gt, args.path_pred, args.scene_name, args.log, args.exp_name)
