# Dr. Splat Pipeline

This document lists the runnable pipeline used by the repository scripts. Paths are relative to the repository root.

## Common Inputs

- `ckpts/sam_vit_h_4b8939.pth`: SAM ViT-H checkpoint.
- `ckpts/pq_index.faiss`: PQ index used by Dr. Splat feature registration and evaluation.
- `assets/text_features.json`: CLIP text features for ScanNet NYU40 class names.
- `assets/text_negative.npy`: negative text features for ScanNet localization.
- `gaussian-pretrain/<scene>/chkpnt30000.pth`: pretrained 3DGS checkpoint. This can be a real directory or a symlink to another 3DGS output directory.

The repository expects pretrained 3DGS checkpoints as inputs. Train 3DGS separately, then place or symlink its output under `gaussian-pretrain/<scene>`.

## LeRF / LangSplat Scenes

Expected scene layout:

```text
langsplat/
  figurines/
    images/
    sparse/
  label/
    figurines/
      frame_*.json
      frame_*.jpg
gaussian-pretrain/
  figurines/
    chkpnt30000.pth
```

Run one scene:

```bash
GPU_ID=0 SCENE_NAME=figurines bash shell/run_lerf.sh
```

Run all four scenes:

```bash
GPU_ID=0 bash shell/run_lerf_all.sh
```

The script performs:

1. SAM mask and CLIP feature extraction into `langsplat/<scene>/language_features`.
2. Dr. Splat training from `gaussian-pretrain/<scene>/chkpnt30000.pth`.
3. Text query rendering with `render_lerf_by_text.py`.
4. IoU evaluation with `eval_object_selection/compute_lerf_iou.py`.

Default output:

```text
output/<scene>_3_pq_openclip_topk40_weight_128/
  chkpnt0.pth
  point_cloud/iteration_0/point_cloud.ply
  text2obj/ours_0/renders_cluster_silhouette/
```

## ScanNet Scenes

Expected source layout after downloading the OpenScene pre-processed ScanNet data:

```text
scannet/
  scannet_2d/
    scene0000_00/
      color/
      pose/
  scannet_3d/
    train|val|test/
      scene0000_00_vh_clean_2.pth
gaussian-pretrain/
  scene0000_00/
    chkpnt30000.pth
```

The raw ScanNet release is not consumed directly by the current scripts. It must first be converted into the `scannet_2d/<scene>/{color,pose}` and `scannet_3d/{train,val,test}/*_vh_clean_2.pth` layout used by OpenScene, or an equivalent layout.

Run one scene:

```bash
GPU_ID=0 SCENE_NAME=scene0000_00 bash shell/run_scannet.sh
```

Run the scene list in `shell/scannet_all_scenes.txt`:

```bash
GPU_ID=0 bash shell/run_scannet_all.sh
```

The script performs:

1. ScanNet preprocessing with `scripts/preprocess_scannet.py`.
2. SAM mask and CLIP feature extraction into `scannet/scannet_2d/<scene>/language_features`.
3. Dr. Splat training from `gaussian-pretrain/<scene>/chkpnt30000.pth`.
4. Pseudo GT generation with `labeling_gaussians.py`.
5. 3D segmentation evaluation with `eval_scannet_nonfix.py`.
6. Localization evaluation with `eval_scannet_localization_nonfix.py`.

Default output:

```text
output/<scene>_3_pq_openclip_topk40_weight_128/
result_nonfix3_3dseg/labels_19/<model-name>
result/localization_negative/weight/<model-name>
```

## Reusing Existing Outputs

The scripts reuse existing outputs when possible. If a scene already has `language_features`, `chkpnt0.pth`, or `labeled_gaussians/<scene>/labels.npy`, the corresponding expensive step is skipped by the single-scene scripts.
