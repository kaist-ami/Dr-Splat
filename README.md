# Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration
<h3>CVPR 2025 <mark>Highlight</mark></h3>

### [Project Page](https://drsplat.github.io/) | [Paper](https://arxiv.org/abs/2502.16652)

![github_teaser](https://github.com/user-attachments/assets/eb6f12cf-152a-4fe9-96b7-7717a9c1a543)

This repository is official implementation for the CVPR 2025 highlight paper, Dr. Splat.


## Download
```bash
git clone git@github.com:kaist-ami/Dr-Splat.git --recursive
```

## Setup
```bash
conda create -n drsplat python=3.9

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install submodules/langsplat-rasterization
pip install submodules/segment-anything-langsplat
pip install submodules/simple-knn
pip install --no-build-isolation submodules/ashawkey-diff-gaussian-rasterization

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install kmeans_pytorch

pip install faiss-cpu
pip install colormaps
pip install open3d
pip install bitarray
```

> **Note (CUDA 13.0).** The pipeline was also run through successfully with Python 3.10, PyTorch 2.11.0, torchvision 0.26.0, torchaudio 2.11.0, and CUDA 13.0. Install the matching `cu130` PyTorch wheels and set `CUDA_HOME` when building the CUDA extensions:
>
> ```bash
> export CUDA_HOME=/usr/local/cuda-13.0
> export PATH="${CUDA_HOME}/bin:${PATH}"
> pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
>   --index-url https://download.pytorch.org/whl/cu130
> ```
>
> On CUDA 13.0, older rasterizer sources may also need `#include <cstdint>` in headers that use fixed-width integer types before they compile.

## Download Checkpoint
- Downalod SAM checkpoint from [here](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth) and move into `ckpts` directory.

## Preliminary
- Prepare camera pose of the scenes (e.g., COLMAP) and trained 3DGS.
- Put each pretrained 3DGS checkpoint at `gaussian-pretrain/<scene>/chkpnt30000.pth`, or symlink `gaussian-pretrain` to your 3DGS output directory.

> **Note.** Use the same NumPy major version when saving the preliminary 3DGS checkpoint and loading it in Dr. Splat. A checkpoint saved with NumPy 2.x can fail in a NumPy 1.x environment with `No module named 'numpy._core'`.
>
> Keep `chkpnt30000.pth` together with the matching `point_cloud/iteration_30000/point_cloud.ply`. Mixing outputs from different training runs changes the Gaussian count and causes ScanNet labeling/evaluation shape mismatches.

## Feature (SAM Mask + CLIP embedding) Extraction
- To construct feature embedded 3DGS with Dr. Splat, first need to prepare CLIP embeddings per sam masks.

```bash
mkdir -p "${CAMERA_PATH}/language_features"
CUDA_VISIBLE_DEVICES=${GPU_ID} python preprocessing.py \
                        --dataset_path ${CAMERA_PATH}
```

## Training
- Training Dr. Splat with direct CLIP embedding registration to 3DGS

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
                        -s ${CAMERA_PATH} \
                        -m ${OUTPUT_PATH} \
                        --start_checkpoint ${TRAINED_3DGS_PATH}/chkpnt30000.pth \
                        --feature_level 1 \
                        --name_extra pq_openclip \
                        --use_pq \
                        --pq_index ckpts/pq_index.faiss \
                        --port 55560 \
                        --topk 45 \
                        --eval  # enable if you want to split your dataset with training and validation sets else, disable this
```

## Feature PCA Visualization
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_pca.py \
                        -s ${CAMERA_PATH} \
                        -m ${TRAINED_DRSPLAT_PATH} \
                        --pq_index ckpts/pq_index.faiss \
                        --feature_level 1 \
                        -l language_features_dim3
```

## Activation Visualization
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_activation.py \
                        -s ${CAMERA_PATH} \
                        -m ${TRAINED_DRSPLAT_PATH} \
                        --semantic_model sam \
                        --feature_level 1 \
                        --pq_index ckpts/pq_index.faiss \
                        --img_label sofa \
                        --img_save_label sofa_test \
                        --threshold 0.5 \
                        -l language_features_dim3
```

`--img_label` is the text query, `--img_save_label` names the output directory, and `--threshold 0.0` renders full activation. Values greater than `0.0` produce an alpha-blended result with the 3D scene.

## Dataset
In the experiments section of our paper, we primarily utilized two datasets: the LeRF dataset and scannet dataset.

For the LeRF dataset, you can directly download from [link](https://github.com/minghanqin/LangSplat?tab=readme-ov-file)

For Scannet dataset, we utilize uniformly sampled, down-scaled version organized by [link](https://github.com/pengsongyou/openscene?tab=readme-ov-file#data-preparation).
Use the OpenScene pre-processed 3D&2D data layout, not the raw ScanNet release directly.
To make it applicable for the training, first run `scripts/preprocess_scannet.py` to generate `points3d.ply` and Blender-style transforms for 3DGS/Dr. Splat.

> **Note.** `labeling_gaussians.py` is CPU intensive. It uses up to 24 CPU threads by default and took about two hours for `scene0000_00` with approximately 2.5 million Gaussians in our run-through. Override `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` before invoking a shell script if the machine requires a lower limit.
>
> If an external Gaussian Splatting checkout fails in Pillow with `Cannot handle this data type: ... |i1`, replace `dtype=np.byte` with `dtype=np.uint8` in its RGB `Image.fromarray` conversion. The corresponding loader in this repository is already fixed.

For the full expected directory layout and script flow, see [docs/PIPELINE.md](docs/PIPELINE.md).

## Evaluation
Set `GPU_ID` to an available GPU before running the scripts below.

### LeRF evaluation
Run a single LangSplat/LeRF scene:

```bash
GPU_ID=0 SCENE_NAME=figurines bash shell/run_lerf.sh
```

Run all four scenes:

```bash
GPU_ID=0 bash shell/run_lerf_all.sh
```

To evaluate Object Selection manually, first render silhouette masks from selected activated 3D Gaussians:

```
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_lerf_by_text.py \
                        -m ${MODEL_PATH} \
                        --scene_name "figurines" \
                        --use_pq \
                        --pq_index "./ckpts/pq_index.faiss" \
                        --checkpoint_iter "0" \
                        --threshold 0.55
```

Then calculate IoU with the LangSplat GT masks:

```
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_object_selection/compute_lerf_iou.py \
                        --scene_name figurines \
                        --path_gt ./langsplat/label/figurines/ \
                        --log . \
                        --exp_name ${MODEL_PATH} \
                        --path_pred ${MODEL_PATH}/text2obj/ours_0/renders_cluster_silhouette
```

### Scannet evaluation
Run a single ScanNet scene:

```bash
GPU_ID=0 SCENE_NAME=scene0000_00 bash shell/run_scannet.sh
```

Run all scenes listed in `shell/scannet_all_scenes.txt`:

```bash
GPU_ID=0 bash shell/run_scannet_all.sh
```

The all-scenes script reuses existing language features, Dr. Splat checkpoints, and Gaussian labels. Set `RUN_VISUALIZATIONS=1` only when PCA and activation renders are also required; they are skipped by default during metric evaluation.

To run the ScanNet evaluation manually after preprocessing, feature extraction, Dr. Splat training, and pseudo GT generation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_scannet_nonfix.py \
                        -m ${MODEL_PATH} \
                        --use_pq \
                        --scene_name scene0000_00 \
                        --pq_index ./ckpts/pq_index.faiss

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_scannet_localization_nonfix.py \
                        -m ${MODEL_PATH} \
                        --use_pq \
                        --scene_name scene0000_00 \
                        --pq_index ./ckpts/pq_index.faiss
```

## Citation
```
@inproceedings{drsplat25,
    title={Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration},
    author={Jun-Seong, Kim and Kim GeonU and Yu-Ji, Kim and Yu-Chiang Frank Wang and Jaesung Choe and Oh, Tae-Hyun},
    booktitle=CVPR,
    year={2025}
}
```
