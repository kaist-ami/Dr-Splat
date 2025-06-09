# Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration
<h3>CVPR 2025 <mark>Highlight</mark></h3>

### [Project Page](https://drsplat.github.io/) | [Paper](https://arxiv.org/abs/2502.16652)

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


pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install kmeans_pytorch

pip install faiss-cpu
```

## Download Checkpoint
- Downalod SAM checkpoint from [here](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth) and move into `ckpts` directory.

## Preliminary
- Prepare camera pose of the scenes (e.g., COLMAP) and trained 3DGS.

## Feature (SAM Mask + CLIP embedding) Extraction
- To construct feature embedded 3DGS with Dr. Splat, first need to prepare CLIP embeddings per sam masks.

```bash
mkdir "${COLMAP_PATH}/language_features"
CUDA_VISIBLE_DEVICES=${GPU_ID} python preprocessing.py \
                        --dataset_path ${COLMAP_PATH} \

echo "All commands executed."
```

## Training
- Training Dr. Splat with direct CLIP embedding registration to 3DGS

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
                        -s ${COLMAP_PATH} \
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
                        -s ${COLMAP_PATH} \
                        -m ${TRAINED_DRSPLAT_PATH} \
                        --pq_index ckpts/pq_index.faiss \
                        --feature_level 1 \
                        -l language_features_dim3
```

## Activation Visualization
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_activation.py \
                        -s ${COLMAP_PATH} \
                        -m ${TRAINED_DRSPLAT_PATH} \
                        --semantic_model sam \
                        --feature_level 1 \
                        --pq_index ckpts/pq_index.faiss \
                        --img_label sofa \  # text query
                        --img_save_label sofa_test \  # save directory name
                        --threshold 0.5 \  # 0.0 - full activation render, 
                        # greater than 0.0 - alpha-blended result with 3D scene 
                        -l language_features_dim3
```

## Evaluation
- TBA

## Citation
```
@inproceedings{drsplat25,
    title={Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration},
    author={Jun-Seong, Kim and Kim GeonU and Yu-Ji, Kim and Yu-Chiang Frank Wang and Jaesung Choe and Oh, Tae-Hyun},
    booktitle=CVPR,
    year={2025}
}
```