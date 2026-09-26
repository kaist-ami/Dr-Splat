#!/bin/sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"

SCENE_NAME="${SCENE_NAME:-scene0000_00}"
GPU_ID="${GPU_ID:-0}"
CAMERA_PATH="./scannet/scannet_2d/${SCENE_NAME}"
OUTPUT_PATH="./output/${SCENE_NAME}"
TRAINED_3DGS_PATH="./gaussian-pretrain/${SCENE_NAME}"
feature_level=3
topk=40
MODEL_PATH="${OUTPUT_PATH}_${feature_level}_pq_openclip_topk${topk}_weight_128"

if [ ! -f "${CAMERA_PATH}/transforms_train.json" ] || [ ! -f "${CAMERA_PATH}/points3d.ply" ]; then
  python scripts/preprocess_scannet.py --scannet_root ./scannet ${SCENE_NAME}
fi

if [ ! -d "${CAMERA_PATH}/language_features" ]; then
  mkdir -p "${CAMERA_PATH}/language_features"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python preprocessing.py \
                          --dataset_path ${CAMERA_PATH} \
                          --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth
fi

if [ ! -f "${MODEL_PATH}/chkpnt0.pth" ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
                          -s ${CAMERA_PATH} \
                          -m ${OUTPUT_PATH} \
                          --start_checkpoint ${TRAINED_3DGS_PATH}/chkpnt30000.pth \
                          --feature_level ${feature_level} \
                          --name_extra pq_openclip \
                          --use_pq \
                          --pq_index ckpts/pq_index.faiss \
                          --port 55560 \
                          --topk ${topk} \
                          --eval
fi


# # Visualize PCA results
# CUDA_VISIBLE_DEVICES=${GPU_ID} python render_pca.py \
#                         -s ${CAMERA_PATH} \
#                         -m ${OUTPUT_PATH}_${feature_level}_pq_openclip_topk${topk}_weight_128 \
#                         --pq_index ckpts/pq_index.faiss \
#                         --feature_level ${feature_level} \
#                         -l language_features_dim3

# # Visualize activation for text query "sofa"
# CUDA_VISIBLE_DEVICES=${GPU_ID} python render_activation.py \
#                         -s ${CAMERA_PATH} \
#                         -m ${OUTPUT_PATH}_${feature_level}_pq_openclip_topk${topk}_weight_128 \
#                         --semantic_model sam \
#                         --feature_level ${feature_level} \
#                         --pq_index ckpts/pq_index.faiss \
#                         --img_label sofa \
#                         --img_save_label sofa_test \
#                         --threshold 0.0 \
#                         -l language_features_dim3

# # Calculate metric results
# To do this we first need
if [ ! -f "./labeled_gaussians/${SCENE_NAME}/labels.npy" ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python labeling_gaussians.py \
                          -s ${CAMERA_PATH} \
                          -m ${TRAINED_3DGS_PATH}
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_scannet_nonfix.py \
                        -m ${MODEL_PATH} \
                        --use_pq \
                        --scene_name ${SCENE_NAME} \
                        --pq_index ckpts/pq_index.faiss

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_scannet_localization_nonfix.py \
                        -m ${MODEL_PATH} \
                        --use_pq \
                        --scene_name ${SCENE_NAME} \
                        --pq_index ckpts/pq_index.faiss
