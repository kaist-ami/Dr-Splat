#!/bin/sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"

SCENE_NAME="${SCENE_NAME:-figurines}"
GPU_ID="${GPU_ID:-0}"
CAMERA_PATH="./langsplat/${SCENE_NAME}"
OUTPUT_PATH="./output/${SCENE_NAME}"
TRAINED_3DGS_PATH="./gaussian-pretrain/${SCENE_NAME}"
feature_level=3
topk=40
MODEL_PATH="${OUTPUT_PATH}_${feature_level}_pq_openclip_topk${topk}_weight_128"

# Training Dr.Splat
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
                        --eval  # enable if you want to split your dataset with training and validation sets else, disable this

# # Visualize PCA results
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_pca.py \
                        -s ${CAMERA_PATH} \
                        -m ${MODEL_PATH} \
                        --pq_index ckpts/pq_index.faiss \
                        --feature_level ${feature_level} \
                        -l language_features_dim3

# Visualize activation for text query "rubics cube"
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_activation.py \
                        -s ${CAMERA_PATH} \
                        -m ${MODEL_PATH} \
                        --semantic_model sam \
                        --feature_level ${feature_level} \
                        --pq_index ckpts/pq_index.faiss \
                        --img_label "rubics cube" \
                        --img_save_label rubics_cube_test \
                        --threshold 0.0 \
                        -l language_features_dim3

# Calculate metric results
# First render 3D activated renderings
CUDA_VISIBLE_DEVICES=${GPU_ID} python render_lerf_by_text.py \
                        -m ${MODEL_PATH} \
                        --scene_name "${SCENE_NAME}" \
                        --use_pq \
                        --pq_index "./ckpts/pq_index.faiss" \
                        --checkpoint_iter "0" \
                        --threshold 0.55

# Then calculate iou with GT mask
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_object_selection/compute_lerf_iou.py \
                        --scene_name "${SCENE_NAME}" \
                        --path_gt "./langsplat/label/${SCENE_NAME}/" \
                        --log . \
                        --exp_name ${MODEL_PATH} \
                        --path_pred ${MODEL_PATH}/text2obj/ours_0/renders_cluster_silhouette
