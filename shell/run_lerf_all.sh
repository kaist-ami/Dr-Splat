#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"
GPU_ID="${GPU_ID:-0}"

#### Checklist for running this script:
# Make sure you have generated the pre-trained 3DGS checkpoints for each scene and placed them in the correct path (./gaussian-pretrain/${scene}/chkpnt30000.pth).
# For text query activation visualization, ensure that you have put the correct text query in the --img_label argument and that the corresponding images will be saved with the label specified in --img_save_label.
# For better metric evaluation results, you can tune the threshold value in the render_lerf_by_text.py command. The current value is set to 0.55, but you may want to adjust it based on your specific dataset and requirements.

for scene in figurines ramen teatime waldo_kitchen
do
    CAMERA_PATH="./langsplat/${scene}"
    OUTPUT_PATH="./output/${scene}"
    TRAINED_3DGS_PATH="./gaussian-pretrain/${scene}"
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

    # # # Visualize PCA results
    CUDA_VISIBLE_DEVICES=${GPU_ID} python render_pca.py \
                            -s ${CAMERA_PATH} \
                            -m ${MODEL_PATH} \
                            --pq_index ckpts/pq_index.faiss \
                            --feature_level ${feature_level} \
                            -l language_features_dim3

    # # Visualize activation for text query "rubics cube"
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python render_activation.py \
    #                         -s ${CAMERA_PATH} \
    #                         -m ${OUTPUT_PATH}_1_pq_openclip_topk45_weight_128 \
    #                         --semantic_model sam \
    #                         --feature_level 3 \
    #                         --pq_index ckpts/pq_index.faiss \
    #                         --img_label "rubics cube" \
    #                         --img_save_label rubics_cube_test \
    #                         --threshold 0.0 \
    #                         -l language_features_dim3

    # Calculate metric results
    # First render 3D activated renderings
    CUDA_VISIBLE_DEVICES=${GPU_ID} python render_lerf_by_text.py \
                            -m ${MODEL_PATH} \
                            --scene_name ${scene} \
                            --use_pq \
                            --pq_index "./ckpts/pq_index.faiss" \
                            --checkpoint_iter "0" \
                            --threshold 0.55

    # Then calculate iou with GT mask
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_object_selection/compute_lerf_iou.py \
                            --scene_name ${scene} \
                            --path_gt ./langsplat/label/${scene}/ \
                            --log . \
                            --exp_name ${MODEL_PATH} \
                            --path_pred ${MODEL_PATH}/text2obj/ours_0/renders_cluster_silhouette
done
