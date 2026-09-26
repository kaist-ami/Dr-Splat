#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
FEATURE_LEVEL="${FEATURE_LEVEL:-3}"
TOPK="${TOPK:-40}"
PORT="${PORT:-55560}"
PQ_INDEX="${PQ_INDEX:-./ckpts/pq_index.faiss}"

SCENE_LIST_FILE="${SCENE_LIST_FILE:-./shell/scannet_all_scenes.txt}"
SCANNET_ROOT="${SCANNET_ROOT:-./scannet}"
CAMERA_ROOT="${CAMERA_ROOT:-${SCANNET_ROOT}/scannet_2d}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-./gaussian-pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_scannet_all}"
RUN_VISUALIZATIONS="${RUN_VISUALIZATIONS:-0}"

IMG_LABEL="${IMG_LABEL:-sofa}"
IMG_SAVE_LABEL="${IMG_SAVE_LABEL:-sofa_test}"
ACT_THRESHOLD="${ACT_THRESHOLD:-0.0}"

echo "[run_scannet_all] repo root: ${ROOT_DIR}"
echo "[run_scannet_all] scene list: ${SCENE_LIST_FILE}"

if [[ ! -f "${SCENE_LIST_FILE}" ]]; then
  echo "[run_scannet_all][error] missing scene list file: ${SCENE_LIST_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

while IFS= read -r scene || [[ -n "${scene}" ]]; do
  scene="${scene%%#*}"
  scene="$(echo "${scene}" | xargs)"
  if [[ -z "${scene}" ]]; then
    continue
  fi

  camera_path="${CAMERA_ROOT}/${scene}"
  trained_3dgs_path="${PRETRAIN_ROOT}/${scene}"
  output_path="${OUTPUT_ROOT}/${scene}"
  model_path="${output_path}_${FEATURE_LEVEL}_pq_openclip_topk${TOPK}_weight_128"

  echo
  echo "[run_scannet_all] ================================================"
  echo "[run_scannet_all] scene: ${scene}"

  if [[ ! -d "${camera_path}" ]]; then
    echo "[run_scannet_all][warn] missing camera path: ${camera_path}" >&2
    continue
  fi

  if [[ ! -f "${trained_3dgs_path}/chkpnt30000.pth" ]]; then
    echo "[run_scannet_all][warn] missing 3DGS checkpoint: ${trained_3dgs_path}/chkpnt30000.pth" >&2
    continue
  fi

  if [[ ! -f "${camera_path}/transforms_train.json" || \
        ! -f "${camera_path}/points3d.ply" || \
        ! -f "${camera_path}/points3d_with_label.ply" ]]; then
    echo "[run_scannet_all] preprocessing"
    python scripts/preprocess_scannet.py --scannet_root "${SCANNET_ROOT}" "${scene}"
  fi

  if ! find "${camera_path}/language_features" -maxdepth 1 -name '*_s.npy' -print -quit 2>/dev/null | grep -q .; then
    mkdir -p "${camera_path}/language_features"
    echo "[run_scannet_all] feature extraction"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python preprocessing.py \
      --dataset_path "${camera_path}" \
      --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth
  else
    echo "[run_scannet_all] reuse existing language features"
  fi

  if [[ ! -f "${model_path}/chkpnt0.pth" ]]; then
    echo "[run_scannet_all] Dr. Splat registration"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
      -s "${camera_path}" \
      -m "${output_path}" \
      --start_checkpoint "${trained_3dgs_path}/chkpnt30000.pth" \
      --feature_level "${FEATURE_LEVEL}" \
      --name_extra pq_openclip \
      --use_pq \
      --pq_index "${PQ_INDEX}" \
      --port "${PORT}" \
      --topk "${TOPK}" \
      --eval
  else
    echo "[run_scannet_all] reuse existing Dr. Splat output"
  fi

  if [[ "${RUN_VISUALIZATIONS}" == "1" ]]; then
    echo "[run_scannet_all] feature PCA visualization"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python render_pca.py \
      -s "${camera_path}" \
      -m "${model_path}" \
      --pq_index "${PQ_INDEX}" \
      --feature_level "${FEATURE_LEVEL}" \
      -l language_features_dim3

    echo "[run_scannet_all] activation visualization"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python render_activation.py \
      -s "${camera_path}" \
      -m "${model_path}" \
      --semantic_model sam \
      --feature_level "${FEATURE_LEVEL}" \
      --pq_index "${PQ_INDEX}" \
      --img_label "${IMG_LABEL}" \
      --img_save_label "${IMG_SAVE_LABEL}" \
      --threshold "${ACT_THRESHOLD}" \
      -l language_features_dim3
  fi

  if [[ ! -f "./labeled_gaussians/${scene}/labels.npy" ]]; then
    echo "[run_scannet_all] labeling gaussians"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python labeling_gaussians.py \
      -s "${camera_path}" \
      -m "${trained_3dgs_path}"
  else
    echo "[run_scannet_all] reuse existing Gaussian labels"
  fi

  echo "[run_scannet_all] ScanNet evaluation"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python eval_scannet_nonfix.py \
    -m "${model_path}" \
    --use_pq \
    --scene_name "${scene}" \
    --pq_index "${PQ_INDEX}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python eval_scannet_localization_nonfix.py \
    -m "${model_path}" \
    --use_pq \
    --scene_name "${scene}" \
    --pq_index "${PQ_INDEX}"
done < "${SCENE_LIST_FILE}"

echo
echo "[run_scannet_all] completed"
