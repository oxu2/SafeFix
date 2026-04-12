#!/bin/bash

# =============================================================================
# Run ALL AL strategies for a SINGLE ratio on a SINGLE GPU.
# Usage: ./run_all_strategies_single_gpu.sh <GPU_ID> <SUB_RATIO>
# =============================================================================

GPU_ID=$1
SUB_RATIO=$2

if [ -z "$GPU_ID" ] || [ -z "$SUB_RATIO" ]; then
    echo "Usage: $0 <GPU_ID> <SUB_RATIO>"
    exit 1
fi

mkdir -p logs_imagenet

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="save/models/weights/model_imagenet_resnet_base.pth"
NUM_EPOCHS=15
RATIO_STR="ratio${SUB_RATIO}"
LOG_PREFIX="logs_imagenet/resume_lr5_epoch15"

COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs $NUM_EPOCHS --verbose False --resume_model $BASE_MODEL --sub_ratio $SUB_RATIO"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Starting ALL strategies for ratio=${SUB_RATIO}"

# 1. Random Baseline
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Random Baseline"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --use_active_learning False \
    > ${LOG_PREFIX}_random_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 2. Stratified
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Stratified"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy stratified \
    > ${LOG_PREFIX}_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 3. Coreset
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Coreset"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy coreset \
    --al_initial_labeled 400 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_coreset_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 4. Uncertainty Stratified
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Uncertainty Stratified"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy uncertainty_stratified \
    > ${LOG_PREFIX}_uncertainty_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 5. Diversity
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Diversity"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy diversity \
    --al_clusters 100 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_diversity_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 6. Hierarchical
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Hierarchical"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy hierarchical \
    --al_alpha 0.4 --al_beta 0.4 --al_gamma 0.2 \
    > ${LOG_PREFIX}_hierarchical_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 7. Hybrid
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Hybrid"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy hybrid \
    --al_coreset_ratio 0.3 \
    --al_stratified_ratio 0.5 \
    --al_initial_labeled 200 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_hybrid_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

# 8. Attribute (aggro_exp_p10_c4: R=0.8, P=10.0, C=4.0, Entropy=True)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: Attribute (aggro_exp_p10_c4)"
CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --al_strategy attribute \
    --al_attribute_focus color,background,texture \
    --al_attribute_rarity_power 0.8 \
    --al_attribute_frequency_penalty 10.0 \
    --al_attribute_class_balance 4.0 \
    --al_attribute_use_entropy True \
    > ${LOG_PREFIX}_attribute_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID}: ALL strategies DONE for ratio=${SUB_RATIO}"
