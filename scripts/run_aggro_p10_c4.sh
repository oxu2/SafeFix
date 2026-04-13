#!/bin/bash
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

# Using optimized aggro_exp_p10_c4 parameters (R=0.8, P=10.0, C=4.0, Entropy=True)
# Removed --verbose as requested
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 15 \
  --resume_model save/models/weights/model_imagenet_resnet_base.pth \
  --al_strategy attribute \
  --al_attribute_focus color,background,texture \
  --al_attribute_rarity_power 0.8 \
  --al_attribute_frequency_penalty 10.0 \
  --al_attribute_class_balance 4.0 \
  --al_attribute_use_entropy True"

RATIOS=(0.1 0.3 0.5 0.9)

# GPU 2: Qwen3
(
for ratio in "${RATIOS[@]}"; do
  echo "--- [GPU 2] Starting Qwen3 Ratio $ratio ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py $COMMON_ARGS \
    --sub_ratio $ratio \
    --attribute_file save/attribute10_qwen3_8B.json \
    > logs_imagenet/aggro_p10_c4_qwen3_ratio${ratio}.log 2>&1
done
echo "=== [GPU 2] Qwen3 Runs Finished ==="
) &

# GPU 3: InternVL3
(
for ratio in "${RATIOS[@]}"; do
  echo "--- [GPU 3] Starting InternVL3 Ratio $ratio ---"
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py $COMMON_ARGS \
    --sub_ratio $ratio \
    --attribute_file save/attribute10_internvl3_8B.json \
    > logs_imagenet/aggro_p10_c4_internvl3_ratio${ratio}.log 2>&1
done
echo "=== [GPU 3] InternVL3 Runs Finished ==="
) &

wait
echo "All runs completed."
