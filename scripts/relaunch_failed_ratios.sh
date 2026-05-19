#!/bin/bash
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 15 \
  --resume_model save/weights/model_imagenet_resnet_base.pth \
  --al_strategy attribute \
  --al_attribute_focus color,background,texture \
  --al_attribute_rarity_power 0.8 \
  --al_attribute_frequency_penalty 10.0 \
  --al_attribute_class_balance 4.0 \
  --al_attribute_use_entropy True"

# GPU 2: Qwen3 Ratios 0.5, 0.9
(
for ratio in 0.5 0.9; do
  echo "--- [GPU 2] Starting Qwen3 Ratio $ratio ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py $COMMON_ARGS \
    --sub_ratio $ratio \
    --attribute_file save/attribute10_qwen3_8B.json \
    > logs/imagenet/aggro_p10_c4_qwen3_ratio${ratio}.log 2>&1
done
) &

# GPU 3: InternVL3 Ratio 0.9 (0.5 already finished)
(
for ratio in 0.9; do
  echo "--- [GPU 3] Starting InternVL3 Ratio $ratio ---"
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py $COMMON_ARGS \
    --sub_ratio $ratio \
    --attribute_file save/attribute10_internvl3_8B.json \
    > logs/imagenet/aggro_p10_c4_internvl3_ratio${ratio}.log 2>&1
done
) &

wait
echo "All remaining runs completed."
