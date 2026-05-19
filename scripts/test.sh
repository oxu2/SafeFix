#!/usr/bin/env bash

set -euo pipefail

sub_ratios=(0.1 0.3 0.5 0.7 0.9)

run_slice() {
  local ratio="$1"

  CUDA_VISIBLE_DEVICES=0 nohup python -u alpred.py \
    --dataset imagenet --model resnet --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture --num_epochs 30 \
    --sub_ratio "$ratio" \
    --al_attribute_focus color,background,texture \
    --verbose False \
    --al_attribute_rarity_power 0 \
    --al_attribute_log_acc_scale 0 \
    --al_attribute_balance_strength 0 \
    --al_attribute_class_balance INF \
    > "logs/imagenet/AL_resnet_attr_imagenet_${ratio}_R0_L0_B0_class_INF_1000.log" 2>&1 < /dev/null &

  CUDA_VISIBLE_DEVICES=1 nohup python -u alpred.py \
    --dataset imagenet --model resnet --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture --num_epochs 30 \
    --sub_ratio "$ratio" \
    --al_attribute_focus color,background,texture \
    --verbose False \
    --al_attribute_rarity_power 1 \
    --al_attribute_log_acc_scale 0 \
    --al_attribute_balance_strength 0 \
    --al_attribute_class_balance INF \
    > "logs/imagenet/AL_resnet_attr_imagenet_${ratio}_R1_L0_B0_class_INF_1000.log" 2>&1 < /dev/null &

  CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \
    --dataset imagenet --model resnet --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture --num_epochs 30 \
    --sub_ratio "$ratio" \
    --al_attribute_focus color,background,texture \
    --verbose False \
    --al_attribute_rarity_power 0 \
    --al_attribute_log_acc_scale 1 \
    --al_attribute_balance_strength 0 \
    --al_attribute_class_balance INF \
    > "logs/imagenet/AL_resnet_attr_imagenet_${ratio}_R0_L1_B0_class_INF_1000.log" 2>&1 < /dev/null &

  CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \
    --dataset imagenet --model resnet --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture --num_epochs 30 \
    --sub_ratio "$ratio" \
    --al_attribute_focus color,background,texture \
    --verbose False \
    --al_attribute_rarity_power 0 \
    --al_attribute_log_acc_scale 0 \
    --al_attribute_balance_strength 5 \
    --al_attribute_class_balance INF \
    > "logs/imagenet/AL_resnet_attr_imagenet_${ratio}_R0_L0_B5_class_INF_1000.log" 2>&1 < /dev/null &

  wait
}

for ratio in "${sub_ratios[@]}"; do
  run_slice "$ratio"
done