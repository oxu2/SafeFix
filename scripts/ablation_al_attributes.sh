#!/bin/bash

# Ablation study for attribute-driven AL parameters on ImageNet
# GPU 1: alpha=1 (rarity_power), beta=0 (uncertainty_boost), gamma=0 (log_acc_scale)
CUDA_VISIBLE_DEVICES=1 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio 0.7 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background \
  --al_attribute_rarity_power 1 \
  --al_attribute_uncertainty_boost 0 \
  --al_attribute_log_acc_scale 0 \
  > logs/imagenet/AL_resnet_imagenet_attr_0.7_alpha1_beta0_gamma0_1000.log 2>&1 < /dev/null &

# GPU 2: alpha=0, beta=1, gamma=0
CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio 0.7 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background \
  --al_attribute_rarity_power 0 \
  --al_attribute_uncertainty_boost 1 \
  --al_attribute_log_acc_scale 0 \
  > logs/imagenet/AL_resnet_imagenet_attr_0.7_alpha0_beta1_gamma0_1000.log 2>&1 < /dev/null &

# GPU 3: alpha=0, beta=0, gamma=1
CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio 0.7 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background \
  --al_attribute_rarity_power 0 \
  --al_attribute_uncertainty_boost 0 \
  --al_attribute_log_acc_scale 1 \
  > logs/imagenet/AL_resnet_imagenet_attr_0.7_alpha0_beta0_gamma1_1000.log 2>&1 < /dev/null &

# GPU 0: alpha=0, beta=0, gamma=0
CUDA_VISIBLE_DEVICES=0 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio 0.7 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background \
  --al_attribute_rarity_power 0 \
  --al_attribute_uncertainty_boost 0 \
  --al_attribute_log_acc_scale 0 \
  > logs/imagenet/AL_resnet_imagenet_attr_0.7_alpha0_beta0_gamma0_1000.log 2>&1 < /dev/null &