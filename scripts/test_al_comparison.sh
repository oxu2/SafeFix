#!/bin/bash

# Active Learning vs Random Selection Comparison Test
# Lightweight configuration for quick testing

echo "Starting Active Learning vs Random Selection comparison test..."

# 1. 轻量级 Active Learning 配置（1-2分钟完成）
echo "Step 1: Running Active Learning method..."
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 500 \
  --attributes pinkcolor_fabrictexture --num_epochs 15 \
  --al_initial_labeled 100 \
  --al_embedding_dim 64 \
  --sub_ratio 0.2 \
  --use_active_learning True \
  > logs/imagenet/AL_resnet_imagenet_balanced_500.log 2>&1

echo "Active Learning completed. Starting Random Selection..."

# 2. 对应的随机选择配置
echo "Step 2: Running Random Selection method..."
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 500 \
  --attributes pinkcolor_fabrictexture --num_epochs 15 \
  --al_initial_labeled 100 \
  --al_embedding_dim 64 \
  --sub_ratio 0.2 \
  --use_active_learning False \
  > logs/imagenet/AL_resnet_imagenet_random_500.log 2>&1

echo "Random Selection completed."

# 3. 再次运行 Active Learning 来触发对比分析
echo "Step 3: Running final Active Learning analysis with comparison..."
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 500 \
  --attributes pinkcolor_fabrictexture --num_epochs 15 \
  --al_initial_labeled 100 \
  --al_embedding_dim 64 \
  --sub_ratio 0.2 \
  --use_active_learning True \
  > logs/imagenet/AL_resnet_imagenet_balanced_500_final.log 2>&1

echo "Comparison test completed!"
echo "Check logs/imagenet/AL_resnet_imagenet_balanced_500_final.log for the comparison results."
echo ""
echo "To run the heavy configuration (if needed):"
echo "CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \\"
echo "  --dataset imagenet --model resnet --num_added_img 1000 \\"
echo "  --attributes pinkcolor_fabrictexture --num_epochs 30 \\"
echo "  --al_initial_labeled 400 --al_embedding_dim 256 --sub_ratio 0.5 \\"
echo "  --use_active_learning True \\"
echo "  > logs/imagenet/AL_resnet_imagenet_balanced_1000.log 2>&1 < /dev/null &"
echo ""
echo "And then run the random baseline:"
echo "CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \\"
echo "  --dataset imagenet --model resnet --num_added_img 1000 \\"
echo "  --attributes pinkcolor_fabrictexture --num_epochs 30 \\"
echo "  --al_initial_labeled 400 --al_embedding_dim 256 --sub_ratio 0.5 \\"
echo "  --use_active_learning False \\"
echo "  > logs/imagenet/AL_resnet_imagenet_random_1000.log 2>&1 < /dev/null &"
