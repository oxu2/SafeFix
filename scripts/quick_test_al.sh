#!/bin/bash
"""
Quick test script for Active Learning integration
Tests with ImageNet 13K dataset (1300 images × 10 classes)
"""

echo "Testing Active Learning Integration in alpred.py"
echo "Dataset: ImageNet 13K (1300 images per class × 10 classes)"
echo "================================================"

# Test 1: ImageNet with Active Learning (sub_ratio=0.5 means 6500 images)
echo "Test 1: ImageNet 13K with Active Learning (sub_ratio=0.5 = 6500 images)"
python alpred.py \
    --dataset imagenet \
    --model resnet \
    --use_active_learning True \
    --sub_ratio 0.5 \
    --al_initial_labeled 400 \
    --al_embedding_dim 256 \
    --num_epochs 5 \
    --batch_size 16 \
    --num_added_img 100 \
    --verbose True \
    --attributes test_al_13k

echo ""
echo "Test 1 completed. Check logs for Active Learning selection."

# Test 2: Compare with random sampling
echo ""
echo "Test 2: ImageNet 13K with Random Sampling (baseline, sub_ratio=0.5 = 6500 images)"
python alpred.py \
    --dataset imagenet \
    --model resnet \
    --use_active_learning False \
    --sub_ratio 0.5 \
    --num_epochs 5 \
    --batch_size 16 \
    --num_added_img 100 \
    --verbose True \
    --attributes test_random_13k

echo ""
echo "Both tests completed on ImageNet 13K dataset!"
echo "Active Learning should show better performance with the same 6500 selected images."
echo ""
echo "Dataset Scale:"
echo "- Total images: 13,000 (1300 per class × 10 classes)"
echo "- Training images selected: 6,500 (50% of total)"
echo "- AL initial labeled: 400 (manageable for 13K scale)"
echo "- Embedding dimension: 256 (optimized for ResNet)"
