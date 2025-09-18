#!/usr/bin/env python3
"""
Example script demonstrating Active Learning integration in alpred.py
This shows how to use the new Active Learning Core-set strategy
"""

import subprocess
import sys
import os

def run_example(example_name, command, description):
    """Run an example and show the command"""
    print(f"\n{'='*80}")
    print(f"Example: {example_name}")
    print(f"Description: {description}")
    print(f"Command: {command}")
    print('='*80)
    
    print("To run this example, execute:")
    print(f"  {command}")
    print()

def main():
    print("Active Learning Integration Examples for alpred.py")
    print("=" * 80)
    print("The following examples show how to use the new Active Learning features")
    print()
    
    # Example 1: Optimized Active Learning for ImageNet 13K dataset
    example1_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model resnet \\
    --use_active_learning True \\
    --sub_ratio 0.5 \\
    --al_initial_labeled 400 \\
    --al_embedding_dim 256 \\
    --num_epochs 25 \\
    --batch_size 32 \\
    --attributes redhair \\
    --verbose True"""
    
    run_example(
        "Optimized Active Learning for ImageNet 13K",
        example1_cmd,
        "Use Active Learning to select 50% (6500) of ImageNet training data (13K total) using Core-set strategy"
    )
    
    # Example 2: Compare different sub_ratio values for ImageNet 13K
    example2a_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model resnet \\
    --use_active_learning True \\
    --sub_ratio 0.3 \\
    --al_initial_labeled 300 \\
    --al_embedding_dim 256 \\
    --num_epochs 25 \\
    --attributes redhair_brownskin \\
    --verbose True"""
    
    example2b_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model resnet \\
    --use_active_learning False \\
    --sub_ratio 0.3 \\
    --num_epochs 25 \\
    --attributes redhair_brownskin \\
    --verbose True"""
    
    run_example(
        "Active Learning vs Random (30% of 13K = 3900 samples)",
        example2a_cmd,
        "Train with Active Learning selection (30% of ImageNet 13K)"
    )
    
    run_example(
        "Random Baseline (30% of 13K = 3900 samples)",
        example2b_cmd,
        "Train with random selection (30% of ImageNet 13K) for comparison"
    )
    
    # Example 3: CLIP model with Active Learning
    example3_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model clip \\
    --use_active_learning True \\
    --sub_ratio 0.4 \\
    --al_initial_labeled 800 \\
    --al_embedding_dim 512 \\
    --num_epochs 15 \\
    --batch_size 16 \\
    --attributes yellowhair \\
    --pretrained True \\
    --verbose True"""
    
    run_example(
        "CLIP with Active Learning",
        example3_cmd,
        "Use CLIP model with Active Learning for better feature representation"
    )
    
    # Example 4: CelebA with Active Learning
    example4_cmd = """python alpred.py \\
    --dataset celeba \\
    --model resnet \\
    --use_active_learning True \\
    --sub_ratio 0.5 \\
    --al_initial_labeled 1000 \\
    --al_embedding_dim 256 \\
    --num_epochs 20 \\
    --attributes redhair_brownskin \\
    --num_added_img 2000 \\
    --verbose True"""
    
    run_example(
        "CelebA with Active Learning",
        example4_cmd,
        "Apply Active Learning to CelebA dataset with generated images"
    )
    
    # Example 5: Different embedding dimensions
    example5_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model vit \\
    --use_active_learning True \\
    --sub_ratio 0.6 \\
    --al_initial_labeled 1200 \\
    --al_embedding_dim 768 \\
    --num_epochs 15 \\
    --batch_size 24 \\
    --attributes brownskin \\
    --pretrained True \\
    --verbose True"""
    
    run_example(
        "ViT with Higher Embedding Dimension",
        example5_cmd,
        "Use ViT model with higher embedding dimension for Active Learning"
    )
    
    print("\n" + "="*80)
    print("KEY PARAMETERS FOR ACTIVE LEARNING:")
    print("="*80)
    print("--use_active_learning True/False  : Enable/disable Active Learning")
    print("--sub_ratio 0.0-1.0              : Fraction of original data to select")
    print("--al_initial_labeled N           : Initial labeled samples for Core-set")
    print("--al_embedding_dim N             : Embedding dimension for similarity")
    print()
    print("BENEFITS:")
    print("- More informative sample selection compared to random sampling")
    print("- Better performance with limited training data")
    print("- Especially effective when sub_ratio < 1.0")
    print()
    print("WHEN TO USE:")
    print("- When you want to train with a subset of data (sub_ratio < 1.0)")
    print("- When computational resources are limited")
    print("- When you want to improve model performance with same amount of data")
    print()

if __name__ == "__main__":
    main()
