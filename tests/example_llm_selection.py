#!/usr/bin/env python3
"""
Example usage of LLM-based data selection in alpred.py
Demonstrates how to use Qwen/LLaVA for intelligent data selection
"""

def main():
    print("LLM-based Data Selection Examples for alpred.py")
    print("=" * 80)
    print("The following examples show how to use the new LLM-based data selection features")
    print()
    
    # Example 1: LLM Diversity-Difficulty Strategy for ImageNet
    example1_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model resnet \\
    --use_llm_selection True \\
    --llm_model qwen \\
    --llm_strategy diversity_difficulty \\
    --sub_ratio 0.5 \\
    --num_epochs 25 \\
    --batch_size 32 \\
    --attributes redhair \\
    --verbose True"""
    
    print("Example 1: LLM Diversity-Difficulty Strategy for ImageNet")
    print("-" * 60)
    print("Use Qwen model to select 50% of ImageNet training data based on visual diversity and difficulty")
    print()
    print(example1_cmd)
    print()
    
    # Example 2: LLM vs Active Learning Comparison
    example2a_cmd = """python alpred.py \\
    --dataset celeba \\
    --model clip \\
    --use_llm_selection True \\
    --llm_model qwen \\
    --llm_strategy diversity \\
    --sub_ratio 0.3 \\
    --attributes redhair_brownskin \\
    --verbose True"""
    
    example2b_cmd = """python alpred.py \\
    --dataset celeba \\
    --model clip \\
    --use_active_learning True \\
    --al_strategy coreset \\
    --sub_ratio 0.3 \\
    --attributes redhair_brownskin \\
    --verbose True"""
    
    print("Example 2: LLM vs Active Learning Comparison")
    print("-" * 60)
    print("Compare LLM-based selection with traditional Active Learning on CelebA")
    print()
    print("LLM Selection:")
    print(example2a_cmd)
    print()
    print("Active Learning:")
    print(example2b_cmd)
    print()
    
    # Example 3: Different LLM Models
    example3_cmd = """python alpred.py \\
    --dataset imagenet \\
    --model vit \\
    --use_llm_selection True \\
    --llm_model llava-7b \\
    --llm_strategy difficulty \\
    --sub_ratio 0.7 \\
    --attributes fabrictexture \\
    --verbose True"""
    
    print("Example 3: Using LLaVA Model for Selection")
    print("-" * 60)
    print("Use LLaVA-7B model to select challenging samples")
    print()
    print(example3_cmd)
    print()
    
    # Available strategies
    print("Available LLM Selection Strategies:")
    print("-" * 60)
    print("1. diversity:            Select most visually diverse samples")
    print("2. difficulty:           Select most challenging samples for training")
    print("3. diversity_difficulty: Combined approach (recommended)")
    print("4. llm_random:           Random selection using LLM pipeline (baseline)")
    print()
    
    # Available models
    print("Available LLM Models:")
    print("-" * 60)
    print("1. qwen:     Qwen2.5-VL-7B-Instruct (recommended)")
    print("2. llava-7b: LLaVA-1.5-7B")
    print("3. llava-13b: LLaVA-1.5-13B")
    print()
    
    # Performance expectations
    print("Expected Performance Benefits:")
    print("-" * 60)
    print("- LLM selection analyzes visual content intelligently")
    print("- Better than random selection, competitive with Active Learning")
    print("- Especially effective for diverse datasets with rich visual content")
    print("- diversity_difficulty strategy typically performs best")
    print()
    
    # Usage tips
    print("Usage Tips:")
    print("-" * 60)
    print("- Start with --llm_strategy diversity_difficulty and --llm_model qwen")
    print("- Use --verbose True to see selection process details")
    print("- LLM selection takes longer than AL but provides interpretable results")
    print("- Cannot combine --use_llm_selection and --use_active_learning")
    print("- For large datasets, analysis is done on a subset first for efficiency")
    print()

if __name__ == "__main__":
    main()