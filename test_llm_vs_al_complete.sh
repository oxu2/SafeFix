#!/bin/bash
"""
Complete comparison script for LLM selection vs Active Learning vs Random
Tests different data selection strategies on ImageNet dataset
"""

# Common parameters
DATASET="imagenet"
MODEL="resnet"
ATTRIBUTES="pinkcolor_fabrictexture"
NUM_IMGS=1000
EPOCHS=30
SUB_RATIO=0.5
SEED=42

echo "=== Comprehensive Data Selection Comparison ==="
echo "Dataset: $DATASET, Model: $MODEL, Attributes: $ATTRIBUTES"
echo "Images: $NUM_IMGS, Epochs: $EPOCHS, Sub-ratio: $SUB_RATIO"
echo ""

# Test 1: LLM Diversity-Difficulty Strategy (Recommended)
echo "1. LLM Diversity-Difficulty Strategy (Qwen)"
CUDA_VISIBLE_DEVICES=0 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_llm_selection True \
  --llm_model qwen \
  --llm_strategy diversity_difficulty \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/LLM_qwen_diversity_difficulty_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

# Test 2: LLM Diversity Strategy  
echo "2. LLM Diversity Strategy (Qwen)"
CUDA_VISIBLE_DEVICES=1 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_llm_selection True \
  --llm_model qwen \
  --llm_strategy diversity \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/LLM_qwen_diversity_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

# Test 3: LLM Difficulty Strategy
echo "3. LLM Difficulty Strategy (Qwen)"  
CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_llm_selection True \
  --llm_model qwen \
  --llm_strategy difficulty \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/LLM_qwen_difficulty_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

# Test 4: Traditional Active Learning (Core-set)
echo "4. Traditional Active Learning (Core-set)"
CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_active_learning True \
  --al_strategy coreset \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/AL_coreset_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

echo ""
echo "Waiting for first batch to complete..."
wait

# Test 5: LLM Random Strategy (LLM Baseline)
echo "5. LLM Random Strategy (LLM Baseline)"
CUDA_VISIBLE_DEVICES=0 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_llm_selection True \
  --llm_model qwen \
  --llm_strategy llm_random \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/LLM_qwen_random_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

# Test 6: Pure Random Selection (Control)
echo "6. Pure Random Selection (Control)"
CUDA_VISIBLE_DEVICES=1 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_active_learning False \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/Random_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

# Test 7: LLaVA Model for comparison
echo "7. LLM with LLaVA-7B Model"
CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \
  --dataset $DATASET --model $MODEL --num_added_img $NUM_IMGS \
  --attributes $ATTRIBUTES --num_epochs $EPOCHS \
  --use_llm_selection True \
  --llm_model llava-7b \
  --llm_strategy diversity_difficulty \
  --sub_ratio $SUB_RATIO \
  --seed $SEED \
  --verbose True \
  > logs_imagenet/LLM_llava_diversity_difficulty_${MODEL}_${DATASET}_${NUM_IMGS}.log 2>&1 < /dev/null &

echo ""
echo "All experiments started!"
echo ""
echo "Expected Performance Ranking:"
echo "1. LLM Diversity-Difficulty (best)"
echo "2. Traditional Active Learning"  
echo "3. LLM Diversity"
echo "4. LLM Difficulty"
echo "5. LLM Random"
echo "6. Pure Random (worst)"
echo ""
echo "Monitor logs with:"
echo "tail -f logs_imagenet/LLM_qwen_diversity_difficulty_${MODEL}_${DATASET}_${NUM_IMGS}.log"