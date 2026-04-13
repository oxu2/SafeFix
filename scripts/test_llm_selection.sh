#!/bin/bash
"""
Test script for LLM-based data selection integration
Compares LLM selection strategies with traditional Active Learning and random sampling
"""

echo "=== Testing LLM-based Data Selection ==="
echo "Comparing LLM selection with Active Learning and random sampling"
echo "Dataset: ImageNet 13K subset (testing with small subset for speed)"

# Test parameters
DATASET="imagenet"
MODEL="resnet"
ATTRIBUTES="pinkcolor"
NUM_IMGS=200  # Small number for testing
EPOCHS=5      # Few epochs for testing
SUB_RATIO=0.3
SEED=42

echo ""
echo "Test 1: LLM Diversity-Difficulty Strategy"
echo "=========================================="
python alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_llm_selection True \
    --llm_model qwen \
    --llm_strategy diversity_difficulty \
    --sub_ratio $SUB_RATIO \
    --seed $SEED \
    --verbose True \
    --batch_size 16

echo ""
echo "Test 2: LLM Diversity Strategy"
echo "==============================="
python alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_llm_selection True \
    --llm_model qwen \
    --llm_strategy diversity \
    --sub_ratio $SUB_RATIO \
    --seed $SEED \
    --verbose True \
    --batch_size 16

echo ""
echo "Test 3: LLM Random Strategy (Baseline)"
echo "======================================="
python alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_llm_selection True \
    --llm_model qwen \
    --llm_strategy llm_random \
    --sub_ratio $SUB_RATIO \
    --seed $SEED \
    --verbose True \
    --batch_size 16

echo ""
echo "Test 4: Traditional Active Learning (Comparison)"
echo "================================================"
python alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_active_learning True \
    --al_strategy coreset \
    --sub_ratio $SUB_RATIO \
    --seed $SEED \
    --verbose True \
    --batch_size 16

echo ""
echo "Test 5: Pure Random Selection (Control)"
echo "========================================"
python alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_active_learning False \
    --sub_ratio $SUB_RATIO \
    --seed $SEED \
    --verbose True \
    --batch_size 16

echo ""
echo "=== All Tests Complete ==="
echo "Expected Order of Performance:"
echo "1. LLM Diversity-Difficulty (best)"
echo "2. Traditional Active Learning"
echo "3. LLM Diversity"
echo "4. LLM Random"
echo "5. Pure Random (worst)"
echo ""
echo "Check the final validation accuracies to compare methods."