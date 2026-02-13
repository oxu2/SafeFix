#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
PYTHON_EXEC="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

echo "Using Python: $PYTHON_EXEC"

echo "Testing Uncertainty Stratified Strategy..."
$PYTHON_EXEC -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 0 \
  --attributes pinkcolor_fabrictexture --num_epochs 1 \
  --sub_ratio 0.01 \
  --al_initial_labeled 20 \
  --al_strategy uncertainty_stratified \
  --verbose True > test_uncertainty.log 2>&1

if [ $? -eq 0 ]; then
    echo "Uncertainty Stratified Strategy: SUCCESS"
    tail -n 20 test_uncertainty.log
else
    echo "Uncertainty Stratified Strategy: FAILED"
    tail -n 20 test_uncertainty.log
    grep "ImportError" test_uncertainty.log
    grep "ModuleNotFoundError" test_uncertainty.log
fi

echo "------------------------------------------------"

echo "Testing Conditional Rarity Attribute Strategy..."
$PYTHON_EXEC -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 0 \
  --attributes pinkcolor_fabrictexture --num_epochs 1 \
  --sub_ratio 0.01 \
  --al_initial_labeled 20 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background \
  --al_conditional_rarity True \
  --verbose True > test_conditional.log 2>&1

if [ $? -eq 0 ]; then
    echo "Conditional Rarity Strategy: SUCCESS"
    tail -n 20 test_conditional.log
else
    echo "Conditional Rarity Strategy: FAILED"
    tail -n 20 test_conditional.log
    grep "ImportError" test_conditional.log
    grep "ModuleNotFoundError" test_conditional.log
fi
