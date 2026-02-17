#!/bin/bash
PYTHON_EXEC="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

echo "Using Python: $PYTHON_EXEC"

echo "Starting Uncertainty Stratified Strategy on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_EXEC -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --sub_ratio 0.5 \
  --al_initial_labeled 400 \
  --al_strategy uncertainty_stratified \
  --verbose True > test_uncertainty.log 2>&1 &
PID1=$!

echo "Starting Conditional Rarity Attribute Strategy on GPU 3..."
CUDA_VISIBLE_DEVICES=3 $PYTHON_EXEC -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --sub_ratio 0.5 \
  --al_strategy attribute \
  --attribute_file save/attribute10.json \
  --al_attribute_focus color,background,texture \
  --al_attribute_log_acc_scale 1 \
  --al_conditional_rarity True \
  --verbose True > test_conditional.log 2>&1 &
PID2=$!

echo "Jobs started. Waiting for completion..."

wait $PID1
STATUS1=$?
if [ $STATUS1 -eq 0 ]; then
    echo "Uncertainty Stratified Strategy (GPU 2): SUCCESS"
    tail -n 10 test_uncertainty.log
else
    echo "Uncertainty Stratified Strategy (GPU 2): FAILED"
    tail -n 20 test_uncertainty.log
fi

wait $PID2
STATUS2=$?
if [ $STATUS2 -eq 0 ]; then
    echo "Conditional Rarity Strategy (GPU 3): SUCCESS"
    tail -n 10 test_conditional.log
else
    echo "Conditional Rarity Strategy (GPU 3): FAILED"
    tail -n 20 test_conditional.log
fi
