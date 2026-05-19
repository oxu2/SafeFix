#!/bin/bash
# Run Conditional Rarity strategy for all ratios sequentially
export CUDA_VISIBLE_DEVICES=1
PYTHON_EXEC="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

echo "Starting Conditional Rarity Sweep (Class Balance: INF) on GPU $CUDA_VISIBLE_DEVICES"
echo "Using Python: $PYTHON_EXEC"

for RATIO in 0.1 0.3 0.5 0.7 0.9; do
    LOG_FILE="logs/imagenet/AL_resnet_condRarity_imagenet_${RATIO}_1000_classINF.log"
    echo "----------------------------------------------------------------"
    echo "Running ratio $RATIO... Logging to $LOG_FILE"
    
    $PYTHON_EXEC -u alpred.py \
      --dataset imagenet \
      --model resnet \
      --num_added_img 1000 \
      --attributes pinkcolor_fabrictexture \
      --num_epochs 30 \
      --sub_ratio $RATIO \
      --al_strategy attribute \
      --attribute_file save/attribute10.json \
      --al_attribute_focus color,background,texture \
      --al_attribute_log_acc_scale 1 \
      --al_conditional_rarity True \
      --al_attribute_class_balance inf \
      --verbose False \
      > "$LOG_FILE" 2>&1
      
    if [ $? -eq 0 ]; then
        echo "Finished ratio $RATIO (classINF) successfully."
    else
        echo "Error running ratio $RATIO. Check log: $LOG_FILE"
    fi
done

echo "All ratios completed."
