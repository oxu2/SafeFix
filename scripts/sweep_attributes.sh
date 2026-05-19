#!/bin/bash

# =============================================================================
# Attribute Strategy Grid Search (Per-GPU Queue)
# Each GPU independently runs ALL scenarios for its assigned ratio.
# GPU 0 -> ratio 0.1, GPU 1 -> ratio 0.3, GPU 2 -> ratio 0.5, GPU 3 -> ratio 0.9
# =============================================================================

mkdir -p logs/imagenet

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable $PYTHON_EXE not found. Falling back to 'python3'."
    PYTHON_EXE="python3"
fi

BASE_MODEL="save/weights/model_imagenet_resnet_base.pth"
NUM_EPOCHS=15

if [ ! -f "$BASE_MODEL" ]; then
    echo "Error: Base model not found at $BASE_MODEL"
    exit 1
fi

COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs $NUM_EPOCHS --verbose False --resume_model $BASE_MODEL"

# Define scenarios: "Name R P C Entropy"
declare -a SCENARIOS=(
    "baseline_def 0.01 5.0 inf False"
    "strict_strat_entropy 0.01 5.0 inf True"
    "soft_conservative_e0 0.1 5.0 15.0 False"
    "soft_conservative_e1 0.1 5.0 15.0 True"
    "balanced_exp_p5_c8 0.3 5.0 8.0 True"
    "balanced_exp_p10_c8 0.3 10.0 8.0 True"
    "aggro_exp_p10_c4 0.8 10.0 4.0 True"
    "aggro_exp_p5_c4 0.8 5.0 4.0 True"
    "pure_greedy 1.0 10.0 0.0 False"
)

# Each GPU gets one ratio and runs ALL scenarios sequentially
run_gpu_queue() {
    local gpu=$1
    local ratio=$2
    local ratio_str="ratio${ratio}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: Starting queue for ratio=${ratio} (${#SCENARIOS[@]} scenarios)"

    for scenario in "${SCENARIOS[@]}"; do
        IFS=' ' read -r name r_val p_val c_val entropy_val <<< "$scenario"

        LOG_FILE="logs/imagenet/sweep_attr_${name}_${ratio_str}.log"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: ${name} (R=${r_val} P=${p_val} C=${c_val} E=${entropy_val}) -> ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=${gpu} $PYTHON_EXE -u alpred.py \
            $COMMON_ARGS \
            --sub_ratio ${ratio} \
            --al_strategy attribute \
            --al_attribute_focus color,background,texture \
            --al_attribute_rarity_power ${r_val} \
            --al_attribute_frequency_penalty ${p_val} \
            --al_attribute_class_balance ${c_val} \
            --al_attribute_use_entropy ${entropy_val} \
            > "${LOG_FILE}" 2>&1

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: ${name} done."
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: All scenarios for ratio=${ratio} COMPLETE."
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching 4 independent GPU queues..."

run_gpu_queue 0 0.1 &
run_gpu_queue 1 0.3 &
run_gpu_queue 2 0.5 &
run_gpu_queue 3 0.9 &

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Grid Search Complete! All 4 GPUs finished."
