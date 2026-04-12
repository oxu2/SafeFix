#!/bin/bash
# Active Learning改进测试脚本

# 基础参数
DATASET="imagenet"
MODEL="resnet"
ATTRIBUTES="pinkcolor_fabrictexture"
NUM_IMGS=1000
EPOCHS=30
SEED=42

echo "=== Active Learning 改进测试 ==="
echo "测试不同策略对比随机选择的性能提升"

# 创建结果目录
mkdir -p logs_improvements
cd logs_improvements

# 1. 测试纯Core-set策略 (最有希望的改进)
echo "1. 测试纯Core-set策略..."
python ../alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_active_learning True \
    --al_strategy coreset \
    --al_initial_labeled 500 \
    --al_embedding_dim 512 \
    --seed $SEED \
    --verbose True \
    > pure_coreset_${ATTRIBUTES}_${NUM_IMGS}.log 2>&1

# 2. 测试改进的Hybrid策略 (60% core-set + 30% stratified + 10% random)
echo "2. 测试改进的Hybrid策略..."
python ../alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_active_learning True \
    --al_strategy hybrid \
    --al_coreset_ratio 0.6 \
    --al_stratified_ratio 0.3 \
    --al_initial_labeled 500 \
    --al_embedding_dim 512 \
    --seed $SEED \
    --verbose True \
    > improved_hybrid_${ATTRIBUTES}_${NUM_IMGS}.log 2>&1

# 3. 测试Uncertainty策略
echo "3. 测试Uncertainty策略..."
if [ -f "../enhanced_active_learning.py" ]; then
    python ../enhanced_active_learning.py \
        --strategy uncertainty \
        --initial-labeled 500 \
        --embedding-dim 512 \
        --num-epochs $EPOCHS \
        --seed $SEED \
        > uncertainty_${ATTRIBUTES}_${NUM_IMGS}.log 2>&1
else
    echo "enhanced_active_learning.py not found, skipping uncertainty test"
fi

# 4. 重新运行原始Hybrid策略作为对比
echo "4. 重新运行原始Hybrid策略作为对比..."
python ../alpred.py \
    --dataset $DATASET \
    --model $MODEL \
    --attributes $ATTRIBUTES \
    --num_added_img $NUM_IMGS \
    --num_epochs $EPOCHS \
    --use_active_learning True \
    --al_strategy hybrid \
    --al_coreset_ratio 0.3 \
    --al_stratified_ratio 0.5 \
    --al_initial_labeled 200 \
    --al_embedding_dim 256 \
    --seed $SEED \
    --verbose True \
    > original_hybrid_${ATTRIBUTES}_${NUM_IMGS}.log 2>&1

# 5. 提取并比较结果
echo "=== 结果比较 ==="
echo "策略 | 整体准确率 | 改进幅度"
echo "---|---|---"

# 提取各个实验的准确率
for log_file in *.log; do
    if [ -f "$log_file" ]; then
        strategy=$(echo $log_file | cut -d'_' -f1-2)
        accuracy=$(grep "Model Validation ACC:" $log_file | tail -1 | grep -o "0\.[0-9]*" | head -1)
        if [ ! -z "$accuracy" ]; then
            echo "$strategy | $accuracy | TBD"
        else
            echo "$strategy | 运行中或失败 | -"
        fi
    fi
done

echo ""
echo "详细日志文件："
ls -la *.log

echo ""
echo "比较基准 (来自原始日志)："
echo "- 随机选择: 0.666538"
echo "- 原始Hybrid AL: 0.647308"
echo ""
echo "如果新策略超过0.666538，则表示改进成功！"
