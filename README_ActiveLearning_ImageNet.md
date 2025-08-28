# Active Learning with ImageNet Dataset

## 概述

本项目实现了基于 Core-set 策略的 Active Learning 系统，支持 ImageNet 数据集和多种模型（ResNet、CLIP）。

## 功能特性

- **Core-set Selection**: 基于 Sener & Savarese (2017) 论文的核心集选择策略
- **多模型支持**: ResNet50 和 CLIP ViT-B/32
- **数据集支持**: CelebA 和 ImageNet
- **GPU 自动管理**: 自动检测 GPU 可用性，支持 CPU 降级
- **进度控制**: 可控制的输出详细程度

## 文件结构

```
├── active_learning_coreset.py      # 主要的 Active Learning 实现
├── run_active_learning.sh          # 完整实验脚本
├── run_quick_al.sh                 # 快速测试脚本
├── results/                        # 结果输出目录
│   ├── active_learning/            # 完整实验结果
│   └── quick_test/                 # 快速测试结果
└── logs/                           # 日志文件
    ├── active_learning/
    └── quick_test/
```

## 使用方法

### 1. 快速测试（推荐先运行）

```bash
./run_quick_al.sh
```

这会运行：
- ResNet 和 CLIP 的快速测试
- 小参数设置（100 初始样本，2轮，3个epoch）
- 约 5-10 分钟完成

### 2. 完整实验

```bash
./run_active_learning.sh
```

这会运行：
- ResNet 和 CLIP 的基准实验
- 不同参数组合的对比实验
- 包含多种配置的完整测试

### 3. 单独运行

```bash
# 基本用法
python active_learning_coreset.py --dataset imagenet --model resnet

# 自定义参数
python active_learning_coreset.py \
    --dataset imagenet \
    --model resnet \
    --verbose \
    --al-rounds 5 \
    --initial-labeled 1000 \
    --samples-per-round 500 \
    --num-epochs 10 \
    --batch-size 32

# CPU 模式
python active_learning_coreset.py \
    --dataset imagenet \
    --model resnet \
    --device cpu \
    --batch-size 16
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | imagenet | 数据集选择 (imagenet/celeba) |
| `--model` | resnet | 模型类型 (resnet/clip) |
| `--al-rounds` | 5 | Active Learning 轮数 |
| `--initial-labeled` | 1000 | 初始标记样本数 |
| `--samples-per-round` | 500 | 每轮选择的样本数 |
| `--num-epochs` | 10 | 每轮训练的 epoch 数 |
| `--batch-size` | 32 | 批次大小 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--device` | auto | 设备选择 (auto/cuda/cpu) |
| `--verbose` | False | 详细输出 |
| `--quiet` | True | 安静模式 |

## 输出说明

### 结果文件

- `al_results_[model].png`: 准确率曲线图
- `al_history_[model].json`: 详细训练历史
- `logs/[experiment]_[timestamp].log`: 完整日志

### 性能指标

每轮实验会输出：
- 训练准确率
- 验证准确率
- 选择的样本数量
- Core-set 选择时间

## 技术细节

### Core-set 策略

基于贪心 k-center 算法：
1. 提取所有样本的特征嵌入
2. 使用贪心算法选择最具代表性的样本
3. 最大化新选择样本与已标记样本的最小距离

### GPU 内存管理

- 自动检测 GPU 可用性
- 内存不足时自动降低批次大小
- CUDA 错误时自动切换到 CPU

### 模型架构

- **ResNet**: 预训练 ResNet50 + 分类头
- **CLIP**: 预训练 CLIP ViT-B/32 + 分类头

## 常见问题

### 1. GPU 内存不足

```bash
# 减小批次大小
python active_learning_coreset.py --batch-size 16

# 或使用 CPU
python active_learning_coreset.py --device cpu
```

### 2. 数据集路径问题

确保 ImageNet 数据存在于：
```
exampleData/imagenet_subset/
├── class1/
├── class2/
└── ...
```

### 3. 依赖问题

```bash
# 安装必要依赖
pip install torch torchvision transformers scikit-learn matplotlib numpy pillow
```

## 实验结果示例

完整实验完成后，会生成如下总结：

```
=== EXPERIMENT SUMMARY ===
resnet_baseline      | Rounds:  5 | Train: 85.20% | Val: 78.40%
clip_baseline        | Rounds:  5 | Train: 82.10% | Val: 76.80%
resnet_small_init    | Rounds:  5 | Train: 83.50% | Val: 77.20%
clip_small_init      | Rounds:  5 | Train: 81.30% | Val: 75.90%
==========================
```

## 论文引用

如果使用此代码，请引用原始论文：

```bibtex
@article{sener2017active,
  title={Active learning for convolutional neural networks: A core-set approach},
  author={Sener, Ozan and Savarese, Silvio},
  journal={arXiv preprint arXiv:1708.00489},
  year={2017}
}
```
