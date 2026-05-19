# Active Learning with ImageNet Dataset

## Overview

This project implements an Active Learning system based on the Core-set strategy, supporting ImageNet dataset and multiple models (ResNet, CLIP).

## Features

- **Core-set Selection**: Core-set selection strategy based on Sener & Savarese (2017) paper
- **Multi-model Support**: ResNet50 and CLIP ViT-B/32
- **Dataset Support**: CelebA and ImageNet
- **GPU Auto Management**: Automatic GPU detection with CPU fallback
- **Progress Control**: Controllable output verbosity

## File Structure

```
├── active_learning_coreset.py      # Main Active Learning implementation
├── run_active_learning.sh          # Complete experiment script
├── run_quick_al.sh                 # Quick test script
├── results/                        # Results output directory
│   ├── active_learning/            # Complete experiment results
│   └── quick_test/                 # Quick test results
└── logs/                           # Log files
    ├── active_learning/
    └── quick_test/
```

## Usage

### 1. Quick Test (Recommended to run first)

```bash
./run_quick_al.sh
```

This will run:
- Quick test for ResNet and CLIP
- Small parameter settings (100 initial samples, 2 rounds, 3 epochs)
- Approximately 5-10 minutes to complete

### 2. Complete Experiment

```bash
./run_active_learning.sh
```

This will run:
- Baseline experiments for ResNet and CLIP
- Comparative experiments with different parameter combinations
- Complete testing with various configurations

### 3. Individual Run

```bash
# Basic usage
python active_learning_coreset.py --dataset imagenet --model resnet

# Custom parameters
python active_learning_coreset.py \
    --dataset imagenet \
    --model resnet \
    --verbose \
    --al-rounds 5 \
    --initial-labeled 1000 \
    --samples-per-round 500 \
    --num-epochs 10 \
    --batch-size 32

# CPU mode
python active_learning_coreset.py \
    --dataset imagenet \
    --model resnet \
    --device cpu \
    --batch-size 16
```

## Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | imagenet | Dataset selection (imagenet/celeba) |
| `--model` | resnet | Model type (resnet/clip) |
| `--al-rounds` | 5 | Active Learning rounds |
| `--initial-labeled` | 1000 | Initial labeled sample count |
| `--samples-per-round` | 500 | Samples selected per round |
| `--num-epochs` | 10 | Training epochs per round |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 1e-4 | Learning rate |
| `--device` | auto | Device selection (auto/cuda/cpu) |
| `--verbose` | False | Verbose output |
| `--quiet` | True | Quiet mode |

## Output Description

### Result Files

- `al_results_[model].png`: Accuracy curve plots
- `al_history_[model].json`: Detailed training history
- `logs/[experiment]_[timestamp].log`: Complete logs

### Performance Metrics

Each round outputs:
- Training accuracy
- Validation accuracy
- Number of selected samples
- Core-set selection time

## Technical Details

### Core-set Strategy

Based on greedy k-center algorithm:
1. Extract feature embeddings for all samples
2. Use greedy algorithm to select most representative samples
3. Maximize minimum distance between newly selected and already labeled samples

### GPU Memory Management

- Automatic GPU availability detection
- Automatic batch size reduction when memory insufficient
- Automatic CPU fallback on CUDA errors

### Model Architecture

- **ResNet**: Pre-trained ResNet50 + classification head
- **CLIP**: Pre-trained CLIP ViT-B/32 + classification head

## Common Issues

### 1. GPU Memory Insufficient

```bash
# Reduce batch size
python active_learning_coreset.py --batch-size 16

# Or use CPU
python active_learning_coreset.py --device cpu
```

### 2. Dataset Path Issues

Ensure ImageNet data exists at:
```
data/imagenet_subset/
├── class1/
├── class2/
└── ...
```

### 3. Dependency Issues

```bash
# Install necessary dependencies
pip install torch torchvision transformers scikit-learn matplotlib numpy pillow
```

## Example Experimental Results

After complete experiment, the following summary will be generated:

```
=== EXPERIMENT SUMMARY ===
resnet_baseline      | Rounds:  5 | Train: 85.20% | Val: 78.40%
clip_baseline        | Rounds:  5 | Train: 82.10% | Val: 76.80%
resnet_small_init    | Rounds:  5 | Train: 83.50% | Val: 77.20%
clip_small_init      | Rounds:  5 | Train: 81.30% | Val: 75.90%
==========================
```

## Paper Citation

If using this code, please cite the original paper:

```bibtex
@article{sener2017active,
  title={Active learning for convolutional neural networks: A core-set approach},
  author={Sener, Ozan and Savarese, Silvio},
  journal={arXiv preprint arXiv:1708.00489},
  year={2017}
}
```
