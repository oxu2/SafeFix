# Active Learning with Core-set Strategy

This repository implements an Active Learning framework using the Core-set approach for image classification tasks, specifically designed for the CelebA dataset.

## Features

- **Multiple Sampling Strategies**: Core-set, Uncertainty, Diversity, Random
- **Model Support**: ResNet, CLIP
- **Dataset Integration**: CelebA dataset with your existing data structure
- **Visualization**: Training progress plots and sample selection analysis

## Quick Start

### 1. Basic Demo
Run a quick demonstration with small-scale parameters:

```bash
python demo_active_learning.py
```

### 2. Full Active Learning Training
Using the bash script (recommended):

```bash
# Make script executable
chmod +x run_active_learning.sh

# Run with default parameters (coreset strategy, resnet model, 5 rounds)
./run_active_learning.sh

# Run with custom parameters
./run_active_learning.sh coreset resnet 10
```

Using Python directly:

```bash
python active_learning_coreset.py \
    --model resnet \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --num-epochs 10 \
    --initial-labeled 1000 \
    --al-rounds 5 \
    --samples-per-round 500 \
    --device cuda
```

### 3. Strategy Comparison
Compare different Active Learning strategies:

```bash
python enhanced_active_learning.py
```

## Data Structure

The implementation follows your existing CelebA data structure:

```
data/celebA/
├── list_attr_celeba.txt
├── train_idx.npy
├── labels.npy
└── predictions.npy
```

Image directory paths:
- `/data/oxx220000/img_align_celeba/` (primary)
- `/home/jovyan/ouser/dataset/img_align_celeba/` (fallback)

## Core-set Strategy

Based on the paper:
> Sener, Ozan, and Silvio Savarese. "Active learning for convolutional neural networks: A core-set approach." arXiv preprint arXiv:1708.00489 (2017).

The Core-set strategy selects samples that are most representative of the unlabeled data distribution using a greedy k-center algorithm.

## Available Strategies

1. **Core-set**: Selects diverse and representative samples
2. **Uncertainty**: Selects samples with highest prediction uncertainty
3. **Diversity**: Selects diverse samples using clustering
4. **Random**: Random sampling (baseline)

## Models

1. **ResNet**: ResNet-50 backbone with linear classifier
2. **CLIP**: CLIP ViT-B/32 with linear classifier

## Output Files

- `al_results_{model}.png`: Training progress visualization
- `al_history_{model}.json`: Detailed training history
- `best_model.pth`: Best performing model
- `logs/active_learning/`: Training logs

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model type (resnet, clip) | resnet |
| `--batch-size` | Batch size | 32 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--num-epochs` | Epochs per round | 10 |
| `--initial-labeled` | Initial labeled samples | 1000 |
| `--al-rounds` | Active Learning rounds | 5 |
| `--samples-per-round` | Samples selected per round | 500 |
| `--device` | Device (cuda/cpu) | cuda |

## Data Statistics

Your current data split:
- Training set: 80,000 samples
- Validation set: 20,000 samples  
- Unlabeled set: 80,000 samples

## Example Usage

```python
from active_learning_coreset import ActiveLearningTrainer, load_celeba_data

# Load data
all_image_paths, all_labels, all_splits = load_celeba_data()

# Initialize trainer
trainer = ActiveLearningTrainer(
    model_type='resnet',
    num_classes=2,
    device='cuda'
)

# Run Active Learning
history = trainer.active_learning_loop(
    all_image_paths=all_image_paths,
    all_labels=all_labels,
    all_splits=all_splits,
    initial_labeled_size=1000,
    al_rounds=5,
    samples_per_round=500
)

# Plot results
trainer.plot_results('results.png')
```

## Requirements

- PyTorch
- torchvision
- transformers (for CLIP)
- scikit-learn
- matplotlib
- numpy
- PIL

## Expected Results

The Core-set strategy typically achieves:
- Better sample efficiency compared to random sampling
- More stable learning curves
- Improved final accuracy with fewer labeled samples

## Tips

1. Start with smaller parameters for testing
2. Monitor GPU memory usage with large batch sizes
3. Use CLIP for better feature representations
4. Compare multiple strategies to find the best for your specific task

## Troubleshooting

1. **Out of memory**: Reduce batch size or use CPU
2. **Data not found**: Check CelebA dataset paths
3. **Slow training**: Reduce epochs or use smaller dataset subset

## References

```bibtex
@article{sener2017active,
  title={Active learning for convolutional neural networks: A core-set approach},
  author={Sener, Ozan and Savarese, Silvio},
  journal={arXiv preprint arXiv:1708.00489},
  year={2017}
}
```
