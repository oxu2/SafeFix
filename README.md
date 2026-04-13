# Safefix

This repo is the implementation of Safefix. Safefix is a data-centric tool for investigating bugs in visual deep learning, such as rare cases in datasets and model correlations on visual features.

<!-- ![Workflow](save/plot/pictures/Safefix.png) -->

## Features

- Utilizes large language models and vision-language models for bug discovery and repair.
- Provides a human-understandable debugging process.
- Equipped with a user interface.
- Supports multiple model architectures: **ResNet-18**, **ViT**, and **CLIP**.
- Supports multiple datasets: **CelebA** (binary classification) and **ImageNet** (multi-class classification).
- Active Learning strategies for intelligent data selection (CoreSet, Stratified, Diversity, Hybrid, Hierarchical, Attribute-driven).
- LLM-based data selection as an alternative to traditional Active Learning (Qwen, LLaVA).
- ControlNet-based data augmentation for targeted bug repair.
- Automated model diagnostics and accuracy comparison.

## Installation

1. Clone the repository.
2. Set up the environment.

```bash
conda create -n Safefix python=3.7
conda activate Safefix
pip install -r requirements.txt
```

## Getting Started

1. Get a ChatGPT API key. (Optional)
2. Follow the steps in `run.ipynb`.

## Example

We provide a use case on the CelebA dataset in [celebA_use_case.ipynb](celebA_use_case.ipynb).

---

## `alpred.py` - Active Learning Prediction Pipeline

`alpred.py` is the main training and evaluation script. It integrates data loading, Active Learning / LLM-based data selection, model training, prediction saving, and model diagnostics into a single end-to-end pipeline.

### Quick Start

```bash
# CelebA with ResNet (default settings)
python alpred.py --dataset celeba --model resnet --attributes redhair_brownskin

# ImageNet with CLIP, pretrained, Active Learning enabled
python alpred.py --dataset imagenet --model clip --pretrained True --use_active_learning True

# CelebA with ViT, 1000 generated images, 30 epochs
python alpred.py --dataset celeba --model vit --num_added_img 1000 --num_epochs 30

# Use LLM-based selection instead of Active Learning
python alpred.py --dataset celeba --model resnet --use_llm_selection True --llm_model qwen --llm_strategy diversity_difficulty
```

### Command-Line Arguments

#### General

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `celeba` | Dataset to use: `celeba` or `imagenet` |
| `--model` | str | `resnet` | Model architecture: `resnet`, `vit`, or `clip` |
| `--attributes` | str | `redhair_brownskin` | Attributes to detect (e.g., `redhair_sademotion`, `brownskin`, `yellowhair`) |
| `--pretrained` | bool | `False` | Use pretrained weights (auto-enabled for CLIP/ViT on ImageNet) |
| `--verbose` | bool | `False` | Enable verbose output |

#### Training

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lr` | float | `1e-4` | Learning rate (auto-adjusted per model/dataset) |
| `--lr_head` | float | `None` | Learning rate for CLIP classification head (scratch mode) |
| `--lr_backbone` | float | `None` | Learning rate for CLIP backbone (scratch mode) |
| `--num_epochs` | int | `20` | Number of training epochs (auto-adjusted to 15 for ViT and CLIP on ImageNet) |
| `--batch_size` | int | `64` | Mini-batch size for training and validation |
| `--workers` | int | `4` | Number of DataLoader workers |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--gpu` | int | `0` | Index of the CUDA GPU to use |

#### Data Augmentation

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_added_img` | int | `1000` | Number of generated images to add |
| `--gen_image_dir` | str | `None` | Directory containing generated images (default: `exampleData/<attributes>`) |
| `--gen_only` | str | `false` | Generated-image usage mode: `false`, `train_only`, or `train_val_test` |
| `--sub_ratio` | float | `1.0` | Subsampling ratio (0-1) for original training data when combining with generated images |
| `--no_controlnet` | bool | `False` | Ablation: disable ControlNet-based augmentation |
| `--simple` | bool | `False` | Simple mode: disable ControlNet and attribute editing (original Safefix behavior) |
| `--skip_feasible` | bool | `False` | Skip loading feasible JSON for any attributes |

#### Feasible Images

| Argument | Type | Default | Description |
|---|---|---|---|
| `--feasible_file` | str | `None` | Path to the feasible images JSON file (default: `save/feasible_<attributes>.json`) |
| `--lvlm` | str | `qwen` | LVLM model name (e.g., `llava`) to modify feasible file naming |

#### Active Learning

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_active_learning` | bool | `True` | Enable Active Learning strategy for data selection |
| `--al_strategy` | str | `attribute` | Strategy: `coreset`, `stratified`, `diversity`, `hybrid`, `hierarchical`, or `attribute` |
| `--al_embedding_dim` | int | `512` | Embedding dimension for Active Learning |
| `--al_initial_labeled` | int | `400` | Initial number of labeled samples |
| `--al_combined` | bool | `False` | Apply AL on combined dataset (generated + original) |
| `--combined_ratio` | float | `0.7` | Ratio of samples to select from combined dataset |

##### Hybrid Strategy Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--al_coreset_ratio` | float | `0.4` | Ratio of samples selected by core-set in hybrid strategy |
| `--al_stratified_ratio` | float | `0.4` | Ratio selected by stratified sampling in hybrid strategy |

##### Diversity Strategy Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--al_clusters` | int | `None` | Number of clusters (default: `n_select // 10`) |

##### Hierarchical Strategy Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--al_w_err` | float | `0.6` | Weight for error rate in difficulty calculation |
| `--al_w_unc` | float | `0.4` | Weight for uncertainty in difficulty calculation |
| `--al_alpha` | float | `0.4` | Weight for difficulty in priority score |
| `--al_beta` | float | `0.4` | Weight for coverage gap in priority score |
| `--al_gamma` | float | `0.2` | Weight for correlation penalty in priority score |
| `--al_lambda_div` | float | `0.5` | Weight for diversity in instance selection |
| `--al_round1_ratio` | float | `0.7` | Fraction of budget for Round 1 |
| `--al_min_slice_quota` | int | `5` | Minimum samples per slice |
| `--al_max_slice_quota` | int | `100` | Maximum samples per slice |

##### Attribute-Driven Strategy Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--al_attribute_focus` | str | `None` | Comma-separated attribute keys to prioritize |
| `--al_attribute_rarity_power` | float | `1.0` | Exponent controlling emphasis on rare attribute values |
| `--al_attribute_uncertainty_boost` | float | `1.0` | Score bonus for uncertain samples in attribute metadata |
| `--al_attribute_log_path` | str | `None` | Path to attribute diagnostics log for Rare Case accuracies |
| `--al_attribute_log_acc_scale` | float | `1.0` | Scaling factor for log-derived failure weights |
| `--al_attribute_extra_counts` | str | `None` | Extra counts per attribute value (JSON path or inline entries) |
| `--al_attribute_balance_strength` | float | `0.5` | Penalty strength for over-represented attribute values |
| `--al_attribute_class_balance` | str | `0` | Class-level stratified weight (`"inf"` for hard per-class quotas) |

#### LLM-Based Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_llm_selection` | bool | `False` | Use LLM-based data selection instead of AL |
| `--llm_model` | str | `qwen` | LLM model: `qwen`, `llava-7b`, or `llava-13b` |
| `--llm_strategy` | str | `diversity_difficulty` | Selection strategy: `diversity`, `difficulty`, `diversity_difficulty`, or `llm_random` |
| `--llm_max_memory` | int | `23` | Maximum memory per GPU in GB for LLM model |

#### Output and Diagnostics

| Argument | Type | Default | Description |
|---|---|---|---|
| `--plot` | bool | `False` | Enable plotting training metrics (saved to `save/plot/`) |
| `--save_report` | bool | `False` | Generate and save diagnostic report (saved to `save/reports/`) |
| `--save_pred_model` | str | `best` | Which model to use for saving predictions: `best` or `last` |
| `--attribute_file` | str | `save/attribute10.json` | Path to the attribute JSON file for diagnostics |
| `--pad_missing` | bool | `True` | Pad missing predictions with -1 when subsampling |
| `--skip_train_unlabeled` | bool | `False` | Skip processing TRAIN and UNLABELED splits in diagnostics |

### Pipeline Overview

```
1. Parse arguments and set random seed
       |
2. Load dataset (CelebA or ImageNet)
       |
3. Load generated/augmented images
       |
4. Data selection (Active Learning / LLM-based / Random)
       |
5. Combine original + generated data into training set
       |
6. Define model (ResNet-18 / ViT / CLIP), loss, optimizer, scheduler
       |
7. Train model with validation tracking (best model saved)
       |
8. Evaluate on held-out test set (if available)
       |
9. Save predictions to save/pred/
       |
10. Run model diagnostics (failure-by-label, prediction correlation, etc.)
       |
11. Compare accuracy against baseline log files
       |
12. Compare Active Learning vs Random Selection results
```

### Automatic Parameter Adjustments

`alpred.py` automatically adjusts several parameters based on the model and dataset:

| Condition | Adjustment |
|---|---|
| ViT model | Epochs: 20 -> 15, LR: 1e-4 -> 5e-5 |
| CLIP on ImageNet | Epochs: 20 -> 15 |
| Pretrained ViT on ImageNet | LR -> 2e-5 |
| ResNet on ImageNet | LR: 1e-4 -> 1e-3 |
| Pretrained CLIP on ImageNet | LR -> 1e-5 |
| Scratch CLIP on ImageNet | Head LR: 5e-3, Backbone LR: 5e-4 |
| CLIP/ViT on CelebA | LR -> 1e-5 |
| ImageNet dataset | Attribute file -> `save/attribute10.json` |
| CLIP/ViT on ImageNet (pretrained) | Auto-enable pretrained weights |
| `emotion` in attributes | Auto-enable `skip_feasible` |

### Model Architectures

| Model | CelebA (Binary) | ImageNet (Multi-class) |
|---|---|---|
| **ResNet-18** | Simplified (layers 3-4 removed), Dropout 0.7, Linear(128, 1) | Full ResNet-18, Linear(512, num_classes) |
| **ViT** | `vit_base_patch16_224`, Dropout 0.7, Linear(768, 1) | `vit_base_patch16_224`, Dropout 0.5, Linear(768, num_classes) |
| **CLIP** | CLIPForBinary, Dropout 0.5, output=1 | CLIPForBinary, Dropout 0.5, output=num_classes |

### Data Selection Modes

`alpred.py` supports three data selection strategies:

1. **Active Learning** (`--use_active_learning True`): Uses the specified `--al_strategy` to intelligently select training samples. Available strategies:
   - `attribute`: Attribute-driven selection prioritizing rare attribute values
   - `coreset`: Core-set selection for maximum coverage
   - `stratified`: Stratified sampling across classes
   - `diversity`: Cluster-based diversity sampling
   - `hybrid`: Combines coreset + stratified + diversity
   - `hierarchical`: Two-round selection with difficulty and coverage prioritization

2. **LLM-Based Selection** (`--use_llm_selection True`): Uses a vision-language model (Qwen or LLaVA) to analyze images and select diverse/difficult samples. Strategies: `diversity`, `difficulty`, `diversity_difficulty`, `llm_random`.

3. **Random Sampling** (default when both AL and LLM are disabled with `sub_ratio < 1.0`): Random subsampling from the original training set.

> Note: If both `--use_active_learning` and `--use_llm_selection` are enabled, LLM selection takes priority.

### Generated Image Modes (`--gen_only`)

| Mode | Description |
|---|---|
| `false` (default) | Standard mode: combine original training data with generated images |
| `train_only` | Use only generated images, split 8:1 for train/validation |
| `train_val_test` | Use only generated images, split 8:1:1 for train/validation/test |

### Output Files

| File Pattern | Description |
|---|---|
| `save/pred/predictions_<dataset>_<model>_<attr>_<n>imgs[_g][_h]_<timestamp>.npy` | Saved predictions |
| `save/plot/plot_<model>_<attr>_<n>imgs_<timestamp>.png` | Training metrics plot (if `--plot True`) |
| `save/reports/<dataset>_<model>_<attr>_<n>imgs_<timestamp>/` | Diagnostic report (if `--save_report True`) |
| `save/idx/train_idx_<attr>_<n>imgs.npy` | Combined training indices |
| `save/labels/labels_<attr>_<n>imgs.npy` | Combined labels |

Prediction file suffixes:
- `_g`: generated-only mode
- `_h`: ablation / no-controlnet mode

### Diagnostics

After training, `alpred.py` runs model diagnostics using `ModelDiagnose`:

- **`detect_failure_by_label`**: Identifies labels associated with high failure rates (CelebA only).
- **`detect_prediction_correlation`**: Finds correlations between predictions and attributes.
- **`detect_failure_prediction_correlation`**: Discovers correlations between model failures and attribute combinations (CelebA only).

Diagnostic thresholds:
```python
acc_diff_threshold = 0.01
distribution_diff_threshold = 0.3
dominant_label_in_failure_threshold = 0.8
dominant_label_in_prediction_threshold = 0.8
rare_label_threshold = 0.93
dominant_error_threshold = 0.3
error_cover_threshold = 0.9
top_error = 3
```

### Example Commands

```bash
# Basic CelebA training with default settings
python alpred.py

# ImageNet with ViT, pretrained, attribute-driven Active Learning
python alpred.py --dataset imagenet --model vit --pretrained True \
    --al_strategy attribute --num_added_img 2000

# CelebA with CLIP, LLM-based selection using Qwen
python alpred.py --dataset celeba --model clip \
    --use_active_learning False --use_llm_selection True \
    --llm_model qwen --llm_strategy diversity_difficulty

# Ablation: no ControlNet, simple mode
python alpred.py --no_controlnet True --simple True --attributes redhair_brownskin

# Generated-only mode (train+val+test from generated images)
python alpred.py --gen_only train_val_test --num_added_img 5000

# Combined Active Learning on original + generated data
python alpred.py --al_combined True --combined_ratio 0.7 --al_strategy coreset

# Full pipeline with diagnostics and plotting
python alpred.py --dataset celeba --model resnet --attributes redhair_brownskin \
    --num_added_img 1000 --num_epochs 20 --plot True --save_report True --verbose True

# Sub-sampling original data with hierarchical AL
python alpred.py --sub_ratio 0.3 --al_strategy hierarchical \
    --al_alpha 0.4 --al_beta 0.4 --al_gamma 0.2
```

### Project Structure

```
Safefix/
├── alpred.py                    # Main training & evaluation pipeline
├── train.py                     # Standalone training script
├── predict.py                   # Prediction scripts
├── controlnet.py                # ControlNet integration
├── instruct_pix2pix.py          # InstructPix2Pix image editing
├── config.py                    # Global configuration
│
├── utils/
│   ├── imports.py               # Centralized imports
│   ├── defines.py               # Constants (TRAIN, VALID, TEST, UNLABELED, etc.)
│   ├── models.py                # CLIPForBinary, SmoothBCEWithLogitsLoss
│   ├── training.py              # Training and evaluation utilities
│   ├── data_utils.py            # Dataset classes and transforms
│   ├── active_learning.py       # Active Learning strategies
│   ├── hierarchical_al.py       # Hierarchical Active Learning
│   ├── llm_selection.py         # LLM-based data selection
│   ├── qwen.py                  # Qwen VLM integration
│   ├── gpt.py                   # ChatGPT chatbot for attribute generation
│   └── ...
│
├── processData/
│   ├── Diagnose.py              # Model diagnostics / error analysis
│   ├── AttributeSelection.py    # Attribute selection logic
│   ├── Corpus.py                # Corpus management
│   └── report.py                # Error report generation
│
├── cldm/                        # ControlNet diffusion model
├── ldm/                         # Latent diffusion model
├── annotator/                   # Image annotation tools
│
├── exampleData/                 # Example datasets (CelebA, ImageNet subsets)
├── save/                        # Saved artifacts (predictions, indices, plots, reports)
├── logs/                        # Experiment logs
└── misc/                        # Additional documentation
```

## Citing

If you find this repository useful for your work, please consider citing it as follows:


