# LLM-based Data Selection for Safefix

This document describes the new LLM-based data selection feature that can replace traditional Active Learning strategies in `alpred.py`.

## Overview

Instead of using mathematical approaches like coreset or uncertainty sampling, the LLM-based selection uses vision-language models (Qwen/LLaVA) to intelligently analyze images and select the most informative samples for training.

## Key Features

- **Visual Content Analysis**: Uses LLM to understand image diversity, difficulty, and training value
- **Multiple Strategies**: Choose from diversity, difficulty, or combined approaches
- **Multiple Models**: Support for Qwen and LLaVA models
- **Interpretable Results**: LLM provides explanations for why samples were selected
- **Maintains sub_ratio**: Works with the existing subsampling ratio functionality

## Usage

### Basic Usage

Replace Active Learning with LLM selection:

```bash
# Traditional Active Learning (old way)
python alpred.py --use_active_learning True --al_strategy coreset --sub_ratio 0.5

# LLM-based selection (new way)
python alpred.py --use_llm_selection True --llm_strategy diversity_difficulty --sub_ratio 0.5
```

### Available Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--use_llm_selection` | True/False | False | Enable LLM-based data selection |
| `--llm_model` | qwen, llava-7b, llava-13b | qwen | LLM model to use |
| `--llm_strategy` | diversity, difficulty, diversity_difficulty, llm_random | diversity_difficulty | Selection strategy |
| `--llm_max_memory` | int | 23 | Max memory per GPU in GB |

### Selection Strategies

1. **diversity**: Selects visually diverse samples (varied viewpoints, backgrounds, lighting)
2. **difficulty**: Selects challenging samples (blurred, occluded, unusual angles)
3. **diversity_difficulty**: Combined approach (recommended, 60% diversity + 40% difficulty)
4. **llm_random**: Random selection using LLM pipeline (baseline for comparison)

### Examples

#### ImageNet with Qwen Model
```bash
python alpred.py \
    --dataset imagenet \
    --model resnet \
    --use_llm_selection True \
    --llm_model qwen \
    --llm_strategy diversity_difficulty \
    --sub_ratio 0.5 \
    --attributes pinkcolor \
    --verbose True
```

#### CelebA with LLaVA Model
```bash
python alpred.py \
    --dataset celeba \
    --model clip \
    --use_llm_selection True \
    --llm_model llava-7b \
    --llm_strategy diversity \
    --sub_ratio 0.3 \
    --attributes redhair_brownskin \
    --verbose True
```

## How It Works

1. **Subset Analysis**: For efficiency, analyzes 3x the target number of samples
2. **Visual Analysis**: LLM rates each image on 6 aspects (diversity/difficulty)
3. **Scoring**: Combines scores based on selected strategy
4. **Selection**: Chooses top-ranked samples maintaining sub_ratio
5. **Class Balance**: Attempts to maintain reasonable class distribution

### Analysis Dimensions

**For CelebA (Faces):**
- Facial features distinctiveness
- Hair style/color uniqueness  
- Skin tone variety
- Clothing/accessories
- Background complexity
- Overall visual uniqueness

**For ImageNet (Objects):**
- Viewpoint/angle variety
- Lighting conditions
- Background complexity
- Object state/condition
- Color/texture diversity
- Training informativeness

## Performance Comparison

Based on testing, typical performance ranking:

1. **LLM Diversity-Difficulty** (best)
2. **Traditional Active Learning**
3. **LLM Diversity**
4. **LLM Difficulty**
5. **LLM Random**
6. **Pure Random** (worst)

## Advantages

- **Interpretable**: LLM explains why samples were selected
- **Content-Aware**: Understands visual semantics, not just embeddings
- **Flexible**: Easy to modify selection criteria
- **No Training Required**: Uses pretrained vision-language models

## Limitations

- **Slower**: LLM inference takes longer than mathematical AL methods
- **Memory**: Requires GPU memory for large vision-language models
- **Approximation**: Analyzes subset for efficiency on large datasets
- **Model Dependency**: Quality depends on chosen LLM model

## Testing

Run the test script to compare all methods:

```bash
./test_llm_selection.sh
```

See example usage:

```bash
python example_llm_selection.py
```

## Implementation Details

- Located in `utils/llm_selection.py`
- Integrated into `alpred.py` alongside existing AL methods
- Uses same inference pipeline as `filter.py`
- Maintains compatibility with existing arguments and workflows

## Migration from Active Learning

To switch from Active Learning to LLM selection:

```bash
# Old command
python alpred.py --use_active_learning True --al_strategy coreset --sub_ratio 0.3

# New command
python alpred.py --use_llm_selection True --llm_strategy diversity_difficulty --sub_ratio 0.3
```

The `sub_ratio` parameter works identically - it controls what fraction of the original training data to select.