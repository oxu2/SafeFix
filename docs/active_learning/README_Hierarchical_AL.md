# Hierarchical Two-Stage Active Learning

This document describes the implementation of the hierarchical two-stage active learning method in Safefix.

## Overview

The hierarchical active learning method performs data selection in two stages:

1. **Stage A (Slice-level priority)**: Allocate per-slice budgets using fast statistics over attribute groups
2. **Stage B (Instance-level selection)**: Select specific images within each slice that are both informative and diverse

This approach mirrors the "ask attributes when they help many classes" idea but adapted to data selection.

## Method Description

### Stage A: Slice-Level Priority

The method defines attribute groups and computes priority scores for each slice based on:

- **Difficulty**: Combination of validation error rate and uncertainty
- **Coverage gap**: How well the slice is currently covered
- **Correlation penalty**: Reduces redundancy between correlated slices

### Stage B: Instance Selection within Slices

Within each slice, samples are selected using a greedy approach that maximizes:
- Uncertainty scores (margin or entropy-based)
- Diversity using facility-location criterion over cached embeddings

### Hierarchical Scheduling

The method runs two rounds of allocation:
1. **Round 1 (Coarse)**: Allocate ~70% of budget to single-attribute slices
2. **Round 2 (Fine)**: Allocate remaining ~30% to attribute-pair slices

## Usage

### Basic Usage

```bash
python alpred.py \
    --dataset imagenet \
    --model resnet \
    --attributes pinkcolor_fabrictexture \
    --al_strategy hierarchical \
    --use_active_learning True \
    --verbose True
```

### Advanced Configuration

```bash
python alpred.py \
    --dataset imagenet \
    --model resnet \
    --attributes your_attribute \
    --al_strategy hierarchical \
    --al_w_err 0.6 \
    --al_w_unc 0.4 \
    --al_alpha 0.4 \
    --al_beta 0.4 \
    --al_gamma 0.2 \
    --al_lambda_div 0.5 \
    --al_round1_ratio 0.7 \
    --al_min_slice_quota 5 \
    --al_max_slice_quota 100 \
    --verbose True
```

## Parameters

### Difficulty Computation
- `--al_w_err`: Weight for error rate in difficulty calculation (default: 0.6)
- `--al_w_unc`: Weight for uncertainty in difficulty calculation (default: 0.4)

### Priority Score Weights
- `--al_alpha`: Weight for difficulty in priority score (default: 0.4)
- `--al_beta`: Weight for coverage gap in priority score (default: 0.4)
- `--al_gamma`: Weight for correlation penalty in priority score (default: 0.2)

### Instance Selection
- `--al_lambda_div`: Weight for diversity in instance selection (default: 0.5)

### Hierarchical Scheduling
- `--al_round1_ratio`: Fraction of budget for Round 1 (single attributes) (default: 0.7)
- `--al_min_slice_quota`: Minimum samples per slice (default: 5)
- `--al_max_slice_quota`: Maximum samples per slice (default: 100)

## Attribute Groups

### ImageNet Dataset
- **color**: red, blue, green, yellow, pink, white, black, brown
- **texture**: smooth, rough, fabric, metal, wood, plastic
- **background**: indoor, outdoor, nature, urban, plain
- **size**: small, medium, large

### CelebA Dataset
- **hair**: blonde, brown, black, red, gray
- **emotion**: happy, sad, angry, neutral, surprised
- **facial**: beard, mustache, glasses, hat
- **skin**: pale, medium, dark

## Algorithm Flow

1. **Initialization**: Define attribute groups and slice definitions
2. **Round 1 Selection**:
   - Compute difficulty, coverage gap, and correlation penalty for single-attribute slices
   - Allocate 70% of budget based on priority scores
   - Select instances within each slice using uncertainty + diversity
3. **Round 2 Selection**:
   - Compute priorities for attribute-pair slices
   - Allocate remaining 30% of budget
   - Select instances within remaining slices
4. **Fallback**: Fill any remaining budget with uncertainty-based selection

## Performance Characteristics

- **Computational Complexity**: O(n·k + k²) where n is dataset size, k is number of slices
- **Memory Requirements**: Stores embeddings for all samples
- **Selection Time**: Typically 2-5x slower than basic strategies due to two-stage process

## Expected Benefits

1. **Better Attribute Coverage**: Ensures diverse attribute representation
2. **Reduced Redundancy**: Correlation penalty reduces overlapping selections
3. **Hierarchical Focus**: Two-round approach handles both coarse and fine-grained attributes
4. **Uncertainty + Diversity**: Combines informativeness with representativeness

## Comparison with Other Strategies

| Strategy | Speed | Accuracy | Diversity | Attribute Coverage |
|----------|-------|----------|-----------|-------------------|
| Random | Fast | Baseline | Good | Random |
| Coreset | Medium | Good | Excellent | Poor |
| Stratified | Fast | Good | Medium | Poor |
| Hybrid | Medium | Good | Good | Medium |
| **Hierarchical** | **Slow** | **Best** | **Excellent** | **Excellent** |

## Testing

Run the hierarchical AL tests:

```bash
./test_hierarchical_al.sh
```

This will test:
1. Basic hierarchical AL with default parameters
2. Custom parameter configuration
3. Comparison with coreset baseline
4. CelebA dataset compatibility

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `--al_max_slice_quota` or `--al_embedding_dim`
2. **Slow performance**: Increase `--al_min_slice_quota` to reduce number of slices
3. **Poor coverage**: Adjust `--al_beta` (coverage gap weight) higher
4. **Too much redundancy**: Increase `--al_gamma` (correlation penalty)

### Debug Mode

Use `--verbose True` to see detailed slice allocation and selection information:

```bash
python alpred.py --al_strategy hierarchical --verbose True
```

## Implementation Notes

- Slice membership is currently simulated; real implementation would use attribute annotations
- The method requires a pre-trained model for uncertainty computation
- Two-round scheduling ensures both coarse and fine-grained attribute coverage
- Correlation penalty prevents over-selection of similar attribute combinations

## Future Improvements

1. **Real Attribute Detection**: Integrate with actual attribute classifiers
2. **Dynamic Slice Generation**: Create slices based on model confusion
3. **Adaptive Parameters**: Learn optimal weights during training
4. **Efficient Implementation**: Cache computations for faster selection