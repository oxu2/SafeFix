# Active Learning Performance Improvement Plan

Based on log analysis, the current Active Learning overall accuracy (64.73%) is approximately 1.9% lower than random selection (66.65%). Here are specific improvement recommendations:

## Problem Analysis

1. **Hybrid Strategy Ratio Issues**: Current ratio of 30% core-set + 50% stratified + 20% random, with excessive stratified proportion potentially causing selection bias toward certain classes
2. **Insufficient Initial Labeled Samples**: 200 initial samples may be inadequate for establishing good feature representations
3. **Low Embedding Dimension**: Current 256 dimensions may be insufficient for complex tasks like ImageNet
4. **Class Imbalance**: Some attribute classes have sparse samples, affecting learning effectiveness

## Improvement Proposals

### 1. Adjust Hybrid Strategy Ratios

Recommend testing the following ratios:
- **Option A**: 50% core-set + 30% stratified + 20% random (increase core-set proportion)
- **Option B**: 60% core-set + 25% stratified + 15% random (further strengthen core-set)
- **Option C**: Pure core-set strategy (100% core-set)

### 2. Increase Initial Labeled Samples

- Increase from 200 to 500-1000 initial samples
- Ensure initial samples cover all classes

### 3. Enhance Feature Representation Capability

- Increase embedding dimension from 256 to 512 or 1024
- Consider using pre-trained ResNet features

### 4. Improve Sampling Strategies

- Try uncertainty-based strategies, especially for hard-to-classify attributes
- Combine diversity sampling to ensure sample diversity

## Specific Implementation Steps

### Step 1: Test Pure Core-set Strategy
```bash
python alpred.py --dataset imagenet --model resnet --attributes pinkcolor_fabrictexture \
    --num_added_img 1000 --use_active_learning True --al_strategy coreset \
    --al_initial_labeled 500 --al_embedding_dim 512
```

### Step 2: Adjust Hybrid Ratios
```bash
python alpred.py --dataset imagenet --model resnet --attributes pinkcolor_fabrictexture \
    --num_added_img 1000 --use_active_learning True --al_strategy hybrid \
    --al_coreset_ratio 0.6 --al_stratified_ratio 0.3 --al_initial_labeled 500
```

### Step 3: Test Uncertainty Strategy
```bash
python enhanced_active_learning.py --strategy uncertainty --initial-labeled 500 \
    --embedding-dim 512
```

### Step 4: Optimize Training Parameters
- Increase training epochs from 20 to 30
- Adjust learning rate scheduling strategy
- Add early stopping mechanism

## Expected Improvement Effects

1. **Pure Core-set Strategy**: May perform better in diversity sampling, expected improvement of 2-3%
2. **Increase Initial Samples**: Better feature space initialization, expected improvement of 1-2%
3. **Increase Embedding Dimension**: Richer feature representation, expected improvement of 1-2%
4. **Combined Optimization**: Overall expected improvement of 3-5%, surpassing random selection baseline

## Monitoring Metrics

1. Overall validation accuracy
2. Accuracy distribution across attribute classes
3. Training convergence speed
4. Sample selection diversity metrics

## Risk Mitigation

1. If pure Core-set does not perform well, revert to adjusted Hybrid strategy
2. Keep random selection as backup option
3. Implement in phases, adjusting only one parameter at a time

Let us start with the most promising improvements: adjusting Hybrid ratios and increasing initial sample count.
