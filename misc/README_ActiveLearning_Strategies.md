# Active Learning Strategy Usage Guide

## Overview

I have implemented multiple Active Learning strategies for the HiBug project to address the poor performance of the original Core-set method.

## Available Strategies

### 1. CoreSetStrategy (Original Method)
```bash
python alpred.py --al_strategy coreset --al_initial_labeled 400 --al_embedding_dim 256
```
- **Principle**: Selects samples farthest from labeled data (diversity-first)
- **Use Cases**: Uniform data distribution, need to maximize coverage
- **Issues**: May select outlier samples, disrupting class balance

### 2. StratifiedStrategy (Stratified Sampling)
```bash
python alpred.py --al_strategy stratified
```
- **Principle**: Stratified sampling by class proportion, maintaining class balance
- **Use Cases**: Imbalanced data, need representation from every class
- **Advantages**: Maintains original data distribution, simple and efficient
- **Speed**: Fastest (no feature extraction needed)

### 3. DiversityStrategy (Diversity Clustering)
```bash
python alpred.py --al_strategy diversity --al_clusters 100 --al_embedding_dim 256
```
- **Principle**: K-means clustering followed by representative sample selection from each cluster
- **Use Cases**: Data with natural clustering structure, need to cover different data patterns
- **Advantages**: Balances diversity and representativeness

### 4. HybridStrategy (Mixed Strategy)
```bash
python alpred.py --al_strategy hybrid --al_coreset_ratio 0.3 --al_stratified_ratio 0.5 --al_embedding_dim 256
```
- **Principle**: Combines advantages of multiple methods
  - 30% Core-set (diversity)
  - 50% Stratified (class balance)  
  - 20% Random (randomness)
- **Use Cases**: Complex data distributions, need comprehensive consideration
- **Advantages**: Most robust, highly adaptive

### 5. AttributeFocusedStrategy (Attribute-Driven Sampling)
```bash
python alpred.py --al_strategy attribute --attribute_file save/attribute10.json \
  --al_attribute_focus color,background --al_attribute_rarity_power 1.5 --al_attribute_uncertainty_boost 1.0
```
- **Principle**: Scores samples using external attribute metadata, prioritizing rare or uncertain attribute slices
- **Use Cases**: Vision models with known attribute weaknesses (e.g., color, background, texture)
- **Advantages**: No feature extraction required, directly targets challenging attributes
- **Speed**: Fast (metadata lookup only)
> Tip: Provide a diagnostic log (for example `logs/compare/stat_imagenet_resnet_compare.log`) via `--al_attribute_log_path` so that slices flagged as “Rare Case”/“Hard case” receive additional selection boost.

## Parameter Description

### General Parameters
- `--al_strategy`: AL strategy selection [coreset|stratified|diversity|hybrid|hierarchical|attribute]
- `--al_embedding_dim`: Feature vector dimension (default: 512)
- `--al_initial_labeled`: Initial labeled sample count (default: 500)

### Hybrid Strategy Parameters
- `--al_coreset_ratio`: Core-set sampling ratio (default: 0.4)
- `--al_stratified_ratio`: Stratified sampling ratio (default: 0.4)
- Remaining portion automatically allocated to random sampling

### Diversity Strategy Parameters
- `--al_clusters`: Number of clusters (default: n_select//10)

### Attribute Strategy Parameters
- `--attribute_file`: Path to attribute metadata JSON (e.g., `save/attribute10.json`)
- `--al_attribute_focus`: Optional comma-separated attribute keys to prioritize (default: all keys in metadata)
- `--al_attribute_rarity_power`: Emphasis on rare attribute values (>=0, higher = stronger focus on rare slices)
- `--al_attribute_uncertainty_boost`: Score bonus for samples marked as uncertain in metadata (>=0)
- `--al_attribute_log_path`: Optional path to diagnostic log highlighting rare/hard attribute slices
- `--al_attribute_include_hard`: Whether to boost “Hard case” slices from the log (default: True)
- `--al_attribute_log_rare_boost`: Additional score for each rare case match (default: 1.0)
- `--al_attribute_log_hard_boost`: Additional score for each hard case match (default: 0.0)

## Recommended Configurations

### For ImageNet Dataset (Multi-class Classification)
```bash
# Recommended: Hybrid strategy, balancing performance and efficiency
python alpred.py \
  --dataset imagenet --model resnet \
  --al_strategy hybrid \
  --al_coreset_ratio 0.2 \
  --al_stratified_ratio 0.6 \
  --al_embedding_dim 256 \
  --sub_ratio 0.5

# Alternative: Pure stratified sampling, fastest and most stable
python alpred.py \
  --dataset imagenet --model resnet \
  --al_strategy stratified \
  --sub_ratio 0.5
```

### For CelebA Dataset (Binary Classification)
```bash
# Recommended: Diversity strategy
python alpred.py \
  --dataset celeba --model resnet \
  --al_strategy diversity \
  --al_clusters 50 \
  --al_embedding_dim 128 \
  --sub_ratio 0.5
```

## Performance Expectations

Based on analysis, expected performance ranking:

1. **StratifiedStrategy**: Most stable, maintains class balance
2. **HybridStrategy**: Best overall performance
3. **DiversityStrategy**: Moderate performance, adapts to complex distributions
4. **CoreSetStrategy**: Possibly worst performance (original issue)

## Testing Scripts

Run comparative tests of all strategies:
```bash
./test_al_strategies.sh
```

View result comparisons:
```bash
python compare_al_results.py
```

## Debugging Suggestions

1. **Use --verbose True** to view detailed class distribution information
2. **Start with small datasets** to verify strategy functionality
3. **Compare training times** to select strategy suitable for your computational resources
4. **Check class balance** to ensure selected samples have sufficient representativeness

## Code Structure

New AL strategies are all in `utils/active_learning.py`:
- `CoreSetStrategy`: Original core-set method
- `StratifiedStrategy`: Stratified sampling
- `UncertaintyStrategy`: Uncertainty-based sampling (future extension)
- `DiversityStrategy`: Clustering-based diversity sampling
- `HybridStrategy`: Mixed multiple strategies

Each strategy is an independent class that can be easily extended and modified.
