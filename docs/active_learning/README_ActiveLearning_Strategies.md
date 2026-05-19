# Active Learning Guide

## Overview

This is the canonical Active Learning document for the Safefix repository. The older `md/README_ActiveLearning.md` overview has been merged into this file so the current `alpred.py` workflow and the older round-based demos are documented in one place.

The repository contains multiple Active Learning strategies for the Safefix project to address the poor performance of the original Core-set method.

`alpred.py` applies data selection before training. The dispatch order is:

1. `--use_llm_selection True`: use the VLM-based selector and disable traditional AL.
2. `--use_active_learning True`: run the requested AL strategy on either the original train pool or the combined pool when `--al_combined True`.
3. Otherwise: use random sub-sampling when `--sub_ratio < 1.0`.

Within the AL branch, the current strategies fall into three families:

- **Geometry / coverage-driven**: `coreset`, `diversity`, `hybrid`
- **Uncertainty-aware batch selection**: `badge`, `uncertainty_stratified`, `gen_probe`
- **Metadata / slice-driven**: `attribute`, `hierarchical`

## Available Strategies

### 1. CoreSetStrategy (Original Method)
```bash
python alpred.py --al_strategy coreset --al_initial_labeled 400 --al_embedding_dim 256
```
- **Principle**: Selects samples farthest from labeled data (diversity-first)
- **Use Cases**: Uniform data distribution, need to maximize coverage
- **Issues**: May select outlier samples, disrupting class balance

### 2. BadgeStrategy (Gradient-Embedding Batch Selection)
```bash
python alpred.py --al_strategy badge --al_embedding_dim 256
```
- **Principle**: Builds BADGE-style gradient embeddings and uses k-means++ seeding to choose a batch that is both uncertain and diverse.
- **Current Implementation Detail**: `alpred.py` estimates the class probabilities from validation-set class centroids in the pretrained embedding space, so BADGE can run before task-specific fine-tuning.
- **Use Cases**: Strong baseline when a batch selector should mix uncertainty with diversity rather than optimizing either one alone.
- **Advantages**: Closer to a standard batch-AL baseline than pure coreset or pure clustering.

### 3. StratifiedStrategy (Stratified Sampling)
```bash
python alpred.py --al_strategy stratified
```
- **Principle**: Stratified sampling by class proportion, maintaining class balance
- **Use Cases**: Imbalanced data, need representation from every class
- **Advantages**: Maintains original data distribution, simple and efficient
- **Speed**: Fastest (no feature extraction needed)

### 4. DiversityStrategy (Diversity Clustering)
```bash
python alpred.py --al_strategy diversity --al_clusters 100 --al_embedding_dim 256
```
- **Principle**: K-means clustering followed by representative sample selection from each cluster
- **Use Cases**: Data with natural clustering structure, need to cover different data patterns
- **Advantages**: Balances diversity and representativeness

### 5. HybridStrategy (Mixed Strategy)
```bash
python alpred.py --al_strategy hybrid --al_coreset_ratio 0.3 --al_stratified_ratio 0.5 --al_embedding_dim 256
```
- **Principle**: Combines advantages of multiple methods
  - 30% Core-set (diversity)
  - 50% Stratified (class balance)  
  - 20% Random (randomness)
- **Use Cases**: Complex data distributions, need comprehensive consideration
- **Advantages**: Most robust, highly adaptive

### 6. AttributeFocusedStrategy (Attribute-Driven Sampling)
```bash
python alpred.py --al_strategy attribute --attribute_file save/attribute10.json \
  --al_attribute_focus color,background --al_attribute_rarity_power 1.5 --al_attribute_uncertainty_boost 1.0
```
- **Principle**: Builds a score for every unlabeled sample by combining attribute rarity, metadata-level uncertainty, and optional diagnostic weights derived from logs.
- **Use Cases**: Scenarios where human annotations describe attribute slices (color, background, hairstyle, etc.) and the model underperforms on specific combinations.
- **Advantages**: Lightweight (no feature extraction), controllable focus on high-value slices, deterministic once metadata is fixed.

Let $i$ index a candidate sample and $A$ denote the set of attributes contained in the metadata (e.g., $A = \{\text{color}, \text{background}\}$). For attribute $a \in A$, let $v_{i,a}$ be the attribute value of sample $i$, and $f_{a}(v)$ be the frequency of value $v$ among the current pool. The strategy computes four additive components:

1. **Rarity term** (controlled by `--al_attribute_rarity_power`):
  $$R_{i,a} = \left(\frac{1}{f_{a}(v_{i,a}) + \varepsilon}\right)^{\alpha}$$
  where $\alpha \geq 0$ is the rarity power and $\varepsilon$ avoids division by zero. Higher $\alpha$ exaggerates the contrast between common and rare attributes.

2. **Uncertainty bonus** (driven by the `uncertain_idx` list in the metadata):
  $$U_{i,a} = \begin{cases}
  \beta, & \text{if sample $i$ is marked uncertain for attribute $a$}\\
  0, & \text{otherwise}
  \end{cases}$$
  with $\beta \geq 0$ set by `--al_attribute_uncertainty_boost`.

3. **Log-derived weight** (optional, requires `--al_attribute_log_path` and `--al_attribute_log_acc_scale`): diagnostic logs record rare cases along with their classification accuracy. The parser converts each entry into a failure weight $w_{a}(v) = 1 - \text{ACC}$. For attribute $a$ and value $v$, the contribution becomes
  $$L_{i,a} = \gamma \cdot w_{a}(v_{i,a})$$
  where $\gamma$ is the scaling factor. Lower accuracies therefore produce larger boosts, while missing entries default to zero.

4. **Frequency penalty** (optional, requires `--al_attribute_extra_counts` and `--al_attribute_frequency_penalty`): when generated data injects thousands of samples with specific attribute values, pass those counts through `--al_attribute_extra_counts` (inline `attr:value=count` list or JSON file). Frequencies are augmented with the supplied counts, and the balance term subtracts
  $$B_{i,a} = -\lambda \cdot \frac{f^{*}_{a}(v_{i,a})}{\sum_{v} f^{*}_{a}(v)}$$
  where $\lambda$ is `--al_attribute_frequency_penalty` and $f^{*}$ denotes the augmented frequency. Larger augmented shares therefore receive a larger negative adjustment.

The total score for sample $i$ is the sum across all attributes:
  $$S_i = \sum_{a \in A} \big(R_{i,a} + U_{i,a} + L_{i,a} + B_{i,a}\big).$$

Samples are sorted by $S_i$ (with a small injected noise term for tie breaking) and the top $n_{\text{select}}$ indices are returned. When a sample exhibits multiple rare or low-accuracy attributes, the contributions accumulate, ensuring multi-attribute failures are prioritized.

Optionally, enable an additional **class stratification score** via `--al_attribute_class_balance`. After computing $S_i$, the selector performs a greedy re-ranking that rewards classes whose current selections fall below their per-class quota. When the flag is set to a finite value $w>0$, each class $c$ keeps a running count $n_c$ and target quota $t = n_{\text{select}} / N_{\text{classes}}$. The next candidate from class $c$ receives an additive bonus
$$\Delta_{c} = w \times \frac{t - n_c}{t}$$
which is added to its base score inside `_soft_class_selection` (see `utils/active_learning.py`). Classes under their quota get positive bonuses, while classes at or above the quota see the term shrink toward zero or become negative, letting pure attribute scores win. Candidates with unknown labels (e.g., synthetic-only entries) sit in a separate queue and are inserted only when their base score exceeds every class-adjusted option, so unlabeled data does not distort the balancing pass.

During iteration $k$ of `_soft_class_selection`, every labeled candidate $i$ from class $c = y_i$ uses the effective score
$$\tilde{S}_i^{(k)} = S_i + w \times \frac{t - n_c^{(k)}}{t},$$
where $n_c^{(k)}$ is the number of times class $c$ has already been chosen. The algorithm picks the candidate with the largest $\tilde{S}_i^{(k)}$ (breaking ties with the injected noise from the base ranking), appends it to the selection list, and then updates $n_c^{(k+1)} = n_c^{(k)} + 1$ for that class. If the highest raw-score unlabeled item $u$ satisfies $S_u > \max_i \tilde{S}_i^{(k)}$, the iteration chooses $u$ instead and leaves all class counts unchanged, where $S_u$ denotes the raw attribute score of the unlabeled candidate (no class bonus exists for unlabeled entries). That iteration simply appends $u$ to `selected`. The loop stops once either $|\text{selected}| = n_{\text{select}}$ or all class and unlabeled queues are empty; in the latter case `_finalize_balanced_selection` backfills the remaining slots from the global ranking. “All other class and unlabeled queues are empty” literally means every per-class list (and the unlabeled list) has been drained—no candidate remains to evaluate—so the loop has no further element to pull even if some deficits are still positive.

This deficit-based soft balancing is a contribution inspired by proportional-fair and deficit round-robin schedulers: we convert each class shortfall $(t - n_c)$ into a temporary additive bonus while preserving the metadata-driven ordering within every class, providing a controllable bridge between pure attribute scoring and hard stratification.
The motivation was empirical: attribute scores alone over-pick head classes, yet strict round-robin ignores the large score spread across classes, so we needed a lightweight middle ground that can lean toward balance without throwing away the metadata signal—a criterion none of the standard AL heuristics offered out of the box.

Setting `--al_attribute_class_balance inf` bypasses the soft bonus and triggers `_round_robin_class_selection`, the literal round-robin implementation. Each class queue contributes its current best-scoring sample, then the queue moves to the back. Exhausted queues are removed, and the process repeats until either the requested budget is filled or only unlabeled/leftover samples remain (at which point the code falls back to global sorting in `_finalize_balanced_selection`). This guarantees per-class counts differ by at most one (subject to availability) before any remaining slots revert to global scores.

**Round-robin mechanics (as implemented in `_round_robin_class_selection`):**
1. Build per-class queues sorted by the raw attribute score + noise.
2. Walk the queues in order; on each step pop the head of the current class and append it to `selected`.
3. Move that class to the back of the list; if the queue is empty, drop it permanently.
4. Repeat until `n_select` items are collected or every class queue is empty, then rely on `_finalize_balanced_selection` to fill any residual slots from the global score ranking.
This produces the same effect as traditional stratified sampling but preserves the attribute ranking within each class.

**Example (ImageNet, `sub_ratio=0.3`, ten classes):** the quota is $t = 3{,}120 / 10 = 312$ samples per class. Running with `--al_attribute_class_balance 0` let the desk class drop to 133 samples (4.3 %) because it had few high attribute scores. Re-running the same configuration with `--al_attribute_class_balance 4` gave each desk candidate an early bonus of $4 \times (312-133)/312 \approx 2.29$, so they entered the greedy loop sooner. The final desk count rose to 273 samples (8.8 %), while large classes such as backpack and purse were nudged down toward the quota. Desk growth stopped at 273 because the global budget $n_{\text{select}} = 3{,}120$ was filled once the bonus had already reduced the biggest deficits; the algorithm cannot insert additional desk samples after the budget is exhausted even though $n_{\text{desk}} < t$. This illustrates how finite weights gently pull distributions toward balance without forcing perfect equality.

**Weight tuning versus sub-ratio:** because $\Delta_c = w \times (t - n_c)/t$ scales with the quota $t = n_{\text{select}} / N_{\text{classes}}$, a fixed $w$ becomes weaker as the sub-sampling ratio grows. To preserve roughly the same maximum bonus seen at `sub_ratio = 0.3`, scale $w$ in proportion to $t$: $w_{\text{new}} \approx w_{0.3} \times (t_{\text{new}} / 312)$. For ImageNet’s ten classes this gives practical anchors:
- `sub_ratio = 0.5` (quota $\approx 520$): use $w \approx 6$–7 instead of 4.
- `sub_ratio = 0.7` (quota $\approx 728$): use $w \approx 9$–10.
- `sub_ratio = 0.9` (quota $\approx 936$): use $w \approx 12$.
Smaller weights still work but apply weaker nudges; larger weights push the distribution closer to stratified counts. Set `inf` if the run truly needs round-robin behavior.

The selector still picks the same total budget—$(10{,}400 \times 0.3 = 3{,}120)$ samples in this setup. Turning on `--al_attribute_class_balance 4` does not add or remove items; it just changes which indices fill those 3,120 slots. `_finalize_balanced_selection` appends the highest remaining base scores until the list has exactly $n_{\text{select}}$ entries, ensuring the class-aware re-ranking is purely a reordering step.

**Workflow**
- Provide an attribute JSON (for example `save/attribute10.json`) that maps each attribute to per-sample values and optional uncertain indices.
- Optionally restrict attention to a subset via `--al_attribute_focus` (comma-separated keys).
- (Optional) Supply a diagnostic log with `--al_attribute_log_path`; each `Rare Case` entry contributes a failure weight $(1 - \text{ACC})$ through the parser in `utils.active_learning.parse_attribute_priority_log`.
- Tune `--al_attribute_rarity_power`, `--al_attribute_uncertainty_boost`, and `--al_attribute_log_acc_scale` to balance between metadata-driven rarity, known uncertain slices, and empirical accuracy signals.
- (Optional) Provide pre-counted generated samples via `--al_attribute_extra_counts color:pink,texture:fabric`; when counts are omitted they default to `--num_added_img`, and if the flag itself is omitted the selector now parses `--attributes` (tokens such as `pinkcolor` or `fabrictexture`) to auto-inject the same `--num_added_img` count for those attribute-value pairs.

Because the method operates purely on metadata, it integrates cleanly with the rest of the pipeline—no embedding extraction or model forward passes are required during selection.

## Why the Attribute Strategy Is Different

The `attribute` strategy is not just another uncertainty or diversity heuristic. Standard AL methods in this repo behave as follows:

- `coreset` / `diversity`: rank samples from geometry in feature space.
- `badge` / `uncertainty_stratified` / `gen_probe`: rank samples from uncertainty-like signals or gradient embeddings.
- `hybrid`: mixes geometric coverage, class balance, and randomness.
- `hierarchical`: allocates budget over slices, but still uses model-side uncertainty and embeddings during selection.

By contrast, `attribute` starts from **semantic metadata** rather than feature geometry. It explicitly asks: which human-named slices are rare, uncertain, underperforming, or artificially flooded by generated data? The selector then scores each image by those slice-level signals.

The main methodological contributions in the current `attribute` path are:

- **Class-conditional rarity**: prefer values that are rare within a class, not merely rare globally.
- **Generated-data frequency penalty**: suppress oversampling of slices that were already heavily injected by synthetic data.
- **Log-aware failure weighting**: upweight samples from attribute values that earlier diagnostics marked as low-accuracy rare cases.
- **Deficit-based class balancing**: interpolate between pure attribute ranking and hard stratification without discarding the metadata signal.

This makes the method a better fit for Safefix than standard AL baselines when the goal is not only to improve average accuracy, but also to surface and repair semantically interpretable failure slices.

## Parameter Description

### General Parameters
- `--al_strategy`: AL strategy selection [attribute|badge|coreset|stratified|diversity|hybrid|hierarchical|uncertainty_stratified|gen_probe]
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
- `--al_attribute_use_entropy`: If True, computes Information Entropy from model inference and uses it for the uncertainty term instead of static metadata flags
- `--al_attribute_log_path`: Optional path to diagnostic log providing Rare Case accuracies
- `--al_attribute_log_acc_scale`: Scaling factor applied to each log-derived failure weight (default: 0.0)
- `--al_attribute_extra_counts`: Inline list or JSON file describing additional counts to inject per attribute value (useful when synthetic data floods specific slices); inline entries without `=count` inherit `--num_added_img`, and when omitted entirely the code infers counts from `--attributes` tokens (value+attribute) using `--num_added_img`
- `--al_attribute_frequency_penalty`: Multiplier controlling how strongly over-represented values are penalized (0 disables the term)
- `--al_attribute_class_balance`: Stratified weight for enforcing class balance after attribute scoring (0 disables, `inf` enforces near-perfect round-robin per class)
- `--al_conditional_rarity`: Use class-conditional rarity `P(value | class)` instead of global rarity

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

## Legacy Round-Based Demos

The production selector now lives in `alpred.py`, but the repo still contains earlier round-based demos that can be useful for debugging or small-scale checks.

### Quick Start

```bash
python demo_active_learning.py
```

```bash
chmod +x run_active_learning.sh
./run_active_learning.sh
./run_active_learning.sh coreset resnet 10
```

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

```bash
python enhanced_active_learning.py
```

### Legacy CelebA Layout

The older demos follow the original CelebA-oriented layout:

```text
data/celebA/
├── list_attr_celeba.txt
├── train_idx.npy
├── labels.npy
└── predictions.npy
```

Image directories used by those demos:
- `/data/oxx220000/img_align_celeba/`
- `/home/jovyan/ouser/dataset/img_align_celeba/`

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
- `BadgeStrategy`: BADGE-style gradient-embedding batch selection
- `StratifiedStrategy`: Stratified sampling
- `UncertaintyStrategy`: Uncertainty-based sampling (future extension)
- `DiversityStrategy`: Clustering-based diversity sampling
- `HybridStrategy`: Mixed multiple strategies

Each strategy is an independent class that can be easily extended and modified.

### 6. Improved Strategies (New)

I have implemented two robust strategies to correct the weaknesses of previous methods (ignoring class balance or ignoring semantic rarity).

#### A. Uncertainty-Weighted Stratified Strategy (`uncertainty_stratified`)
```bash
python alpred.py --al_strategy uncertainty_stratified
```
**Motivation**: Standard uncertainty sampling often picks outliers from a single "hard" class, causing imbalance. Standard stratified sampling ignores sample difficulty. This strategy fuses them.

**Algorithm**:
1.  **Inference**: Compute uncertainty for all unlabeled samples.
    *   *Multi-class*: Entropy $H(p) = -\sum p_i \log p_i$.
    *   *Binary*: Confidence gap $1 - |p - 0.5| \times 2$.
2.  **Quota Calculation**: Determine strict sample quotas for each class $c$: $k_c \approx N_{total} / C$.
3.  **Selection**: 
    *   For each class $c$, retrieve all samples predicted as $c$ (or labeled as $c$ if labels available).
    *   Sort them by uncertainty (High to Low).
    *   Select top $k_c$ samples.

**Key Benefit**: Guarantees diversity across classes (like Stratified) whilst maximizing information gain (like Uncertainty).

#### B. Class-Conditional Attribute Rarity (`conditional_rarity`)
```bash
python alpred.py --al_strategy attribute --al_conditional_rarity True
```
**Motivation**: Global rarity ($P(v)$) fails when an attribute is common globally but rare for a specific object (e.g., "Red" is common, but "Red Banana" is rare). We need to find *semantically* interesting outliers.

**Algorithm** (Modification to `AttributeFocusedStrategy`):
Instead of scoring based on global frequency $f_a(v)$, we score based on **class-conditional frequency** $f_a(v|y)$:
$$S_{i} = \sum_{a \in Attributes} \left( \left( \frac{1}{f_a(v_{i,a} | y_i) + \epsilon} \right)^\alpha + U_{i,a} + L_{i,a} + B_{i,a} \right)$$

1.  **Grouping**: Split the unlabeled pool and labeled metadata by class label $y$.
2.  **Counting**: Inside each class group $c$, count attribute value frequencies $f_a(v|c)$.
3.  **Scoring**: For a sample $i$ with predicted class $y_i$, use the frequencies of that class. A sample with value $v$ gets a high score only if $v$ is rare *within its specific class*.

**Example**:
*   *Global Rarity*: "White" is common -> Low score.
*   *Conditional Rarity*: "White" is common for *Sheep*, but rare for *Crows*. A "White Crow" gets a massive score boost.

**Key Benefit**: Discovers subtle, context-dependent anomalies that global statistics miss.
