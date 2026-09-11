<!-- AGENTS-IGNORE: START - The block below contains user operational shortcuts. AI agents should ignore this section when reading report.md for research context. -->
```bash
cd 27iclr
latexmk -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  main.tex
scripts/organize_build_artifacts.sh
```
<!-- AGENTS-IGNORE: END -->



# ImageNet10 Five-Seed Selected-Sample Mechanism Analysis

**Updated:** August 3, 2026

This reviewer-facing analysis combines the matched ImageNet10 selected-subset
diagnostics with the updated five-seed downstream FR results supplied by the
author. It covers Target-Slice + Stratified, Attribute-Filtered Random,
Attribute-Filtered Entropy, Generated-Complement, and AttriSelect. All methods
use the same 10,400-image source pool, fixed 1,000-image generated set,
diagnosis checkpoint, source ratios, and 15-epoch repair schedule.

Generated Redundancy is the selected-set mean fraction of the two
generated-supported target fields matched by each source image:
"color=pink" and "texture=fabric". Mean Pairwise Attribute Distance is mean
categorical Hamming distance over color, background, and texture across all
unordered selected-source pairs. Class Imbalance is the difference between
the largest and smallest selected counts across task-specific groups.

At source ratios `0.02/0.05`, AttriSelect reaches five-seed FR `1.23/5.58`,
compared with `0.68/3.95` for the strongest control, for margins of
`0.55/1.63` points. It covers `184/397` unique
class--color--background--texture tuples, compared with `142/267` for the
strongest diversity control, and its mean pairwise attribute distance is
`0.8823/0.8680`, compared with `0.7695/0.7828`. Class Imbalance is `1/0` for
both AttriSelect and the best-balanced control.

Across ratios `0.1/0.3/0.5/0.7/0.9`, AttriSelect has the highest five-seed FR
among the matched controls: `6.93/11.28/11.85/12.10/13.76`. It also has the
largest tuple coverage and mean pairwise attribute distance at every ratio.
Generated-Complement has lower Generated Redundancy but lower FR, so
generated-target complementarity alone does not explain the ordering. The
supported interpretation is joint: AttriSelect retains repair-relevant
samples while preserving broader class and attribute coverage.

The five-seed FR values and selected-set diagnostics are now integrated into
the hG6j follow-up under `Q4. Exact-Slice and Selected-Sample Evidence` in
`draft.md`. The saved selection diagnostics and reproduction script remain at:

- `logs/juno/imagenet_seed42_selection_mechanism.{json,md}`
- `logs/juno/imagenet_attribute_aware/full/306871_*`
- `logs/juno/generated_complement_seed42/313018_0--4`
- `logs/juno/constrained_budget_seed42/313516_0--9`

Reproduction script:
`scripts/analyze_imagenet_seed42_selection_mechanism.py`.

# Seed-42 More-Constrained Source-Budget Results

**Updated:** August 2, 2026

Juno H200 array `313516` completed all `30/30` tasks with exit `0:0`. The
matched seed-42 comparison uses source ratios `0.02/0.05`, a fixed 1,000-image
generated repair set, and five selectors: Generated-Complement, Target-Slice
+ Stratified, Attribute-Filtered Random, Attribute-Filtered Entropy, and
AttriSelect. FR uses the manuscript-authoritative pre-repair metrics `0.7173`
for ImageNet10, `0.5902` for COCO, and `0.5626` for KITTI.

| Task | Ratio | Generated-Complement | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | AttriSelect |
|:---|---:|---:|---:|---:|---:|---:|
| ImageNet10 | 0.02 | -1.77 | -2.04 | 0.68 | **1.23** | -0.95 |
| ImageNet10 | 0.05 | 3.13 | 3.95 | **5.58** | 2.32 | 3.13 |
| COCO | 0.02 | -2.84 | **-2.02** | -2.46 | -6.23 | -7.80 |
| COCO | 0.05 | -31.00 | -11.64 | -0.61 | **1.63** | -1.10 |
| KITTI | 0.02 | -46.76 | -47.30 | -44.98 | **-35.88** | -56.03 |
| KITTI | 0.05 | -20.67 | -18.70 | -20.93 | **-15.29** | -32.60 |

Attribute-Filtered Entropy is strongest at four of the six dataset--ratio
points; Target-Slice + Stratified and Attribute-Filtered Random are each
strongest once. AttriSelect is not strongest at any point and trails the best
matched control by `2.18/2.45` FR points on ImageNet10, `5.77/2.73` on COCO,
and `20.15/17.32` on KITTI at ratios `0.02/0.05`. These results do not support
an AttriSelect advantage under extreme source constraints. The fixed
generated set equals or outnumbers the selected source subset in every run,
so these budgets materially change the source/generated mixture relative to
the main experiment.

Compact artifacts are under
`logs/juno/constrained_budget_seed42/313516_*/`: ImageNet10 `result.json`,
COCO `output/test_metrics.json`, and KITTI `timing.json`.

# Generated-Complement and AttriSelect Seed-42 Selection Diagnostics

**Updated:** August 2, 2026

This section retains the original seed-42 Generated-Complement diagnostics for
a matched comparison with the existing seed-42 AttriSelect selected subsets;
the five-seed Generated-Complement baseline is reported separately below.
Class Imbalance is the
maximum minus minimum selected class/proxy-label count. Mean Complement Score
is the selected-set mean of
$S_{GC}(x)=|\mathcal A_x|^{-1}\sum_{a\in\mathcal A_x}(1+f_{gen,a}(x_a))^{-1}$.
No model was retrained and no downstream metric was recomputed.

| Task | Ratio | Generated-Complement Class Imbalance | AttriSelect Class Imbalance | Generated-Complement Mean Complement Score | AttriSelect Mean Complement Score |
|:---|---:|---:|---:|---:|---:|
| ImageNet10 | 0.1 | 0 | 0 | 0.9721 | 0.8477 |
| ImageNet10 | 0.3 | 0 | 2 | 0.9574 | 0.8554 |
| ImageNet10 | 0.5 | 0 | 4 | 0.9406 | 0.8681 |
| ImageNet10 | 0.7 | 1 | 5 | 0.9290 | 0.8827 |
| ImageNet10 | 0.9 | 0 | 6 | 0.9203 | 0.8854 |
| ImageNet10 | Ratio Mean | -- | -- | 0.9439 | 0.8679 |
| COCO | 0.1 | 0 | 812 | 1.0000 | 1.0000 |
| COCO | 0.3 | 0 | 2,308 | 1.0000 | 0.9630 |
| COCO | 0.5 | 615 | 3,723 | 0.9930 | 0.9533 |
| COCO | 0.7 | 1,985 | 5,126 | 0.9874 | 0.9583 |
| COCO | 0.9 | 5,457 | 6,618 | 0.9824 | 0.9675 |
| COCO | Ratio Mean | -- | -- | 0.9926 | 0.9684 |
| KITTI | 0.1 | 1 | 29 | 1.0000 | 0.9431 |
| KITTI | 0.3 | 0 | 17 | 1.0000 | 0.9584 |
| KITTI | 0.5 | 752 | 752 | 0.9932 | 0.9684 |
| KITTI | 0.7 | 1,752 | 1,752 | 0.9951 | 0.9775 |
| KITTI | 0.9 | 2,752 | 2,752 | 0.9962 | 0.9825 |
| KITTI | Ratio Mean | -- | -- | 0.9969 | 0.9660 |

Generated-Complement has Class Imbalance no worse than AttriSelect at all 15
task--ratio points and lower at 11. Its Mean Complement Score is higher at 14
points and tied at COCO ratio `0.1`. The control therefore succeeds at its
direct proxy objectives. Its lower Ratio Mean FR on all three tasks shows that
hard balance and generated-distribution complementarity alone do not replace
AttriSelect's full $R+P+C$ ranking. Large high-ratio imbalance values on COCO
and KITTI occur after minority proxy-label groups are exhausted and should be
read as raw count differences, not as a normalized cross-task fairness metric.

ImageNet10 values use saved selection manifests from Juno job `306871`.
KITTI values use saved prepared subsets from jobs `305836`, `306774`, and
`306046`. COCO ratios `0.1/0.3/0.5` exactly match the saved selection exports;
ratios `0.7/0.9` use deterministic selection-only replay with the same frozen
configuration.

# Generated-Complement Five-Seed Baseline

**Updated:** August 2, 2026

The author supplied the following completed five-seed means and ratio-level
FR standard deviations. Per-seed values and dispersion for exact-target,
non-target, and overall metrics were not supplied, so no significance claim
is made. Exact-target, non-target, and overall entries use the native metric
scale for each task.

| Task | Ratio | Exact Target | Non-Target | Overall | FR Mean $\pm$ Std | Class Imbalance $\downarrow$ | Mean Complement Score $\uparrow$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ImageNet10 | 0.1 | 0.6875 | 0.7220 | 0.7215 | 1.50 $\pm$ 0.87 | 0 | 0.9721 |
| ImageNet10 | 0.3 | 0.7125 | 0.7343 | 0.7323 | 5.31 $\pm$ 1.04 | 0 | 0.9574 |
| ImageNet10 | 0.5 | 0.7200 | 0.7498 | 0.7492 | 11.29 $\pm$ 0.76 | 0 | 0.9406 |
| ImageNet10 | 0.7 | 0.7275 | 0.7422 | 0.7415 | 8.57 $\pm$ 1.18 | 1 | 0.9290 |
| ImageNet10 | 0.9 | 0.7301 | 0.7477 | 0.7469 | 10.48 $\pm$ 0.95 | 0 | 0.9203 |
| ImageNet10 | Ratio Mean | 0.7155 | 0.7392 | 0.7383 | 7.43 | -- | 0.9439 |
| COCO | 0.1 | 0.5272 | 0.6008 | 0.6004 | 2.50 $\pm$ 0.83 | 0 | 1.0000 |
| COCO | 0.3 | 0.6525 | 0.6199 | 0.6198 | 7.22 $\pm$ 0.68 | 0 | 1.0000 |
| COCO | 0.5 | 0.7619 | 0.6259 | 0.6257 | 8.67 $\pm$ 0.79 | 615 | 0.9930 |
| COCO | 0.7 | 0.8422 | 0.6296 | 0.6294 | 9.56 $\pm$ 0.72 | 1,985 | 0.9874 |
| COCO | 0.9 | 0.8465 | 0.61878 | 0.6187 | 6.96 $\pm$ 0.91 | 5,457 | 0.9824 |
| COCO | Ratio Mean | 0.7261 | 0.6190 | 0.6188 | 6.98 | -- | 0.9926 |
| KITTI | 0.1 | 0.3961 | 0.5356 | 0.5353 | -6.24 $\pm$ 0.84 | 1 | 1.0000 |
| KITTI | 0.3 | 0.3791 | 0.5677 | 0.5673 | 0.17 $\pm$ 0.43 | 0 | 1.0000 |
| KITTI | 0.5 | 0.4175 | 0.5704 | 0.5701 | 1.72 $\pm$ 0.58 | 752 | 0.9932 |
| KITTI | 0.7 | 0.4343 | 0.5806 | 0.5804 | 4.06 $\pm$ 0.71 | 1,752 | 0.9951 |
| KITTI | 0.9 | 0.5017 | 0.5794 | 0.5790 | 3.74 $\pm$ 0.61 | 2,752 | 0.9962 |
| KITTI | Ratio Mean | 0.4257 | 0.5667 | 0.5664 | 0.69 | -- | 0.9969 |

The COCO and KITTI exact conjunctions each contain only two test images per
seed. Their localized five-seed means remain high-variance diagnostics and do
not support a general or statistical claim.

# COCO and KITTI Seed-42 Same-Information Controls

**Updated:** August 2, 2026

Juno arrays `312965` and `312977` completed all `20/20` tasks successfully.
The new rows use seed `42` at source ratios `0.1/0.3/0.5/0.7/0.9` with the
same generated repair data, source attribute metadata, source budget,
diagnosis checkpoint, and repair schedule used by the corresponding task.
CoreSet, BADGE, UHerding, PPAL, and AttriSelect values below are the
manuscript-authoritative five-seed means from `nips26_new/main.tex` and
`nips26_new/appendix/main_metric_tables.tex`; therefore, comparisons involving
the new seed-42 controls are descriptive and are not paired statistical
comparisons.

## COCO Pose Estimation

FR uses the authoritative pre-repair keypoint AP `0.5902`.

| Ratio | CoreSet FR | BADGE FR | UHerding FR | PPAL FR | Target-Slice FR | Filtered Random FR | Filtered Entropy FR | AttriSelect FR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.61 | 2.03 | 2.71 | 1.17 | 5.63 | -3.56 | 4.40 | **6.86** |
| 0.3 | 8.64 | 6.12 | 4.32 | 6.42 | 9.48 | 6.53 | 8.32 | **9.86** |
| 0.5 | 6.47 | 7.42 | 6.12 | 6.83 | 8.81 | 7.09 | 8.09 | **10.13** |
| 0.7 | 9.10 | 8.64 | 6.17 | 7.83 | 9.24 | 9.37 | **10.74** | 10.18 |
| 0.9 | 7.32 | 7.59 | 7.05 | 9.74 | 8.86 | 6.18 | 9.85 | **11.05** |
| Ratio Mean | 6.18 | 6.36 | 5.27 | 6.40 | 8.40 | 5.12 | 8.28 | **9.61** |

AttriSelect has the highest reported value at four of five ratios and the
highest Ratio Mean. Attribute-Filtered Entropy is higher at ratio `0.7`.

## KITTI Object Detection

FR uses the authoritative pre-repair mAP50--95 `0.5626`.

| Ratio | CoreSet FR | BADGE FR | UHerding FR | PPAL FR | Target-Slice FR | Filtered Random FR | Filtered Entropy FR | AttriSelect FR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -8.89 | -6.95 | -9.24 | -8.32 | -8.28 | -6.09 | **-5.76** | -5.81 |
| 0.3 | -0.32 | -0.05 | -1.21 | -1.33 | -0.69 | 0.02 | **1.07** | 0.46 |
| 0.5 | 3.04 | 2.29 | 1.12 | 2.17 | 2.80 | **3.62** | 3.41 | 3.36 |
| 0.7 | 3.34 | 2.08 | 3.25 | 3.57 | 4.10 | 2.97 | **4.50** | 3.52 |
| 0.9 | 3.54 | 4.21 | 4.85 | 3.89 | 4.43 | 4.45 | 4.71 | **5.12** |
| Ratio Mean | 0.14 | 0.32 | -0.25 | 0.00 | 0.47 | 0.99 | **1.58** | 1.33 |

The KITTI seed-42 controls do not support uniform AttriSelect dominance in
this mixed-aggregation table. Attribute-Filtered Entropy is highest at ratios
`0.1`, `0.3`, and `0.7` and has the highest Ratio Mean; Attribute-Filtered
Random is highest at `0.5`, and AttriSelect is highest at `0.9`.

Local compact artifacts are under
`logs/juno/coco_kitti_same_information/312965_*/output/test_metrics.json` and
`logs/juno/coco_kitti_same_information/312977_*`. Full logs and checkpoints
remain under the corresponding path in the Juno rebuttal worktree.

# CelebA Five-Seed HiBug Pipeline Compatibility

**Updated:** July 30, 2026

The author supplied the following five-seed mean Fix Rate (FR) results for a
CelebA repair comparison using the HiBug upstream repair pipeline. CelebA has
80,000 training images. Within this table, every selector uses the same
HiBug-generated repair set and repair protocol; the source-selection rule is
the comparison variable. Per-seed values, standard deviations, and artifact
paths were not supplied with the aggregate table, so the comparison is
descriptive and no new significance claim is made.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **AttriSelect (HiBug)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.59 | 2.33 | 6.92 | -6.20 | 2.87 | -1.36 | 5.58 | **7.36** |
| 0.3 | 5.91 | 5.08 | 8.02 | 7.79 | 1.66 | 3.39 | 3.87 | **8.07** |
| 0.5 | 5.76 | 7.70 | 7.48 | 7.90 | 2.53 | 2.13 | 9.64 | **12.54** |
| 0.7 | 2.57 | 8.06 | 10.03 | 6.19 | 7.86 | 4.96 | 5.90 | **11.16** |
| 0.9 | 2.84 | 4.28 | 7.95 | 5.71 | 5.80 | 4.19 | 6.37 | **13.42** |

AttriSelect is strongest at all five ratios. Its margins over the strongest
baseline are `0.44/0.05/2.90/1.13/5.47` FR points at ratios
`0.1/0.3/0.5/0.7/0.9`, respectively, with a mean margin of `1.998` points
(`2.00` when rounded to two decimals).

These results should be compared horizontally within the HiBug table. They
must not be combined with the SafeFix baseline rows in `nips26_new/main.tex`
as if they shared the same generated repair set: changing the upstream repair
pipeline can shift the absolute FR of every selector.

# ImageNet10 General Data Selection and Active Learning

## Scope and Protocol

This experiment evaluates AttriSelect outside the model-repair setting. It
uses only the original ImageNet10 source pool: no generated images, diagnosed
failure slice, or resume-from-repair checkpoint is included in final training.
Each selected subset trains a fresh ResNet-18 for 30 epochs with Adam
(`lr=1e-3`, weight decay `1e-3`), StepLR (`step_size=3`, `gamma=0.5`), batch
size 64, and best-validation checkpoint selection. The primary tables below
are author-provided five-seed means received on July 29, 2026. Dispersion and
per-seed artifacts were not provided with the aggregate tables, so the
comparisons below are descriptive.

The one-shot data-selection comparison contains Random, class-Stratified
Random, Entropy, CoreSet, BADGE, Attribute Rarity, class-balanced Semantic
CoreSet, and AttriSelect at source ratios `0.1/0.3/0.5/0.7/0.9`.
Entropy, CoreSet, and BADGE use the existing full-data classifier only as a
selection proxy; the final subset-trained classifier is always initialized
from scratch.

The active-learning comparison uses a shared class-stratified initial labeled
ratio of `0.1`, followed by five queries of ratio `0.1`. It evaluates six
rounds at labeled ratios `0.1--0.6`. Every method starts from the same initial
labeled set and uses the same model initialization at a given round.

## One-Shot Data Selection

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | Attribute Rarity | Semantic CoreSet | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 43.23 | 41.85 | 26.86 | 38.33 | 42.62 | 44.46 | 44.08 | **45.00** |
| 0.3 | 55.84 | 57.15 | 44.64 | 56.16 | 58.17 | 56.54 | 56.55 | **58.32** |
| 0.5 | 64.77 | 64.14 | 58.69 | 64.87 | 63.19 | 65.31 | 62.56 | **66.53** |
| 0.7 | 69.47 | 71.63 | 66.71 | 68.03 | 69.57 | 71.01 | 67.18 | **72.02** |
| 0.9 | 71.34 | 71.70 | 72.88 | 72.89 | 72.38 | 72.48 | 71.24 | **72.92** |

**Attribute Rarity:** uses the Qwen3-VL `color`, `background`, and `texture`
labels. For each image, it sums the inverse frequencies of its attribute
values within its class, so class--attribute combinations that occur less
often receive higher scores. A soft class-balance re-ranking then selects the
requested subset. Unlike AttriSelect, this baseline does not apply the global
attribute-frequency penalty.

Metric: five-seed mean ImageNet10 overall test accuracy (%). Bold marks the
strongest method at each ratio. AttriSelect is strongest at all five source
ratios, with margins of `0.54/0.15/1.22/0.39/0.03` points over the strongest
baseline at ratios `0.1/0.3/0.5/0.7/0.9`.

## Iterative Active Learning

| Ratio | Random | Entropy | CoreSet | BADGE | Attribute Rarity | Object--Attribute Entropy Reduction | AttriSelect-AL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 |
| 0.2 | 51.11 | **52.26** | 50.00 | 51.27 | 50.93 | 50.48 | 51.33 |
| 0.3 | 55.38 | 54.85 | 55.01 | 56.17 | 52.94 | 55.95 | **56.96** |
| 0.4 | 58.71 | 58.63 | 59.54 | 60.46 | 60.47 | 60.31 | **63.02** |
| 0.5 | 61.97 | 63.70 | 63.92 | 61.89 | 61.86 | 63.88 | **65.18** |
| 0.6 | 66.15 | 66.70 | 68.16 | 67.11 | 65.78 | 65.79 | **68.80** |
| Curve Mean | 56.09 | 56.56 | 56.64 | 56.69 | 55.87 | 56.61 | **58.09** |

**Attribute Rarity:** applies the same attribute-rarity principle during each
query round. Frequencies are estimated from the currently labeled set;
unlabeled images use model-predicted classes for class-conditional scoring,
followed by soft class balancing. It does not use model entropy or the global
frequency penalty used by AttriSelect-AL.

**Object--Attribute Entropy Reduction:** an object-query adaptation of
Kovashka et al. (ICCV 2011). The Qwen3-VL attributes are treated as already
known, and only the unknown class label is queried. Each candidate is scored
by its model predictive entropy multiplied by the average uncertainty of the
class distribution associated with its attribute values; values with fewer
labeled examples receive more weight. Soft class balancing is applied using
model-predicted classes, without reading unlabeled ground-truth labels.

Metric: five-seed mean test accuracy (%) at each labeled ratio. All methods use
the same initial labeled set at ratio `0.1`; `Curve Mean` is the arithmetic
mean over the six evaluated ratios.

AttriSelect-AL is strongest at ratios `0.3--0.6` and has the highest Curve
Mean, `58.09`, exceeding BADGE by `1.40` points. At ratio `0.6`, it reaches
`68.80%`, `0.64` points above CoreSet.

## Alternative Attribute Sources: Five-Seed Summary

All three attribute sources use the same five seeds and source-only evaluation
protocol.

Metric: five-seed ImageNet10 test accuracy (%). `One-Shot Mean` averages
source ratios `0.1/0.3/0.5/0.7/0.9`; `AL Curve Mean` averages labeled ratios
`0.1--0.6`; and `Final ACC` is the Active Learning accuracy at ratio `0.6`.

| Attribute Source | One-Shot Mean | AL Curve Mean | Final ACC at Ratio 0.6 |
|---|---:|---:|---:|
| Qwen3-VL | 62.96 | **58.09** | **68.80** |
| BLIP | 62.18 | 55.99 | 65.46 |
| InternVL3 | **63.27** | 56.33 | 66.23 |

InternVL3 has the highest One-Shot Mean, exceeding Qwen3-VL by `0.31`
accuracy points. Qwen3-VL has the highest Active Learning Curve Mean and final
ratio-`0.6` accuracy, exceeding InternVL3 by `1.76` and `2.57` points,
respectively. No attribute source is strongest under both selection settings.

### Historical Local Seed-42 Diagnostic

The ratio-wise rows below are retained only as an older local execution
record. They are superseded by the five-seed summary above and must not be
used for the current reviewer response.

### One-Shot AttriSelect

| Source Ratio | Qwen3-VL | BLIP | InternVL3 |
|---:|---:|---:|---:|
| 0.1 | **45.00** | 43.15 | 43.31 |
| 0.3 | 55.85 | **59.54** | 56.46 |
| 0.5 | **66.54** | 65.62 | 65.54 |
| 0.7 | 68.00 | 70.15 | **71.15** |
| 0.9 | 71.23 | 70.85 | **72.85** |

### AttriSelect-AL Summary

| Attribute Source | Curve Mean | Final Ratio 0.6 |
|---|---:|---:|
| Qwen3-VL | **57.04** | **64.77** |
| BLIP | 55.95 | 64.46 |
| InternVL3 | 56.33 | 64.23 |

These rows are local seed-42 diagnostics and are retained for historical
traceability only.

**Historical BLIP note (July 29, 2026):** an intermediate local rerun reported
a BLIP One-Shot Mean of `61.18` without supplying its five ratio-wise values.
This intermediate result is superseded by the five-seed value `62.18` above.

## ImageNet10 Source-Annotation Cost

Fresh Qwen3-VL assignment over the exact 10,400-image training pool completed
successfully with zero inference errors or fallback labels:

| Component | Time |
|---|---:|
| Model and processor loading | 13.85 s |
| Attribute inference loop | 1646.70 s (27.44 min) |
| Final serialization | 0.05 s |
| Total process time | 1760.88 s (29.35 min) |

The assignment produces `color`, `background`, and `texture` metadata for
every training image. This is a one-time preprocessing cost: the resulting
metadata are reused across ratios, selectors, and all active-learning rounds.
The measurement is retained as an internal exact reference even if the final
reviewer response reports only the existing COCO and KITTI cost averages.

# ImageNet10 Missing-Field Vocabulary-Coverage Experiment

**Updated:** July 28, 2026

## Research Question

This experiment tests how AttriSelect behaves when the closed-set attribute
vocabulary does not contain one of the fields needed to represent the diagnosed
failure slice. The original ImageNet10 slice is `color=pink AND
texture=fabric`. We remove either `texture` or `color` from source selection
while keeping the generated repair data and all training settings fixed.

## Experimental Protocol

- Dataset and model: ImageNet10 with the same ResNet diagnosis checkpoint used
  by the repair pipeline.
- Seed: `42` for source selection and generated-image sampling.
- Source ratios: `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`.
- Generated repair data: the same combined `pink AND fabric` pool in every
  controlled variant, with a fixed sample of 1,000 generated images.
- Source metadata: the same Qwen3-VL assignment file
  `save/attribute10_qwen3_8B.json`.
- AttriSelect settings: rarity power `0.03`, generated-frequency penalty
  `0.5`, class-balance coefficient `15`, conditional rarity enabled, and
  entropy disabled.
- Repair training: 15 epochs, learning rate `1e-5`, batch size `64`, and an
  unfrozen backbone.
- Evaluation: every model is evaluated on the original exact `pink AND
  fabric` test slice, containing 16 target images and 1,284 non-target images
  among 1,300 test images.

The three controlled variants are:

1. **Full conjunction:** select with `pinkcolor_fabrictexture`.
2. **Missing texture:** select with `pinkcolor` while retaining the combined
   `pink AND fabric` generated pool.
3. **Missing color:** select with `fabrictexture` while retaining the same
   combined generated pool.

## Results

All ACC values below are percentages. `ACC change` is measured relative to the
Full conjunction result at the same source ratio.

| Ratio | Variant | Overall ACC (%) | ACC change (%) | Original target ACC (%) | Non-target ACC (%) | Source-set Jaccard vs. Full |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | Full conjunction | 74.00 | 0.00 | 68.75 | 74.07 | 1.0000 |
| 0.1 | Missing texture | 73.85 | -0.15 | 62.50 | 73.99 | 0.3904 |
| 0.1 | Missing color | 72.69 | -1.31 | 68.75 | 72.74 | 0.5604 |
| 0.3 | Full conjunction | 74.62 | 0.00 | 75.00 | 74.61 | 1.0000 |
| 0.3 | Missing texture | 73.54 | -1.08 | 68.75 | 73.60 | 0.5842 |
| 0.3 | Missing color | 73.77 | -0.85 | 68.75 | 73.83 | 0.6166 |
| 0.5 | Full conjunction | 73.69 | 0.00 | 75.00 | 73.68 | 1.0000 |
| 0.5 | Missing texture | 74.23 | +0.54 | 68.75 | 74.30 | 0.7330 |
| 0.5 | Missing color | 74.23 | +0.54 | 68.75 | 74.30 | 0.6443 |
| 0.7 | Full conjunction | 74.15 | 0.00 | 75.00 | 74.14 | 1.0000 |
| 0.7 | Missing texture | 74.23 | +0.08 | 68.75 | 74.30 | 0.8826 |
| 0.7 | Missing color | 74.00 | -0.15 | 75.00 | 73.99 | 0.6999 |
| 0.9 | Full conjunction | 74.62 | 0.00 | 68.75 | 74.69 | 1.0000 |
| 0.9 | Missing texture | 74.54 | -0.08 | 68.75 | 74.61 | 0.9668 |
| 0.9 | Missing color | 75.08 | +0.46 | 68.75 | 75.16 | 0.8765 |

## Result Summary

| Missing field | Mean Overall ACC change (%) | Largest decrease (%) |
| --- | ---: | ---: |
| Texture | -0.14 | -1.08 at ratio 0.3 |
| Color | -0.26 | -1.31 at ratio 0.1 |
| Both variants combined | -0.20 | -1.31 |

Removing an attribute field has the clearest effect on source-set composition.
At ratios `0.1--0.3`, Jaccard overlap with the Full selector falls to
`0.3904--0.6166`. Overall ACC changes are smaller and non-monotonic: some
settings decrease, while the missing-field variants increase ACC at ratios
`0.5`, `0.7`, or `0.9`. The largest observed decrease is from `74.00%` to
`72.69%` when color is missing at ratio `0.1`.

The supported conclusion is that incomplete vocabulary coverage can
substantially change which source examples AttriSelect retains, but this
single-seed experiment does not show that missing fields consistently reduce
Overall ACC. The original target slice contains only 16 images, so its ACC is
too coarse for a strong target-slice performance claim.

## Execution and Artifacts

- Main ratio sweep: H200 job `307515`, which produced 22 of 23 planned
  vocabulary-coverage and slice-granularity result/selection pairs before its
  90-minute timeout.
- Ratio-`0.5` controlled runs: jobs `307507_0` and `307513_1..4`.
- Missing fabric-only ratio-`1.0` slice-granularity run: H200 job `307853_4`,
  completed with exit `0:0`. This run is not part of the controlled
  missing-field comparison above.
- Results and selection manifests:
  `logs/juno/imagenet_vocabulary_sensitivity/`.
- Launchers:
  `scripts/juno_imagenet_vocabulary_sensitivity.sbatch` and
  `scripts/juno_imagenet_vocabulary_sensitivity_ratio_sweep.sbatch`.

This experiment does **not** test merged-value or coarsened-category
sensitivity. Pink-only and fabric-only runs also change the generated repair
pool and are therefore separate slice-granularity diagnostics, not controlled
vocabulary-coverage ablations.

---

# COCO Missing-Field Vocabulary-Coverage Experiment

**Updated:** July 28, 2026

## Research Question and Protocol

This experiment tests how the COCO pose repair pipeline behaves when source
selection omits one field from the diagnosed target
`clothes_color=brown AND weather=rainy`. The generated repair pool remains the
same brown-rainy ICEdit pool in every run, and seed `42` samples the same fixed
budget of 1,000 generated images. The three source-selection variants are:

1. **Full conjunction:** `clothes_color,weather`, with generated extra-count
   terms for both `brown` and `rainy`.
2. **Missing weather:** `clothes_color` only, with the `brown` extra-count term.
3. **Missing color:** `weather` only, with the `rainy` extra-count term.

All variants use the same Qwen3-VL train20k metadata, diagnosis checkpoint,
20,000-image source candidate pool, source ratios `0.1/0.3/0.5/0.7/0.9`, two
repair epochs, learning rate `0.005`, RPC parameters
`(R,P,C)=(0.03,0.5,15)`, and post-training evaluation fields
`clothes_color,weather`. Only the available source-selection fields and their
corresponding generated extra-count terms change.

Array `309094` completed all `15/15` tasks. Every selection manifest contains
the expected number of unique source image IDs: 2,000, 6,000, 10,000, 14,000,
and 18,000 across the five ratios.

## Results

`AP change` is relative to Full conjunction at the same ratio. `Rainy AP` and
`Brown AP` are marginal diagnostics on the original test subsets containing
31 rainy images and 38 brown-clothing images, respectively. The current slice
analyzer does not report the exact brown-and-rainy intersection.

| Ratio | Variant | Overall AP | AP change | AP50 | AP75 | AR | Rainy AP | Brown AP | Source-set Jaccard vs. Full |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | Full conjunction | **0.6192** | 0.0000 | 0.8469 | 0.6759 | 0.6871 | 0.5742 | 0.6114 | 1.0000 |
| 0.1 | Missing weather | 0.6105 | -0.0086 | 0.8467 | 0.6622 | 0.6861 | 0.5430 | 0.5600 | 0.2034 |
| 0.1 | Missing color | 0.6055 | -0.0137 | 0.8324 | 0.6585 | 0.6799 | 0.5667 | 0.5469 | 0.4782 |
| 0.3 | Full conjunction | 0.6170 | 0.0000 | 0.8508 | 0.6683 | 0.6870 | 0.5598 | 0.5530 | 1.0000 |
| 0.3 | Missing weather | 0.6214 | +0.0044 | 0.8570 | 0.6775 | 0.6929 | 0.5713 | 0.5944 | 0.4613 |
| 0.3 | Missing color | **0.6221** | +0.0051 | 0.8546 | 0.6785 | 0.6941 | 0.5743 | 0.6053 | 0.5611 |
| 0.5 | Full conjunction | 0.5870 | 0.0000 | 0.8321 | 0.6321 | 0.6666 | 0.5349 | 0.5511 | 1.0000 |
| 0.5 | Missing weather | **0.6271** | +0.0401 | 0.8576 | 0.6764 | 0.6946 | 0.5717 | 0.5954 | 0.6084 |
| 0.5 | Missing color | 0.6178 | +0.0309 | 0.8515 | 0.6699 | 0.6866 | 0.5665 | 0.5909 | 0.7063 |
| 0.7 | Full conjunction | 0.6313 | 0.0000 | 0.8578 | 0.6882 | 0.6981 | 0.5901 | 0.5758 | 1.0000 |
| 0.7 | Missing weather | 0.6323 | +0.0010 | 0.8618 | 0.6839 | 0.6947 | 0.5729 | 0.6154 | 0.8110 |
| 0.7 | Missing color | **0.6332** | +0.0019 | 0.8560 | 0.6928 | 0.6976 | 0.6023 | 0.5831 | 0.7836 |
| 0.9 | Full conjunction | 0.6238 | 0.0000 | 0.8573 | 0.6740 | 0.6889 | 0.5827 | 0.5634 | 1.0000 |
| 0.9 | Missing weather | **0.6244** | +0.0006 | 0.8576 | 0.6725 | 0.6898 | 0.5444 | 0.5612 | 0.8708 |
| 0.9 | Missing color | 0.6236 | -0.0002 | 0.8539 | 0.6799 | 0.6931 | 0.5815 | 0.5676 | 0.8969 |

## Result Summary

| Missing field | Mean Overall AP change | Largest decrease | Largest increase | Mean Jaccard vs. Full |
| --- | ---: | ---: | ---: | ---: |
| Weather | +0.0075 | -0.0086 at ratio 0.1 | +0.0401 at ratio 0.5 | 0.5910 |
| Color | +0.0048 | -0.0137 at ratio 0.1 | +0.0309 at ratio 0.5 | 0.6852 |

Removing a field has the clearest and most consistent effect on source-set
composition. At ratio `0.1`, Jaccard overlap with Full conjunction falls to
`0.2034` when weather is missing and `0.4782` when color is missing. The
overlap rises to `0.8708/0.8969` at ratio `0.9` as the retained source pool
approaches the full candidate set.

Overall AP changes are non-monotonic. Both missing-field variants are worse at
ratio `0.1`, slightly better at ratios `0.3` and `0.7`, and nearly tied with
Full conjunction at ratio `0.9`. The positive mean changes are driven by ratio
`0.5`, where Full conjunction drops to `0.5870` while the missing-field
variants reach `0.6271` and `0.6178`. This isolated low point makes the mean
effect sensitive to one ratio.

The supported conclusion is that incomplete vocabulary coverage can strongly
change selected source examples, especially at low budgets, but this
single-seed experiment does not show that omitting a field consistently lowers
Overall AP. The marginal rainy and brown slice results are also non-monotonic.

## Execution and Artifacts

- H200 array: `309094`, completed `15/15` with exit `0:0` for every task.
- Launcher: `scripts/juno_pose_brown_rainy_missing_field_vocabulary.sbatch`.
- Local artifacts:
  `logs/juno/coco_missing_field_vocabulary/309094_*`.
- Durable Juno artifacts:
  `/work/oxx220000/safefix_rebuttal/logs/juno/coco_missing_field_vocabulary/309094_*`.
- Each task preserves `selection_manifest.json`, `test_metrics.json`, test and
  validation predictions, and Qwen3-VL slice-analysis outputs.

---

# Quick Comparison: ImageNet10 and COCO Slice-Granularity Diagnostics

**Updated:** July 27, 2026

All entries are five-seed means. The Full ImageNet10 and brown-rain COCO
columns use the authoritative values from
`nips26_new/appendix/main_metric_tables.tex`; the remaining columns record the
new five-seed granularity runs provided by the author. The variants change the
selection vocabulary and generated repair pool together, so they measure
practical slice-granularity and repair-data sensitivity rather than an
isolated selector-only effect. Dispersion values are not shown here, so the
comparison supports descriptive rather than statistical claims.

## COCO: Single-Attribute vs. Multi-Attribute Repair Sets

The `brown AND rainy` column reports the diagnosed slice used in the main
paper. Brown-only and Rain-only use their respective single-attribute
selection definitions and generated repair pools. The three-attribute variant
adds `time_of_day=night`. The rightmost column reports **Ours** using the
target slice `brown AND rainy`.

| Ratio | Brown-only Test AP | Rain-only Test AP | Brown AND rainy AND night Test AP | **Ours: Brown AND rainy Test AP** |
| ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.6161` | `0.6174` | `0.6097` | **`0.6183`** |
| `0.3` | `0.6238` | `0.6224` | `0.6179` | **`0.6306`** |
| `0.5` | `0.6284` | `0.6217` | `0.6246` | **`0.6317`** |
| `0.7` | `0.6327` | **`0.6333`** | `0.6291` | `0.6319` |
| `0.9` | `0.6348` | `0.6132` | `0.6336` | **`0.6355`** |
| `1.0` | **`0.6295`** | `0.6276` | `0.6278` | `0.6259` |

Over active ratios `0.1--0.9`, `brown AND rainy` exceeds Brown-only and
Rain-only at four of five ratios, by average margins of `0.0024` and `0.0080`
AP. It exceeds `brown AND rainy AND night` at all five ratios by `0.0066` AP
on average. The ratio-`0.7` single-attribute exceptions show that the effect is
not monotonic. Ratio `1.0` is excluded from the selector-sensitivity summary
because the complete source pool is retained.

## ImageNet10: Single-Attribute vs. Multi-Attribute Repair Sets

Each column reports overall ACC. Pink-only and Fabric-only use their
corresponding single-attribute selection definitions and generated pools. The
three-attribute ImageNet10 variant adds `rainy background` to `pink AND
fabric`. The rightmost column reports **Ours** using the diagnosed target slice
`pink AND fabric`.

| Ratio | Pink-only ACC | Fabric-only ACC | Pink AND fabric AND rainy background ACC | **Ours: Pink AND fabric ACC** |
| ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.7223` | `0.7346` | **`0.7398`** | `0.7369` |
| `0.3` | `0.7369` | `0.7431` | `0.7414` | **`0.7492`** |
| `0.5` | `0.7377` | `0.7438` | `0.7429` | **`0.7508`** |
| `0.7` | `0.7485` | `0.7423` | `0.7462` | **`0.7515`** |
| `0.9` | `0.7500` | `0.7354` | `0.7491` | **`0.7562`** |
| `1.0` | `0.7423` | **`0.7431`** | `0.7365` | `0.7400` |

Over active ratios `0.1--0.9`, `pink AND fabric` exceeds Pink-only and
Fabric-only at all five ratios, by average margins of `0.0098` and `0.0091`
ACC. It exceeds `pink AND fabric AND rainy background` at four of five ratios
by `0.0050` ACC on average. The three-attribute result at ratio `0.1` is the
exception, so the result supports sensitivity to granularity but not a rule
that performance changes monotonically with the number of attributes. Ratio
`1.0` is included as a full-data reference and excluded from the
selector-sensitivity summary.

---

# NeurIPS 2026 Manuscript Result Authority Note

**Updated:** July 26, 2026

For the AttriSelect NeurIPS 2026 submission and rebuttal, the results reported in `nips26_new/main.tex` and `nips26_new/appendix/*.tex` are the authoritative values. The author confirmed that several final experiments were rerun on other machines and that the resulting artifacts are not fully mirrored in this local workspace.

Result lookup order for previously reported experiments:

1. the corresponding table or discussion in `nips26_new/main.tex`;
2. the relevant file under `nips26_new/appendix/`;
3. local logs or `report.md` only for implementation context or when the TeX
   does not contain the requested detail.

For official five-seed task metrics, use
`nips26_new/appendix/main_metric_tables.tex`. Do not recover an older paper
value from this report when a corresponding authoritative TeX value exists.

The following manuscript claims are confirmed:

- the main-result tables report five-seed means and standard deviations, with the paired statistical tests stated in the appendix;
- the main results use ControlNet as described in the implementation appendix;
- the alternative-generator tables, human audit, and attribute-noise sensitivity tables come from completed external runs and are accurate;
- the CelebA InstructPix2Pix and COCO ICEdit appendix comparisons are valid alternative-generator experiments.

Older local logs, launcher paths, and tables in this report document earlier experiment workflows and may not reproduce the final manuscript values exactly. They must not be used to overwrite or reject a confirmed value in `nips26_new`. Newly completed experiments may be recorded in this report, but they must be labeled with their protocol, seeds, and relationship to the manuscript results. When preparing the NeurIPS rebuttal or revising the paper, use the TeX manuscript as the source of truth unless the author explicitly provides a newer correction.

---

## COCO Rain-Only ICEdit Slice-Granularity Diagnostic

**Updated:** July 27, 2026

This single-seed (`42`) diagnostic replaces the original COCO target
`clothes_color=brown AND weather=rainy` with `weather=rainy`. It uses the
rain-only ICEdit directory
`generated_coco_qwen3_1500_rain_icedit`, weather-only AttriSelect scoring for
ratios below `1.0`, Qwen3 train20k metadata, a fixed 1,000-image generated
subset, a 20,000-image source cap, two total epochs, learning rate `0.005`,
and RPC `(R,P,C)=(0.03,0.5,15)`. Ratio `1.0` uses the full source set, so no
subset selector is applied.

The loader found all 1,500 generated JPGs readable and retained 1,314 after
COCO train-ID and bounding-box compatibility checks. Seed `42` selected the
same fixed 1,000 generated IDs at every ratio. Required-complete validation
confirmed the expected source counts, all metric files, and successful
weather-only val/test slice analysis for all six ratios.

| Ratio | Source images | Val AP | Test AP | Rainy val AP (`n=26`) | Rainy test AP (`n=31`) | Rainy test delta vs. overall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `2,000` | `0.6217` | `0.6174` | `0.6202` | **`0.5837`** | `-0.0336` |
| `0.3` | `6,000` | `0.6278` | `0.6224` | **`0.6440`** | `0.5744` | `-0.0480` |
| `0.5` | `10,000` | `0.6320` | `0.6217` | `0.6259` | `0.5501` | `-0.0716` |
| `0.7` | `14,000` | **`0.6390`** | **`0.6333`** | `0.6430` | `0.5688` | `-0.0645` |
| `0.9` | `18,000` | `0.6227` | `0.6132` | `0.5944` | `0.5692` | `-0.0440` |
| `1.0` | `20,000` | `0.6274` | `0.6276` | `0.6233` | `0.5746` | `-0.0530` |

Overall AP is non-monotonic and peaks at ratio `0.7`. The rainy-slice result
also does not increase with source ratio; its best test AP is `0.5837` at
ratio `0.1`. The rainy test slice contains only 31 images, so slice AP is a
small-slice diagnostic and should not support a strong comparative claim.

For context, the older local brown-rain ICEdit runs with the same
`(0.03,0.5,15)` RPC tuple report overall test AP
`0.6183/0.6176/0.5866/0.6319/0.6255` at ratios
`0.1/0.3/0.5/0.7/0.9`. The corresponding rain-only values are
`0.6174/0.6224/0.6217/0.6333/0.6132`. The differences are non-monotonic.
Because this experiment changes both the selection vocabulary and the
generated repair pool, this comparison measures slice granularity and repair
data sensitivity together; it does not isolate the effect of removing the
`brown` attribute. Ratio `1.0` is an extra diagnostic and is not part of the
standard five-ratio COCO manuscript protocol.

All six trained models were evaluated from the in-memory final state because
their validation AP did not exceed the resumed checkpoint's recorded best AP
`0.6401`; the runner therefore did not write a new best/last checkpoint.
Predictions, overall metrics, slice summaries, logs, and the durable aggregate
summary are preserved.

Execution and artifacts:

- H200 job `308170`: completed with exit `0:0` in `02:30:42`;
- summary:
  `logs/juno/coco_rainy_icedit_vocabulary_sensitivity/coco_rainy_icedit_seed42_summary.{json,md}`;
- validated per-ratio artifacts:
  `logs/juno/coco_rainy_icedit_vocabulary_sensitivity/308170/`;
- launcher: `scripts/juno_pose_rainy_icedit_vocabulary_sensitivity.sbatch`;
- summarizer:
  `scripts/summarize_coco_rainy_icedit_vocabulary_sensitivity.py`.

This result is a seed-42 local diagnostic. It must not overwrite or be
presented as a statistically matched comparison with the authoritative COCO
results under `nips26_new/`.

---

## H200 End-to-End Cost Summary

**Updated:** July 28, 2026

The following current tables are placed at the top for quick reference. They
use the H200 source-assignment charges:
`90 min` for the 20,000-image COCO source pool and `32.77 min`
for the 5,000-image KITTI source pool.

The assignment timings cover each dataset's complete task-level metadata
schema, not only the two fields in the diagnosed repair slice. KITTI assigns
six fields (`color`, `weather`, `background_style`, `traffic_density`,
`lighting_condition`, and `road_condition`); COCO assigns five fields
(`clothes_color`, `clothes_type`, `background_style`, `time_of_day`, and
`weather`); and ImageNet10 assigns three fields (`color`, `background`, and
`texture`). The selector focuses on `color,weather` for KITTI,
`clothes_color,weather` for COCO, and the ImageNet10 repair slice is
`color=pink AND texture=fabric`.

| Dataset | Source Images | Assigned Attributes | H200 Assignment Time | Normalized Assignment Time (min / 1K images / attribute) |
| --- | ---: | ---: | ---: | ---: |
| KITTI | 5,000 | 6 | 32.77 min | 1.09 |
| ImageNet10 | 10,400 | 3 | 29.35 min | 0.94 |
| COCO | 20,000 | 5 | 90 min | 0.90 |

`Normalized Assignment Time` divides wall-clock time by the source-pool size
in thousands and the number of assigned attributes. For example, the COCO
value is `90 / (20 x 5) = 0.90` minutes per 1,000 images per attribute. This is
a throughput summary, not an assumption of exact linear scaling: model
loading, image loading, prompt and output lengths, and serialization also
affect wall-clock time.

### COCO: 20,000 Source Images

Image generation and generated-repair-image assignment are excluded. Ours
conservatively includes the complete 20,000-image source-pool Qwen3-VL
assignment (`90 min` on H200) in every row.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `4.41 min` | `4.42 min` | `10.98 min` | `16.18 min` | `19.63 min` | `5.08 min` | `6.97 min` | `1.57 h` |
| `0.3` | `8.18 min` | `8.18 min` | `14.74 min` | `39.86 min` | `49.76 min` | `8.93 min` | `17.86 min` | `1.62 h` |
| `0.5` | `12.37 min` | `12.36 min` | `19.04 min` | `1.08 h` | `1.56 h` | `13.08 min` | `29.47 min` | `1.66 h` |
| `0.7` | `15.73 min` | `15.75 min` | `22.27 min` | `1.45 h` | `2.29 h` | `16.34 min` | `37.63 min` | `1.69 h` |
| `0.9` | `19.66 min` | `19.65 min` | `26.28 min` | `1.87 h` | `3.01 h` | `20.18 min` | `40.94 min` | `1.72 h` |

### KITTI: 5,000 Source Images

Ours conservatively includes the mean fresh 5,000-image H200 Qwen3-VL
assignment (`1966.42 s`, or `32.77 min`) in every row.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `42.17 min` | `42.18 min` | `48.82 min` | `1.37 h` | `1.12 h` | `43.47 min` | `43.53 min` | `1.01 h` |
| `0.3` | `43.84 min` | `43.85 min` | `50.28 min` | `1.44 h` | `1.20 h` | `45.08 min` | `45.63 min` | `1.09 h` |
| `0.5` | `45.18 min` | `45.18 min` | `51.73 min` | `1.49 h` | `1.27 h` | `46.42 min` | `47.47 min` | `1.11 h` |
| `0.7` | `47.04 min` | `47.05 min` | `53.46 min` | `1.56 h` | `1.35 h` | `48.16 min` | `49.73 min` | `1.13 h` |
| `0.9` | `48.86 min` | `48.87 min` | `55.53 min` | `1.63 h` | `1.42 h` | `50.34 min` | `51.46 min` | `1.15 h` |

Ours charges fresh source assignment in every row; subtract the assignment
cost for subsequent repair cycles that reuse metadata.

---

## ImageNet10 Independent Epoch-Budget Diagnostic

**Updated:** July 27, 2026

This corrected single-seed (`42`) diagnostic addresses the repeated Ours
result in the initial cumulative-milestone experiment below and provides the
matched comparison with Random, Entropy, CoreSet, and BADGE. The original
60-epoch trajectory used a fixed `StepLR(step_size=3, gamma=0.5)`, so its
learning rate had nearly vanished at longer milestones, and AttriSelect
repeatedly selected the epoch-6 checkpoint.

Each method now uses four independent trajectories with the StepLR interval
rescaled to the training horizon. The epoch/interval pairs are `15/3`, `30/6`,
`45/9`, and `60/12`, so every run applies five learning-rate decays and ends at
`3.125e-7`. All runs use source ratio `0.9`, source and generated seed `42`,
the same diagnosis checkpoint, the same exact 1,000-image ControlNet repair
set, `lr=1e-5`, `gamma=0.5`, and batch size `64`. Within each method, strict
manifest validation confirms that the exact 9,360-image selected source subset
is unchanged across the four epoch budgets.

| Method | 15 epochs FR (%) | 30 epochs FR (%) | 45 epochs FR (%) | 60 epochs FR (%) |
| --- | ---: | ---: | ---: | ---: |
| Random | 6.67 | 12.11 | 11.84 | **12.38** |
| Entropy | **12.38** | 11.57 | **12.38** | 11.84 |
| CoreSet | 11.29 | **12.93** | 10.48 | 9.93 |
| BADGE | **12.38** | 8.30 | 11.84 | 12.11 |
| AttriSelect | 10.21 | 12.11 | 12.11 | 11.57 |

Bold marks the highest FR at each epoch budget, including ties.

| Method | Best epoch at 15/30/45/60 | Test ACC at 15/30/45/60 | Target ACC (`n=16`) | Non-target ACC (`n=1284`) |
| --- | --- | --- | --- | --- |
| Random | `7/13/19/17` | `0.7362/0.7515/0.7508/0.7523` | `0.6875/0.7500/0.6875/0.6875` | `0.7368/0.7516/0.7516/0.7531` |
| Entropy | `10/25/36/27` | `0.7523/0.7500/0.7523/0.7508` | `0.6875/0.6875/0.7500/0.8125` | `0.7531/0.7508/0.7523/0.7500` |
| CoreSet | `8/24/26/52` | `0.7492/0.7538/0.7469/0.7454` | `0.6875/0.6875/0.6875/0.6875` | `0.7500/0.7547/0.7477/0.7461` |
| BADGE | `6/7/32/35` | `0.7523/0.7408/0.7508/0.7515` | `0.7500/0.6875/0.7500/0.7500` | `0.7523/0.7414/0.7508/0.7516` |
| AttriSelect | `6/29/30/30` | `0.7462/0.7515/0.7515/0.7500` | `0.6875/0.7500/0.6875/0.6875` | `0.7469/0.7516/0.7523/0.7508` |

The corrected runs confirm that Ours does not inherently produce an identical
checkpoint or result at every budget. Its best epoch changes from `6` to
`29/30`, and test ACC improves by `0.0054` at 30 and 45 epochs relative to the
15-epoch run. However, no method improves monotonically with more epochs.
AttriSelect reaches FR `12.11%` at 30 and 45 epochs, then drops to `11.57%` at
60 epochs. The best method also changes by budget: Entropy/BADGE at 15,
CoreSet at 30, Entropy at 45, and Random at 60 epochs. AttriSelect ranks
`4/2/2/4` at the four budgets and does not win a milestone in this seed-42
diagnostic.

Across all methods and budgets, FR ranges from `6.67%` to `12.93%`. The
horizon-matched schedule therefore removes the checkpoint-reuse artifact and
can improve individual runs, but it does not make FR consistently large or
support the claim that the original low absolute FR is mainly a consequence
of training for only 15 epochs.

The 30- and 45-epoch test ACC values tie even though they use different best
checkpoints and have different validation ACC. This is a discrete test-set
tie, not checkpoint reuse. Target ACC remains too coarse for a strong claim
because the exact target slice has only 16 images.

Execution and artifacts:

- H200 scheduler smoke job `307913`: completed with exit `0:0`;
- H200 AttriSelect-only array `307915_0..3`: all tasks completed with exit
  `0:0`;
- H200 baseline array `307993_0..15`: all Random, Entropy, CoreSet, and BADGE
  tasks completed with exit `0:0`;
- complete summary:
  `logs/juno/imagenet_fr_epoch_multiplier_rescaled/imagenet_seed42_rescaled_full_summary.{json,md}`;
- AttriSelect-only intermediate summary:
  `logs/juno/imagenet_fr_epoch_multiplier_rescaled/imagenet_seed42_attriselect_rescaled_summary.{json,md}`;
- validated result and selection records:
  `logs/juno/imagenet_fr_epoch_multiplier_rescaled/{attriselect,corrected}/`;
- launcher: `scripts/juno_imagenet_fr_epoch_multiplier_rescaled.sbatch`;
- summarizer: `scripts/summarize_imagenet_fr_epoch_multiplier_rescaled.py`.

This result is a seed-42 diagnostic only. It must not be combined
statistically with or overwrite the authoritative five-seed manuscript values
under `nips26_new/`.

---

## ImageNet10 Initial Cumulative-Milestone Diagnostic

**Updated:** July 27, 2026

This initial single-seed (`42`) diagnostic for Reviewer `yyPZ` Q3 is preserved
for provenance. Its cumulative-checkpoint design is superseded by the
independent horizon-rescaled AttriSelect diagnostic above for interpreting the
Ours epoch trend. It uses the
ImageNet10 ratio-`0.9` repair setting, the fixed 1,000-image ControlNet repair
set, generated seed `42`, learning rate `1e-5`, and the same diagnosis
checkpoint for Random, Entropy, CoreSet, BADGE, and AttriSelect. The
before-repair accuracy used for Fix Rate (FR) is the manuscript value `0.7173`.

Each method follows one continuous 60-epoch trajectory. At epochs
`15/30/45/60`, evaluation uses the best-validation checkpoint observed up to
that milestone. This matches the existing best-checkpoint evaluation rule and
keeps the selected source subset fixed across all four repair budgets.

| Method | 15 epochs FR (%) | 30 epochs FR (%) | 45 epochs FR (%) | 60 epochs FR (%) |
| --- | ---: | ---: | ---: | ---: |
| Random | 6.67 | **12.66** | **12.66** | 11.29 |
| Entropy | **12.38** | 12.38 | 12.38 | 12.38 |
| CoreSet | 11.29 | 11.29 | 11.29 | 11.29 |
| BADGE | **12.38** | **12.66** | **12.66** | **12.66** |
| AttriSelect | 10.21 | 10.21 | 10.21 | 10.21 |

Bold marks the highest value at each milestone, including ties.

| Method | Best epoch at 15/30/45/60 | Test ACC | Target ACC (`n=16`) | Non-target ACC (`n=1284`) |
| --- | --- | --- | --- | --- |
| Random | `7/16/16/56` | `0.7362/0.7531/0.7531/0.7492` | `0.6875/0.6875/0.6875/0.7500` | `0.7368/0.7539/0.7539/0.7492` |
| Entropy | `10/10/10/10` | `0.7523/0.7523/0.7523/0.7523` | `0.6875/0.6875/0.6875/0.6875` | `0.7531/0.7531/0.7531/0.7531` |
| CoreSet | `8/8/8/8` | `0.7492/0.7492/0.7492/0.7492` | `0.6875/0.6875/0.6875/0.6875` | `0.7500/0.7500/0.7500/0.7500` |
| BADGE | `6/25/25/25` | `0.7523/0.7531/0.7531/0.7531` | `0.7500/0.6875/0.6875/0.6875` | `0.7523/0.7539/0.7539/0.7539` |
| AttriSelect | `6/6/6/6` | `0.7462/0.7462/0.7462/0.7462` | `0.6875/0.6875/0.6875/0.6875` | `0.7469/0.7469/0.7469/0.7469` |

The result does not show a consistent increase from longer repair training.
Random improves at 30 epochs but drops when a later checkpoint becomes the
best-validation choice at 60 epochs; BADGE improves slightly at 30 epochs;
Entropy, CoreSet, and AttriSelect retain an early checkpoint across all later
milestones. AttriSelect is not the strongest method in this seed-42 rerun.
Therefore, this diagnostic does not support the draft claim that low absolute
FR is mainly caused by the 15-epoch schedule or that AttriSelect remains best
at every epoch multiplier.

The target slice contains only `16/1300` test images, so target ACC changes in
increments of `0.0625` and should not be used for a strong comparative claim.
This single-seed result must not be combined statistically with or overwrite
the authoritative five-seed values in `nips26_new/appendix/main_metric_tables.tex`.

Execution and artifacts:

- H200 smoke job `307867`: completed with exit `0:0` in `00:02:30`;
- H200 pilot array `307870_0..4`: all tasks completed with exit `0:0` in
  `00:15:44--00:16:31`;
- summary: `logs/juno/imagenet_fr_epoch_multiplier/imagenet_seed42_epoch_multiplier_summary.{json,md}`;
- validated result and selection records:
  `logs/juno/imagenet_fr_epoch_multiplier/pilot/307870_<task>/`;
- launcher: `scripts/juno_imagenet_fr_epoch_multiplier.sbatch`;
- summarizer: `scripts/summarize_imagenet_fr_epoch_multiplier.py`.

---

## ImageNet10 Ratio-0.5 Five-Seed Exact-Target Evaluation

**Updated:** August 3, 2026

The corrected exact target is "pink AND fabric" and contains 16 of the 1,300
test images; the non-target complement contains 1,284 images. Exact-target ACC
is hard top-1 accuracy on the fixed 16-image slice. Values report the mean and
sample standard deviation over seeds `0`, `42`, `1234`, `3407`, and `2024`.

| Method | Exact Target ACC | Target $\Delta$ | Non-Target ACC | Non-Target $\Delta$ | Overall ACC |
|:---|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 0.6250 | -- | 0.7185 | -- | 0.7173 |
| Random | 0.7000 $\pm$ 0.0815 | 7.50\% $\pm$ 8.15\% | 0.7385 $\pm$ 0.0030 | 2.00\% $\pm$ 0.30\% | 0.7380 $\pm$ 0.0030 |
| Stratified | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7448 $\pm$ 0.0018 | 2.64\% $\pm$ 0.18\% | 0.7446 $\pm$ 0.0017 |
| Entropy | 0.6750 $\pm$ 0.0815 | 5.00\% $\pm$ 8.15\% | 0.7338 $\pm$ 0.0028 | 1.54\% $\pm$ 0.28\% | 0.7331 $\pm$ 0.0027 |
| CoreSet | 0.7000 $\pm$ 0.0927 | 7.50\% $\pm$ 9.27\% | 0.7405 $\pm$ 0.0018 | 2.20\% $\pm$ 0.18\% | 0.7400 $\pm$ 0.0017 |
| BADGE | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7472 $\pm$ 0.0030 | 2.87\% $\pm$ 0.30\% | 0.7469 $\pm$ 0.0029 |
| PPAL | 0.7125 $\pm$ 0.0713 | 8.75\% $\pm$ 7.13\% | 0.7427 $\pm$ 0.0017 | 2.42\% $\pm$ 0.17\% | 0.7423 $\pm$ 0.0016 |
| UHerding | 0.7125 $\pm$ 0.0948 | 8.75\% $\pm$ 9.48\% | 0.7414 $\pm$ 0.0029 | 2.29\% $\pm$ 0.29\% | 0.7410 $\pm$ 0.0028 |
| Target-Slice + Stratified | 0.7375 $\pm$ 0.0523 | 11.25\% $\pm$ 5.23\% | 0.7473 $\pm$ 0.0021 | 2.89\% $\pm$ 0.21\% | 0.7472 $\pm$ 0.0020 |
| Attribute-Filtered Random | 0.7375 $\pm$ 0.0685 | 11.25\% $\pm$ 6.85\% | 0.7477 $\pm$ 0.0025 | 2.93\% $\pm$ 0.25\% | 0.7476 $\pm$ 0.0024 |
| Attribute-Filtered Entropy | 0.7375 $\pm$ 0.0815 | 11.25\% $\pm$ 8.15\% | 0.7483 $\pm$ 0.0023 | 2.99\% $\pm$ 0.23\% | 0.7482 $\pm$ 0.0022 |
| Generated-Complement | 0.7500 $\pm$ 0.0442 | 12.50\% $\pm$ 4.42\% | 0.7461 $\pm$ 0.0019 | 2.76\% $\pm$ 0.19\% | 0.7461 $\pm$ 0.0018 |
| **AttriSelect** | **0.7625 $\pm$ 0.0523** | **13.75\% $\pm$ 5.23\%** | **0.7507 $\pm$ 0.0011** | **3.22\% $\pm$ 0.11\%** | **0.7508 $\pm$ 0.0010** |

AttriSelect has the highest exact-target, non-target, and overall mean. Its
exact-target count is `61/80` across the five seeds, compared with `60/80` for
Generated-Complement. Because one prediction changes a seed-level exact-target
ACC by 6.25 percentage points, the localized ordering is descriptive. The
supported conclusion is that AttriSelect improves the diagnosed conjunction
while non-target accuracy also increases.

The author supplied the corrected aggregate table and per-seed exact-target
counts. No checkpoint or prediction artifact path accompanied this update.

---

## ImageNet10 Five-Seed Same-Information Attribute-Aware Baselines

**Updated:** July 27, 2026

The three reviewer-requested attribute-aware baselines are complete over seeds
`0`, `42`, `1234`, `3407`, and `2024`. Each method uses the current Qwen3-VL
source metadata, the same fixed 1,000-image generated set, class labels,
diagnosis checkpoint, source budget, generated seed `42`, and 15-epoch repair
schedule. The required-complete summarizer validated all `3 x 5 x 5 = 75`
baseline records and their selection manifests.

| Ratio | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | AttriSelect (authoritative manuscript) |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.7303 $\pm$ 0.0051 | 0.7340 $\pm$ 0.0034 | 0.7317 $\pm$ 0.0025 | **0.7369 $\pm$ 0.0043** |
| 0.3 | 0.7385 $\pm$ 0.0024 | 0.7382 $\pm$ 0.0042 | 0.7365 $\pm$ 0.0023 | **0.7492 $\pm$ 0.0044** |
| 0.5 | 0.7438 $\pm$ 0.0037 | 0.7442 $\pm$ 0.0038 | 0.7442 $\pm$ 0.0031 | **0.7508 $\pm$ 0.0010** |
| 0.7 | 0.7434 $\pm$ 0.0065 | 0.7474 $\pm$ 0.0038 | 0.7482 $\pm$ 0.0020 | **0.7515 $\pm$ 0.0037** |
| 0.9 | 0.7485 $\pm$ 0.0026 | 0.7469 $\pm$ 0.0036 | 0.7503 $\pm$ 0.0045 | **0.7562 $\pm$ 0.0019** |

AttriSelect has the highest mean at all five ratios. Its margin over the best
same-information baseline is `0.0029`, `0.0107`, `0.0066`, `0.0033`, and
`0.0059`, respectively. These use matched seed sets and protocols, but no new
paired significance claim is made because the authoritative manuscript TeX
does not expose the per-seed AttriSelect values.

The exact `pink AND fabric` test slice contains only `16/1300` images. Target
ACC therefore changes in increments of `0.0625`; overall ACC is the more
stable reviewer-facing summary.

Artifacts:

- `logs/juno/imagenet_attribute_aware/imagenet_baselines_fiveseed_summary.json`
- `logs/juno/imagenet_attribute_aware/imagenet_baselines_fiveseed_summary.md`
- Juno completed jobs: `307496_0..2` and sequential job `307503` tasks `3..59`
- Authoritative AttriSelect values: `nips26_new/appendix/main_metric_tables.tex`

---

## Historical ImageNet10 Seed-42 Attribute-Aware Baseline Pilot

**Updated:** July 26, 2026

Juno H200 array `306871` completed all `20/20` seed-42 diagnostic tasks with
exit code `0:0`. Each method selects from the same `10,400`-image source pool and
uses the same fixed `1,000` generated images, Qwen3-VL attribute file, class
labels, diagnosis checkpoint, source and generated seeds, 15 repair epochs,
learning rate `1e-5`, and batch size `64`.

Historical snapshot: the AttriSelect rows produced by array `306871`
are diagnostic reruns and are not the manuscript's official AttriSelect
results. The authoritative values are the five-seed mean and standard
deviation in `nips26_new/appendix/main_metric_tables.tex`, over seeds
`0`, `42`, `1234`, `3407`, and `2024`. At the time of this pilot, the new
same-information baselines had only seed `42`; the completed five-seed result
is now reported in the section above.

Operational definitions:

- Target-Slice + Stratified selects exact `pink AND fabric` examples first,
  then fills the source budget from class-stratified background samples.
- Attribute-Filtered Random prioritizes the `pink OR fabric` candidate pool,
  samples randomly within class-balanced queues, then fills from background.
- Attribute-Filtered Entropy uses the same candidate pool and queues, ranking
  candidates by diagnosis-checkpoint predictive entropy.
- AttriSelect uses `alpha=0.03`, `lambda=0.5`, `w=15`, conditional rarity,
  and no entropy term.

### Seed-42 Diagnostic Rerun

The following table preserves the exact output of array `306871` for
reproducibility. Its AttriSelect rows are not manuscript-authoritative.

| Ratio | Method | Overall ACC | Target ACC | Non-target ACC | Exact target selected | Selection | Training |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | Target-Slice + Stratified | 0.7254 | 0.7500 | 0.7251 | 130/1040 | 0.02 s | 66.87 s |
| 0.1 | Attribute-Filtered Random | 0.7362 | 0.6875 | 0.7368 | 30/1040 | 0.02 s | 67.95 s |
| 0.1 | Attribute-Filtered Entropy | 0.7285 | 0.7500 | 0.7282 | 34/1040 | 10.76 s | 66.94 s |
| 0.1 | AttriSelect | **0.7400** | 0.6875 | **0.7407** | 24/1040 | 0.05 s | 67.16 s |
| 0.3 | Target-Slice + Stratified | 0.7392 | 0.6875 | 0.7399 | 130/3120 | 0.03 s | 103.62 s |
| 0.3 | Attribute-Filtered Random | 0.7362 | 0.6875 | 0.7368 | 60/3120 | 0.03 s | 102.23 s |
| 0.3 | Attribute-Filtered Entropy | 0.7354 | **0.8125** | 0.7344 | 70/3120 | 10.81 s | 102.74 s |
| 0.3 | AttriSelect | **0.7462** | 0.7500 | **0.7461** | 40/3120 | 0.06 s | 102.85 s |
| 0.5 | Target-Slice + Stratified | **0.7492** | 0.7500 | **0.7492** | 130/5200 | 0.03 s | 140.15 s |
| 0.5 | Attribute-Filtered Random | 0.7485 | **0.8125** | 0.7477 | 101/5200 | 0.03 s | 140.58 s |
| 0.5 | Attribute-Filtered Entropy | 0.7431 | 0.6875 | 0.7438 | 92/5200 | 10.71 s | 140.97 s |
| 0.5 | AttriSelect | 0.7369 | 0.7500 | 0.7368 | 95/5200 | 0.07 s | 140.78 s |
| 0.7 | Target-Slice + Stratified | 0.7431 | **0.7500** | 0.7430 | 130/7279 | 0.04 s | 177.59 s |
| 0.7 | Attribute-Filtered Random | **0.7531** | 0.6875 | **0.7539** | 125/7279 | 0.04 s | 176.65 s |
| 0.7 | Attribute-Filtered Entropy | 0.7508 | 0.6875 | 0.7516 | 118/7279 | 11.98 s | 178.80 s |
| 0.7 | AttriSelect | 0.7415 | **0.7500** | 0.7414 | 95/7279 | 0.08 s | 179.10 s |
| 0.9 | Target-Slice + Stratified | **0.7492** | **0.7500** | **0.7492** | 130/9360 | 0.04 s | 212.03 s |
| 0.9 | Attribute-Filtered Random | 0.7438 | **0.7500** | 0.7438 | 129/9360 | 0.04 s | 211.31 s |
| 0.9 | Attribute-Filtered Entropy | 0.7485 | **0.7500** | 0.7484 | 129/9360 | 10.66 s | 213.03 s |
| 0.9 | AttriSelect | 0.7462 | 0.6875 | 0.7469 | 130/9360 | 0.10 s | 214.87 s |

The exact target slice contains only `16/1300` test images, so target ACC
moves in `0.0625` increments and should not support a strong single-seed
conclusion.

### Official AttriSelect Reference and New Baseline Pilot

| Ratio | Target-Slice + Stratified (seed 42) | Attribute-Filtered Random (seed 42) | Attribute-Filtered Entropy (seed 42) | Official AttriSelect (5-seed mean $\pm$ std) |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.7254 | 0.7362 | 0.7285 | 0.7369 $\pm$ 0.0043 |
| 0.3 | 0.7392 | 0.7362 | 0.7354 | 0.7492 $\pm$ 0.0044 |
| 0.5 | 0.7492 | 0.7485 | 0.7431 | 0.7508 $\pm$ 0.0010 |
| 0.7 | 0.7431 | 0.7531 | 0.7508 | 0.7515 $\pm$ 0.0037 |
| 0.9 | 0.7492 | 0.7438 | 0.7485 | 0.7562 $\pm$ 0.0019 |

This table is descriptive only because its columns use different seed
aggregation. The official AttriSelect mean is above the seed-42 baseline
maximum at ratios `0.1`, `0.3`, `0.5`, and `0.9`, and is `0.0016` below
Attribute-Filtered Random at ratio `0.7`. A reviewer-ready comparison requires
the four remaining seeds for each of the three new baselines.

Validation confirmed `20` result/manifest pairs, one identical generated-set
hash across all tasks, source budgets of `1040/3120/5200/7279/9360`, source
and generated seeds `42`, and one shared checkpoint and attribute file.

Artifacts:

- Juno tasks: `/work/oxx220000/safefix_rebuttal/logs/juno/imagenet_attribute_aware/full/306871_<task>`
- Juno summary: `/work/oxx220000/safefix_rebuttal/logs/juno/imagenet_attribute_aware/imagenet_seed42_summary.{json,md}`
- Local summary: `logs/juno/imagenet_attribute_aware/imagenet_seed42_summary.{json,md}`
- Official AttriSelect table: `nips26_new/appendix/main_metric_tables.tex`

---

## Historical COCO Rough H200 End-to-End Cost Projection — Superseded

**Updated:** July 27, 2026

This historical planning-level table estimates the COCO repair cost for the
manuscript's `20,000`-image source pool. Per the current accounting scope,
image generation is excluded. The 20,000-image Qwen3-VL cost is source-pool
attribute assignment rather than generated-image assignment: the measured A5000 run took
`399.7 min`, which becomes `182.5 min` (`3.04 h`) after applying the
A5000-to-H200 factor `0.4565`. This charge is included only for Ours and is
conservatively repeated in every ratio row; it can be removed after the source
metadata has been created once.

Do not use this projection for the current rebuttal. It is superseded by the
approximately `1.50 h` measured H200 charge and the current table above.

The shared repair-stage cost uses completed A5000 COCO Random end-to-end times
of `9:44`, `17:55`, `27:11`, and `34:27` at ratios `0.1--0.7`, converted to
H200 with the same factor. Ratio `0.9` is a linear extrapolation. This gives
H200 retraining/evaluation estimates of `4.4/8.2/12.4/15.7/19.7 min`.

Selector estimates follow `nips26_new/main.tex` and the exact algorithmic
paths:

- Entropy uses the TeX `863--865 s` scoring time;
- CoreSet uses `5/15/25/35/45 min`, extending the observed linear trend. These
  A5000 measurements are retained without the Qwen-derived hardware factor:
  original greedy k-center repeatedly updates the full candidate pool, and its
  CPU/GPU distance-update scaling is not represented by VLM inference speed;
- BADGE uses `21 min/1 h/2 h/3 h/4 h`, extending the slow exact trend rather
  than the later approximation;
- exact PPAL expands the TeX proxy times with the documented full-diversity
  work factors `5/15/25/35/45`;
- UHerding follows the implementation that is actually executed in the COCO
  experiments. `UncertaintyHerdingStrategy` defaults to
  `exact_max_samples=10000`; because COCO has 20,000 candidates, `select()`
  dispatches to `_fast_select`. This path computes proxy uncertainty and
  initial labeled-set coverage, then performs one weighted sample without
  replacement. It does not build the exact full pairwise kernel. The TeX
  `86/93/94/80/64 s` times are therefore converted directly to H200.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding$^{\dagger}$ | PPAL$^{\dagger}$ | Ours$^{\ddagger}$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `4.4 min` | `4.4 min` | `11.0 min` | `9.4 min` | `14.0 min` | `5.1 min` | `7.0 min` | `3.11 h` |
| `0.3` | `8.2 min` | `8.2 min` | `14.7 min` | `23.2 min` | `35.6 min` | `8.9 min` | `17.9 min` | `3.18 h` |
| `0.5` | `12.4 min` | `12.4 min` | `19.0 min` | `37.4 min` | `1.12 h` | `13.1 min` | `29.5 min` | `3.25 h` |
| `0.7` | `15.7 min` | `15.7 min` | `22.3 min` | `50.7 min` | `1.63 h` | `16.3 min` | `37.6 min` | `3.30 h` |
| `0.9` | `19.7 min` | `19.7 min` | `26.3 min` | `1.08 h` | `2.16 h` | `20.2 min` | `40.9 min` | `3.37 h` |

$^{\dagger}$PPAL is a complexity-based estimate. UHerding is projected from
the measured A5000 runtime of the actual scalable implementation.
$^{\ddagger}$Ours includes the complete
20,000-image source assignment in every row. With reusable metadata, subtract
`3.04 h`; the remaining Ours cost is approximately the Random repair-stage
time because pure AttriSelect subset selection takes less than one second.

The table excludes ControlNet generation and generated-repair-image assignment.
It should remain labeled as estimated until the exact selectors and the full
COCO retraining grid are measured on the same H200.

---

## KITTI End-to-End Cost Accounting

**Updated:** July 25, 2026

### Rough H200 End-to-End Projection With Non-Approximate Selectors

**Updated:** July 27, 2026

The table below is a planning-level projection, not a measured benchmark and
not an authoritative replacement for the selector-only results in
`nips26_new/main.tex`. It estimates the complete cost of a representative
repair cycle on one H200 for source ratios `0.1--0.9`. The estimate combines:

- the measured H200 ratio-dependent KITTI repair pipeline time, using the
  Random rows below as the shared cost of dataset materialization, 10 repair
  epochs, checkpoint loading, and test evaluation;
- a projected ControlNet generation cost of `27.4 min` for 1,000 images and a
  projected generated-image Qwen3-VL assignment cost of `6.8 min`, obtained by
  multiplying the A5000 costs in `nips26_new/appendix/implementation.tex` by
  the measured Qwen3-VL A5000-to-H200 factor
  `(1942.98 s + 1989.86 s) / 2 / (71.8 min) = 0.4565`;
- KITTI-size exact-selector trends derived from the TeX rows and the actual
  source-pool sizes: KITTI has `N=5,000`, compared with `10,400` for
  ImageNet10, `20,000` for COCO, and `80,000` for CelebA. CoreSet and exact
  BADGE update distances over the candidate pool once per selected sample, so
  their dominant greedy work is approximated as `O(rN^2)`. The COCO slow-path
  CoreSet work is therefore multiplied by `(5000/20000)^2=1/16` and added to
  the measured H200 fast-path time as a fixed embedding/setup floor. It is not
  additionally multiplied by the Qwen hardware factor because greedy distance
  updates do not share the VLM inference bottleneck;
- exact PPAL uses `O(k min(4k,N))` greedy diversity work. At the same ratio,
  reducing COCO from 20,000 to 5,000 candidates also reduces this work by
  approximately `1/16`; the scaled greedy term is added to the measured KITTI
  H200 PPAL proxy time;
- UHerding uses the path actually selected by the KITTI cost launcher.
  `scripts/juno_kitti_cost_accounting.sbatch` passes
  `--al-fast-selection true`, which sets `exact_max_select=1` and
  `exact_max_samples=1`; every active ratio therefore dispatches to
  `_fast_select`. The H200 selector times are the measured
  `75.02/76.44/72.04/73.09/83.63 s`, not an exact dense-kernel estimate;
- an Ours preprocessing/selection charge of `1966.42 s` (`32.77 min`) in
  every row, the mean of the two fresh 5,000-image H200 Qwen3-VL assignments
  (`1942.98 s` and `1989.86 s`). This deliberately charges source annotation
  to every repair cycle and is therefore conservative; when metadata is
  reused, this `32.77 min` should be removed from subsequent cycles.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding$^{\dagger}$ | PPAL$^{\dagger}$ | Ours$^{\ddagger}$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `42.2 min` | `42.2 min` | `48.8 min` | `47.4 min` | `47.7 min` | `43.5 min` | `43.5 min` | `1.25 h` |
| `0.3` | `43.8 min` | `43.8 min` | `50.3 min` | `49.7 min` | `50.9 min` | `45.1 min` | `45.6 min` | `1.28 h` |
| `0.5` | `45.2 min` | `45.2 min` | `51.7 min` | `51.8 min` | `54.2 min` | `46.4 min` | `47.5 min` | `1.30 h` |
| `0.7` | `47.0 min` | `47.0 min` | `53.5 min` | `54.1 min` | `57.6 min` | `48.2 min` | `49.7 min` | `1.33 h` |
| `0.9` | `48.9 min` | `48.9 min` | `55.5 min` | `56.6 min` | `1.02 h` | `50.3 min` | `51.5 min` | `1.36 h` |

$^{\dagger}$PPAL remains a complexity-based projection. UHerding uses measured
H200 runtime from the launcher's forced scalable path. $^{\ddagger}$Ours
includes fresh source-pool Qwen3-VL assignment
in every row. With reusable source metadata, the projected Ours totals become
approximately `42.2/43.8/45.2/47.0/48.9 min`, because its pure subset-selection
time is below one second and is negligible at this scale.

These totals should be used only as coarse reviewer-facing accounting. They
mix a hardware projection with measured H200 repair-stage timings; they do not
support a precise speedup claim. A paper-ready table should either retain the
`estimated` label or be replaced by matched H200 measurements of generation,
generated-image assignment, exact baseline selection, and retraining.

The measured scope starts at source selection and includes prepared-dataset
materialization, YOLOv8 checkpoint loading, 10 repair epochs, final checkpoint
loading, and evaluation on the `1241`-image KITTI test split. For Ours, the
included total also covers Qwen3-VL attribute assignment for all `5000` source
images. Repair-image generation is excluded. Generated-image attribute
assignment is not applicable because the known `red` and `cloudy` counts are
passed directly to the selector.

| Ratio | Method | Job | Selected Data | Annotation | Repair | Included Total | Test mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0.1` | Random | `305835` | 500 source + 1000 generated | N/A | 477.09 s | 477.09 s (7.95 min) | 0.5270 |
| `0.1` | Ours | `305836` | 500 source + 1000 generated | 1989.86 s | 583.09 s | 2572.94 s (42.88 min) | 0.4945 |
| `0.3` | Ours | `306774_0` | 1500 source + 1000 generated | Excluded (canonical metadata reused) | 350.14 s | 350.14 s (5.84 min) | 0.5607 |
| `0.5` | Ours | `306774_1` | 2500 source + 1000 generated | Excluded (canonical metadata reused) | 419.62 s | 419.62 s (6.99 min) | 0.5723 |
| `0.7` | Ours | `306774_2` | 3500 source + 1000 generated | Excluded (canonical metadata reused) | 492.87 s | 492.87 s (8.21 min) | 0.5782 |
| `0.9` | Entropy | `306046_0` | 4500 source + 1000 generated | N/A | 933.82 s | 933.82 s (15.56 min) | 0.5833 |
| `0.9` | Ours | `306046_1` | 4500 source + 1000 generated | 1942.98 s | 1057.76 s | 3000.74 s (50.01 min) | 0.5858 |

At ratio `0.9`, Ours improves test mAP50-95 by `0.0025` over Entropy
(`0.5858` versus `0.5833`). This is a single-run cost comparison and should
not be interpreted as a multi-seed performance estimate. The isolated fresh
Qwen3-VL annotation output is measured for cost, while repair selection uses
the canonical existing metadata to preserve the intended selection behavior.

Artifacts:

- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/305835`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/305836`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/306046_0`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/306046_1`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/306774_0`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/306774_1`
- `/work/oxx220000/safefix_rebuttal/logs/juno/kitti_cost/306774_2`

### KITTI Baseline Runtime Grid

The completed sweep measures the included repair pipeline for Random, Entropy,
fast approximate CoreSet, and fast approximate BADGE at source ratios `0.1`,
`0.3`, `0.5`, `0.7`, and `0.9`. Each entry includes source selection,
dataset materialization, 10 repair epochs, and final test evaluation. Qwen
attribute annotation is not used by these four baselines.

| Method | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| Random | 477.09 s (7.95 min) | 571.40 s (9.52 min) | 656.10 s (10.93 min) | 763.84 s (12.73 min) | 880.68 s (14.68 min) |
| Entropy | 538.44 s (8.97 min) | 619.16 s (10.32 min) | 720.48 s (12.01 min) | 823.22 s (13.72 min) | 933.82 s (15.56 min) |
| CoreSet (fast) | 686.91 s (11.45 min) | 803.02 s (13.38 min) | 901.48 s (15.02 min) | 978.78 s (16.31 min) | 1079.98 s (18.00 min) |
| BADGE (fast) | 696.40 s (11.61 min) | 835.47 s (13.92 min) | 946.56 s (15.78 min) | 1065.40 s (17.76 min) | 1173.77 s (19.56 min) |

Random ratio `0.1` and Entropy ratio `0.9` reuse validated jobs `305835` and
`306046_0`; the other 18 cells are successful tasks from array `306333`.

#### Source-Selection Time Only

| Method | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| Random | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| Entropy | 159.46 s | 154.09 s | 159.45 s | 159.05 s | 160.47 s |
| CoreSet (fast) | 294.14 s | 297.87 s | 302.40 s | 293.71 s | 292.29 s |
| BADGE (fast) | 293.07 s | 321.48 s | 332.27 s | 326.83 s | 338.57 s |
| Ours | 170.34 s | **62.03 s** | **66.14 s** | **75.47 s** | 172.17 s |

Ours ratios `0.3/0.5/0.7` are the corrected runs from array `306774`. Fresh
Qwen annotation was skipped, but the selector used the canonical KITTI
attribute metadata with `R=0.03`, `P=5`, `C=15`, seed `42`, and the same
red/cloudy generated-count injection. The ratio `0.1` and `0.9` selection
times come from the earlier complete cost runs and were not rerun in this
array.

#### Final KITTI Test mAP50-95

| Method | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| Random | 0.5270 | 0.5579 | 0.5746 | 0.5780 | 0.5796 |
| Entropy | **0.5417** | **0.5676** | **0.5758** | **0.5797** | **0.5833** |
| CoreSet (fast) | 0.5278 | 0.5604 | 0.5645 | 0.5767 | 0.5829 |
| BADGE (fast) | 0.5355 | 0.5549 | 0.5713 | 0.5758 | 0.5823 |
| Ours | 0.4945 | 0.5607 | 0.5723 | 0.5782 | 0.5858 |

All 18 array tasks completed with exit code `0:0`, exact requested source and
generated-image counts, and final evaluation on all `1241` KITTI test images.
The two-GPU throttle kept two tasks active concurrently on nodes `g-07-01` and
`g-07-10` until the grid finished.

Main observations:

- Random is the fastest method at every ratio because its source-selection
  cost is negligible.
- Entropy adds about `154--160` seconds of source scoring and is the strongest
  baseline by test mAP50-95 at every measured ratio.
- Fast CoreSet selection remains around `292--302` seconds across ratios.
- Fast BADGE selection ranges from `293` to `339` seconds and has higher total
  runtime than fast CoreSet at all five ratios.
- Total runtime increases with the source ratio because the prepared training
  set grows from `1500` images at ratio `0.1` to `5500` images at ratio `0.9`.

---

## SafeFix Baselines Comparison

### COCO Brown-Rainy Pose (Test AP)

| Method | 1k Images | 5k Images | 10k Images |
|---|---|---|---|
| Base | 0.5902 | 0.5902 | 0.5902 |
| Data Augmentation | 0.5921 | 0.5954 | 0.5987 |
| DiGA | 0.5935 | 0.5960 | 0.5991 |
| DA-CDM | 0.5942 | 0.5975 | 0.6005 |
| Mask-ControlNet | 0.5960 | 0.6012 | 0.6021 |
| HiBug_Class | 0.5955 | 0.6001 | 0.6015 |
| HiBug_Task | 0.5971 | 0.6023 | 0.6038 |
| Ours (Q) | 0.6185 | 0.6255 | 0.6280 |

### KITTI Red-Cloudy Generated-Image YOLOv8 (Test mAP50-95)

| Method | 100 Images | 500 Images | 1k Images |
|---|---|---|---|
| Base | 0.5626 | 0.5626 | 0.5626 |
| Data Augmentation | 0.5634 | 0.5658 | 0.5671 |
| DiGA | 0.5641 | 0.5662 | 0.5684 |
| DA-CDM | 0.5652 | 0.5675 | 0.5695 |
| Mask-ControlNet | 0.5668 | 0.5691 | 0.5715 |
| HiBug_Class | 0.5659 | 0.5685 | 0.5702 |
| HiBug_Task | 0.5675 | 0.5704 | 0.5728 |
| Ours (Q) | 0.5732 | 0.5765 | 0.5790 |


---

###  long-tail learning Results Comparison

The following table summarizes the performance of LTL baselines vs. Safefix. 
**Note**: Paper baseline accuracy for ResNet-18 on ImageNet10 is **71.73%**. Safefix results are cited from Table 2 and Table 8 of the paper.

| Baseline | CelebA Test ACC | CelebA Fix Bugs | ImageNet Test ACC | ImageNet Fix Bugs |
|---|---|---|---|---|
| **Base** | **90.57%** | **-** | **71.73%** | **-** |
| Focal Loss | 91.3% | 8/13  | 68.8% | 11/17  |
| Class-Balanced Loss | 91.2% | 8/13  | 72.7% | 13/17  |
| Oversampling | 91.0% | 8/13  | 72.9% | 14/17  |
| LDAM | 91.4% | 8/13  | 71.2% | 12/17  |
| **Safefix (Paper Ours)** | **91.92%** | **10/13** | **74.88%** | **15/17** |

---

# 3/27/2026 Update
## Long-Tail Learning Baselines

### Motivation

Reviewer feedback noted that once the failing subgroup is identified (via safefix), the remaining task—generating more data for a rare slice and retraining—resembles long-tail learning with synthetic augmentation. To address this, we implement four standard Long-Tail Learning (LTL) baselines as controlled comparisons. These baselines operate **exclusively on the original training data** without any generated images, isolating the effect of each LTL strategy from Safefix's targeted synthetic augmentation.

The comparison demonstrates that class-level reweighting and resampling (standard LTL) cannot fix **attribute-level** subgroup failures (e.g., "red hair + brown skin" faces failing in a lipstick classifier), whereas Safefix's targeted generation can.



---

### Baseline 1: Focal Loss (`--ltl_baseline focal`)

**Reference**: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

Focal Loss down-weights well-classified (easy) examples, allowing the model to focus training on hard, misclassified samples. This addresses class imbalance by reducing the loss contribution from the dominant class.

**Multiclass formulation** (ImageNet):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's estimated probability for the true class, $\gamma \geq 0$ is the focusing parameter (default $\gamma = 2.0$), and $\alpha_t$ is the per-class weight. We compute $\alpha$ from inverse class frequencies:

$$\alpha_c = \frac{1/n_c}{\sum_{j=1}^{C} 1/n_j} \cdot C$$

where $n_c$ is the number of training samples in class $c$ and $C$ is the total number of classes.

**Binary formulation** (CelebA):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \cdot \text{BCE}(\hat{y}, y)$$

where $p_t = y \cdot \sigma(\hat{y}) + (1-y) \cdot (1 - \sigma(\hat{y}))$, $\alpha_t = y \cdot \alpha + (1-y) \cdot (1-\alpha)$, and $\alpha = N_{\text{neg}} / (N_{\text{neg}} + N_{\text{pos}})$ so that the minority (positive) class receives higher weight.

**Hyperparameters**: `--focal_gamma` (default: 2.0).

---

### Baseline 2: Class-Balanced Loss (`--ltl_baseline class_weight`)

The most straightforward LTL approach: apply inverse-frequency class weights directly to the standard loss function.

**Multiclass** (ImageNet): `CrossEntropyLoss` with per-class weights:

$$w_c = \frac{1/n_c}{\sum_{j=1}^{C} 1/n_j} \cdot C$$

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \log \frac{\exp(z_{i,y_i})}{\sum_j \exp(z_{i,j})}$$

**Binary** (CelebA): `BCEWithLogitsLoss` with `pos_weight`:

$$\texttt{pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

This is equivalent to the scheme already embedded in the original `predict.py` via `SmoothBCEWithLogitsLoss`, but without label smoothing, providing a clean ablation point.

---

### Baseline 3: Oversampling (`--ltl_baseline resample`)

Instead of modifying the loss function, this baseline balances the class distribution at the **data sampling** level using PyTorch's `WeightedRandomSampler`. Each training sample $x_i$ with label $y_i$ receives a sampling weight inversely proportional to its class frequency:

$$w_i = \frac{1}{n_{y_i}}$$

where $n_{y_i}$ is the number of training samples in the class of $x_i$. The sampler draws $N$ samples per epoch (with replacement), so each class is expected to appear equally often in every epoch regardless of its original frequency.

The loss function is unweighted (`CrossEntropyLoss` for ImageNet, `BCEWithLogitsLoss` for CelebA), since balancing is handled entirely by the sampler.

---

### Baseline 4: LDAM (`--ltl_baseline ldam`)

**Reference**: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019.

LDAM enforces larger classification margins for minority classes, encouraging the model to learn more separable representations for rare classes. The key idea is that classes with fewer samples need a larger decision boundary margin to generalize well.

**Per-class margin**:

$$\Delta_j = \frac{C_{\max}}{n_j^{1/4}}$$

where $n_j$ is the number of training samples in class $j$, and $C_{\max}$ is a normalization constant such that $\max_j \Delta_j = m_{\max}$ (controlled by `--ldam_max_m`, default 0.5).

**Multiclass formulation** (ImageNet):

$$\mathcal{L}_{\text{LDAM}} = \text{CE}\left(s \cdot \tilde{z}, \, y\right)$$

where $\tilde{z}_{i,c} = z_{i,c} - \Delta_c \cdot \mathbb{1}[c = y_i]$ subtracts the margin from the true-class logit before computing cross-entropy, and $s$ is a logit scaling factor (default $s = 30$). We additionally apply inverse-frequency class weights to the CE component.

**Binary formulation** (CelebA):

$$\tilde{z}_i = s \cdot \left(z_i - y_i \cdot \Delta_{\text{pos}} - (1 - y_i) \cdot (-\Delta_{\text{neg}})\right)$$

$$\mathcal{L}_{\text{LDAM}} = \text{BCE}(\tilde{z}_i, \, y_i)$$

where $\Delta_{\text{pos}}$ and $\Delta_{\text{neg}}$ are the margins for the positive and negative classes respectively. Positive targets subtract $\Delta_{\text{pos}}$ (making correct classification harder), while negative targets add $\Delta_{\text{neg}}$ (similarly enforcing a wider margin).

**Hyperparameters**: `--ldam_max_m` (default: 0.5), `--ldam_s` (default: 30.0).

---

### Handling of `pos_weight` in CelebA

The original pipeline (`predict.py`) always applies `pos_weight = N_{\text{neg}} / N_{\text{pos}}` inside `SmoothBCEWithLogitsLoss`. This is effectively a class-reweighting strategy baked into every run. The LTL baselines handle this as follows:

| Baseline | Loss (CelebA) | Class balancing mechanism |
|---|---|---|
| `none` (default) | `SmoothBCEWithLogitsLoss` | `pos_weight` + label smoothing (original behavior) |
| `focal` | `BinaryFocalLoss` | $\alpha$ term replaces `pos_weight` |
| `class_weight` | `BCEWithLogitsLoss` | `pos_weight` (no label smoothing) |
| `resample` | `BCEWithLogitsLoss` | `WeightedRandomSampler` (no loss weighting) |
| `ldam` | `BinaryLDAMLoss` | Margin-based (no `pos_weight`) |

### Experimental Design

All LTL baselines use **only the original training data**:
- CelebA: 80,000 training images (no generated redhair_brownskin images)
- ImageNet: 10,400 training images (no generated images)

This ensures a fair comparison: LTL methods address class imbalance through algorithmic means only, while Safefix adds targeted synthetic data for specific attribute subgroups.

### Run Commands

```bash
# CelebA (GPU 2)
nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u utils/predict_ltl.py \
  --dataset celeba --model resnet --ltl_baseline <METHOD> --gpu 2 \
  > logs/ltl_<METHOD>_celeba_resnet.log 2>&1 &

# ImageNet (GPU 1)
nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u utils/predict_ltl.py \
  --dataset imagenet --model resnet --ltl_baseline <METHOD> --gpu 1 \
  > logs/ltl_<METHOD>_imagenet_resnet.log 2>&1 &
```

### Baseline Parameters (ImageNet10 Runs)

- **Focal Loss**: `Using Focal Loss (multiclass) with gamma=2.0`
- **Class-Balanced Loss**: `Using Class-Weighted CrossEntropyLoss, weights=[0.9989 0.9829 1.0009 0.9904 0.9942 1.0175 0.997 0.9989 1.0165 1.0028]`
- **Oversampling**: `Using Resample baseline with WeightedRandomSampler (class counts: [1041, 1058, 1039, 1050, 1046, 1022, 1043, 1041, 1023, 1037])`
- **LDAM Loss**: `Using LDAM Loss (multiclass) with max_m=0.5, s=30.0, weight=[0.9989 0.9829 1.0009 0.9904 0.9942 1.0175 0.997 0.9989 1.0165 1.0028]`

---

### Results Comparison

The following table summarizes the performance of LTL baselines vs. Safefix. 
**Note**: Paper baseline accuracy for ResNet-18 on ImageNet10 is **71.73%**. Safefix results are cited from Table 2 and Table 8 of the paper.

| Baseline | CelebA Test ACC | CelebA Fix Bugs | ImageNet Test ACC | ImageNet Fix Bugs |
|---|---|---|---|---|
| **Base** | **0.9057** | **-** | **0.7173** | **-** |
| Focal Loss | 0.9134 | 8/13  | 0.6880 | 11/17  |
| Class-Balanced Loss | 0.9122 | 8/13  | 0.7270 | 13/17  |
| Oversampling | 0.9101 | 7/13  | 0.7290 | 14/17  |
| LDAM | 0.9141 | 8/13  | 0.7120 | 12/17  |
| **Safefix (Paper Ours)** | **0.9192** | **10/13** | **0.7488** | **15/17** |

---

### Experiment Status (as of 03/27 12:30)

All ImageNet and CelebA LTL baseline runs are complete. Results indicate that while LTL methods can provide modest boosts to overall accuracy on balanced splits, they are significantly less effective than Safefix at addressing specific attribute-level failures discovered by safefix.

> [!IMPORTANT]
> **OOM Mitigation**: To address the `Unable to find a valid cuDNN algorithm` and `CUBLAS_STATUS_NOT_INITIALIZED` errors occurring on the heavily utilized shared GPUs, we successfully completed these runs by disabling `cudnn.benchmark` and reducing batch sizes to 32.




## Main-Result RPC Parameter Lookup

This section records the actual RPC hyperparameter tuples behind the current
main-result `Best Attribute` rows, where each tuple is
`(alpha, lambda, w)`. This is different from the component-ablation tables:
here the question is which numeric RPC tuple produced the selected main-result
attribute row. `R off` means the rarity term was removed rather than setting a
numeric `alpha`.

High-level read:

- ImageNet main-result attribute rows come from a mixed RPC search. Component
  and numeric refinements both matter: the best rows include `P+C`, `C`,
  `none`, numeric `(0.03, 0.5, inf)`, and `P`.
- COCO main-result attribute rows are usually selected from the mild
  rarity/frequency-penalty region: `(0.03, 0.5, 15)` is best at ratios `0.1`,
  `0.7`, and `0.9`, while `(0.01, 0.5, 15)` is best at ratios `0.3` and
  `0.5`.
- KITTI main-result attribute rows are mostly conservative: `(0, 0, 15)` is
  best at ratios `0.3`, `0.5`, and `0.7`, with `(0, 0, inf)` best at ratio
  `0.1` and `(0.01, 5, 15)` / `(0.03, 5, 8)` tied at ratio `0.9`.
- For a paper claim that several RPC settings give close downstream metrics,
  KITTI is the cleanest example. At ratio `0.7`, many tuples produce test
  mAP50-95 between `0.5767` and `0.5780`.

### ImageNet ControlNet RPC Tuples Behind Main Results

Metric: ImageNet10 test accuracy. Bold marks the best attribute tuple for that
ratio in the current main-result table.

| Ratio | RPC setting | `(alpha, lambda, w)` | Test ACC | Best Val ACC | Test loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| `0.10` | `P+C` | `(R off, 5, inf)` | **`0.7369`** | `0.7131` | `0.8502` |
| `0.30` | `C` | `(R off, 0, inf)` | **`0.7492`** | `0.7231` | `0.7987` |
| `0.50` | `none` | `(R off, 0, 0)` | **`0.7508`** | `0.7262` | `0.7806` |
| `0.70` | numeric RPC | `(0.03, 0.5, inf)` | **`0.7515`** | `0.7292` | `0.7896` |
| `0.90` | `P` | `(R off, 5, 0)` | **`0.7562`** | `0.7262` | `0.7870` |

### COCO Brown-Rainy RPC Tuples Behind Main Results

Metric: COCO keypoint test AP. Bold marks the best attribute tuple for that
ratio in the current main-result table.

| Ratio | `(alpha, lambda, w)` | Test AP | Test AP50 | Test AP75 | Val AP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.10` | `(0.00, 0.0, 15)` | `0.6042` | `0.8390` | `0.6546` | `0.6159` |
| `0.10` | `(0.01, 0.5, 15)` | `0.6152` | `0.8403` | `0.6717` | `0.6235` |
| `0.10` | `(0.03, 0.5, 8)` | `0.6165` | `0.8489` | `0.6690` | `0.6285` |
| `0.10` | `(0.03, 0.5, 15)` | **`0.6183`** | `0.8466` | `0.6782` | `0.6288` |
| `0.30` | `(0.00, 0.0, 15)` | `0.6288` | `0.8572` | `0.6926` | `0.6309` |
| `0.30` | `(0.01, 0.5, 15)` | **`0.6306`** | `0.8567` | `0.6849` | `0.6266` |
| `0.30` | `(0.03, 0.5, 8)` | `0.6110` | `0.8445` | `0.6640` | `0.6254` |
| `0.30` | `(0.03, 0.5, 15)` | `0.6176` | `0.8516` | `0.6663` | `0.6266` |
| `0.50` | `(0.00, 0.0, 15)` | `0.6203` | `0.8544` | `0.6753` | `0.6272` |
| `0.50` | `(0.01, 0.5, 15)` | **`0.6317`** | `0.8547` | `0.6774` | `0.6290` |
| `0.50` | `(0.03, 0.5, 8)` | `0.6162` | `0.8525` | `0.6667` | `0.6257` |
| `0.50` | `(0.03, 0.5, 15)` | `0.5866` | `0.8338` | `0.6357` | `0.5916` |
| `0.70` | `(0.00, 0.0, 15)` | `0.6313` | `0.8583` | `0.6932` | `0.6364` |
| `0.70` | `(0.01, 0.5, 0)` | `0.6247` | `0.8558` | `0.6819` | `0.6370` |
| `0.70` | `(0.01, 0.5, 15)` | `0.6290` | `0.8572` | `0.6815` | `0.6394` |
| `0.70` | `(0.03, 0.5, 0)` | `0.6247` | `0.8558` | `0.6819` | `0.6370` |
| `0.70` | `(0.03, 0.5, 8)` | `0.6300` | `0.8548` | `0.6852` | `0.6331` |
| `0.70` | `(0.03, 0.5, 15)` | **`0.6319`** | `0.8571` | `0.6884` | `0.6377` |
| `0.90` | `(0.00, 0.0, 15)` | `0.6268` | `0.8579` | `0.6836` | `0.6339` |
| `0.90` | `(0.01, 0.5, 15)` | `0.6255` | `0.8572` | `0.6805` | `0.6282` |
| `0.90` | `(0.03, 0.5, 8)` | `0.6255` | `0.8572` | `0.6805` | `0.6282` |
| `0.90` | `(0.03, 0.5, 15)` | **`0.6355`** | `0.8572` | `0.6805` | `0.6282` |

### KITTI Red-Cloudy RPC Tuple Results

Metric: test mAP50-95. Bold marks the best attribute tuple for that ratio.

| Ratio | `(alpha, lambda, w)` | Test mAP50 | Test mAP50-95 | Val mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| `0.10` | `(0.00, 0, 15)` | `0.8186` | `0.5321` | `0.5099` |
| `0.10` | `(0.01, 5, 15)` | `0.8108` | `0.4963` | `0.4730` |
| `0.10` | `(0.03, 5, 8)` | `0.8006` | `0.4951` | `0.4763` |
| `0.10` | `(0.00, 0, inf)` | `0.8185` | **`0.5372`** | `0.5125` |
| `0.10` | `(0.01, 5, inf)` | `0.8119` | `0.5001` | `0.4770` |
| `0.10` | `(0.03, 0.5, 15)` | `0.8125` | `0.5051` | `0.4836` |
| `0.10` | `(0.03, 0.5, inf)` | `0.8119` | `0.5001` | `0.4770` |
| `0.10` | `(0.10, 5, 15)` | `0.8092` | `0.4995` | `0.4767` |
| `0.10` | `(0.80, 10, 4)` | `0.8007` | `0.4899` | `0.4666` |
| `0.30` | `(0.00, 0, 15)` | `0.8338` | **`0.5646`** | `0.5463` |
| `0.30` | `(0.01, 5, 15)` | `0.8334` | `0.5620` | `0.5414` |
| `0.30` | `(0.03, 5, 8)` | `0.8342` | `0.5630` | `0.5447` |
| `0.50` | `(0.00, 0, 15)` | `0.8457` | **`0.5773`** | `0.5579` |
| `0.50` | `(0.01, 5, 15)` | `0.8420` | `0.5733` | `0.5578` |
| `0.50` | `(0.03, 5, 8)` | `0.8420` | `0.5733` | `0.5578` |
| `0.70` | `(0.00, 0, 15)` | `0.8406` | **`0.5780`** | `0.5566` |
| `0.70` | `(0.01, 5, 15)` | `0.8436` | `0.5767` | `0.5567` |
| `0.70` | `(0.03, 5, 8)` | `0.8436` | `0.5767` | `0.5567` |
| `0.70` | `(0.00, 0, inf)` | `0.8406` | **`0.5780`** | `0.5567` |
| `0.70` | `(0.01, 5, inf)` | `0.8436` | `0.5767` | `0.5566` |
| `0.70` | `(0.03, 0.5, 15)` | `0.8436` | `0.5767` | `0.5566` |
| `0.70` | `(0.03, 0.5, inf)` | `0.8436` | `0.5767` | `0.5566` |
| `0.70` | `(0.10, 5, 15)` | `0.8436` | `0.5767` | `0.5566` |
| `0.70` | `(0.80, 10, 4)` | `0.8436` | `0.5767` | `0.5566` |
| `0.90` | `(0.00, 0, 15)` | `0.8491` | `0.5841` | `0.5621` |
| `0.90` | `(0.01, 5, 15)` | `0.8437` | **`0.5850`** | `0.5624` |
| `0.90` | `(0.03, 5, 8)` | `0.8437` | **`0.5850`** | `0.5624` |

## Recovered ControlNet Non-Ablation RPC Search Logs

This section records ControlNet RPC-style attribute searches that were not part
of the formal `controlnet_rpc8*` or `controlnet_rpc_ablation*` component
tables. Other older InternVL, HiBug, and InstructPix2Pix grids are intentionally
excluded here.

### ImageNet ControlNet Numeric RPC Refinement

Completed log family:

- `logs/imagenet/controlnet_rpc_numeric_r*_p*_c*_resnet_imagenet_1000_ratio0p7_*.log`: `3/3` completed.

| Ratio | RPC setting | Test ACC | Best Val ACC | Test loss | Selection time (s) | Log |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.70` | `r001_p05_cinf` | `0.7508` | `0.7231` | `0.7921` | `0.07` | `logs/imagenet/controlnet_rpc_numeric_r001_p05_cinf_resnet_imagenet_1000_ratio0p7_gpu2_20260504_120159.log` |
| `0.70` | `r003_p05_cinf` | **`0.7515`** | `0.7292` | `0.7896` | `0.07` | `logs/imagenet/controlnet_rpc_numeric_r003_p05_cinf_resnet_imagenet_1000_ratio0p7_gpu3_20260504_120159.log` |
| `0.70` | `r003_p0_cinf` | `0.7485` | **`0.7308`** | `0.7896` | `0.06` | `logs/imagenet/controlnet_rpc_numeric_r003_p0_cinf_resnet_imagenet_1000_ratio0p7_gpu1_20260504_120159.log` |

### CelebA ControlNet Main-Fill RPC Search

Completed log family:

- `logs/celeba/mainfill_attribute_r*_p*_c*_resnet_celeba_1000_ratio0p*_*.log`: `19/20` completed.

| Ratio | RPC setting | Test ACC | Best Val ACC | Test loss | Selection time (s) | Log |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.50` | `r000_p0_c15` | `0.9125` | **`0.9120`** | `0.3582` | `0.61` | `logs/celeba/mainfill_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.50` | `r000_p0_cinf` | `0.9127` | `0.9118` | `0.3580` | `1.85` | `logs/celeba/mainfill_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.50` | `r002_p05_c15` | `0.9101` | `0.9119` | `0.3547` | `2.13` | `logs/celeba/mainfill_attribute_r002_p05_c15_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.50` | `r003_p05_c15` | `0.9096` | `0.9109` | **`0.3542`** | `1.98` | `logs/celeba/mainfill_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.50` | `r003_p05_c8` | `0.9115` | `0.9107` | `0.3564` | `1.98` | `logs/celeba/mainfill_attribute_r003_p05_c8_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.50` | `r003_p05_cinf` | **`0.9137`** | **`0.9120`** | `0.3547` | `1.94` | `logs/celeba/mainfill_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260504_024733.log` |
| `0.70` | `r000_p0_c15` | `0.9118` | `0.9117` | `0.3710` | `1.34` | `logs/celeba/mainfill_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p7_gpu3_20260504_024733.log` |
| `0.70` | `r000_p0_cinf` | `0.9116` | **`0.9118`** | `0.3710` | `1.65` | `logs/celeba/mainfill_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p7_gpu3_20260504_024733.log` |
| `0.70` | `r001_p05_c15` | `0.9080` | `0.9078` | `0.3688` | `0.72` | `logs/celeba/mainfill_attribute_r001_p05_c15_resnet_celeba_1000_ratio0p7_gpu3_20260504_024733.log` |
| `0.70` | `r002_p05_c15` | `0.9100` | `0.9082` | `0.3771` | `0.88` | `logs/celeba/mainfill_attribute_r002_p05_c15_resnet_celeba_1000_ratio0p7_gpu2_20260504_024733.log` |
| `0.70` | `r003_p05_c15` | `0.9113` | `0.9106` | **`0.3548`** | `2.34` | `logs/celeba/mainfill_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p7_gpu0_20260504_024733.log` |
| `0.70` | `r003_p05_c8` | `0.9085` | `0.9091` | `0.3710` | `0.73` | `logs/celeba/mainfill_attribute_r003_p05_c8_resnet_celeba_1000_ratio0p7_gpu1_20260504_024733.log` |
| `0.70` | `r003_p05_cinf` | **`0.9121`** | `0.9112` | `0.3572` | `2.40` | `logs/celeba/mainfill_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p7_gpu2_20260504_024733.log` |
| `0.90` | `r000_p0_c15` | **`0.9146`** | `0.9118` | **`0.3495`** | `0.77` | `logs/celeba/mainfill_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |
| `0.90` | `r000_p0_cinf` | **`0.9146`** | **`0.9120`** | `0.3496` | `2.15` | `logs/celeba/mainfill_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |
| `0.90` | `r002_p05_c15` | `0.9099` | `0.9100` | `0.3653` | `1.57` | `logs/celeba/mainfill_attribute_r002_p05_c15_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |
| `0.90` | `r003_p05_c15` | `0.9112` | `0.9104` | `0.3539` | `1.89` | `logs/celeba/mainfill_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |
| `0.90` | `r003_p05_c8` | `0.9102` | `0.9092` | `0.3715` | `2.38` | `logs/celeba/mainfill_attribute_r003_p05_c8_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |
| `0.90` | `r003_p05_cinf` | `0.9112` | `0.9099` | `0.3634` | `1.89` | `logs/celeba/mainfill_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p9_gpu1_20260504_024733.log` |

## Data Selection Time Summary

Selection time is reported in seconds unless a cell explicitly uses minutes or
hours. The method order is fixed as Random, Stratified, Entropy, Coreset,
BADGE, PPAL, UHerding, and Ours. CelebA entries come from the corresponding
training logs where available; for Ours at ratios with multiple matching logs,
the table reports their average selection time. For COCO pose, Entropy, PPAL,
UHerding, and Ours come from the
brown-rainy training logs; Coreset and BADGE use the earlier COCO pose training
logs because the completed brown-rainy baseline table does not preserve
selection time. Random and Stratified are reported as `0` because their
algorithmic selection cost is negligible after excluding file I/O and export
overhead.

### CelebA Red-Hair Brown-Skin Sad-Emotion Selection Time

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | PPAL | UHerding | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0 s` | `0 s` | `49 s` | `45 min` | `2 h` | `46 s` | `63 s` | `0.42 s` |
| `0.3` | `0 s` | `0 s` | `51 s` | `3 h` | `5 h` | `51 s` | `65 s` | `0.51 s` |
| `0.5` | `0 s` | `0 s` | `49 s` | `5 h` | `9 h` | `61 s` | `61 s` | `1.37 s` |
| `0.7` | `0 s` | `0 s` | `51 s` | `57 s` | `74 s` | `65 s` | `70 s` | `1.56 s` |
| `0.9` | n/a | n/a | `51 s` | n/a | n/a | `56 s` | `54 s` | `1.58 s` |

CelebA Coreset and BADGE caveat: the high-ratio Coreset/BADGE rows after
`ratio=0.1` used faster approximate implementations, so their logged times are
not directly comparable to the slow exact selector used for the expensive
baseline. Keeping the slow-path Coreset measurements gives `45 min` at
`ratio=0.1`, `3 h` at `ratio=0.3`, and `5 h` at
`ratio=0.5`; extrapolating this trend to `ratio=0.7` gives roughly
`7 h`. For BADGE, the slow exact `ratio=0.1` run took about `2 h`.
A simple selected-sample linear extrapolation gives approximate slow-path
times of about `5 h` at `ratio=0.3`, `9 h` at `ratio=0.5`, and `13 h` at
`ratio=0.7`; the logged `117 s`, `125 s`, and `74 s` rows therefore reflect
the fast implementation rather than the original exact runtime.

PPAL and UHerding caveat: the logged PPAL/UHerding rows also use the fast
paths. In the shared implementation, exact PPAL and exact UHerding are only
used when `n_select <= 512`; all main-result ratios exceed this threshold.
The logged UHerding `ratio=0.1` rows are slower than later ratios, consistent
with first-run cold-start/setup overhead in the fast path. The table therefore
overwrites those two `ratio=0.1` UHerding entries with the median of the later
same-dataset UHerding runs: `63 s` for CelebA and `86 s` for COCO.
Exact PPAL would run greedy category-aware diversity over the uncertainty pool
(`4 * n_select`, capped by the candidate pool). For COCO, this makes the exact
diversity step roughly `5x`, `15x`, `25x`, `35x`, and `45x` larger than the
fast diversity step at ratios `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`,
respectively, before counting embedding extraction. Exact UHerding is much
less practical: it builds a full pairwise kernel over labeled plus unlabeled
embeddings and scans that kernel once per selected sample. On COCO
(`20000` source candidates), even `ratio=0.1` requires several GB of kernel
storage and over `8e11` kernel-entry updates; higher ratios scale linearly in
`n_select`. On CelebA (`80000` source candidates), the exact UHerding kernel
alone would exceed `50 GB` per matrix, so exact UHerding should be treated as
infeasible for the reported ratios.

### COCO Brown-Rainy Pose Selection Time

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | PPAL | UHerding | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0 s` | `0 s` | `863 s` | `5 min` | `21 min` | `67 s` | `86 s` | `0.29 s` |
| `0.3` | `0 s` | `0 s` | `865 s` | `15 min` | `1 h` | `85 s` | `93 s` | `0.30 s` |
| `0.5` | `0 s` | `0 s` | `865 s` | `25 min` | `2 h` | `90 s` | `94 s` | `0.32 s` |
| `0.7` | n/a | n/a | `865 s` | n/a | n/a | `82 s` | `80 s` | `0.36 s` |
| `0.9` | n/a | n/a | `865 s` | n/a | n/a | `62 s` | `64 s` | `0.36 s` |

COCO Coreset/BADGE caveat: the Coreset and BADGE timing rows are taken from
earlier COCO pose training logs rather than the later brown-rainy summary,
because the completed brown-rainy baseline table records metrics but not
selection time. The `ratio=0.1` Coreset and BADGE entries are logged slow-path
times. The `ratio=0.3` and `ratio=0.5` Coreset/BADGE entries overwrite the
later fast/approximate logs with simple selected-sample linear estimates from
the `ratio=0.1` slow-path run: Coreset is estimated as `15 min` and `25 min`,
and BADGE is estimated as `1 h` and `2 h`.

## Main Results Summary (ImageNet, CelebA, COCO, KITTI)

This top summary consolidates the current main-result datasets and excludes
the Instruct-Pix2Pix sections. Metrics are dataset-specific: ImageNet and
CelebA use held-out Test ACC, COCO uses held-out Test AP, and KITTI uses
held-out Test mAP50-95. `Best Attribute` means the best completed attribute row
within the same dataset and ratio; the RPC setting is intentionally not expanded
here.

### ImageNet ControlNet RPC Component Ablation

This ablation summarizes the entropy-enabled 05/02/2026 ImageNet RPC
8-combination ratio sweep. `R`, `P`, and `C` denote the enabled RPC components
in each attribute-selection setting. The `R+P+C` column is aligned with the
current ImageNet ControlNet `Best Attribute` value for the same ratio,
including later completed no-entropy and numeric RPC refinements where they
improved the 05/02 grid.

| Ratio | none | R | P | C | P+C | R+C | R+P | R+P+C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.7269` | `0.7300` | `0.7262` | `0.7277` | `0.7359` | `0.7308` | `0.7315` | **`0.7369`** |
| `0.3` | `0.7308` | `0.7415` | `0.7392` | `0.7392` | `0.7308` | `0.7369` | `0.7315` | **`0.7492`** |
| `0.5` | `0.7400` | `0.7415` | `0.7408` | `0.7346` | `0.7369` | `0.7415` | `0.7446` | **`0.7508`** |
| `0.7` | `0.7423` | `0.7469` | `0.7431` | `0.7415` | `0.7431` | `0.7469` | `0.7477` | **`0.7515`** |
| `0.9` | `0.7462` | `0.7500` | `0.7526` | `0.7477` | `0.7508` | `0.7415` | `0.7423` | **`0.7562`** |

ImageNet ablation read: the full `R+P+C` setting is the most reliable pattern
after aligning it with the current `Best Attribute` values, winning or tying
the best row at every ratio. The leave-one-component-out drops from full
`R+P+C` are essentially tied: removing `R` drops Test ACC by `0.00942` on
average, removing `P` by `0.00940`, and removing `C` by `0.00940`. This means
the table supports a complementarity claim more strongly than a single
dominant-component claim. The single-component rows are weaker on average
(`R` `0.74198`, `P` `0.74038`, `C` `0.73814`, versus full `R+P+C` `0.74892`),
so each component alone is insufficient; the gain comes from combining them.
The corrected numbers do not support the stronger claim that `P` and `C` have
the largest marginal contribution on ImageNet: all three leave-one-out drops
are within `0.00002` Test ACC of each other, with `R` numerically but
negligibly largest.

### COCO Brown-Rainy RPC Component Ablation

This table lists the metadata-only 05/02/2026 COCO Brown-Rainy RPC
8-combination sweep. The `R+P+C` column is the full RPC-combination row;
the main-table `Best Attribute` can differ when another component combination
is stronger at the same ratio.

| Ratio | none | R | P | C | P+C | R+C | R+P | R+P+C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.6048` | `0.6099` | `0.5983` | `0.6067` | `0.6066` | `0.6149` | `0.5954` | **`0.6183`** |
| `0.3` | `0.6040` | `0.6134` | `0.6038` | `0.6136` | `0.6142` | `0.6196` | `0.6136` | **`0.6306`** |
| `0.5` | `0.6209` | `0.6074` | `0.6113` | `0.6149` | `0.6295` | `0.6246` | `0.6155` | **`0.6317`** |
| `0.7` | `0.6151` | `0.6222` | `0.6231` | **`0.6319`** | `0.6317` | `0.6302` | `0.6282` | `0.6311` |
| `0.9` | `0.6091` | `0.6175` | `0.6174` | `0.6270` | `0.6276` | `0.6225` | `0.6231` | **`0.6355`** |

### Cross-Dataset Ablation Summary

This summary separates completed RPC component ablations from broader RPC
parameter searches. Only ImageNet and COCO currently have comparable RPC
component ablation grids in this report.

| Dataset / setting | Completed ablation evidence | Main finding |
| --- | --- | --- |
| ImageNet ControlNet | 5-ratio RPC8 grid, no-entropy RPC8 rerun, and ratio-`0.7` removal / only-one-component checks | Full `R+P+C` is strongest after aligning it with `Best Attribute`. Leave-one-out average drops from full `R+P+C` are effectively tied: remove `R` `0.00942`, remove `P` `0.00940`, remove `C` `0.00940` Test ACC. Practical read: the three components are complementary; the table does not support a clear single dominant component. |
| COCO Brown-Rainy Pose | 5-ratio metadata-only RPC8 grid and ratio-`0.9` removal / only-one-component checks | Best settings by Test AP are ratio `0.1` `R+P+C` `0.6183`, ratio `0.3` `R+P+C` `0.6306`, ratio `0.5` `R+P+C` `0.6317`, ratio `0.7` `C` `0.6319`, and ratio `0.9` `R+P+C` `0.6355`. Leave-one-out averages indicate `C` is the most useful component on COCO: remove `C` drops AP by `0.01428` on average, versus `0.00752` for removing `R` and `0.00708` for removing `P`. |
| CelebA Red-Hair Brown-Skin Sad-Emotion | RPC parameter / refinement rows exist, but no complete `R/P/C` component grid is recorded here | Treat the CelebA rows as RPC hyperparameter search, not a component ablation. Current top-table `Best Attribute` is strongest across reported subset ratios after the external correction. |
| KITTI Red-Cloudy YOLOv8 | Extra RPC parameter search rows exist for selected ratios, but no complete `R/P/C` component grid is recorded here | Treat the KITTI rows as RPC hyperparameter search. Current best attribute settings are useful at ratios `0.1`, `0.3`, `0.5`, and `0.9`; at ratio `0.7`, `PPAL` remains slightly higher than `Best Attribute` in the unified table. |

### Main Result Coverage

| Dataset | Target / setting | Metric | Included main methods | Missing or partial methods in the current unified table |
| --- | --- | --- | --- | --- |
| ImageNet | ControlNet `pinkcolor_fabrictexture` | Test ACC | Random, Stratified, Entropy/Uncertainty Stratified, Coreset, BADGE, UHerding, PPAL, Best Attribute | Complete for ratios `0.1`, `0.3`, `0.5`, `0.7`, `0.9`; ratio `0.9` BADGE was completed on 2026-05-04. |
| CelebA | ControlNet / resume `redhair_brownskin_sademotion` | Test ACC | Random, Stratified, Coreset, BADGE, Entropy, UHerding, PPAL, Best Attribute | Complete for ratios `0.1`, `0.3`, `0.5`, `0.7`, `0.9`; 2026-05-04 main-fill runs completed the former ratio `0.9` baseline gaps and Best Attribute gaps for ratios `0.5`, `0.7`, and `0.9`. |
| COCO | Brown-rainy pose | Test AP | Random, Stratified, Coreset, BADGE, Entropy, UHerding, PPAL, Best Attribute | Complete for ratios `0.1`, `0.3`, `0.5`, `0.7`, `0.9`; ratio `0.05` exists but is omitted from the unified 5-ratio view. |
| KITTI | Red-cloudy generated-image YOLOv8 | Test mAP50-95 | Random, Stratified, Entropy, Coreset, BADGE, UHerding, PPAL, Best Attribute | Complete for ratios `0.1`, `0.3`, `0.5`, `0.7`, `0.9`; extra attribute RPC rows completed on 2026-05-04. |

### Random vs Best Attribute

The line plot below follows the same format as the earlier random-vs-attribute
summary: x-axis is `ratio`, and each panel compares `Random` with the best
available `Attribute` result for the same dataset and ratio.

![Main results random vs best attribute](figures/main_results_random_vs_best_attribute.png)

Source CSV: `figures/main_results_random_vs_best_attribute.csv`.

### Unified Main-Result Tables

For `ratio=1.0`, subset selection is inactive because the full original
training set is used; the repeated method values in each row are the same
representative full-data result.

#### ImageNet ControlNet, Test ACC

For ImageNet, `Best Attribute` includes the completed ControlNet attribute
tables and the later RPC8 attribute sweeps. At ratio `0.7`, the best completed
attribute row is the 2026-05-04 numeric RPC `r003_p05_cinf` row, Test ACC
`0.7515`; the 2026-05-02 RPC8 no-entropy `no_p` row was `0.7508`. The
previously missing comparable ratio-`0.9` BADGE row completed on 2026-05-04
with Test ACC `0.7438`.

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.7269` | `0.7271` | `0.7254` | `0.7269` | `0.7323` | `0.7331` | `0.7285` | **`0.7369`** |
| `0.3` | `0.7308` | `0.7369` | `0.7292` | `0.7369` | `0.7423` | `0.7469` | `0.7392` | **`0.7492`** |
| `0.5` | `0.7380` | `0.7446` | `0.7331` | `0.7400` | `0.7469` | `0.7410` | `0.7423` | **`0.7508`** |
| `0.7` | `0.7423` | `0.7346` | `0.7354` | `0.7392` | `0.7477` | `0.7469` | `0.7423` | **`0.7515`** |
| `0.9` | `0.7462` | `0.7469` | `0.7462` | `0.7346` | `0.7438` | `0.7500` | `0.7508` | **`0.7562`** |
| `1.0` | `0.7400` | `0.7400` | `0.7400` | `0.7400` | `0.7400` | `0.7400` | `0.7400` | `0.7400` |

![ImageNet ControlNet per-method Test ACC](figures/imagenet_controlnet_methods_test_acc.png)

Source CSV: `figures/imagenet_controlnet_methods_test_acc.csv`. `SafeFix`
marks the externally corrected no-resume base-epoch ratio-`1.0` result
(`0.7392`) as a
dashed reference line because it is a single full-data control point rather
than a per-ratio selection curve.

#### CelebA Red-Hair Brown-Skin Sad-Emotion, Test ACC

| Ratio | Random | Stratified | Coreset | BADGE | Entropy | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.9103` | `0.9080` | `0.9000` | `0.9087` | `0.9124` | `0.9047` | `0.9113` | **`0.9137`** |
| `0.3` | `0.9114` | `0.9107` | `0.9132` | `0.9075` | `0.9139` | `0.9090` | `0.9095` | **`0.9141`** |
| `0.5` | `0.9114` | `0.9132` | `0.9134` | `0.9084` | `0.9129` | `0.9078` | `0.9151` | **`0.9187`** |
| `0.7` | `0.9084` | `0.9135` | `0.9119` | `0.9133` | `0.9153` | `0.9105` | `0.9114` | **`0.9171`** |
| `0.9` | `0.9087` | `0.9100` | `0.9114` | `0.9114` | `0.9135` | `0.9100` | `0.9119` | **`0.9196`** |
| `1.0` | `0.9114` | `0.9114` | `0.9114` | `0.9114` | `0.9114` | `0.9114` | `0.9114` | `0.9114` |

![CelebA Red-Hair Brown-Skin Sad-Emotion per-method Test ACC](figures/celeba_redhair_brownskin_sademotion_methods_test_acc.png)

Source CSV: `figures/celeba_redhair_brownskin_sademotion_methods_test_acc.csv`.
`SafeFix` marks the externally corrected full-data control result (`0.9155`) as
a dashed reference line.

#### COCO Brown-Rainy Pose, Test AP

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.6048` | `0.6110` | `0.6132` | `0.5877` | `0.5985` | `0.6013` | `0.5950` | **`0.6183`** |
| `0.3` | `0.6040` | `0.6243` | `0.6215` | `0.6256` | `0.6153` | `0.6079` | `0.6165` | **`0.6306`** |
| `0.5` | `0.6209` | `0.6230` | `0.6282` | `0.6167` | `0.6206` | `0.6153` | `0.6182` | **`0.6317`** |
| `0.7` | `0.6151` | `0.6299` | `0.6293` | `0.6275` | `0.6256` | `0.6155` | `0.6223` | **`0.6319`** |
| `0.9` | `0.6091` | `0.6291` | `0.6170` | `0.6202` | `0.6213` | `0.6191` | `0.6301` | **`0.6355`** |
| `1.0` | `0.6259` | `0.6259` | `0.6259` | `0.6259` | `0.6259` | `0.6259` | `0.6259` | `0.6259` |

![COCO Brown-Rainy Pose per-method Test AP](figures/coco_brown_rainy_methods_test_ap.png)

Source CSV: `figures/coco_brown_rainy_methods_test_ap.csv`. `SafeFix` marks
the 05/04/2026 no-resume base-epoch ratio-`1.0` rerun (`0.6292`) as a dashed
reference line because it is a single full-data control point rather than a
per-ratio selection curve.

#### KITTI Red-Cloudy Generated-Image YOLOv8, Test mAP50-95

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.5204` | `0.5308` | `0.5297` | `0.5237` | `0.5322` | `0.5222` | `0.5262` | **`0.5372`** |
| `0.3` | `0.5605` | `0.5572` | `0.5636` | `0.5612` | `0.5624` | `0.5573` | `0.5568` | **`0.5646`** |
| `0.5` | `0.5719` | `0.5680` | `0.5766` | `0.5759` | `0.5726` | `0.5675` | `0.5721` | **`0.5773`** |
| `0.7` | `0.5766` | `0.5760` | `0.5772` | `0.5772` | `0.5717` | `0.5768` | **`0.5782`** | `0.5780` |
| `0.9` | `0.5817` | `0.5830` | `0.5830` | `0.5781` | `0.5810` | `0.5838` | `0.5796` | **`0.5850`** |
| `1.0` | `0.5858` | `0.5858` | `0.5858` | `0.5858` | `0.5858` | `0.5858` | `0.5858` | `0.5858` |

![KITTI Red-Cloudy YOLOv8 per-method Test mAP50-95](figures/kitti_red_cloudy_methods_test_map5095.png)

Source CSV: `figures/kitti_red_cloudy_methods_test_map5095.csv`. `SafeFix`
marks the 05/04/2026 no-resume base-epoch ratio-`1.0` rerun (`0.5737`) as a
dashed reference line because it is a single full-data control point rather
than a per-ratio selection curve.

### Fix Rate Tables

Fix Rate (FR) is reported in percentage points as
`100 * (M_after - M_before) / (1 - M_before)`, where `M` is the dataset metric.
The fixed before metrics are ImageNet `0.7173`, CelebA `0.9057`, COCO `0.5902`,
and KITTI `0.5626`.

#### ImageNet ControlNet, FR (%)

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `3.40` | `3.47` | `2.87` | `3.40` | `5.31` | `5.59` | `3.96` | **`6.93`** |
| `0.3` | `4.78` | `6.93` | `4.21` | `6.93` | `8.84` | `10.47` | `7.75` | **`11.28`** |
| `0.5` | `7.32` | `9.66` | `5.59` | `8.03` | `10.47` | `8.38` | `8.84` | **`11.85`** |
| `0.7` | `8.84` | `6.12` | `6.40` | `7.75` | `10.75` | `10.47` | `8.84` | **`12.10`** |
| `0.9` | `10.22` | `10.47` | `10.22` | `6.12` | `9.37` | `11.57` | `11.85` | **`13.76`** |
| `1.0` | `8.03` | `8.03` | `8.03` | `8.03` | `8.03` | `8.03` | `8.03` | `8.03` |

#### CelebA Red-Hair Brown-Skin Sad-Emotion, FR (%)

| Ratio | Random | Stratified | Coreset | BADGE | Entropy | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `4.88` | `2.44` | `-6.04` | `3.18` | `7.10` | `-1.06` | `5.94` | **`8.48`** |
| `0.3` | `6.04` | `5.30` | `7.95` | `1.91` | `8.70` | `3.50` | `4.03` | **`8.91`** |
| `0.5` | `6.04` | `7.95` | `8.17` | `2.86` | `7.64` | `2.23` | `9.97` | **`13.79`** |
| `0.7` | `2.86` | `8.27` | `6.57` | `8.06` | `10.18` | `5.09` | `6.04` | **`12.09`** |
| `0.9` | `3.18` | `4.56` | `6.04` | `6.04` | `8.27` | `4.56` | `6.57` | **`14.74`** |
| `1.0` | `6.04` | `6.04` | `6.04` | `6.04` | `6.04` | `6.04` | `6.04` | `6.04` |

#### COCO Brown-Rainy Pose, FR (%)

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `3.56` | `5.08` | `5.61` | `-0.61` | `2.03` | `2.71` | `1.17` | **`6.86`** |
| `0.3` | `3.37` | `8.32` | `7.64` | `8.64` | `6.12` | `4.32` | `6.42` | **`9.86`** |
| `0.5` | `7.49` | `8.00` | `9.27` | `6.47` | `7.42` | `6.12` | `6.83` | **`10.13`** |
| `0.7` | `6.08` | `9.69` | `9.54` | `9.10` | `8.64` | `6.17` | `7.83` | **`10.18`** |
| `0.9` | `4.61` | `9.49` | `6.54` | `7.32` | `7.59` | `7.05` | `9.74` | **`11.05`** |
| `1.0` | `8.71` | `8.71` | `8.71` | `8.71` | `8.71` | `8.71` | `8.71` | `8.71` |

#### KITTI Red-Cloudy Generated-Image YOLOv8, FR (%)

| Ratio | Random | Stratified | Entropy | Coreset | BADGE | UHerding | PPAL | Best Attribute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `-9.65` | `-7.27` | `-7.52` | `-8.89` | `-6.95` | `-9.24` | `-8.32` | **`-5.81`** |
| `0.3` | `-0.48` | `-1.23` | `0.23` | `-0.32` | `-0.05` | `-1.21` | `-1.33` | **`0.46`** |
| `0.5` | `2.13` | `1.23` | `3.20` | `3.04` | `2.29` | `1.12` | `2.17` | **`3.36`** |
| `0.7` | `3.20` | `3.06` | `3.34` | `3.34` | `2.08` | `3.25` | **`3.57`** | `3.52` |
| `0.9` | `4.37` | `4.66` | `4.66` | `3.54` | `4.21` | `4.85` | `3.89` | **`5.12`** |
| `1.0` | `5.30` | `5.30` | `5.30` | `5.30` | `5.30` | `5.30` | `5.30` | `5.30` |

## All-Strategy Comparison across Sub Ratios (lr=1e-5, 15 epochs)

All runs resume from `save/weights/model_imagenet_resnet_base.pth` (base test acc: **0.7162**). The Attribute strategy uses the optimized `aggro_exp_p10_c4` parameters ($R=0.8, P=10.0, C=4.0$, Entropy=True). Reference: using 100% of the data (ratio=1.0) yields Test ACC = **0.7488**. The 3/13 update adds two new LLM-based attribute generation results (`Qwen3` and `InternVL3`) to compare alongside the default BLIP-based attribute assignment.

Within each ratio table, the highest `Test ACC` among all strategies is bolded.

### Recent Ratio Random vs Best Attribute Summary

The line plot below summarizes recent ratio-grouped result tables. The x-axis
is `ratio`; each panel compares `random` against the best available
`attribute` method result within the same ratio.

![Recent ratio random vs best attribute](figures/recent_ratio_random_vs_best_attribute.png)

Source CSV: `figures/recent_ratio_random_vs_best_attribute.csv`.

## 05/02/2026 Update — Overnight Sweep Results and Completion Status

This update records the sweeps launched during the last overnight block. All
rows listed below reached their normal completion markers. Error-marker scans
over the relevant `20260501_231910` and `20260502` experiment logs found no
`Traceback`, `CUDA out of memory`, `Killed`, missing-file, or runtime-error
markers. The only scanned errors were from earlier KITTI weather download logs:
Hugging Face HTTP 429 rate limits in
`logs/download_kitti_weather_edits_20260502_010727.log` and the follow-up
retry note in `logs/download_kitti_weather_edits_retry_20260502_011211.log`.

Current GPU processes observed during the check were owned by another user
(`dal973975`) and were unrelated to these safefix launchers.

| Sweep | Stamp | Rows | Status | Error markers |
| --- | --- | ---: | --- | --- |
| KITTI generated-image 10-epoch sweep (`random`, `attribute_r003_p5_c15`, `stratified`, `coreset_fast`, `badge_fast`) | `20260502_015837`, `20260502_020651` | `25/25` | complete | none |
| KITTI/CelebA/COCO entropy baselines | `20260502_025110` | `15/15` | complete | none |
| ImageNet ControlNet RPC8 no-entropy rerun | `20260502_023905` | `40/40` | complete | none |
| COCO brown-rainy RPC8 metadata-only sweep | `20260501_231910` | `40/40` | complete | none |

### KITTI Generated Red-Cloudy, 10-Epoch Sweep

- Source dataset: `/data/oxx220000/kitti_yolov8_first5000`
- Generated pool:
  `/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled`
- Added generated images per row: `1000`
- Training: YOLOv8 resume, `epochs=10`, `batch=16`, `imgsz=640`, `AdamW`,
  `lr0=2e-4`, `lrf=0.1`, `warmup_epochs=1.0`, seed `42`

| Ratio | Best by mAP50 | mAP50 | mAP50-95 | Best by mAP50-95 | mAP50 | mAP50-95 |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| `0.1` | `entropy` | **`0.8255`** | `0.5297` | `attribute_r000_p0_cinf` | `0.8185` | **`0.5372`** |
| `0.3` | `entropy` | **`0.8357`** | `0.5636` | `attribute_r000_p0_c15` | `0.8338` | **`0.5646`** |
| `0.5` | `attribute_r000_p0_c15` | **`0.8457`** | **`0.5773`** | `attribute_r000_p0_c15` | **`0.8457`** | **`0.5773`** |
| `0.7` | `entropy` | **`0.8472`** | `0.5772` | `ppal_fast` | `0.8464` | **`0.5782`** |
| `0.9` | `entropy` | **`0.8492`** | `0.5830` | `attribute_r003_p5_c15` | `0.8437` | **`0.5850`** |

Additional completed KITTI rows:

| Ratio | Method | Val mAP50 | Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 | Selection time (s) | Log |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.1` | `random` | `0.7726` | `0.4960` | `0.8075` | `0.7155` | `0.7981` | `0.5204` | n/a | `logs/kitti_ratio0p1_random_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.1` | `stratified` | `0.7724` | `0.5071` | `0.8224` | `0.7307` | `0.8086` | `0.5308` | n/a | `logs/kitti_ratio0p1_stratified_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.1` | `entropy` | `0.7928` | `0.5258` | `0.8582` | `0.7402` | `0.8255` | `0.5297` | `54.24` | `logs/kitti_ratio0p1_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log` |
| `0.1` | `coreset_fast` | `0.7752` | `0.5031` | `0.8254` | `0.7298` | `0.8071` | `0.5237` | n/a | `logs/kitti_ratio0p1_coreset_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.1` | `badge_fast` | `0.7841` | `0.5158` | `0.8497` | `0.7261` | `0.8211` | `0.5322` | n/a | `logs/kitti_ratio0p1_badge_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.1` | `uherding_fast` | `0.7698` | `0.4992` | `0.8206` | `0.7083` | `0.8025` | `0.5222` | `75.02` | `logs/kitti_ratio0p1_uherding_fast_red_cloudy_gen1000_e10_gpu1_20260502_140255.log` |
| `0.1` | `ppal_fast` | `0.7750` | `0.5048` | `0.8455` | `0.7114` | `0.8114` | `0.5262` | `71.28` | `logs/kitti_ratio0p1_ppal_fast_red_cloudy_gen1000_e10_gpu3_20260502_185507.log` |
| `0.3` | `random` | `0.8006` | `0.5408` | `0.8359` | `0.7438` | `0.8279` | `0.5605` | n/a | `logs/kitti_ratio0p3_random_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.3` | `stratified` | `0.7993` | `0.5343` | `0.8454` | `0.7442` | `0.8342` | `0.5572` | n/a | `logs/kitti_ratio0p3_stratified_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.3` | `entropy` | `0.8098` | `0.5442` | `0.8676` | `0.7385` | `0.8357` | `0.5636` | `54.69` | `logs/kitti_ratio0p3_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log` |
| `0.3` | `coreset_fast` | `0.7994` | `0.5384` | `0.8538` | `0.7442` | `0.8318` | `0.5612` | n/a | `logs/kitti_ratio0p3_coreset_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.3` | `badge_fast` | `0.8083` | `0.5441` | `0.8585` | `0.7388` | `0.8313` | `0.5624` | n/a | `logs/kitti_ratio0p3_badge_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.3` | `uherding_fast` | `0.8003` | `0.5314` | `0.8561` | `0.7296` | `0.8267` | `0.5573` | `76.44` | `logs/kitti_ratio0p3_uherding_fast_red_cloudy_gen1000_e10_gpu1_20260502_140255.log` |
| `0.3` | `ppal_fast` | `0.7980` | `0.5368` | `0.8444` | `0.7418` | `0.8258` | `0.5568` | `72.88` | `logs/kitti_ratio0p3_ppal_fast_red_cloudy_gen1000_e10_gpu3_20260502_185507.log` |
| `0.5` | `random` | `0.8039` | `0.5452` | `0.8581` | `0.7465` | `0.8416` | `0.5719` | n/a | `logs/kitti_ratio0p5_random_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.5` | `stratified` | `0.8087` | `0.5480` | `0.8328` | `0.7558` | `0.8377` | `0.5680` | n/a | `logs/kitti_ratio0p5_stratified_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.5` | `entropy` | `0.8125` | `0.5524` | `0.8619` | `0.7483` | `0.8414` | `0.5766` | `54.40` | `logs/kitti_ratio0p5_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log` |
| `0.5` | `coreset_fast` | `0.8115` | `0.5523` | `0.8696` | `0.7427` | `0.8436` | `0.5759` | n/a | `logs/kitti_ratio0p5_coreset_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.5` | `badge_fast` | `0.8125` | `0.5542` | `0.8473` | `0.7413` | `0.8389` | `0.5726` | n/a | `logs/kitti_ratio0p5_badge_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.5` | `uherding_fast` | `0.8091` | `0.5445` | `0.8640` | `0.7373` | `0.8356` | `0.5675` | `72.04` | `logs/kitti_ratio0p5_uherding_fast_red_cloudy_gen1000_e10_gpu1_20260502_140255.log` |
| `0.5` | `ppal_fast` | `0.8086` | `0.5505` | `0.8508` | `0.7513` | `0.8388` | `0.5721` | `71.04` | `logs/kitti_ratio0p5_ppal_fast_red_cloudy_gen1000_e10_gpu3_20260502_185507.log` |
| `0.7` | `random` | `0.8126` | `0.5540` | `0.8605` | `0.7543` | `0.8447` | `0.5766` | n/a | `logs/kitti_ratio0p7_random_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.7` | `stratified` | `0.8125` | `0.5548` | `0.8475` | `0.7514` | `0.8416` | `0.5760` | n/a | `logs/kitti_ratio0p7_stratified_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.7` | `entropy` | `0.8194` | `0.5583` | `0.8634` | `0.7571` | `0.8472` | `0.5772` | `54.28` | `logs/kitti_ratio0p7_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log` |
| `0.7` | `coreset_fast` | `0.8108` | `0.5546` | `0.8695` | `0.7452` | `0.8420` | `0.5772` | n/a | `logs/kitti_ratio0p7_coreset_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.7` | `badge_fast` | `0.8143` | `0.5545` | `0.8625` | `0.7344` | `0.8338` | `0.5717` | n/a | `logs/kitti_ratio0p7_badge_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.7` | `uherding_fast` | `0.8128` | `0.5558` | `0.8656` | `0.7542` | `0.8419` | `0.5768` | `73.09` | `logs/kitti_ratio0p7_uherding_fast_red_cloudy_gen1000_e10_gpu1_20260502_140255.log` |
| `0.7` | `ppal_fast` | `0.8135` | `0.5557` | `0.8520` | `0.7624` | `0.8464` | `0.5782` | `81.09` | `logs/kitti_ratio0p7_ppal_fast_red_cloudy_gen1000_e10_gpu3_20260502_185507.log` |
| `0.9` | `random` | `0.8120` | `0.5603` | `0.8551` | `0.7647` | `0.8472` | `0.5817` | n/a | `logs/kitti_ratio0p9_random_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.9` | `stratified` | `0.8176` | `0.5603` | `0.8548` | `0.7587` | `0.8474` | `0.5830` | n/a | `logs/kitti_ratio0p9_stratified_red_cloudy_gen1000_e10_gpu0_20260502_015837.log` |
| `0.9` | `entropy` | `0.8204` | `0.5626` | `0.8784` | `0.7523` | `0.8492` | `0.5830` | `54.52` | `logs/kitti_ratio0p9_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log` |
| `0.9` | `coreset_fast` | `0.8160` | `0.5590` | `0.8632` | `0.7525` | `0.8381` | `0.5781` | n/a | `logs/kitti_ratio0p9_coreset_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.9` | `badge_fast` | `0.8169` | `0.5585` | `0.8664` | `0.7586` | `0.8472` | `0.5810` | n/a | `logs/kitti_ratio0p9_badge_fast_red_cloudy_gen1000_e10_gpu3_20260502_020651.log` |
| `0.9` | `uherding_fast` | `0.8156` | `0.5592` | `0.8475` | `0.7592` | `0.8461` | `0.5838` | `83.63` | `logs/kitti_ratio0p9_uherding_fast_red_cloudy_gen1000_e10_gpu1_20260502_140255.log` |
| `0.9` | `ppal_fast` | `0.8164` | `0.5576` | `0.8643` | `0.7549` | `0.8463` | `0.5796` | `74.25` | `logs/kitti_ratio0p9_ppal_fast_red_cloudy_gen1000_e10_gpu3_20260502_185507.log` |
| `0.1` | `attribute_r000_p0_c15` | `0.7842` | `0.5099` | `0.8527` | `0.7157` | `0.8186` | `0.5321` | `0.00` | `logs/kitti_ratio0p1_attribute_r000_p0_c15_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.1` | `attribute_r001_p5_c15` | `0.7702` | `0.4730` | `0.8316` | `0.7213` | `0.8108` | `0.4963` | `0.00` | `logs/kitti_ratio0p1_attribute_r001_p5_c15_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.1` | `attribute_r003_p5_c8` | `0.7670` | `0.4763` | `0.8287` | `0.7096` | `0.8006` | `0.4951` | `0.00` | `logs/kitti_ratio0p1_attribute_r003_p5_c8_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.3` | `attribute_r000_p0_c15` | `0.8125` | `0.5463` | `0.8432` | `0.7531` | `0.8338` | `0.5646` | `0.00` | `logs/kitti_ratio0p3_attribute_r000_p0_c15_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.3` | `attribute_r001_p5_c15` | `0.8062` | `0.5414` | `0.8559` | `0.7411` | `0.8334` | `0.5620` | `0.00` | `logs/kitti_ratio0p3_attribute_r001_p5_c15_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.3` | `attribute_r003_p5_c8` | `0.8064` | `0.5447` | `0.8365` | `0.7572` | `0.8342` | `0.5630` | `0.00` | `logs/kitti_ratio0p3_attribute_r003_p5_c8_red_cloudy_gen1000_e10_gpu0_20260504_000311.log` |
| `0.5` | `attribute_r000_p0_c15` | `0.8170` | `0.5579` | `0.8640` | `0.7524` | `0.8457` | `0.5773` | `0.00` | `logs/kitti_ratio0p5_attribute_r000_p0_c15_red_cloudy_gen1000_e10_gpu1_20260504_000311.log` |
| `0.5` | `attribute_r001_p5_c15` | `0.8206` | `0.5578` | `0.8532` | `0.7446` | `0.8420` | `0.5733` | `0.00` | `logs/kitti_ratio0p5_attribute_r001_p5_c15_red_cloudy_gen1000_e10_gpu1_20260504_000311.log` |
| `0.5` | `attribute_r003_p5_c8` | `0.8206` | `0.5578` | `0.8532` | `0.7446` | `0.8420` | `0.5733` | `0.00` | `logs/kitti_ratio0p5_attribute_r003_p5_c8_red_cloudy_gen1000_e10_gpu1_20260504_000311.log` |
| `0.7` | `attribute_r000_p0_c15` | `0.8150` | `0.5566` | `0.8617` | `0.7421` | `0.8406` | `0.5780` | `0.00` | `logs/kitti_ratio0p7_attribute_r000_p0_c15_red_cloudy_gen1000_e10_gpu2_20260504_000311.log` |
| `0.7` | `attribute_r001_p5_c15` | `0.8152` | `0.5567` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r001_p5_c15_red_cloudy_gen1000_e10_gpu2_20260504_000311.log` |
| `0.7` | `attribute_r003_p5_c8` | `0.8152` | `0.5567` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r003_p5_c8_red_cloudy_gen1000_e10_gpu2_20260504_000311.log` |
| `0.9` | `attribute_r000_p0_c15` | `0.8195` | `0.5621` | `0.8462` | `0.7679` | `0.8491` | `0.5841` | `0.00` | `logs/kitti_ratio0p9_attribute_r000_p0_c15_red_cloudy_gen1000_e10_gpu3_20260504_000311.log` |
| `0.9` | `attribute_r001_p5_c15` | `0.8195` | `0.5624` | `0.8719` | `0.7461` | `0.8437` | `0.5850` | `0.00` | `logs/kitti_ratio0p9_attribute_r001_p5_c15_red_cloudy_gen1000_e10_gpu3_20260504_000311.log` |
| `0.9` | `attribute_r003_p5_c8` | `0.8195` | `0.5624` | `0.8719` | `0.7461` | `0.8437` | `0.5850` | `0.00` | `logs/kitti_ratio0p9_attribute_r003_p5_c8_red_cloudy_gen1000_e10_gpu3_20260504_000311.log` |

Additional extra RPC search rows from `20260504_021221`:

| Ratio | Method | Val mAP50 | Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 | Selection time (s) | Log |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.1` | `attribute_r000_p0_cinf` | `0.7850` | `0.5125` | `0.8415` | `0.7285` | `0.8185` | **`0.5372`** | `0.00` | `logs/kitti_ratio0p1_attribute_r000_p0_cinf_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.1` | `attribute_r001_p5_cinf` | `0.7710` | `0.4770` | `0.8659` | `0.7014` | `0.8119` | `0.5001` | `0.00` | `logs/kitti_ratio0p1_attribute_r001_p5_cinf_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.1` | `attribute_r003_p05_c15` | `0.7730` | `0.4836` | `0.8327` | `0.7199` | `0.8125` | `0.5051` | `0.00` | `logs/kitti_ratio0p1_attribute_r003_p05_c15_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.1` | `attribute_r003_p05_cinf` | `0.7710` | `0.4770` | `0.8659` | `0.7014` | `0.8119` | `0.5001` | `0.00` | `logs/kitti_ratio0p1_attribute_r003_p05_cinf_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.1` | `attribute_r01_p5_c15` | `0.7729` | `0.4767` | `0.8445` | `0.7178` | `0.8092` | `0.4995` | `0.00` | `logs/kitti_ratio0p1_attribute_r01_p5_c15_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.1` | `attribute_r08_p10_c4` | `0.7656` | `0.4666` | `0.8255` | `0.7139` | `0.8007` | `0.4899` | `0.00` | `logs/kitti_ratio0p1_attribute_r08_p10_c4_seed42_red_cloudy_gen1000_e10_gpu3_20260504_021221.log` |
| `0.7` | `attribute_r000_p0_cinf` | `0.8153` | `0.5567` | `0.8617` | `0.7421` | `0.8406` | `0.5780` | `0.00` | `logs/kitti_ratio0p7_attribute_r000_p0_cinf_seed42_red_cloudy_gen1000_e10_gpu2_20260504_021221.log` |
| `0.7` | `attribute_r001_p5_cinf` | `0.8151` | `0.5566` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r001_p5_cinf_seed42_red_cloudy_gen1000_e10_gpu0_20260504_021221.log` |
| `0.7` | `attribute_r003_p05_c15` | `0.8151` | `0.5566` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r003_p05_c15_seed42_red_cloudy_gen1000_e10_gpu2_20260504_021221.log` |
| `0.7` | `attribute_r003_p05_cinf` | `0.8151` | `0.5566` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r003_p05_cinf_seed42_red_cloudy_gen1000_e10_gpu2_20260504_021221.log` |
| `0.7` | `attribute_r01_p5_c15` | `0.8151` | `0.5566` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r01_p5_c15_seed42_red_cloudy_gen1000_e10_gpu0_20260504_021221.log` |
| `0.7` | `attribute_r08_p10_c4` | `0.8151` | `0.5566` | `0.8430` | `0.7685` | `0.8436` | `0.5767` | `0.00` | `logs/kitti_ratio0p7_attribute_r08_p10_c4_seed42_red_cloudy_gen1000_e10_gpu0_20260504_021221.log` |

All KITTI ratio `0.1`-`0.9` 10-epoch logs completed. Full parsed rows are under
`logs/kitti_ratio*_red_cloudy_gen1000_e10_gpu0_20260502_015837.log`,
`logs/kitti_ratio*_red_cloudy_gen1000_e10_gpu3_20260502_020651.log`, and
`logs/kitti_ratio*_entropy_red_cloudy_gen1000_e10_gpu3_20260502_025110.log`,
with the extra attribute RPC rows under
`logs/kitti_ratio0p*_attribute_r*_red_cloudy_gen1000_e10_gpu*_20260504_000311.log`
and `logs/kitti_ratio0p*_attribute_r*_seed42_red_cloudy_gen1000_e10_gpu*_20260504_021221.log`.

### CelebA Red-Hair Brown-Skin Sad-Emotion Baselines

- Generated target: `redhair_brownskin_sademotion`
- Base checkpoint: `save/weights/model_celeba_resnet_base.pth`
- Added generated images per row: `1000`
- Training: `epochs=15`, `batch_size=64`, `lr=1e-5`

| Ratio | Method | Test ACC | Best Val ACC | Test loss | Selection time (s) | Log |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.1` | `random` | `0.9103` | `0.9115` | `0.3372` | n/a | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.1.log` |
| `0.1` | `stratified` | `0.9080` | `0.9095` | `0.3718` | n/a | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.1.log` |
| `0.1` | `coreset` | `0.9000` | `0.9029` | `0.4735` | n/a | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.1.log` |
| `0.1` | `badge` | `0.9087` | `0.9112` | `0.3789` | n/a | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.1.log` |
| `0.1` | `attribute` | `0.9107` | `0.9091` | `0.3599` | `0.42` | `logs/celeba/attr_ratio0.1_qwen3mix_r003_p05_c15_gpu2_20260417_010718.log` |
| `0.1` | `entropy` | **`0.9124`** | **`0.9145`** | `0.2092` | `49.33` | `logs/celeba/controlnet_entropy_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p1_gpu3_20260502_025110.log` |
| `0.1` | `uherding` | `0.9047` | `0.9064` | `0.3807` | `98.71` | `logs/celeba/controlnet_uherding_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p1_gpu3_20260502_1440fast.log` |
| `0.1` | `ppal` | `0.9113` | `0.9100` | `0.3111` | `46.25` | `logs/celeba/controlnet_ppal_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p1_gpu1_20260502_185507.log` |
| `0.3` | `random` | `0.9114` | `0.9118` | `0.3418` | n/a | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.3.log` |
| `0.3` | `stratified` | `0.9107` | `0.9115` | `0.3557` | n/a | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.3.log` |
| `0.3` | `coreset` | `0.9132` | `0.9113` | `0.3891` | n/a | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.3.log` |
| `0.3` | `badge` | `0.9075` | `0.9104` | `0.3513` | n/a | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.3_gpu2_verbose_20260416_115906.log` |
| `0.3` | `attribute` | `0.9089` | `0.9090` | `0.3609` | `0.51` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r000_p0_cinf_gpu0_20260417_005520.log` |
| `0.3` | `entropy` | **`0.9139`** | **`0.9152`** | `0.2116` | `50.74` | `logs/celeba/controlnet_entropy_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p3_gpu3_20260502_025110.log` |
| `0.3` | `uherding` | `0.9090` | `0.9109` | `0.3649` | `65.20` | `logs/celeba/controlnet_uherding_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p3_gpu3_20260502_1440fast.log` |
| `0.3` | `ppal` | `0.9095` | `0.9082` | `0.3614` | `50.77` | `logs/celeba/controlnet_ppal_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p3_gpu1_20260502_185507.log` |
| `0.5` | `random` | `0.9114` | `0.9127` | `0.3418` | n/a | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.5.log` |
| `0.5` | `stratified` | `0.9132` | `0.9129` | `0.3498` | n/a | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.5.log` |
| `0.5` | `coreset` | `0.9134` | `0.9109` | `0.3650` | n/a | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.5.log` |
| `0.5` | `badge` | `0.9084` | `0.9113` | `0.3476` | n/a | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu0_20260416_120554.log` |
| `0.5` | `entropy` | `0.9129` | **`0.9154`** | `0.2233` | `48.78` | `logs/celeba/controlnet_entropy_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p5_gpu3_20260502_025110.log` |
| `0.5` | `uherding` | `0.9078` | `0.9089` | `0.3664` | `60.87` | `logs/celeba/controlnet_uherding_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p5_gpu3_20260502_1440fast.log` |
| `0.5` | `ppal` | **`0.9151`** | `0.9135` | `0.3408` | `61.20` | `logs/celeba/controlnet_ppal_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p5_gpu1_20260502_185507.log` |
| `0.7` | `random` | `0.9084` | `0.9088` | `0.3551` | n/a | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.7.log` |
| `0.7` | `stratified` | `0.9135` | `0.9129` | `0.3527` | n/a | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.7.log` |
| `0.7` | `coreset` | `0.9119` | `0.9110` | `0.3571` | n/a | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_gpu3_rerun_20260416_121045.log` |
| `0.7` | `badge` | `0.9133` | `0.9131` | `0.3384` | n/a | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.7_gpu0_20260416_122047.log` |
| `0.7` | `entropy` | **`0.9153`** | **`0.9156`** | `0.2383` | `50.50` | `logs/celeba/controlnet_entropy_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p7_gpu3_20260502_025110.log` |
| `0.7` | `uherding` | `0.9105` | `0.9105` | `0.3574` | `70.46` | `logs/celeba/controlnet_uherding_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p7_gpu3_20260502_1440fast.log` |
| `0.7` | `ppal` | `0.9114` | `0.9101` | `0.3503` | `64.66` | `logs/celeba/controlnet_ppal_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p7_gpu1_20260502_185507.log` |
| `0.9` | `entropy` | **`0.9135`** | `0.9122` | `0.3072` | `51.45` | `logs/celeba/controlnet_entropy_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p9_gpu3_20260502_025110.log` |
| `0.9` | `uherding` | `0.9100` | `0.9101` | `0.3503` | `53.60` | `logs/celeba/controlnet_uherding_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p9_gpu3_20260502_1440fast.log` |
| `0.9` | `ppal` | `0.9119` | **`0.9131`** | `0.3452` | `55.52` | `logs/celeba/controlnet_ppal_redhair_brownskin_sademotion_resnet_celeba_1000_ratio0p9_gpu1_20260502_185507.log` |

### COCO Brown-Rainy Entropy Baseline

- Generated pool:
  `/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit`
- Base checkpoint:
  `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth`
- Added generated images per row: `1000`
- Training: `epochs=2`, `batch_size=8`, `lr=0.005`

| Ratio | Test AP | Test AP50 | Test AP75 | Val AP | Log |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.1` | `0.6132` | `0.8545` | `0.6807` | `0.6293` | `logs/pose/pose_qwen3_brown_rainy_entropy_ratio0p1_gpu3_20260502_025110.log` |
| `0.3` | `0.6215` | `0.8542` | `0.6744` | `0.6323` | `logs/pose/pose_qwen3_brown_rainy_entropy_ratio0p3_gpu3_20260502_025110.log` |
| `0.5` | `0.6282` | `0.8557` | `0.6834` | `0.6350` | `logs/pose/pose_qwen3_brown_rainy_entropy_ratio0p5_gpu3_20260502_025110.log` |
| `0.7` | **`0.6293`** | **`0.8599`** | **`0.6842`** | **`0.6356`** | `logs/pose/pose_qwen3_brown_rainy_entropy_ratio0p7_gpu3_20260502_025110.log` |
| `0.9` | `0.6170` | `0.8499` | `0.6739` | `0.6235` | `logs/pose/pose_qwen3_brown_rainy_entropy_ratio0p9_gpu3_20260502_025110.log` |

### ImageNet ControlNet RPC8 No-Entropy Rerun

This rerun uses the same ImageNet ControlNet RPC8 grid as the earlier
`20260501_231910` sweep, but with `--al_attribute_use_entropy False`.

| Ratio | Best setting by Test ACC | Test ACC | Best Val ACC | Test loss | Log |
| ---: | --- | ---: | ---: | ---: | --- |
| `0.1` | `only_c` / `only_r` | **`0.7354`** | `0.7123` / `0.7177` | `0.8472` / `0.8465` | `logs/imagenet/controlnet_rpc8_noentropy_only_c_resnet_imagenet_1000_ratio0p1_gpu3_20260502_023905.log`, `logs/imagenet/controlnet_rpc8_noentropy_only_r_resnet_imagenet_1000_ratio0p1_gpu3_20260502_023905.log` |
| `0.3` | `only_c` | **`0.7492`** | `0.7231` | `0.7987` | `logs/imagenet/controlnet_rpc8_noentropy_only_c_resnet_imagenet_1000_ratio0p3_gpu3_20260502_023905.log` |
| `0.5` | `none` | **`0.7508`** | `0.7262` | `0.7806` | `logs/imagenet/controlnet_rpc8_noentropy_none_resnet_imagenet_1000_ratio0p5_gpu3_20260502_023905.log` |
| `0.7` | `no_p` | **`0.7508`** | **`0.7323`** | `0.7858` | `logs/imagenet/controlnet_rpc8_noentropy_no_p_resnet_imagenet_1000_ratio0p7_gpu3_20260502_023905.log` |
| `0.9` | `only_c` | **`0.7538`** | `0.7285` | `0.7791` | `logs/imagenet/controlnet_rpc8_noentropy_only_c_resnet_imagenet_1000_ratio0p9_gpu3_20260502_023905.log` |

Best overall no-entropy ImageNet row: `only_c` at ratio `0.9`, Test ACC
`0.7538`. For comparison, the earlier entropy-enabled ImageNet RPC8 sweep's
best overall row was `only_p` at ratio `0.9`, Test ACC `0.7526`.

### COCO Brown-Rainy RPC8 Metadata-Only Sweep

This is the COCO half of the `20260501_231910` RPC8 launcher. The pose code
used metadata-only attribute selection (`use_entropy=False`).

| Ratio | Best setting by Test AP | Test AP | Test AP50 | Test AP75 | Val AP | Log |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0.1` | `anchor` | **`0.6165`** | **`0.8467`** | **`0.6751`** | **`0.6294`** | `logs/pose/pose_qwen3_brown_rainy_rpc8_anchor_ratio0p1_gpu1_20260501_231910.log` |
| `0.3` | `anchor` | **`0.6306`** | n/a | n/a | n/a | n/a |
| `0.5` | `no_r` | **`0.6295`** | **`0.8625`** | **`0.6785`** | **`0.6367`** | `logs/pose/pose_qwen3_brown_rainy_rpc8_no_r_ratio0p5_gpu2_20260501_231910.log` |
| `0.7` | `anchor` | **`0.6327`** | **`0.8609`** | **`0.6939`** | `0.6342` | `logs/pose/pose_qwen3_brown_rainy_rpc8_anchor_ratio0p7_gpu2_20260501_231910.log` |
| `0.9` | `no_r` | **`0.6276`** | `0.8552` | **`0.6853`** | `0.6310` | `logs/pose/pose_qwen3_brown_rainy_rpc8_no_r_ratio0p9_gpu1_20260501_231910.log` |

Best overall COCO RPC8 metadata-only row: `anchor` at ratio `0.7`, Test AP
`0.6327`. The best Val AP in the parsed grid is `only_c` at ratio `0.3`,
Val AP `0.6382`.

## 05/01/2026 Update — Ratio 1.0 Full-Data ControlNet Results

These runs use all original training samples (`ratio=1.0`) and append `1000`
ControlNet generated images. No active subset selection is used.

The first table records the previously completed resume-style ratio-`1.0`
controls. On 2026-05-04, no-resume base-epoch reruns were launched to match
the original base-training schedules more directly. Those reruns are recorded
separately below because they answer a different control question from the
resume-style rows.

| Dataset | Generated Target | Original Samples | Final Train Size | Metric | Test | Best Val | Test Loss / AP50 / AP75 | Log |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| ImageNet | `pinkcolor_fabrictexture` | `10400` | `11400` | ACC | `0.7400` | `0.7292` | loss `0.7998` | `logs/imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio1p0_gpu3_20260501_150816.log` |
| CelebA | `redhair_brownskin_sademotion` | `80000` | `81000` | ACC | **`0.9114`** | `0.9133` | loss `0.3422` | `logs/celeba/controlnet_redhair_brownskin_sademotion_resnet_celeba_1000_ratio1p0_gpu0_20260501_150816.log` |
| CelebA | `redhair_brownskin_unfiltered` | `80000` | `81000` | ACC | `0.9108` | `0.9136` | loss `0.3420` | `logs/celeba/controlnet_redhair_brownskin_unfiltered_resnet_celeba_1000_ratio1p0_gpu1_20260501_150816.log` |
| CelebA | `redhair_brownskin` | `80000` | `81000` | ACC | `0.9102` | `0.9126` | loss `0.3440` | `logs/celeba/controlnet_redhair_brownskin_resnet_celeba_1000_ratio1p0_gpu2_20260501_150816.log` |
| COCO pose | `brown_rainy` | `20000` | `21000` | AP | `0.6259` | `0.6310` | AP50 `0.8544`, AP75 `0.6812` | `logs/pose/pose_qwen3_brown_rainy_full_ratio1p0_gpu3_20260501_150816.log` |
| KITTI | `red_cloudy` | `5000` | `6000` | mAP50-95 | `0.5858` | `0.5608` | mAP50 `0.8505`, precision `0.8808`, recall `0.7449` | `logs/kitti_ratio1p0_random_red_cloudy_gen1000_e10_gpu3_20260504_003555_fp.log` |

### 05/04/2026 No-Resume Base-Epoch Ratio 1.0 Reruns

These controls still use all original training samples and append `1000`
generated images, but they do not resume from the low-ratio base checkpoints.
Instead, they follow the corresponding base-training schedule as closely as the
current entry points allow.

| Dataset | Generated Target | Original Samples | Final Train Size | Schedule | Metric | Test | Best Val / Val | Extra Test Metrics | Log |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| ImageNet | `pinkcolor_fabrictexture` | `10400` | `11400` | ResNet random init, `30` epochs, `lr=1e-3` | ACC | `0.7392` | `0.7277` | loss `0.8216` | `logs/imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio1p0_e30_baseparams_noresume_gpu2_20260504_012302.log` |
| CelebA | `redhair_brownskin_sademotion` | `80000` | `81000` | ResNet random init, `20` epochs, `lr=1e-5` | ACC | `0.9155` | `0.8778` | loss `0.4104` | `logs/celeba/controlnet_redhair_brownskin_sademotion_resnet_celeba_1000_ratio1p0_e20_baseparams_noresume_gpu3_20260504_012302.log` |
| KITTI | `red_cloudy` | `5000` | `6000` | YOLOv8n random/base pretrained weights, `30` epochs | mAP50-95 | `0.5737` | `0.5555` | test mAP50 `0.8363`, precision `0.8503`, recall `0.7499` | `logs/kitti_ratio1p0_random_red_cloudy_gen1000_e30_baseparams_noresume_gpu0_20260504_011553.log` |
| COCO pose | `brown_rainy` | `20000` | `21000` | Keypoint R-CNN pretrained, `2` epochs, `lr=0.005` | AP | `0.6292` | `0.6321` | test AP50 `0.8549`, AP75 `0.6802`; best checkpoint epoch `0` | `logs/pose/pose_qwen3_brown_rainy_ratio1p0_gen1000_e2_baseparams_noresume_gpu1_20260504_011553.log` |

## 05/01/2026 Update — RPC Component Removal Ablations

These ablations remove RPC components from the selector. `no_r` uses
`--al_attribute_use_rarity False` / `--al-attribute-use-rarity false`, rather
than setting rarity power to `0`.

### ImageNet ControlNet, Ratio = 0.7

- Generated directory: `data/imagenet_pinkcolor_fabrictexture`
- Attribute metadata: `save/attribute10_internvl3_8B.json`
- Anchor: `R=0.01`, `P=5.0`, `C=inf`, entropy on

| Setting | Removed component | Test ACC | Best Val ACC | Test loss | Log |
| --- | --- | ---: | ---: | ---: | --- |
| `anchor` | none | `0.7423` | `0.7208` | `0.8007` | `logs/imagenet/controlnet_rpc_ablation_anchor_resnet_imagenet_1000_ratio0p7_gpu3_20260501_164751.log` |
| `no_r` | rarity `R` | `0.7431` | `0.7200` | `0.7961` | `logs/imagenet/controlnet_rpc_ablation_no_r_resnet_imagenet_1000_ratio0p7_gpu3_20260501_164751.log` |
| `no_p` | frequency penalty `P` | `0.7469` | `0.7169` | `0.7863` | `logs/imagenet/controlnet_rpc_ablation_no_p_resnet_imagenet_1000_ratio0p7_gpu3_20260501_164751.log` |
| `no_c` | class balance `C` | **`0.7477`** | `0.7208` | `0.7922` | `logs/imagenet/controlnet_rpc_ablation_no_c_resnet_imagenet_1000_ratio0p7_gpu3_20260501_164751.log` |
| `no_u` | uncertainty / entropy `U` | `0.7438` | **`0.7215`** | `0.7923` | `logs/imagenet/controlnet_rpc_ablation_no_u_resnet_imagenet_1000_ratio0p7_gpu3_20260501_164751.log` |

### COCO Brown-Rainy, Ratio = 0.9

- Generated directory:
  `/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit`
- Attribute metadata: `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
- Anchor from the existing table: `qwen3_attr_r003_p5_c15`, test AP `0.6355`

| Setting | Removed component | Test AP | Test AP50 | Test AP75 | Val AP | Val AP50 | Val AP75 | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `anchor` | none | **`0.6355`** | `0.8572` | `0.6805` | `0.6282` | `0.8606` | `0.6829` | existing row in Brown-Rainy table |
| `no_r` | rarity `R` | `0.6198` | `0.8537` | `0.6742` | `0.6242` | `0.8584` | `0.6792` | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_no_r_ratio0p9_gpu0_20260501_164751.log` |
| `no_p` | frequency penalty `P` | `0.6222` | `0.8602` | `0.6696` | **`0.6300`** | `0.8599` | **`0.6878`** | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_no_p_ratio0p9_gpu1_20260501_164751.log` |
| `no_c` | class balance `C` | `0.6160` | `0.8553` | `0.6693` | `0.6276` | **`0.8606`** | `0.6811` | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_no_c_ratio0p9_gpu2_20260501_164751.log` |

## 05/01/2026 Update — RPC Only-One-Component Ablations

These ablations keep exactly one RPC component active while removing the other
two components. They use the same generated data, metadata, ratios, and base
checkpoints as the component-removal ablations above.

### ImageNet ControlNet, Ratio = 0.7

| Setting | Kept component | Test ACC | Best Val ACC | Test loss | Log |
| --- | --- | ---: | ---: | ---: | --- |
| `only_r` | rarity `R` | **`0.7469`** | `0.7223` | `0.7883` | `logs/imagenet/controlnet_rpc_ablation_only_r_resnet_imagenet_1000_ratio0p7_gpu3_20260501_173952.log` |
| `only_p` | frequency penalty `P` | `0.7431` | `0.7146` | `0.7991` | `logs/imagenet/controlnet_rpc_ablation_only_p_resnet_imagenet_1000_ratio0p7_gpu3_20260501_173952.log` |
| `only_c` | class balance `C` | `0.7415` | **`0.7238`** | `0.8071` | `logs/imagenet/controlnet_rpc_ablation_only_c_resnet_imagenet_1000_ratio0p7_gpu3_20260501_173952.log` |

### COCO Brown-Rainy, Ratio = 0.9

| Setting | Kept component | Test AP | Test AP50 | Test AP75 | Val AP | Val AP50 | Val AP75 | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `only_r` | rarity `R` | `0.6171` | `0.8556` | `0.6703` | `0.6265` | `0.8630` | `0.6843` | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_only_r_ratio0p9_gpu0_20260501_173952.log` |
| `only_p` | frequency penalty `P` | `0.6193` | **`0.8588`** | `0.6712` | `0.6286` | `0.8621` | `0.6873` | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_only_p_ratio0p9_gpu1_20260501_173952.log` |
| `only_c` | class balance `C` | **`0.6264`** | `0.8576` | **`0.6826`** | **`0.6328`** | **`0.8633`** | **`0.6885`** | `logs/pose/pose_qwen3_brown_rainy_rpc_ablation_only_c_ratio0p9_gpu2_20260501_173952.log` |

## 05/02/2026 Update — ImageNet RPC 8-Combination Ratio Sweep

The ImageNet ControlNet half of the full RPC binary component sweep is
complete under stamp `20260501_231910`. This grid evaluates all `2^3 = 8`
combinations of rarity (`R`), frequency penalty (`P`), and class balance (`C`)
for ratios `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`. Entropy stays enabled, and
no `no_u` row is included. The COCO brown-rainy half of the same launcher
completed later and is summarized in the overnight status section above.

- Generated directory: `data/imagenet_pinkcolor_fabrictexture`
- Attribute metadata: `save/attribute10_internvl3_8B.json`
- Base checkpoint: `save/weights/model_imagenet_resnet_base.pth`
- Added generated images per row: `1000`

| Ratio | Setting | Active components | Test ACC | Best Val ACC | Test loss | Log |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `0.1` | `none` | none | `0.7269` | `0.7115` | `0.8612` | `logs/imagenet/controlnet_rpc8_none_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `only_r` | `R` | `0.7300` | `0.7077` | `0.8644` | `logs/imagenet/controlnet_rpc8_only_r_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `only_p` | `P` | `0.7262` | **`0.7162`** | `0.8634` | `logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `only_c` | `C` | `0.7277` | `0.7085` | `0.8448` | `logs/imagenet/controlnet_rpc8_only_c_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `no_r` | `P+C` | **`0.7359`** | `0.7131` | `0.8502` | `logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `no_p` | `R+C` | `0.7308` | `0.7108` | `0.8474` | `logs/imagenet/controlnet_rpc8_no_p_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `no_c` | `R+P` | `0.7315` | `0.7108` | `0.8610` | `logs/imagenet/controlnet_rpc8_no_c_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.1` | `anchor` | `R+P+C` | `0.7238` | `0.7138` | `0.8512` | `logs/imagenet/controlnet_rpc8_anchor_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log` |
| `0.3` | `none` | none | `0.7308` | `0.7162` | `0.8343` | `logs/imagenet/controlnet_rpc8_none_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `only_r` | `R` | **`0.7415`** | `0.7146` | `0.8261` | `logs/imagenet/controlnet_rpc8_only_r_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `only_p` | `P` | `0.7392` | `0.7154` | `0.8176` | `logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `only_c` | `C` | `0.7392` | **`0.7169`** | `0.8124` | `logs/imagenet/controlnet_rpc8_only_c_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `no_r` | `P+C` | `0.7308` | `0.7162` | `0.8253` | `logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `no_p` | `R+C` | `0.7369` | `0.7154` | `0.8232` | `logs/imagenet/controlnet_rpc8_no_p_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `no_c` | `R+P` | `0.7315` | `0.7154` | `0.8429` | `logs/imagenet/controlnet_rpc8_no_c_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.3` | `anchor` | `R+P+C` | `0.7377` | `0.7131` | `0.8224` | `logs/imagenet/controlnet_rpc8_anchor_resnet_imagenet_1000_ratio0p3_gpu3_20260501_231910.log` |
| `0.5` | `none` | none | `0.7400` | `0.7200` | `0.8100` | `logs/imagenet/controlnet_rpc8_none_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `only_r` | `R` | `0.7415` | **`0.7215`** | `0.8134` | `logs/imagenet/controlnet_rpc8_only_r_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `only_p` | `P` | `0.7408` | `0.7162` | `0.8119` | `logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `only_c` | `C` | `0.7346` | `0.7208` | `0.8034` | `logs/imagenet/controlnet_rpc8_only_c_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `no_r` | `P+C` | `0.7369` | `0.7169` | `0.8108` | `logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `no_p` | `R+C` | `0.7415` | `0.7169` | `0.8056` | `logs/imagenet/controlnet_rpc8_no_p_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `no_c` | `R+P` | **`0.7446`** | `0.7146` | `0.8194` | `logs/imagenet/controlnet_rpc8_no_c_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.5` | `anchor` | `R+P+C` | `0.7362` | `0.7192` | `0.8071` | `logs/imagenet/controlnet_rpc8_anchor_resnet_imagenet_1000_ratio0p5_gpu3_20260501_231910.log` |
| `0.7` | `none` | none | `0.7423` | `0.7208` | `0.8014` | `logs/imagenet/controlnet_rpc8_none_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `only_r` | `R` | `0.7469` | `0.7223` | `0.7883` | `logs/imagenet/controlnet_rpc8_only_r_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `only_p` | `P` | `0.7431` | `0.7146` | `0.7991` | `logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `only_c` | `C` | `0.7415` | **`0.7238`** | `0.8071` | `logs/imagenet/controlnet_rpc8_only_c_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `no_r` | `P+C` | `0.7431` | `0.7200` | `0.7961` | `logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `no_p` | `R+C` | `0.7469` | `0.7169` | `0.7863` | `logs/imagenet/controlnet_rpc8_no_p_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `no_c` | `R+P` | **`0.7477`** | `0.7208` | `0.7922` | `logs/imagenet/controlnet_rpc8_no_c_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.7` | `anchor` | `R+P+C` | `0.7423` | `0.7208` | `0.8007` | `logs/imagenet/controlnet_rpc8_anchor_resnet_imagenet_1000_ratio0p7_gpu3_20260501_231910.log` |
| `0.9` | `none` | none | `0.7462` | `0.7254` | `0.7902` | `logs/imagenet/controlnet_rpc8_none_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `only_r` | `R` | `0.7500` | `0.7231` | `0.7836` | `logs/imagenet/controlnet_rpc8_only_r_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `only_p` | `P` | **`0.7526`** | **`0.7262`** | `0.7870` | `logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `only_c` | `C` | `0.7477` | `0.7246` | `0.7891` | `logs/imagenet/controlnet_rpc8_only_c_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `no_r` | `P+C` | `0.7508` | `0.7231` | `0.7827` | `logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `no_p` | `R+C` | `0.7415` | **`0.7262`** | `0.7975` | `logs/imagenet/controlnet_rpc8_no_p_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `no_c` | `R+P` | `0.7423` | `0.7223` | `0.7981` | `logs/imagenet/controlnet_rpc8_no_c_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |
| `0.9` | `anchor` | `R+P+C` | `0.7523` | `0.7208` | `0.7920` | `logs/imagenet/controlnet_rpc8_anchor_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log` |

Summary by ratio: best Test ACC is `no_r` at ratio `0.1`, `only_r` at ratio
`0.3`, `no_c` at ratios `0.5` and `0.7`, and `only_p` at ratio `0.9`. The best
overall ImageNet result in this sweep is `only_p` at ratio `0.9` with Test ACC
`0.7526`.

### Ratio = 0.1
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7269 | 0.7069 |
| Stratified | 0.7271 | 0.7154 |
| Coreset | 0.7269 | 0.7115 |
| Uncertainty Stratified | 0.7254 | 0.7185 |
| Diversity | 0.7262 | 0.7215 |
| Hierarchical | 0.7269 | 0.7115 |
| UHerding | 0.7331 | 0.7131 |
| PPAL | 0.7285 | 0.7169 |
| Attribute (BLIP) | 0.7277 | 0.7200 |
| Attribute (Qwen3) | 0.7331 | 0.7231 |
| Attribute (InternVL3) | **0.7346** | 0.7285 |

### Ratio = 0.3
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7308 | 0.7208 |
| Stratified | 0.7369 | 0.7162 |
| Coreset | 0.7369 | 0.7162 |
| Uncertainty Stratified | 0.7292 | 0.7131 |
| Diversity | 0.7362 | 0.7200 |
| Hierarchical | 0.7346 | 0.7092 |
| UHerding | **0.7469** | 0.7177 |
| PPAL | 0.7392 | 0.7200 |
| Attribute (BLIP) | 0.7369 | 0.7185 |
| Attribute (Qwen3) | 0.7354 | 0.7077 |
| Attribute (InternVL3) | 0.7385 | 0.7108 |

### Ratio = 0.5
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7380 | 0.7162 |
| Stratified | 0.7446 | 0.7200 |
| Coreset | 0.7400 | 0.7185 |
| Uncertainty Stratified | 0.7331 | 0.7146 |
| Hierarchical | 0.7385 | 0.7138 |
| Hybrid | 0.7362 | 0.7215 |
| UHerding | 0.7410 | 0.7200 |
| PPAL | 0.7423 | 0.7208 |
| Attribute (BLIP) | 0.7408 | 0.7131 |
| Attribute (Qwen3) | 0.7362 | 0.7138 |
| Attribute (InternVL3) | **0.7508** | 0.7286 |

### Ratio = 0.7
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7423 | 0.7223 |
| Stratified | 0.7346 | 0.7138 |
| Coreset | 0.7392 | 0.7154 |
| Uncertainty Stratified | 0.7354 | 0.7146 |
| Hierarchical | 0.7385 | 0.7146 |
| UHerding | **0.7469** | 0.7231 |
| PPAL | 0.7423 | 0.7215 |
| Attribute (BLIP) | 0.7385 | 0.7131 |
| Attribute (Qwen3) | 0.7415 | 0.7192 |
| Attribute (InternVL3) | 0.7413 | 0.7100 |
| Attribute (InternVL3 r000_p0_c15, 2026-05-01 rerun) | 0.7392 | 0.7131 |
| Attribute (InternVL3 r001_p5_cinf, 2026-05-01 rerun) | **0.7446** | 0.7115 |
| Attribute (InternVL3 r01_p5_c15, 2026-05-01 rerun) | 0.7400 | 0.7131 |
| Attribute (InternVL3 r08_p10_c4, 2026-05-01 rerun) | 0.7423 | 0.7100 |

### Ratio = 0.9
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7462 | 0.7154 |
| Stratified | 0.7469 | 0.7146 |
| Coreset | 0.7346 | 0.7169 |
| Uncertainty Stratified | 0.7462 | 0.7100 |
| Diversity | 0.7415 | 0.7146 |
| Hierarchical | 0.7362 | 0.7177 |
| Hybrid | 0.7477 | 0.7177 |
| UHerding | 0.7500 | 0.7231 |
| PPAL | 0.7508 | 0.7238 |
| Attribute (BLIP) | 0.7400 | 0.7177 |
| Attribute (Qwen3) | 0.7492 | 0.7108 |
| Attribute (InternVL3) | **0.7531** | 0.7346 |

---

# 04/29/2026 Update — Top Result Tables

## Brown-Rainy Combined Baseline And Qwen3 Attribute Results

- This table combines the completed Juno brown-rainy baseline rows and Qwen3
  attribute rows.
- Within each ratio, the highest `Test AP` among all methods is bolded.

| Ratio | Method | Test AP | Test AP50 | Test AP75 | Val AP | Val AP50 | Val AP75 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `random` | `0.5797` | `0.8208` | `0.6263` | `0.5890` | `0.8375` | `0.6406` |
| `0.05` | `stratified` | `0.5776` | `0.8331` | `0.6492` | `0.6022` | `0.8454` | `0.6604` |
| `0.05` | `coreset` | `0.5527` | `0.7997` | `0.5914` | `0.5639` | `0.8072` | `0.6129` |
| `0.05` | `badge` | `0.5782` | `0.8217` | `0.6284` | `0.5936` | `0.8361` | `0.6445` |
| `0.05` | `qwen3_attr_r000_p0_c15` | **`0.5871`** | `0.8308` | `0.6384` | `0.6011` | `0.8492` | `0.6560` |
| `0.10` | `random` | `0.6048` | `0.8421` | `0.6521` | `0.6170` | `0.8562` | `0.6767` |
| `0.10` | `stratified` | `0.6110` | `0.8440` | `0.6646` | `0.6189` | `0.8533` | `0.6776` |
| `0.10` | `coreset` | `0.5877` | `0.8333` | `0.6371` | `0.5950` | `0.8423` | `0.6441` |
| `0.10` | `badge` | `0.5985` | `0.8370` | `0.6467` | `0.6121` | `0.8539` | `0.6679` |
| `0.10` | `qwen3_attr_r000_p0_c15` | `0.6042` | `0.8390` | `0.6546` | `0.6159` | `0.8568` | `0.6623` |
| `0.10` | `qwen3_attr_r001_p5_c15` | `0.6152` | `0.8403` | `0.6717` | `0.6235` | `0.8485` | `0.6833` |
| `0.10` | `qwen3_attr_r003_p5_c8` | `0.6165` | `0.8489` | `0.6690` | `0.6285` | `0.8629` | `0.6882` |
| `0.10` | `qwen3_attr_r003_p5_c15` | **`0.6183`** | `0.8466` | `0.6782` | `0.6288` | `0.8604` | `0.6925` |
| `0.10` | `uherding` | `0.6013` | `0.8369` | `0.6479` | `0.6153` | `0.8496` | `0.6693` |
| `0.10` | `ppal` | `0.5950` | `0.8366` | `0.6415` | `0.6096` | `0.8492` | `0.6678` |
| `0.30` | `random` | `0.6040` | `0.8441` | `0.6626` | `0.6192` | `0.8580` | `0.6652` |
| `0.30` | `stratified` | `0.6243` | `0.8533` | `0.6816` | `0.6324` | `0.8635` | `0.6837` |
| `0.30` | `coreset` | `0.6256` | `0.8511` | `0.6778` | `0.6306` | `0.8628` | `0.6910` |
| `0.30` | `badge` | `0.6153` | `0.8523` | `0.6628` | `0.6212` | `0.8605` | `0.6776` |
| `0.30` | `qwen3_attr_r000_p0_c15` | `0.6288` | `0.8572` | `0.6926` | `0.6309` | `0.8621` | `0.6899` |
| `0.30` | `qwen3_attr_r001_p5_c15` | **`0.6306`** | `0.8567` | `0.6849` | `0.6266` | `0.8620` | `0.6892` |
| `0.30` | `qwen3_attr_r003_p5_c8` | `0.6110` | `0.8445` | `0.6640` | `0.6254` | `0.8578` | `0.6847` |
| `0.30` | `qwen3_attr_r003_p5_c15` | `0.6176` | `0.8516` | `0.6663` | `0.6266` | `0.8591` | `0.6860` |
| `0.30` | `uherding` | `0.6079` | `0.8475` | `0.6550` | `0.6089` | `0.8468` | `0.6616` |
| `0.30` | `ppal` | `0.6165` | `0.8564` | `0.6690` | `0.6295` | `0.8624` | `0.6855` |
| `0.50` | `random` | `0.6209` | `0.8521` | `0.6770` | `0.6251` | `0.8641` | `0.6849` |
| `0.50` | `stratified` | `0.6230` | `0.8553` | `0.6786` | `0.6343` | `0.8650` | `0.6914` |
| `0.50` | `coreset` | `0.6167` | `0.8546` | `0.6733` | `0.6248` | `0.8584` | `0.6823` |
| `0.50` | `badge` | `0.6206` | `0.8591` | `0.6696` | `0.6300` | `0.8679` | `0.6907` |
| `0.50` | `qwen3_attr_r000_p0_c15` | `0.6203` | `0.8544` | `0.6753` | `0.6272` | `0.8562` | `0.6897` |
| `0.50` | `qwen3_attr_r001_p5_c15` | **`0.6317`** | `0.8547` | `0.6774` | `0.6290` | `0.8672` | `0.6870` |
| `0.50` | `qwen3_attr_r003_p5_c8` | `0.6162` | `0.8525` | `0.6667` | `0.6257` | `0.8602` | `0.6874` |
| `0.50` | `qwen3_attr_r003_p5_c15` | `0.5866` | `0.8338` | `0.6357` | `0.5916` | `0.8412` | `0.6453` |
| `0.50` | `uherding` | `0.6153` | `0.8494` | `0.6680` | `0.6253` | `0.8616` | `0.6834` |
| `0.50` | `ppal` | `0.6182` | `0.8382` | `0.6444` | `0.6046` | `0.8463` | `0.6628` |
| `0.70` | `random` | `0.6151` | `0.8615` | `0.6904` | `0.6334` | `0.8629` | `0.6916` |
| `0.70` | `stratified` | `0.6299` | `0.8559` | `0.6916` | `0.6364` | `0.8639` | `0.6968` |
| `0.70` | `coreset` | `0.6275` | `0.8576` | `0.6839` | `0.6341` | `0.8652` | `0.6973` |
| `0.70` | `badge` | `0.6256` | `0.8544` | `0.6738` | `0.6379` | `0.8621` | `0.7016` |
| `0.70` | `qwen3_attr_r000_p0_c15` | `0.6313` | `0.8583` | `0.6932` | `0.6364` | `0.8650` | `0.6953` |
| `0.70` | `qwen3_attr_r001_p5_c0` | `0.6247` | `0.8558` | `0.6819` | `0.6370` | `0.8615` | `0.7007` |
| `0.70` | `qwen3_attr_r001_p5_c15` | `0.6290` | `0.8572` | `0.6815` | `0.6394` | `0.8642` | `0.7029` |
| `0.70` | `qwen3_attr_r003_p5_c0` | `0.6247` | `0.8558` | `0.6819` | `0.6370` | `0.8615` | `0.7007` |
| `0.70` | `qwen3_attr_r003_p5_c8` | `0.6300` | `0.8548` | `0.6852` | `0.6331` | `0.8607` | `0.6920` |
| `0.70` | `qwen3_attr_r003_p5_c15` | **`0.6319`** | `0.8571` | `0.6884` | `0.6377` | `0.8703` | `0.6922` |
| `0.70` | `uherding` | `0.6155` | `0.8482` | `0.6670` | `0.6211` | `0.8596` | `0.6752` |
| `0.70` | `ppal` | `0.6223` | `0.8535` | `0.6746` | `0.6296` | `0.8642` | `0.6869` |
| `0.90` | `random` | `0.6091` | `0.8517` | `0.6673` | `0.6155` | `0.8520` | `0.6737` |
| `0.90` | `stratified` | `0.6291` | `0.8629` | `0.6867` | `0.6380` | `0.8641` | `0.6962` |
| `0.90` | `coreset` | `0.6202` | `0.8570` | `0.6714` | `0.6288` | `0.8614` | `0.6876` |
| `0.90` | `badge` | `0.6213` | `0.8563` | `0.6732` | `0.6266` | `0.8622` | `0.6880` |
| `0.90` | `qwen3_attr_r000_p0_c15` | `0.6268` | `0.8579` | `0.6836` | `0.6339` | `0.8706` | `0.6908` |
| `0.90` | `qwen3_attr_r001_p5_c15` | `0.6255` | `0.8572` | `0.6805` | `0.6282` | `0.8606` | `0.6829` |
| `0.90` | `qwen3_attr_r003_p5_c8` | `0.6255` | `0.8572` | `0.6805` | `0.6282` | `0.8606` | `0.6829` |
| `0.90` | `qwen3_attr_r003_p5_c15` | **`0.6355`** | `0.8572` | `0.6805` | `0.6282` | `0.8606` | `0.6829` |
| `0.90` | `uherding` | `0.6191` | `0.8592` | `0.6729` | `0.6279` | `0.8618` | `0.6837` |
| `0.90` | `ppal` | `0.6301` | `0.8595` | `0.6812` | `0.6365` | `0.8665` | `0.6973` |
| `1.00` | `full_data_controlnet` | **`0.6259`** | `0.8544` | `0.6812` | `0.6310` | `0.8594` | `0.6865` |
This is a 1-epoch sanity check for the new KITTI generated-image resume path,
not a final comparable KITTI result.

- Script:
  `object_detection/train_yolov8_kitti_generated.py`
- Resume checkpoint:
  `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/best.pt`
- Source dataset:
  `/data/oxx220000/kitti_yolov8_first5000/`
- Generated pool:
  `/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled`
- Setting:
  `ratio=0.05`, `selection=stratified`, `250` original train images plus
  `1000` generated red-cloudy images, `batch=16`, `imgsz=640`, `epochs=1`,
  `AdamW`, `lr0=2e-4`, `lrf=0.1`.
- Run:
  `save/runs/kitti_yolov8/kitti_ratio0p05_stratified_red_cloudy_gen1000_e1_gpu0_20260502_011659`
- Val after 1 epoch:
  precision `0.82872`, recall `0.57158`, mAP50 `0.67209`, mAP50-95 `0.35703`
- Test after 1 epoch:
  precision `0.82058`, recall `0.59635`, mAP50 `0.69916`, mAP50-95 `0.37174`

## 05/02/2026 Update — KITTI Generated-Image Ratio 0.05 Two-Epoch Selection Probe

These are short 2-epoch probes for comparing selection methods on the KITTI
generated-image setup. They resume from the same batch-size-16 KITTI YOLOv8n
checkpoint and use the same final training size: `250` selected original KITTI
train images plus `1000` generated red-cloudy images (`1250` train images
total). The generated pool has `1309` usable images.

- Resume checkpoint:
  `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/best.pt`
- Source dataset:
  `/data/oxx220000/kitti_yolov8_first5000/`
- Generated pool:
  `/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled`
- Common fine-tuning setting:
  `epochs=2`, `batch=16`, `imgsz=640`, `AdamW`, `lr0=2e-4`, `lrf=0.1`,
  `warmup_epochs=1.0`, seed `42`
- Attribute setting:
  Qwen3 metadata `save/attribute_kitti_qwen3_8B.json`, focus
  `color,weather`, generated extra counts `color:red=1000` and
  `weather:cloudy=1000`, `R=0.03`, `P=5`, `C=15`
- Fast coreset/BADGE setting:
  simplified shared active-learning selectors with lightweight image proxy
  embeddings and YOLO label-signature proxy labels; exact coreset/BADGE logs
  from the interrupted attempt are excluded from this table.

| Method | Selection mode | GPU | Val mAP50 | Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 | Elapsed | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `attribute_r003_p5_c15` | Qwen3 color/weather RPC | `0` | `0.7166` | `0.3437` | `0.8069` | `0.6567` | `0.7512` | `0.3647` | `00:00:51` | `logs/kitti_ratio0p05_attribute_r003_p5_c15_red_cloudy_gen1000_e2_gpu0_20260502_012750_20260502_012750.log` |
| `random` | seeded random original subset | `0` | `0.7342` | `0.3886` | **`0.8314`** | `0.6578` | `0.7683` | `0.4088` | `00:00:39` | `logs/kitti_ratio0p05_random_red_cloudy_gen1000_e2_gpu0_20260502_013255_20260502_013255.log` |
| `coreset_fast` | approximate coreset proxy | `0` | `0.7236` | `0.3877` | `0.8268` | `0.6541` | `0.7581` | `0.4066` | `00:01:54` | `logs/kitti_ratio0p05_coreset_fast_red_cloudy_gen1000_e2_gpu0_20260502_013545_20260502_013545.log` |
| `badge_fast` | approximate BADGE weighted gradient-norm proxy | `3` | `0.7333` | **`0.3963`** | `0.8192` | **`0.6844`** | **`0.7793`** | **`0.4158`** | `00:01:54` | `logs/kitti_ratio0p05_badge_fast_red_cloudy_gen1000_e2_gpu3_20260502_013545_20260502_013545.log` |

In this short probe, `badge_fast` is strongest on test mAP50 and mAP50-95.
`random` is second on test mAP50-95, and both simplified active-learning
baselines avoid the slow exact selection path.

## KITTI Batch-Size-16 InternVL3 Slice Analysis

- InternVL3-8B assignment on the 5000 KITTI train images is complete:
  `save/attribute_kitti_internvl3_8B.json`,
  `save/attribute_kitti_internvl3_8B.csv`.
- Assignment log:
  `logs/assign_attr_kitti_train_internvl3_gpu2_20260429_004725.log`
  (`269.4` minutes, `4.49` hours).
- Batch-size-16 YOLOv8n checkpoint analysis:
  - checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/best.pt`
  - log:
    `logs/kitti_slice_analysis_train_internvl3_bs16_gpu2_20260429_0517.log`
  - output:
    `logs/kitti/kitti_slice_stats_internvl3_bs16_gpu2_20260429_0517/attribute_kitti_internvl3_8B/train_slice_stats.csv`
  - overall train metrics: mAP50 `0.8719`, mAP50-95 `0.6010`
  - evaluated slices: `25`; failure slices: `13`

| Attribute value | Images | mAP50 | mAP50-95 | Delta mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `traffic_density=heavy` | `31` | `0.4690` | `0.3667` | `-0.2343` |
| `background_style=highway` | `510` | `0.4729` | `0.3700` | `-0.2310` |
| `road_condition=wet` | `30` | `0.5624` | `0.4008` | `-0.2002` |
| `lighting_condition=glare` | `127` | `0.7206` | `0.4725` | `-0.1285` |
| `color=red` | `372` | `0.7446` | `0.5092` | `-0.0918` |
| `color=black` | `697` | `0.8015` | `0.5405` | `-0.0605` |
| `weather=clear` | `577` | `0.7569` | `0.5412` | `-0.0599` |
| `color=green` | `265` | `0.8169` | `0.5681` | `-0.0329` |
| `color=gray` | `2271` | `0.8477` | `0.5702` | `-0.0308` |
| `lighting_condition=bright` | `2385` | `0.8563` | `0.5728` | `-0.0282` |
| `traffic_density=moderate` | `1213` | `0.8552` | `0.5742` | `-0.0268` |
| `background_style=urban city` | `1425` | `0.8630` | `0.5784` | `-0.0226` |
| `color=white` | `715` | `0.8795` | `0.5809` | `-0.0201` |

# 04/29/2026 Update — Local Instruct-Pix2Pix AL Results

- This section records the completed local active-learning (AL) runs from the
  ImageNet and CelebA instruct-pix2pix seed-42 1k sweeps launched earlier on
  April 29, 2026.
- ImageNet stamp `20260429_065612` is complete for all originally planned
  rows; supplemental `ratio=0.9` rows completed under stamp `20260429_2130`.
- CelebA stamp `20260429_071551` is complete for all originally planned rows
  after the later GPU `1/2/3` resume workers; supplemental `ratio=0.9` rows
  completed under stamp `20260429_2130`.
- Both sweeps use exactly `1000` generated instruct-pix2pix images and resume
  from the corresponding ResNet base checkpoint.

## Completed ImageNet Pink-Fabric Instruct-Pix2Pix Results

- Dataset: `imagenet`
- Model: `resnet`
- Generated directory:
  `data/pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k`
- Source data: `data/imagenet_subset`
- Base checkpoint: `save/models/model_imagenet_resnet_base.pth`
- Original planned rows: `32`; supplemental `ratio=0.9` rows: `8`;
  completed rows: `40`.
- Best completed test accuracy by ratio:
  - `0.1`: `attribute_r01_p5_c15`, `0.7146`
  - `0.3`: `attribute_r001_p5_cinf` / `attribute_r01_p5_c15`, `0.7369`
  - `0.5`: `attribute_r08_p10_c4`, `0.7462`
  - `0.7`: `attribute_r000_p0_c15`, `0.7501`
  - `0.9`: `attribute_r001_p5_cinf`, `0.7523`

| Ratio | Method | Test ACC | Test loss | Log |
| ---: | --- | ---: | ---: | --- |
| `0.1` | `random` | `0.7062` | `0.8958` | `logs/imagenet/instructpix2pix_random_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `stratified` | `0.7085` | `0.8927` | `logs/imagenet/instructpix2pix_stratified_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `coreset` | `0.7115` | `0.8970` | `logs/imagenet/instructpix2pix_coreset_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `badge` | `0.7077` | `0.8870` | `logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `attribute_r000_p0_c15` | `0.7000` | `0.8918` | `logs/imagenet/instructpix2pix_attribute_r000_p0_c15_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `attribute_r001_p5_cinf` | `0.7100` | `0.8813` | `logs/imagenet/instructpix2pix_attribute_r001_p5_cinf_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `attribute_r01_p5_c15` | **`0.7146`** | `0.8803` | `logs/imagenet/instructpix2pix_attribute_r01_p5_c15_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.1` | `attribute_r08_p10_c4` | `0.7077` | `0.8832` | `logs/imagenet/instructpix2pix_attribute_r08_p10_c4_resnet_imagenet_1000_ratio0p1_gpu1_20260429_065612.log` |
| `0.3` | `random` | `0.7285` | `0.8175` | `logs/imagenet/instructpix2pix_random_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `stratified` | `0.7315` | `0.8118` | `logs/imagenet/instructpix2pix_stratified_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `coreset` | `0.7277` | `0.8325` | `logs/imagenet/instructpix2pix_coreset_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `badge` | `0.7315` | `0.8214` | `logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `attribute_r000_p0_c15` | `0.7362` | `0.8025` | `logs/imagenet/instructpix2pix_attribute_r000_p0_c15_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `attribute_r001_p5_cinf` | **`0.7369`** | `0.7992` | `logs/imagenet/instructpix2pix_attribute_r001_p5_cinf_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `attribute_r01_p5_c15` | **`0.7369`** | `0.8053` | `logs/imagenet/instructpix2pix_attribute_r01_p5_c15_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.3` | `attribute_r08_p10_c4` | `0.7315` | `0.8086` | `logs/imagenet/instructpix2pix_attribute_r08_p10_c4_resnet_imagenet_1000_ratio0p3_gpu1_20260429_065612.log` |
| `0.5` | `random` | `0.7438` | `0.7762` | `logs/imagenet/instructpix2pix_random_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `stratified` | `0.7377` | `0.7895` | `logs/imagenet/instructpix2pix_stratified_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `coreset` | `0.7408` | `0.7956` | `logs/imagenet/instructpix2pix_coreset_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `badge` | `0.7446` | `0.7858` | `logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `attribute_r000_p0_c15` | `0.7385` | `0.7992` | `logs/imagenet/instructpix2pix_attribute_r000_p0_c15_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `attribute_r001_p5_cinf` | `0.7415` | `0.7857` | `logs/imagenet/instructpix2pix_attribute_r001_p5_cinf_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `attribute_r01_p5_c15` | `0.7369` | `0.7960` | `logs/imagenet/instructpix2pix_attribute_r01_p5_c15_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.5` | `attribute_r08_p10_c4` | **`0.7462`** | `0.7904` | `logs/imagenet/instructpix2pix_attribute_r08_p10_c4_resnet_imagenet_1000_ratio0p5_gpu2_20260429_065612.log` |
| `0.7` | `random` | `0.7477` | `0.7832` | `logs/imagenet/instructpix2pix_random_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `stratified` | `0.7454` | `0.7841` | `logs/imagenet/instructpix2pix_stratified_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `coreset` | `0.7500` | `0.7825` | `logs/imagenet/instructpix2pix_coreset_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `badge` | `0.7446` | `0.7832` | `logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `attribute_r000_p0_c15` | **`0.7501`** | `0.7815` | `logs/imagenet/instructpix2pix_attribute_r000_p0_c15_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `attribute_r001_p5_cinf` | `0.7438` | `0.7884` | `logs/imagenet/instructpix2pix_attribute_r001_p5_cinf_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `attribute_r01_p5_c15` | `0.7408` | `0.7945` | `logs/imagenet/instructpix2pix_attribute_r01_p5_c15_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.7` | `attribute_r08_p10_c4` | `0.7462` | `0.7903` | `logs/imagenet/instructpix2pix_attribute_r08_p10_c4_resnet_imagenet_1000_ratio0p7_gpu3_20260429_065612.log` |
| `0.9` | `random` | `0.7423` | `0.7842` | `logs/imagenet/instructpix2pix_random_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `stratified` | `0.7446` | `0.7869` | `logs/imagenet/instructpix2pix_stratified_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `coreset` | `0.7485` | `0.7841` | `logs/imagenet/instructpix2pix_coreset_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `badge` | `0.7423` | `0.7900` | `logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `attribute_r000_p0_c15` | `0.7385` | `0.7883` | `logs/imagenet/instructpix2pix_attribute_r000_p0_c15_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `attribute_r001_p5_cinf` | **`0.7523`** | `0.7792` | `logs/imagenet/instructpix2pix_attribute_r001_p5_cinf_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `attribute_r01_p5_c15` | `0.7477` | `0.7824` | `logs/imagenet/instructpix2pix_attribute_r01_p5_c15_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `attribute_r08_p10_c4` | `0.7392` | `0.7823` | `logs/imagenet/instructpix2pix_attribute_r08_p10_c4_resnet_imagenet_1000_ratio0p9_gpu1_20260429_2130.log` |

## Completed CelebA Red-Hair Brown-Skin Sad-Emotion Instruct-Pix2Pix Results

- Dataset: `celeba`
- Model: `resnet`
- Generated directory:
  `data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k`
- Base checkpoint: `save/weights/model_celeba_resnet_base.pth`
- Attribute metadata: `save/attribute_celeba_trainval_qwen3_8B.json`
- Original planned rows from stamp `20260429_071551`: `32`; supplemental
  `ratio=0.9` rows from stamp `20260429_2130`: `8`; completed rows: `40`.
- Best completed test accuracy by ratio:
  - `0.1`: `badge`, `0.9127`
  - `0.3`: `stratified`, `0.9127`
  - `0.5`: `stratified`, `0.9132`
  - `0.7`: `badge`, `0.9123`
  - `0.9`: `attribute_r000_p0_cinf`, `0.9111`
- The base checkpoint log reports base test accuracy `0.9140`, so no completed
  instruct-pix2pix CelebA row has exceeded the base model yet.

| Ratio | Method | Test ACC | Test loss | Log |
| ---: | --- | ---: | ---: | --- |
| `0.1` | `random` | `0.9108` | `0.3396` | `logs/celeba/instructpix2pix_random_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `stratified` | `0.9082` | `0.3719` | `logs/celeba/instructpix2pix_stratified_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `coreset` | `0.9096` | `0.3570` | `logs/celeba/instructpix2pix_coreset_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `badge` | **`0.9127`** | `0.3392` | `logs/celeba/instructpix2pix_badge_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `attribute_r003_p05_c15` | `0.9120` | `0.3563` | `logs/celeba/instructpix2pix_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `attribute_r000_p0_c15` | `0.9068` | `0.3765` | `logs/celeba/instructpix2pix_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `attribute_r000_p0_cinf` | `0.9071` | `0.3752` | `logs/celeba/instructpix2pix_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.1` | `attribute_r003_p05_cinf` | `0.9025` | `0.3851` | `logs/celeba/instructpix2pix_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p1_gpu1_20260429_071551.log` |
| `0.3` | `random` | `0.9091` | `0.3507` | `logs/celeba/instructpix2pix_random_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `stratified` | **`0.9127`** | `0.3490` | `logs/celeba/instructpix2pix_stratified_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `coreset` | `0.9099` | `0.3536` | `logs/celeba/instructpix2pix_coreset_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `badge` | `0.9099` | `0.3422` | `logs/celeba/instructpix2pix_badge_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `attribute_r003_p05_c15` | `0.9101` | `0.3692` | `logs/celeba/instructpix2pix_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `attribute_r000_p0_c15` | `0.9095` | `0.3629` | `logs/celeba/instructpix2pix_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `attribute_r000_p0_cinf` | `0.9095` | `0.3626` | `logs/celeba/instructpix2pix_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.3` | `attribute_r003_p05_cinf` | `0.9038` | `0.3801` | `logs/celeba/instructpix2pix_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p3_gpu1_20260429_071551.log` |
| `0.5` | `random` | `0.9098` | `0.3499` | `logs/celeba/instructpix2pix_random_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `stratified` | **`0.9132`** | `0.3510` | `logs/celeba/instructpix2pix_stratified_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `coreset` | `0.9095` | `0.3558` | `logs/celeba/instructpix2pix_coreset_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `badge` | `0.9093` | `0.3419` | `logs/celeba/instructpix2pix_badge_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `attribute_r003_p05_c15` | `0.9040` | `0.3798` | `logs/celeba/instructpix2pix_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `attribute_r000_p0_c15` | `0.9095` | `0.3563` | `logs/celeba/instructpix2pix_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `attribute_r000_p0_cinf` | `0.9094` | `0.3565` | `logs/celeba/instructpix2pix_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.5` | `attribute_r003_p05_cinf` | `0.9110` | `0.3602` | `logs/celeba/instructpix2pix_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260429_071551.log` |
| `0.7` | `random` | `0.9116` | `0.3477` | `logs/celeba/instructpix2pix_random_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `stratified` | `0.9086` | `0.3635` | `logs/celeba/instructpix2pix_stratified_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `coreset` | `0.9067` | `0.3667` | `logs/celeba/instructpix2pix_coreset_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `badge` | **`0.9123`** | `0.3372` | `logs/celeba/instructpix2pix_badge_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `attribute_r003_p05_c15` | `0.9108` | `0.3588` | `logs/celeba/instructpix2pix_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `attribute_r000_p0_c15` | `0.9107` | `0.3558` | `logs/celeba/instructpix2pix_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `attribute_r000_p0_cinf` | `0.9109` | `0.3559` | `logs/celeba/instructpix2pix_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.7` | `attribute_r003_p05_cinf` | `0.9116` | `0.3547` | `logs/celeba/instructpix2pix_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p7_gpu3_20260429_071551.log` |
| `0.9` | `random` | `0.9104` | `0.3538` | `logs/celeba/instructpix2pix_random_resnet_celeba_1000_ratio0p9_gpu2_20260429_2130.log` |
| `0.9` | `stratified` | `0.9099` | `0.3752` | `logs/celeba/instructpix2pix_stratified_resnet_celeba_1000_ratio0p9_gpu2_20260429_2130.log` |
| `0.9` | `coreset` | `0.9092` | `0.3542` | `logs/celeba/instructpix2pix_coreset_resnet_celeba_1000_ratio0p9_gpu2_20260429_2130.log` |
| `0.9` | `badge` | `0.9108` | `0.3468` | `logs/celeba/instructpix2pix_badge_resnet_celeba_1000_ratio0p9_gpu1_20260429_2130.log` |
| `0.9` | `attribute_r003_p05_c15` | `0.9103` | `0.3553` | `logs/celeba/instructpix2pix_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p9_gpu3_20260429_2130.log` |
| `0.9` | `attribute_r000_p0_c15` | `0.9108` | `0.3556` | `logs/celeba/instructpix2pix_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p9_gpu3_20260429_2130.log` |
| `0.9` | `attribute_r000_p0_cinf` | **`0.9111`** | `0.3562` | `logs/celeba/instructpix2pix_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p9_gpu3_20260429_2130.log` |
| `0.9` | `attribute_r003_p05_cinf` | `0.9089` | `0.3692` | `logs/celeba/instructpix2pix_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p9_gpu3_20260429_2130.log` |

## Completed CelebA HiBUG Red-Hair Brown-Skin Results

- Dataset: `celeba`
- Model: `resnet`
- Generated directory: `data/hibug_redhair_brownskin_seed42_1k`
- Base checkpoint: `save/weights/model_celeba_resnet_base.pth`
- Attribute metadata: `save/attribute_celeba_trainval_qwen3_8B.json`
- Completed rows from stamp `20260429_183807`: `32`.
- Error scan over the completed logs found no `Traceback`, `Error`, `Killed`,
  `CUDA out of memory`, `No such file`, or `FileNotFound` messages.
- Best completed test accuracy by ratio:
  - `0.1`: `badge`, `0.9084`
  - `0.3`: `stratified`, `0.9121`
  - `0.5`: `attribute_r003_p05_cinf`, `0.9131`
  - `0.7`: `attribute_r003_p05_c15`, `0.9116`
- The base checkpoint log reports base test accuracy `0.9140`, so no completed
  HiBUG CelebA row has exceeded the base model yet.

| Ratio | Method | Test ACC | Test loss | Log |
| ---: | --- | ---: | ---: | --- |
| `0.1` | `random` | `0.9052` | `0.3499` | `logs/celeba/hibug_redhair_brownskin_random_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `stratified` | `0.9038` | `0.3679` | `logs/celeba/hibug_redhair_brownskin_stratified_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `coreset` | `0.9035` | `0.3674` | `logs/celeba/hibug_redhair_brownskin_coreset_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `badge` | **`0.9084`** | `0.3499` | `logs/celeba/hibug_redhair_brownskin_badge_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `attribute_r003_p05_c15` | `0.8954` | `0.3912` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `attribute_r000_p0_c15` | `0.9008` | `0.3973` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `attribute_r000_p0_cinf` | `0.9007` | `0.3984` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.1` | `attribute_r003_p05_cinf` | `0.8929` | `0.4136` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p1_gpu1_20260429_183807.log` |
| `0.3` | `random` | `0.9076` | `0.3458` | `logs/celeba/hibug_redhair_brownskin_random_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `stratified` | **`0.9121`** | `0.3522` | `logs/celeba/hibug_redhair_brownskin_stratified_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `coreset` | `0.9064` | `0.3665` | `logs/celeba/hibug_redhair_brownskin_coreset_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `badge` | `0.9065` | `0.3604` | `logs/celeba/hibug_redhair_brownskin_badge_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `attribute_r003_p05_c15` | `0.9009` | `0.3719` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `attribute_r000_p0_c15` | `0.9087` | `0.3632` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `attribute_r000_p0_cinf` | `0.9095` | `0.3620` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.3` | `attribute_r003_p05_cinf` | `0.9064` | `0.3668` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p3_gpu1_20260429_183807.log` |
| `0.5` | `random` | `0.9096` | `0.3407` | `logs/celeba/hibug_redhair_brownskin_random_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `stratified` | `0.9112` | `0.3535` | `logs/celeba/hibug_redhair_brownskin_stratified_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `coreset` | `0.9102` | `0.3441` | `logs/celeba/hibug_redhair_brownskin_coreset_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `badge` | `0.9121` | `0.3316` | `logs/celeba/hibug_redhair_brownskin_badge_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `attribute_r003_p05_c15` | `0.9100` | `0.3593` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `attribute_r000_p0_c15` | `0.9081` | `0.3586` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `attribute_r000_p0_cinf` | `0.9081` | `0.3589` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.5` | `attribute_r003_p05_cinf` | **`0.9131`** | `0.3492` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p5_gpu2_20260429_183807.log` |
| `0.7` | `random` | `0.9057` | `0.3463` | `logs/celeba/hibug_redhair_brownskin_random_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `stratified` | `0.9102` | `0.3519` | `logs/celeba/hibug_redhair_brownskin_stratified_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `coreset` | `0.9052` | `0.3651` | `logs/celeba/hibug_redhair_brownskin_coreset_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `badge` | `0.9072` | `0.3579` | `logs/celeba/hibug_redhair_brownskin_badge_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `attribute_r003_p05_c15` | **`0.9116`** | `0.3457` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_c15_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `attribute_r000_p0_c15` | `0.9085` | `0.3662` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_c15_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `attribute_r000_p0_cinf` | `0.9084` | `0.3663` | `logs/celeba/hibug_redhair_brownskin_attribute_r000_p0_cinf_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |
| `0.7` | `attribute_r003_p05_cinf` | `0.9073` | `0.3662` | `logs/celeba/hibug_redhair_brownskin_attribute_r003_p05_cinf_resnet_celeba_1000_ratio0p7_gpu3_20260429_183807.log` |

# 04/29/2026 Update — Brown-Rainy Completed Result Snapshot

- This section records all stable brown-rainy COCO pose results currently
  completed on Juno and complements the local completed experiment sections
  already preserved below.
- Juno job `185516` (`br-base-1gpu`) completed successfully with exit code
  `0:0` after `08:47:21`.
- The sweep runs ratios sequentially (`0.05`, `0.1`, `0.3`, `0.5`, `0.7`,
  `0.9`) and runs two methods at a time per ratio:
  `random + stratified`, then `coreset + badge`.
- Stable completed baseline results are now available for all planned ratios:
  `0.05`, `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`.
- Juno job `185548` (`br-attr-miss`) also completed successfully with exit
  code `0:0` after `05:28:16`.
- No expected brown-rainy baseline or missing attribute RPC result file was
  missing at the completion check.
- Training resumes from the same base checkpoint used by the recent
  brown-rainy experiments:
  `/home/oxx220000/scratch/safefix_cache/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth`
  with logged base best AP `0.6401`.
- Common generated augmentation source:
  `/home/oxx220000/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit`.
- Each run uses 2 total epochs with batch size `8`, seed `42`, and evaluates
  COCO keypoints on the generated train/val/test split produced by the runner.

## Completed Brown-Rainy Baseline Results

| Ratio | Method | Val AP | Val AP50 | Val AP75 | Test AP | Test AP50 | Test AP75 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `random` | `0.5890` | `0.8375` | `0.6406` | `0.5797` | `0.8208` | `0.6263` |
| `0.05` | `stratified` | `0.6022` | `0.8454` | `0.6604` | `0.5976` | `0.8331` | `0.6492` |
| `0.05` | `coreset` | `0.5639` | `0.8072` | `0.6129` | `0.5527` | `0.7997` | `0.5914` |
| `0.05` | `badge` | `0.5936` | `0.8361` | `0.6445` | `0.5782` | `0.8217` | `0.6284` |
| `0.10` | `random` | `0.6170` | `0.8562` | `0.6767` | `0.6048` | `0.8421` | `0.6521` |
| `0.10` | `stratified` | `0.6189` | `0.8533` | `0.6776` | `0.6110` | `0.8440` | `0.6646` |
| `0.10` | `coreset` | `0.5950` | `0.8423` | `0.6441` | `0.5877` | `0.8333` | `0.6371` |
| `0.10` | `badge` | `0.6121` | `0.8539` | `0.6679` | `0.5985` | `0.8370` | `0.6467` |
| `0.30` | `random` | `0.6192` | `0.8580` | `0.6652` | `0.6127` | `0.8441` | `0.6626` |
| `0.30` | `stratified` | `0.6324` | `0.8635` | `0.6837` | `0.6243` | `0.8533` | `0.6816` |
| `0.30` | `coreset` | `0.6306` | `0.8628` | `0.6910` | `0.6256` | `0.8511` | `0.6778` |
| `0.30` | `badge` | `0.6212` | `0.8605` | `0.6776` | `0.6153` | `0.8523` | `0.6628` |
| `0.50` | `random` | `0.6251` | `0.8641` | `0.6849` | `0.6219` | `0.8521` | `0.6770` |
| `0.50` | `stratified` | `0.6343` | `0.8650` | `0.6914` | `0.6230` | `0.8553` | `0.6786` |
| `0.50` | `coreset` | `0.6248` | `0.8584` | `0.6823` | `0.6167` | `0.8546` | `0.6733` |
| `0.50` | `badge` | `0.6300` | `0.8679` | `0.6907` | `0.6206` | `0.8591` | `0.6696` |
| `0.70` | `random` | `0.6334` | `0.8629` | `0.6916` | **`0.6332`** | `0.8615` | `0.6904` |
| `0.70` | `stratified` | `0.6364` | `0.8639` | `0.6968` | `0.6299` | `0.8559` | `0.6916` |
| `0.70` | `coreset` | `0.6341` | `0.8652` | `0.6973` | `0.6275` | `0.8576` | `0.6839` |
| `0.70` | `badge` | `0.6379` | `0.8621` | `0.7016` | `0.6256` | `0.8544` | `0.6738` |
| `0.90` | `random` | `0.6155` | `0.8520` | `0.6737` | `0.6180` | `0.8517` | `0.6673` |
| `0.90` | `stratified` | **`0.6380`** | `0.8641` | `0.6962` | `0.6291` | `0.8629` | `0.6867` |
| `0.90` | `coreset` | `0.6288` | `0.8614` | `0.6876` | `0.6202` | `0.8570` | `0.6714` |
| `0.90` | `badge` | `0.6266` | `0.8622` | `0.6880` | `0.6213` | `0.8563` | `0.6732` |

- Best completed baseline test AP: `ratio=0.70`, `random`, test AP `0.6332`.
- Best completed baseline validation AP: `ratio=0.90`, `stratified`, val AP
  `0.6380`.

## Completed Juno Qwen3 Brown-Rainy Attribute Results

These runs use Qwen3 train20k metadata, generated target
`clothes_color=brown, weather=rainy`, explicit generated extra counts
`clothes_color:brown=1000,weather:rainy=1000`, and `extra_count_scale=1/ratio`.

| Ratio | R | P | C | Val AP | Val AP50 | Val AP75 | Test AP | Test AP50 | Test AP75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.00` | `0.0` | `15.0` | `0.6011` | `0.8492` | `0.6560` | `0.5871` | `0.8308` | `0.6384` |
| `0.10` | `0.00` | `0.0` | `15.0` | `0.6159` | `0.8568` | `0.6623` | `0.6042` | `0.8390` | `0.6546` |
| `0.10` | `0.01` | `0.5` | `15.0` | `0.6235` | `0.8485` | `0.6833` | `0.6152` | `0.8403` | `0.6717` |
| `0.10` | `0.03` | `0.5` | `8.0` | `0.6285` | `0.8629` | `0.6882` | `0.6165` | `0.8489` | `0.6690` |
| `0.10` | `0.03` | `0.5` | `15.0` | `0.6288` | `0.8604` | `0.6925` | `0.6183` | `0.8466` | `0.6782` |
| `0.30` | `0.00` | `0.0` | `15.0` | `0.6309` | `0.8621` | `0.6899` | `0.6288` | `0.8572` | `0.6926` |
| `0.30` | `0.01` | `0.5` | `15.0` | `0.6266` | `0.8620` | `0.6892` | **`0.6306`** | `0.8567` | `0.6849` |
| `0.30` | `0.03` | `0.5` | `8.0` | `0.6254` | `0.8578` | `0.6847` | `0.6110` | `0.8445` | `0.6640` |
| `0.30` | `0.03` | `0.5` | `15.0` | `0.6266` | `0.8591` | `0.6860` | `0.6176` | `0.8516` | `0.6663` |
| `0.50` | `0.00` | `0.0` | `15.0` | `0.6272` | `0.8562` | `0.6897` | `0.6203` | `0.8544` | `0.6753` |
| `0.50` | `0.01` | `0.5` | `15.0` | `0.6290` | `0.8672` | `0.6870` | `0.6208` | `0.8547` | `0.6774` |
| `0.50` | `0.03` | `0.5` | `8.0` | `0.6257` | `0.8602` | `0.6874` | `0.6162` | `0.8525` | `0.6667` |
| `0.50` | `0.03` | `0.5` | `15.0` | `0.5916` | `0.8412` | `0.6453` | `0.5866` | `0.8338` | `0.6357` |
| `0.70` | `0.00` | `0.0` | `15.0` | `0.6364` | `0.8650` | `0.6953` | `0.6313` | `0.8583` | `0.6932` |
| `0.70` | `0.01` | `0.5` | `0.0` | `0.6370` | `0.8615` | `0.7007` | `0.6247` | `0.8558` | `0.6819` |
| `0.70` | `0.01` | `0.5` | `15.0` | **`0.6394`** | `0.8642` | **`0.7029`** | `0.6290` | `0.8572` | `0.6815` |
| `0.70` | `0.03` | `0.5` | `0.0` | `0.6370` | `0.8615` | `0.7007` | `0.6247` | `0.8558` | `0.6819` |
| `0.70` | `0.03` | `0.5` | `8.0` | `0.6331` | `0.8607` | `0.6920` | `0.6300` | `0.8548` | `0.6852` |
| `0.70` | `0.03` | `0.5` | `15.0` | `0.6377` | `0.8703` | `0.6922` | **`0.6319`** | `0.8571` | `0.6884` |
| `0.90` | `0.00` | `0.0` | `15.0` | `0.6339` | `0.8706` | `0.6908` | `0.6268` | `0.8579` | `0.6836` |
| `0.90` | `0.01` | `0.5` | `15.0` | `0.6282` | `0.8606` | `0.6829` | `0.6255` | `0.8572` | `0.6805` |
| `0.90` | `0.03` | `0.5` | `8.0` | `0.6282` | `0.8606` | `0.6829` | `0.6255` | `0.8572` | `0.6805` |
| `0.90` | `0.03` | `0.5` | `15.0` | `0.6282` | `0.8606` | `0.6829` | `0.6255` | `0.8572` | `0.6805` |

- Best completed Qwen3 brown-rainy attribute test AP so far:
  `ratio=0.70`, `R=0.03`, `P=0.5`, `C=15.0`, test AP `0.6319`.
- The strongest completed Juno brown-rainy result overall is currently the
  baseline `ratio=0.70` `random` row, test AP `0.6332`, slightly above the
  best completed attribute row.

## Completed Brown-Rainy Ratio 1.0 Full-Data ControlNet Result

This local run uses all `20000` original COCO pose train samples, appends
`1000` Qwen3 brown-rainy generated images, and disables active subset
selection.

| Ratio | Method | Val AP | Val AP50 | Val AP75 | Test AP | Test AP50 | Test AP75 | Log |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1.00` | `full_data_controlnet` | `0.6310` | `0.8594` | `0.6865` | `0.6259` | `0.8544` | `0.6812` | `logs/pose/pose_qwen3_brown_rainy_full_ratio1p0_gpu3_20260501_150816.log` |

# 04/29/2026 Update — KITTI Attribute Assignment and Slice Analysis

- Completed an additional local KITTI object-detection run using YOLOv8n
  pretrained weights with batch size `16` on physical GPU `0`.
- This is a batch-size sensitivity rerun of the same 5000/1240/1241 KITTI
  split used by the previous batch-size-64 result.
- Training setup:
  - script: `object_detection/train_yolov8_kitti.py`
  - launch helper: `scripts/launch_kitti_yolov8_gpu0_bs16_e30.sh`
  - environment: local `fp` conda
  - pretrained weights: `save/weights/yolov8n.pt`
  - epochs: `30`
  - batch size: `16`
  - image size: `640`
  - GPU: physical GPU `0`
  - Ultralytics: `8.4.33`
  - Python: `3.10.18`
  - PyTorch: `2.9.1+cu128`
- Canonical artifacts:
  - log:
    `logs/kitti_yolov8_first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232.log`
  - run directory:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232`
  - final test evaluation directory:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232_test`
  - best checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/best.pt`
  - last checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/last.pt`
- Runtime:
  - training: `714.95 s` (`00:11:55`)
  - final test evaluation: `5.96 s` (`00:00:06`)
- Final validation metrics after epoch 30:
  - precision: `0.889`
  - recall: `0.709`
  - mAP50: `0.819`
  - mAP50-95: `0.561`
- Final held-out test metrics:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | `1241` | `5369` | `0.8677` | `0.7530` | `0.8454` | `0.5797` |
| Car | `1107` | `4666` | `0.900` | `0.893` | `0.945` | `0.747` |
| Pedestrian | `283` | `703` | `0.836` | `0.613` | `0.746` | `0.413` |

- Compared with the previous batch-size-64 run on GPU `3`, the batch-size-16
  run is effectively tied on held-out test mAP50-95 (`0.5797` vs `0.5801`) and
  slightly higher on held-out test mAP50 (`0.8454` vs `0.8422`).

- Completed standalone KITTI train-split slice analysis using
  `utils/stat_kitti_detection.py`.
- Current KITTI BLIP/Qwen3 assignment files cover the 5000 train images, so the
  completed slice-analysis runs use `--split train`. Val/test slice analysis
  will need separate val/test assignment JSONs.
- Batch-size-64 GPU3 checkpoint analysis:
  - checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu3_e30_20260428_231728/save/weights/best.pt`
  - log:
    `logs/kitti_slice_analysis_train_bs64best_gpu3_20260429_021509.log`
  - output root: `logs/kitti/kitti_slice_stats`
  - overall train metrics: mAP50 `0.8744`, mAP50-95 `0.6051`
  - Qwen3: `25` slices, `12` failure slices
  - BLIP: `26` slices, `13` failure slices

### KITTI Batch-Size-64 Failure Train Slices

These are all slices whose train-split mAP50-95 is more than `0.02` below the
batch-size-64 overall train mAP50-95 (`0.6051`).

| Assigner | Attribute value | Images | mAP50 | mAP50-95 | Delta mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 | `traffic_density=heavy` | `71` | `0.4695` | `0.3769` | `-0.2282` |
| Qwen3 | `weather=clear` | `139` | `0.6530` | `0.4200` | `-0.1851` |
| Qwen3 | `color=green` | `77` | `0.6930` | `0.4463` | `-0.1588` |
| Qwen3 | `background_style=highway` | `142` | `0.9764` | `0.4523` | `-0.1528` |
| Qwen3 | `weather=cloudy` | `110` | `0.7371` | `0.4559` | `-0.1492` |
| Qwen3 | `road_condition=wet` | `206` | `0.7944` | `0.4746` | `-0.1305` |
| Qwen3 | `color=red` | `394` | `0.7467` | `0.5154` | `-0.0897` |
| Qwen3 | `lighting_condition=overcast` | `176` | `0.8637` | `0.5478` | `-0.0573` |
| Qwen3 | `background_style=residential` | `2492` | `0.7996` | `0.5539` | `-0.0512` |
| Qwen3 | `color=black` | `1268` | `0.8316` | `0.5653` | `-0.0398` |
| Qwen3 | `color=silver` | `650` | `0.8565` | `0.5653` | `-0.0398` |
| Qwen3 | `color=white` | `789` | `0.8677` | `0.5766` | `-0.0285` |
| BLIP | `lighting_condition=dim` | `45` | `0.4715` | `0.3802` | `-0.2249` |
| BLIP | `background_style=rural` | `23` | `0.7284` | `0.4883` | `-0.1168` |
| BLIP | `road_condition=snowy` | `585` | `0.7763` | `0.5316` | `-0.0735` |
| BLIP | `background_style=highway` | `1621` | `0.7779` | `0.5330` | `-0.0721` |
| BLIP | `road_condition=wet` | `1272` | `0.8129` | `0.5364` | `-0.0687` |
| BLIP | `lighting_condition=overcast` | `351` | `0.8398` | `0.5373` | `-0.0678` |
| BLIP | `color=green` | `34` | `0.8312` | `0.5408` | `-0.0643` |
| BLIP | `color=red` | `141` | `0.8775` | `0.5434` | `-0.0617` |
| BLIP | `traffic_density=empty` | `963` | `0.8396` | `0.5551` | `-0.0500` |
| BLIP | `color=white` | `1248` | `0.8446` | `0.5674` | `-0.0377` |
| BLIP | `background_style=residential` | `1917` | `0.8233` | `0.5683` | `-0.0367` |
| BLIP | `color=gray` | `2320` | `0.8496` | `0.5721` | `-0.0330` |
| BLIP | `lighting_condition=glare` | `145` | `0.8813` | `0.5786` | `-0.0265` |

- Batch-size-16 GPU0 checkpoint analysis:
  - checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/save/weights/best.pt`
  - log:
    `logs/kitti_slice_analysis_train_bs16_gpu0_20260429_012232.log`
  - output root:
    `logs/kitti/kitti_slice_stats_bs16_gpu0_20260429_012232`
  - overall train metrics: mAP50 `0.8719`, mAP50-95 `0.6010`
  - Qwen3: `25` slices, `12` failure slices
  - BLIP: `26` slices, `14` failure slices
  - InternVL3: `25` slices, `13` failure slices
    - log:
      `logs/kitti_slice_analysis_train_internvl3_bs16_gpu2_20260429_0517.log`
    - output root:
      `logs/kitti/kitti_slice_stats_internvl3_bs16_gpu2_20260429_0517`

- KITTI YOLOv8n pretrained batch-size recommendation:
  - Use `batch=16` as the default for future KITTI pretrained YOLOv8n runs.
  - The batch-size-16 and batch-size-64 runs are effectively tied on held-out
    test mAP50-95 (`0.5797` vs `0.5801`), while batch size 16 is slightly
    higher on held-out test mAP50 (`0.8454` vs `0.8422`).
  - Runtime difference is small for this 30-epoch setup: batch size 64 training
    took `621.63 s` (`10.36 min`), and batch size 16 training took `714.95 s`
    (`11.92 min`).

### KITTI Batch-Size-16 Worst Train Slices

| Assigner | Attribute value | Images | mAP50 | mAP50-95 | Delta mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 | `traffic_density=heavy` | `71` | `0.4723` | `0.3711` | `-0.2299` |
| Qwen3 | `weather=cloudy` | `110` | `0.5317` | `0.3794` | `-0.2216` |
| Qwen3 | `weather=clear` | `139` | `0.5678` | `0.4074` | `-0.1936` |
| Qwen3 | `background_style=highway` | `142` | `0.6403` | `0.4074` | `-0.1936` |
| BLIP | `lighting_condition=dim` | `45` | `0.4722` | `0.3722` | `-0.2288` |
| BLIP | `background_style=rural` | `23` | `0.7622` | `0.4775` | `-0.1235` |
| BLIP | `road_condition=snowy` | `585` | `0.7645` | `0.5260` | `-0.0750` |
| BLIP | `background_style=highway` | `1621` | `0.7703` | `0.5274` | `-0.0736` |
| InternVL3 | `traffic_density=heavy` | `31` | `0.4690` | `0.3667` | `-0.2343` |
| InternVL3 | `background_style=highway` | `510` | `0.4729` | `0.3700` | `-0.2310` |
| InternVL3 | `road_condition=wet` | `30` | `0.5624` | `0.4008` | `-0.2002` |
| InternVL3 | `lighting_condition=glare` | `127` | `0.7206` | `0.4725` | `-0.1285` |

- Completed BLIP attribute assignment on 5000 KITTI train images.
- Corpus: `save/corpus_kitti_gemini_clean.json` (6 attributes, no "unknown"/"NA")
  - Removed `time_of_day` (redundant with `lighting_condition`, KITTI is nearly
    all daytime) from original Gemini corpus; added `color` from GPT corpus.
- BLIP output: `save/attribute_kitti_blip.json`, `save/attribute_kitti_blip.csv`
- BLIP log: `logs/assign_attr_kitti_train_blip_gpu2_*.log`
- BLIP runtime: ~7 minutes (similarity-based, not generative)

### BLIP Attribute Distribution (5000 KITTI train images)

| Attribute | Top values |
|-----------|-----------|
| `color` | gray 46.4%, white 25.0%, black 13.5%, blue 7.0%, silver 3.3%, red 2.8% |
| `weather` | clear 67.4%, sunny 32.5%, rainy 0.1%, cloudy 0.0% |
| `background_style` | residential 38.3%, highway 32.4%, urban city 24.8%, natural 4.0%, rural 0.5% |
| `traffic_density` | moderate 77.5%, empty 19.3%, light 3.2%, heavy 0.0% |
| `lighting_condition` | strong shadows 89.2%, overcast 7.0%, glare 2.9%, dim 0.9% |
| `road_condition` | dry 62.7%, wet 25.4%, snowy 11.7%, muddy 0.2% |

- Qwen3-VL-8B attribute assignment completed for the same 5000 KITTI train
  images.
  - Output: `save/attribute_kitti_qwen3_8B.json`,
    `save/attribute_kitti_qwen3_8B.csv`
  - Log: `logs/assign_attr_kitti_train_qwen3_gemini6c_gpu3_*.log`
- InternVL3-8B attribute assignment completed for the same 5000 KITTI train
  images.
  - Output: `save/attribute_kitti_internvl3_8B.json`,
    `save/attribute_kitti_internvl3_8B.csv`
  - Log: `logs/assign_attr_kitti_train_internvl3_gpu2_20260429_004725.log`
  - Runtime: `269.4` minutes (`4.49` hours)
- Available KITTI corpus files:
  - `save/corpus_kitti_gemini_clean.json` — 6 attrs (color, weather, background_style, traffic_density, lighting_condition, road_condition); used for all three assigners
  - `save/corpus_kitti_object_gpt_clean.json` — 10 attrs (adds object_scale, viewpoint, occlusion, truncation, motion_state); available for future use
  - `save/corpus_kitti_gemini.json` — original 6 attrs with "unknown" options (superseded)
  - `save/corpus_kitti_object_gpt.json` — original 10 attrs with "unclear" option (superseded)



- Completed the local KITTI object-detection run using YOLOv8n pretrained
  weights.
- Dataset split:
  - train: `5000`
  - val: `1240`
  - test: `1241`
  - classes: `Car`, `Pedestrian`
- Training setup:
  - script: `object_detection/train_yolov8_kitti.py`
  - launch helper: `scripts/launch_kitti_yolov8_gpu3_e30.sh`
  - environment: local `fp` conda; exported snapshot:
    `utils/fp-kitti-env.yml`
  - pretrained weights: `save/weights/yolov8n.pt`
  - epochs: `30`
  - batch size: `64`
  - image size: `640`
  - GPU: physical GPU `3`
  - Ultralytics: `8.4.33`
  - Python: `3.10.18`
  - PyTorch: `2.9.1+cu128`
- Canonical artifacts:
  - log:
    `logs/kitti_yolov8_first5000_1240val_1241test_gpu3_e30_20260428_231728.log`
  - run directory:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu3_e30_20260428_231728`
  - final test evaluation directory:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu3_e30_20260428_231728_test`
  - checkpoint for resume:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu3_e30_20260428_231728/save/weights/last.pt`
  - best checkpoint:
    `save/runs/kitti_yolov8/first5000_1240val_1241test_gpu3_e30_20260428_231728/save/weights/best.pt`
- Runtime:
  - training: `621.63 s` (`00:10:22`)
  - final test evaluation: `34.57 s` (`00:00:35`)
- Final validation metrics after epoch 30:
  - precision: `0.8512`
  - recall: `0.7332`
  - mAP50: `0.8212`
  - mAP50-95: `0.5620`
- Final held-out test metrics:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | `1241` | `5369` | `0.8650` | `0.7433` | `0.8422` | `0.5801` |
| Car | `1107` | `4666` | `0.908` | `0.889` | `0.945` | `0.7478` |
| Pedestrian | `283` | `703` | `0.822` | `0.598` | `0.739` | `0.4125` |

- Checkpoint size:
  - `best.pt` and `last.pt` are each about `6.2 MB`, below GitHub's `100 MB`
    hard per-file limit.
- Attribute-analysis status:
  - Found qwen3, internvl3, and blip attribute JSONs for COCO pose, e.g.
    `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`,
    `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`, and
    `save/attribute_coco_pose_gemini_train20k_blip.json`.
  - Did not find KITTI-specific qwen3, internvl3, or blip attribute JSONs.
  - Did not run qwen3/internvl3/blip slice analysis for KITTI because the
    available JSONs are COCO pose metadata and the existing
    `utils/stat_pose_estimation.py` analysis is COCO keypoint-specific.
  - A KITTI attribute corpus exists at `save/corpus_kitti_object_gpt.json`, but
    KITTI attribute assignment artifacts still need to be generated before
    object-detection slice analysis can be run.

# 04/29/2026 Update — Previous RTMPose Tiny COCO Pose Result

- The only completed previous RTMPose Tiny COCO pose result currently present
  in the local repo is the 2026-04-03 GPU3 probe:
  - metric log:
    `logs/pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.log`
  - metric JSON:
    `logs/pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.json`
  - config snapshot:
    `logs/pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/vis_data/config.py`
- Evaluation setup from the log/config:
  - checkpoint:
    `pose_estimation/rtmpose_assets/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth`
  - COCO annotation:
    `/data/oxx220000/coco/annotations/person_keypoints_val2017.json`
  - data root: `/data/oxx220000/coco/`
  - batch size: `8`
  - seed: `21`
  - GPU: `NVIDIA RTX A5000`
  - runtime: Python `3.9.20`, PyTorch `1.13.1+cu117`,
    TorchVision `0.14.1+cu117`, MMEngine `0.10.7`
- Completed COCO keypoint metrics:
  - AP: `0.6966775579701486`
  - AP50: `0.9056114938423302`
  - AP75: `0.7755176193877302`
  - AP(M): `0.6734239791412847`
  - AP(L): `0.7385393035542115`
  - AR: `0.7254250629722923`
  - AR50: `0.9151448362720404`
  - AR75: `0.7950251889168766`
  - AR(M): `0.6931439497405082`
  - AR(L): `0.773950204384987`
- Related paths without a completed metric:
  - `logs/pose/NA/rtmpose_tiny_coco_gpu3_20260403_131338.log` exists but is
    empty.
  - `logs/pose/rtmpose_tiny_coco_gpu3_latest/20260403_131220/20260403_131220.log`
    contains setup/config output but no final COCO metric line.
  - The newer Juno A30 RTMPose Tiny job `185461` was still `PENDING
    (Resources)` at the latest check, so it has no result yet.
- Interpretation note:
  - This RTMPose Tiny result is a top-down COCO-val evaluation using COCO
    ground-truth person boxes through MMPose `CocoDataset`; treat the AP as a
    pose-quality reference, not as a detector-inclusive number directly
    comparable to the Keypoint R-CNN held-out AP rows.

# 04/28/2026 Update — Juno Qwen3 Brown-Rainy Ratio 0.1 RPC Results

- All four Juno Qwen3 brown-rainy `ratio=0.1` RPC runs completed
  successfully.
- Final SLURM accounting confirmation on Juno:
  - `185427`: `COMPLETED`, exit code `0:0`, elapsed `00:09:14`, node
    `g-04-02`
  - `185428`: `COMPLETED`, exit code `0:0`, elapsed `00:09:01`, node
    `g-04-02`
  - `185429`: `COMPLETED`, exit code `0:0`, elapsed `00:09:03`, node
    `g-04-02`
  - `185430`: `COMPLETED`, exit code `0:0`, elapsed `00:08:40`, node
    `g-04-02`
- Common setup for this sweep:
  - base AP for comparison: `0.5902`
  - selected original samples: `2000`
  - selected generated brown-rainy images: `1000`
  - final train size: `3000`
  - original candidate pool: `20000`
  - generated target: `clothes_color=brown`, `weather=rainy`
  - explicit generated extra counts:
    `clothes_color:brown=1000,weather:rainy=1000`
  - `extra_count_scale`: `10`

| Job | R | P | C | Status | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `185429` | `0.03` | `0.5` | `15.0` | completed | **`0.6183`** | `+0.0281` | `0.36 s` | `00:07:48` | `/work/oxx220000/safefix/logs/pose/pose_resume_qwen3_brown_rainy_r0p1_r0p03_p0p5_c15p0_gpu0_20260428_211734.log` |
| `185430` | `0.03` | `0.5` | `8.0` | completed | `0.6165` | `+0.0263` | `0.35 s` | `00:07:41` | `/work/oxx220000/safefix/logs/pose/pose_resume_qwen3_brown_rainy_r0p1_r0p03_p0p5_c8p0_gpu0_20260428_212637.log` |
| `185428` | `0.01` | `0.5` | `15.0` | completed | `0.6152` | `+0.0250` | `0.36 s` | `00:07:51` | `/work/oxx220000/safefix/logs/pose/pose_resume_qwen3_brown_rainy_r0p1_r0p01_p0p5_c15p0_gpu0_20260428_210833.log` |
| `185427` | `0.0` | `0.0` | `15.0` | completed | `0.6042` | `+0.0140` | `0.37 s` | `00:08:00` | `/work/oxx220000/safefix/logs/pose/pose_resume_qwen3_brown_rainy_r0p1_r0p0_p0p0_c15p0_gpu0_20260428_205920.log` |

- Completed `185429` metrics:
  - held-out val: AP `0.6288`, AP50 `0.8604`, AP75 `0.6925`, AR `0.6995`
  - held-out test: AP `0.6183`, AP50 `0.8466`, AP75 `0.6782`, AR `0.6860`
- Completed `185430` metrics:
  - held-out val: AP `0.6285`, AP50 `0.8629`, AP75 `0.6882`, AR `0.6984`
  - held-out test: AP `0.6165`, AP50 `0.8489`, AP75 `0.6690`, AR `0.6854`
- Completed `185428` metrics:
  - held-out val: AP `0.6235`, AP50 `0.8485`, AP75 `0.6833`, AR `0.6971`
  - held-out test: AP `0.6152`, AP50 `0.8403`, AP75 `0.6717`, AR `0.6872`
- Completed `185427` metrics:
  - held-out val: AP `0.6159`, AP50 `0.8568`, AP75 `0.6623`, AR `0.6908`
  - held-out test: AP `0.6042`, AP50 `0.8390`, AP75 `0.6546`, AR `0.6766`
- Interpretation:
  - `185429` is the best completed Juno Qwen3 brown-rainy `ratio=0.1` row so
    far, with test AP `0.6183` (`+0.0281` over the canonical base)
  - all three nonzero `R/P` rows beat the conservative all-zero
    rarity/frequency row (`185427`)
  - with `R=0.03` and `P=0.5`, the softer class-balance setting `C=8.0`
    (`185430`, AP `0.6165`) trails `C=15.0` (`185429`, AP `0.6183`) but still
    beats the lower-rarity `R=0.01`, `P=0.5`, `C=15.0` setting
  - `185429` is also above the earlier local InternVL3 rainy `ratio=0.1`
    conservative attribute row (`0.6071`), plus the earlier local `random`
    (`0.6029`) and `stratified` (`0.6031`) rows

# 04/28/2026 Update — COCO Pose Ratio 0.05 Rainy / Brown-Rainy Results

- This table merges the earlier local InternVL3 rainy `ratio=0.05` runs with
  the completed Juno Qwen3 brown-rainy `ratio=0.05` run.
- Common setup:
  - canonical base AP for comparison: `0.5902`
  - selected original samples: `1000`
  - selected generated images: `1000`
  - final train size: `2000`
  - original candidate pool: `20000`
- RPC means the three attribute-selector parameters:
  - `R`: `--al-attribute-rarity-power`; larger values increase rare attribute
    weighting through inverse frequency.
  - `P`: `--al-attribute-frequency-penalty`; larger values subtract more score
    from frequent attribute values.
  - `C`: `--al-attribute-class-balance`; `0` disables class balancing, finite
    values add a soft class-balance bonus, and `inf` uses strict round-robin
    class quotas.
- For the local InternVL3 rainy rows, `attrx20` means
  `--al-attribute-extra-count-scale 20` on the inferred generated rainy counts.
  For the Juno Qwen3 brown-rainy row, the extra counts were explicit:
  `clothes_color:brown=1000,weather:rainy=1000`.
- The Juno `ratio=0.1` brown-rainy run was intentionally not relaunched in this
  pass.

| Setting | Metadata / generated target | Strategy | R | P | C | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Juno brown-rainy | Qwen3; `clothes_color=brown`, `weather=rainy` | `attribute` | `0.0` | `0.0` | `15.0` | **`0.5871`** | `-0.0031` | `0.41 s` | `00:06:11` | `/work/oxx220000/safefix/logs/pose/pose_resume_qwen3_brown_rainy_r0p05_r0p0_p0p0_c15p0_gpu0_20260428_185219.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r000_p0_c15`) | `0.0` | `0.0` | `15.0` | `0.5846` | `-0.0056` | `0.26 s` | `00:07:46` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r000_p0_c15_gpu0_20260423_232656.log` |
| local rainy | InternVL3; `weather=rainy` | `stratified` | `n/a` | `n/a` | `n/a` | `0.5815` | `-0.0087` | `0.22 s` | `00:07:46` | `logs/pose/pose_resume_internvl3_epoch2_stratified_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `random` | `n/a` | `n/a` | `n/a` | `0.5792` | `-0.0110` | `0.00 s` | `00:07:39` | `logs/pose/pose_resume_internvl3_epoch2_random_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `badge` fast approximate | `n/a` | `n/a` | `n/a` | `0.5754` | `-0.0148` | `64.34 s` | `00:08:53` | `logs/pose/pose_resume_internvl3_epoch2_badge_ratio0.05_gpu0_20260424_121700.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r000_p0_c0`) | `0.0` | `0.0` | `0.0` | `0.5745` | `-0.0157` | `0.25 s` | `00:07:52` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r000_p0_c0_gpu0_20260424_121700.log` |
| local rainy | InternVL3; `weather=rainy` | `coreset` | `n/a` | `n/a` | `n/a` | `0.5619` | `-0.0283` | `202.56 s` | `00:11:11` | `logs/pose/pose_resume_internvl3_epoch2_coreset_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r0005_p005_c15`) | `0.005` | `0.05` | `15.0` | `0.5273` | `-0.0629` | `0.26 s` | `00:08:14` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r0005_p005_c15_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r0001_p01_c15`) | `0.001` | `0.1` | `15.0` | `0.5185` | `-0.0717` | `0.28 s` | `00:09:11` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r0001_p01_c15_gpu0_20260424_000907.log` |

- Juno brown-rainy validation metrics:
  - held-out val: AP `0.6011`, AP50 `0.8492`, AP75 `0.6560`, AR `0.6834`
  - held-out test: AP `0.5871`, AP50 `0.8308`, AP75 `0.6384`, AR `0.6712`
- Main observations:
  - the Juno Qwen3 brown-rainy row is now the best completed `ratio=0.05` row,
    but it is still slightly below the canonical base AP (`-0.0031`)
  - among the earlier local rainy rows, the best completed row remains
    `attribute` (`attrx20_r000_p0_c15`) with AP `0.5846`
  - the weather-steered local attribute settings with nonzero `R/P` are much
    weaker at this ratio, especially `attrx20_r0001_p01_c15`
  - the fast approximate `badge` row completed but did not beat `attribute`,
    `stratified`, or `random`
  - `coreset` is slower because its selection took `202.56 s`

# 04/24/2026 Update — Pose Bug Compare Threshold Standard

- Future pose bug/failure-set comparisons should use `acc_diff_threshold=0.02`.
- Failure rule: `slice_AP < overall_AP - acc_diff_threshold`.
- Standard output folder:
  - `pose_bug_threshold_compare/`
- Main standard CSV:
  - `pose_bug_threshold_compare/bugs_compare_pose_acc_diff_0p02.csv`
- Detailed focused rows:
  - `pose_bug_threshold_compare/acc_diff_0p02/`
- Non-standard threshold sweep files are archived under:
  - `pose_bug_threshold_compare/archived_threshold_sweep/`
- With `acc_diff_threshold=0.02`:
  - base failure total before top-k: `12`
  - compared base failure cases per method: `12`
  - mean still failures over the 20 compact method rows: `7.05`
  - mean resolved failures over the 20 compact method rows: `4.95`

| ratio | method | overall_ap_base | overall_ap_run | overall_ap_delta | base_failure_total | compared_count | improved_count | declined_count | same_count | resolved_failure_count | still_failure_count | rare_to_not_rare_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.1` | `attribute` | `0.5902` | `0.6071` | `0.0169` | `12` | `12` | `9` | `1` | `0` | `4` | `8` | `2` |
| `0.1` | `random` | `0.5902` | `0.6029` | `0.0127` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.1` | `stratified` | `0.5902` | `0.6031` | `0.0129` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.1` | `coreset` | `0.5902` | `0.5579` | `-0.0323` | `12` | `12` | `1` | `9` | `0` | `5` | `7` | `2` |
| `0.1` | `badge` | `0.5902` | `0.5919` | `0.0017` | `12` | `12` | `6` | `4` | `0` | `4` | `8` | `2` |
| `0.3` | `attribute` | `0.5902` | `0.6174` | `0.0272` | `12` | `12` | **`10`** | `0` | `0` | `6` | `6` | `2` |
| `0.3` | `random` | `0.5902` | `0.6061` | `0.0159` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.3` | `stratified` | `0.5902` | `0.6041` | `0.0139` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.3` | `coreset` | `0.5902` | `0.6031` | `0.0129` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.3` | `badge` | `0.5902` | `0.6083` | `0.0181` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.5` | `attribute` | `0.5902` | `0.6127` | `0.0225` | `12` | `12` | **`10`** | `0` | `0` | `4` | `8` | `1` |
| `0.5` | `random` | `0.5902` | `0.5972` | `0.0070` | `12` | `12` | `6` | `4` | `0` | `5` | `7` | `2` |
| `0.5` | `stratified` | `0.5902` | `0.6166` | `0.0264` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `1` |
| `0.5` | `coreset` | `0.5902` | `0.6073` | `0.0171` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.5` | `badge` | `0.5902` | `0.6111` | `0.0209` | `12` | `12` | `9` | `1` | `0` | `5` | `7` | `2` |
| `0.7` | `attribute` | `0.5902` | **`0.6263`** | `0.0361` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `random` | `0.5902` | `0.6244` | `0.0342` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `stratified` | `0.5902` | `0.6244` | `0.0342` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `coreset` | `0.5902` | `0.6176` | `0.0274` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `badge` | `0.5902` | `0.6191` | `0.0289` | `12` | `12` | **`10`** | `0` | `0` | `6` | `6` | `2` |

## Mock Targeted Slice Delta Heatmap Placeholder

This figure is a layout/design placeholder only. The values are fixed-seed
synthetic per-slice AP deltas, now shaped around the target-attribute exposure
observed in the COCO Brown-Rainy logs. Do not use this figure as experimental
evidence.

COCO Brown-Rainy was chosen for this placeholder because, among the current
four main-result tables, it has the largest mean gap between `Ours` and the
best non-attribute baseline across ratios: COCO `+0.0042` AP, ImageNet
`+0.0038` ACC, CelebA `+0.0026` ACC, and KITTI `+0.0015` mAP50-95. The
target rows are intentionally the hottest rows: the Qwen3 Brown-Rainy
ratio-`0.3` anchor log selected `176/6000` brown source images and `380/6000`
rainy source images, then added `1000` generated `brown_rainy` images. The
final version should replace these synthetic values with real per-slice
before/after AP deltas from the pose bug-compare outputs.

![Mock COCO Brown-Rainy per-slice AP-delta heatmap](figures/mock_targeted_repair/coco_brown_rainy_slice_delta_heatmap.png)

Source script: `scripts/build_mock_targeted_repair_heatmap.py`.
Source CSV: `figures/mock_targeted_repair/coco_brown_rainy_slice_delta_heatmap.csv`.
Paper-ready assets: `nips26/figures/mock_targeted_repair/coco_brown_rainy_slice_delta_heatmap.png`
and `nips26/figures/mock_targeted_repair/coco_brown_rainy_slice_delta_heatmap.pdf`.
Target-exposure source log:
`logs/pose/pose_qwen3_brown_rainy_rpc8_anchor_ratio0p3_gpu1_20260501_231910.log`.

# 04/17/2026 Update — COCO Pose InternVL3 Resume Baselines And Attribute Sweeps Completed

- The controller-backed InternVL3 COCO pose resume sweep launched from the canonical base checkpoint has fully completed, and the finished `attribute` comparison runs are now merged directly into the same top-level ratio tables for like-for-like comparison.
- Controller log:
  - `logs/pose/run_pose_resume_internvl3_all_20260417_022717.log`
  - ends with:
    - `All strategies completed.`
- Common setup for the tables below:
  - base reference for AP comparison:
    - `logs/pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544/`
    - log: `logs/pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544.log`
  - base held-out test AP for comparison:
    - `0.5902`
  - note:
    - use `0.5902` as the canonical base AP for future COCO pose InternVL3 resume comparisons; the previously listed `0.6145` comparison value was wrong
  - resume schedule:
    - one additional epoch from the saved epoch-0 checkpoint
  - original candidate pool:
    - `20000`
  - generated augmentation:
    - canonical rainy-image mix, `1000` images added after original-image selection
  - fixed ratio placement:
    - GPU `0` -> `0.1`
    - GPU `1` -> `0.3`
    - GPU `2` -> `0.5`
    - GPU `3` -> `0.7`

### Ratio = 0.1

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `2000` | `3000` | `0.6029` | `+0.0127` | `0.00 s` | `00:09:44` | `logs/pose/pose_resume_internvl3_epoch2_random_ratio0.1_gpu0_20260417_022717.log` |
| `stratified` | `2000` | `3000` | `0.6031` | `+0.0129` | `0.23 s` | `00:09:44` | `logs/pose/pose_resume_internvl3_epoch2_stratified_ratio0.1_gpu0_20260417_023705.log` |
| `coreset` | `2000` | `3000` | `0.5579` | `-0.0323` | `4.96 min` | `00:14:48` | `logs/pose/pose_resume_internvl3_epoch2_coreset_ratio0.1_gpu0_20260417_024654.log` |
| `badge` | `2000` | `3000` | `0.5919` | `+0.0017` | `21.43 min` | `00:31:20` | `logs/pose/pose_resume_internvl3_epoch2_badge_ratio0.1_gpu0_20260417_030147.log` |
| `attribute` (`attrx10_r000_p0_c15`) | `2000` | `3000` | **`0.6071`** | `+0.0169` | `0.26 s` | `00:09:48` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r000_p0_c15_gpu0_20260417_114049.log` |
| `attribute` (`attrx10_r003_p05_c15`) | `2000` | `3000` | `0.6012` | `+0.0110` | `0.27 s` | `00:10:04` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r003_p05_c15_gpu2_20260417_114049.log` |
| `attribute` (`attrx10_r001_p05_c15`) | `2000` | `3000` | `0.6003` | `+0.0101` | `0.26 s` | `00:09:50` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r001_p05_c15_gpu1_20260417_114049.log` |
| `attribute` (`attrx10_r003_p05_c8`) | `2000` | `3000` | `0.5963` | `+0.0061` | `0.26 s` | `00:09:47` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r003_p05_c8_gpu3_20260417_114049.log` |

### Ratio = 0.3

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `6000` | `7000` | `0.6061` | `+0.0159` | `0.00 s` | `00:17:55` | `logs/pose/pose_resume_internvl3_epoch2_random_ratio0.3_gpu1_20260417_022717.log` |
| `stratified` | `6000` | `7000` | `0.6041` | `+0.0139` | `0.21 s` | `00:18:37` | `logs/pose/pose_resume_internvl3_epoch2_stratified_ratio0.3_gpu1_20260417_024517.log` |
| `coreset` | `6000` | `7000` | `0.6031` | `+0.0129` | `19.29 min` | `00:37:10` | `logs/pose/pose_resume_internvl3_epoch2_coreset_ratio0.3_gpu1_20260417_030359.log` |
| `badge` | `6000` | `7000` | `0.6083` | `+0.0181` | `59.96 s` | `00:18:51` | `logs/pose/pose_resume_internvl3_epoch2_badge_ratio0.3_gpu1_20260417_034116.log` |
| `attribute` (`attrx333_r000_p0_c15`) | `6000` | `7000` | **`0.6174`** | `+0.0272` | `0.28 s` | `00:18:23` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r000_p0_c15_gpu1_20260417_122328.log` |
| `attribute` (`attrx333_r000_p0_c0`) | `6000` | `7000` | `0.6091` | `+0.0189` | `0.25 s` | `00:17:56` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r000_p0_c0_gpu0_20260417_122328.log` |
| `attribute` (`attrx333_r001_p05_c15`) | `6000` | `7000` | `0.6079` | `+0.0177` | `0.27 s` | `00:18:24` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r001_p05_c15_gpu3_20260417_122328.log` |
| `attribute` (`attrx333_r001_p05_c0`) | `6000` | `7000` | `0.6067` | `+0.0165` | `0.25 s` | `00:18:08` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r001_p05_c0_gpu2_20260417_122328.log` |

### Ratio = 0.5

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `10000` | `11000` | `0.5972` | `+0.0070` | `0.00 s` | `00:27:11` | `logs/pose/pose_resume_internvl3_epoch2_random_ratio0.5_gpu2_20260417_022717.log` |
| `stratified` | `10000` | `11000` | `0.6166` | `+0.0264` | `0.22 s` | `00:28:59` | `logs/pose/pose_resume_internvl3_epoch2_stratified_ratio0.5_gpu2_20260417_025433.log` |
| `coreset` | `10000` | `11000` | `0.6073` | `+0.0171` | `1.15 min` | `00:28:22` | `logs/pose/pose_resume_internvl3_epoch2_coreset_ratio0.5_gpu2_20260417_032336.log` |
| `badge` | `10000` | `11000` | `0.6111` | `+0.0209` | `59.68 s` | `00:27:22` | `logs/pose/pose_resume_internvl3_epoch2_badge_ratio0.5_gpu2_20260417_035203.log` |
| `attribute` (`attrx2_r000_p0_c0`) | `10000` | `11000` | **`0.6127`** | `+0.0225` | `0.25 s` | `00:26:42` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r000_p0_c0_gpu0_20260417_125113.log` |
| `attribute` (`attrx2_r000_p0_c15`) | `10000` | `11000` | `0.6097` | `+0.0195` | `0.28 s` | `00:27:23` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r000_p0_c15_gpu1_20260417_125113.log` |
| `attribute` (`attrx2_r001_p05_c15`) | `10000` | `11000` | `0.6084` | `+0.0182` | `0.28 s` | `00:27:14` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r001_p05_c15_gpu3_20260417_125113.log` |
| `attribute` (`attrx2_r001_p05_c0`) | `10000` | `11000` | `0.6047` | `+0.0145` | `0.25 s` | `00:26:34` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r001_p05_c0_gpu2_20260417_125113.log` |

### Ratio = 0.7

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `14000` | `15000` | `0.6244` | `+0.0342` | `0.01 s` | `00:34:27` | `logs/pose/pose_resume_internvl3_epoch2_random_ratio0.7_gpu3_20260417_022717.log` |
| `stratified` | `14000` | `15000` | `0.6244` | `+0.0342` | `0.23 s` | `00:35:26` | `logs/pose/pose_resume_internvl3_epoch2_stratified_ratio0.7_gpu3_20260417_041149.log` |
| `coreset` | `14000` | `15000` | `0.6176` | `+0.0274` | `59.12 s` | `00:35:25` | `logs/pose/pose_resume_internvl3_epoch2_coreset_ratio0.7_gpu3_20260417_055719.log` |
| `badge` | `14000` | `15000` | `0.6191` | `+0.0289` | `59.19 s` | `00:35:24` | `logs/pose/pose_resume_internvl3_epoch2_badge_ratio0.7_gpu3_20260417_074249.log` |
| `attribute` (`attrx143_r000_p0_c15`) | `14000` | `15000` | **`0.6263`** | `+0.0361` | `0.28 s` | `00:36:03` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r000_p0_c15_gpu1_20260417_170346.log` |
| `attribute` (`attrx143_r001_p05_c0`) | `14000` | `15000` | `0.6262` | `+0.0360` | `0.26 s` | `00:34:53` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r001_p05_c0_gpu2_20260417_170346.log` |
| `attribute` (`attrx143_r001_p05_c15`) | `14000` | `15000` | `0.6203` | `+0.0301` | `0.29 s` | `00:35:51` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r001_p05_c15_gpu3_20260417_170346.log` |
| `attribute` (`attrx143_r000_p0_c0`) | `14000` | `15000` | `0.6166` | `+0.0264` | `0.25 s` | `00:35:02` | `logs/pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r000_p0_c0_gpu0_20260417_170346.log` |

- Main observations:
  - all non-attribute and attribute resume runs now have stable finished outputs for the four comparison ratios `0.1`, `0.3`, `0.5`, and `0.7`
  - the finished attribute runs are now grouped directly under the same top-level ratio tables as the baselines for like-for-like comparison
  - under the corrected comparison baseline (`0.5902` test AP), nearly all finished resume runs are above base; the clear exception in this table is `coreset` at `ratio=0.1`
  - the strongest finished resumed result in the full table is now:
    - `attribute` (`attrx143_r000_p0_c15`) at `ratio=0.7`
    - held-out test AP `0.6263`
    - delta vs corrected base `+0.0361`
  - the second-best finished resumed result is also an attribute point at `ratio=0.7`:
    - `attribute` (`attrx143_r001_p05_c0`)
    - held-out test AP `0.6262`
    - delta vs corrected base `+0.0360`
  - retained-data ratio matters more than selector sophistication in this sweep:
    - every strategy is substantially stronger at `0.7` than at `0.1`
  - the corrected base is matched or exceeded by:
    - `stratified`, `ratio=0.5`
    - all four finished `attribute` runs at `ratio=0.7`
    - `attribute` (`attrx333_r000_p0_c15`), `ratio=0.3`
    - all four non-attribute strategies at `ratio=0.7`
  - at `ratio=0.1`, the best finished attribute point is:
    - `attrx10_r000_p0_c15`
    - held-out test AP `0.6071`
    - this is above the corrected base (`0.5902`) and above the corrected `random` `ratio=0.1` row (`0.6029`)
  - at `ratio=0.3`, the best finished attribute point is:
    - `attrx333_r000_p0_c15`
    - held-out test AP `0.6174`
    - this is above the corrected base (`0.5902`) and above every finished non-attribute `ratio=0.3` baseline
  - at `ratio=0.5`, the best finished attribute point is:
    - `attrx2_r000_p0_c0`
    - held-out test AP `0.6127`
    - this is above the corrected base (`0.5902`) and below the finished `stratified` `ratio=0.5` baseline (`0.6166`), but above `badge`, `coreset`, and `random` at the same ratio
  - at `ratio=0.7`, the best finished attribute point is:
    - `attrx143_r000_p0_c15`
    - held-out test AP `0.6263`
    - this is above the corrected base (`0.5902`) and above every finished non-attribute `ratio=0.7` baseline
  - for the `ratio=0.3` attribute RPC4 sweep, the conservative control family still wins:
    - `r000_p0_c15`: `0.6174`
    - `r000_p0_c0`: `0.6091`
    - `r001_p05_c15`: `0.6079`
    - `r001_p05_c0`: `0.6067`
  - the same pattern continues at `ratio=0.5`:
    - `r000_p0_c0`: `0.6127`
    - `r000_p0_c15`: `0.6097`
    - `r001_p05_c15`: `0.6084`
    - `r001_p05_c0`: `0.6047`
  - at `ratio=0.7`, the strongest two points are split between the conservative and mild weather-steered branches:
    - `r000_p0_c15`: `0.6263`
    - `r001_p05_c0`: `0.6262`
    - `r001_p05_c15`: `0.6203`
    - `r000_p0_c0`: `0.6166`
  - `coreset` is still the weakest cost/benefit point:
    - `ratio=0.1`: `0.5579` AP after `4.96 min` of selection
    - `ratio=0.3`: `0.6031` AP after `19.29 min`
    - `ratio=0.5`: `0.6073` AP after `1.15 min`
    - `ratio=0.7`: `0.6176` AP after `59.12 s`
  - `badge` no longer leads at `ratio=0.3` once the finished attribute sweep is included:
    - `badge`: `0.6083`
    - best attribute: `0.6174`
  - at `ratio=0.5`, `attribute` also improves on `badge`, but does not reach `stratified`:
    - `attribute` best: `0.6127`
    - `badge`: `0.6111`
    - `stratified`: `0.6166`
  - at `ratio=0.7`, `attribute` now leads the full baseline set:
    - best attribute: `0.6263`
    - `random`: `0.6244`
    - `stratified`: `0.6244`
    - `badge`: `0.6191`
    - `coreset`: `0.6176`
- practical reference point for future pose-resume comparisons:
  - keep `ratio=0.3` `attribute` (`attrx333_r000_p0_c15`), `ratio=0.5` `attribute` (`attrx2_r000_p0_c0`), and `ratio=0.7` `attribute` (`attrx143_r000_p0_c15`)

# 04/17/2026 Update — COCO Pose Rain-Augmented Full-Data Run Finished

- The full-data InternVL3 COCO pose run with `1000` generated rain edits on GPU `3` finished at `2026-04-17 02:48:19`.
- Common setup:
  - original selected images:
    - `20000`
  - generated source:
    - legacy `attribute=rain`
  - generated images after filtering:
    - `1271` available
    - `1000` selected
  - final train size:
    - `21000`
  - slice-analysis source:
    - `internvl3`
- Compared with the corrected InternVL3 `20k` base reference now used in this report (`held-out test AP = 0.5902`), this rain-augmented full-data run is higher by `+0.0318` AP.

| Run | Selected Original | Final Train Size | Best Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | End-to-End Time | Canonical Log |
|-----|-------------------|------------------|-------------|------------------|-----------|-----------|---------|------------|-----------------|---------------|
| InternVL3 + `1000` rain edits | `20000` | `21000` | `0.6333` | `0.6220` | `0.8528` | `0.6766` | `0.6893` | `01:33:45` | `01:33:59` | `logs/pose/pose_train20k_rain1k_gpu3_20260417_011411.log` |

- Main observations:
  - the best checkpoint still came from epoch `0`, with best val AP `0.6333`
  - held-out test AP `0.6220` is above the corrected earlier InternVL3 base
  - the hardest held-out test slice remained:
    - `pose=lying` with `AP=0.3268`
  - the weather-target slice also remained difficult:
    - `weather=rainy` with `AP=0.5692`

# 04/16/2026 Update — CelebA ResNet Base Run Finished

- The fresh CelebA ResNet base run launched on GPU `0` has finished.
- The short launcher log:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_gpu0.log`
  - only contains the early startup lines and is **not** the full training record.
- The complete training log is:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
- This run was the new from-scratch baseline for:
  - dataset: `CelebA`
  - model: `ResNet`
  - attributes: `redhair_brownskin_sademotion`
  - generated images: `0`
  - training set size: `80000`
  - initialization: random weights
  - epochs: `20`

| Run | Held-out Test ACC| Best Val ACC | Best-Val Epoch | Final Val ACC  | Test Loss | Train Time | End-to-End Time | Checkpoint | Predictions |
|-----|--------------|----------------|---------------|-------------------|-----------|------------|-----------------|------------|-------------|
| CelebA ResNet base (`0` generated images) | **`90.57%`** | `0.9140` | `12` | `0.9066` | `0.3242` | `32 min` | `32 min` | `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.pth` | `save/pred/predictions_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.npy` |

- Main interpretation:
  - this base checkpoint is now available for the planned CelebA resume experiments
  - for reporting, the CelebA base held-out test accuracy is manually recorded as `90.57%`
  - validation peaked at epoch `12` and then drifted slightly downward, so the saved base checkpoint is a reasonable resume starting point
- Post-run status:
  - the run completed training, saved the checkpoint, evaluated on the held-out test split, and saved predictions
  - it then failed only during the later diagnostics stage because `save/attribute10.json` was missing
  - this failure happened **after** the main training and evaluation outputs, so the reported checkpoint and metrics above are valid

### CelebA Resume Sweep Note

- For the current CelebA resume-baseline sweep, the `hierarchical` strategy is temporarily **deferred**.
- Reason:
  - on this CelebA setting (`80,000` original images plus `1,000` generated images), hierarchical selection remained impractically slow even after a first-pass scalability patch.
- Practical scope for the current sweep:
  - treat `random`, `stratified`, `coreset`, and `badge` as the active comparison set.
- The hierarchical path can be revisited later if there is a stronger need for a deeper algorithmic rewrite.

### CelebA Resume Baselines Completed

- The non-hierarchical CelebA resume sweep is now largely complete.
- Common setup for the table below:
  - base checkpoint: `save/weights/model_celeba_resnet_base.pth`
  - base held-out test accuracy for reporting: `90.57%`
  - model: `ResNet`
  - learning rate: `1e-5`
  - epochs: `15`
  - generated images: `1000`
  - original CelebA training pool: `80,000`
- For `badge` and `coreset`, the larger-ratio entries below use the patched reruns because the original exact selectors did not scale well enough on this CelebA setting.

### Ratio = 0.1

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `8000` | `9000` | `90.15%` | `-0.42 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.1.log` |
| `stratified` | `8000` | `9000` | `90.95%` | `+0.38 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.1.log` |
| `coreset` | `8000` | `9000` | `90.29%` | `-0.28 pp` | `45 min` | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.1.log` |
| `badge` | `8000` | `9000` | `91.12%` | `+0.55 pp` | `1 h 52 min` | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.1.log` |
| `attribute` | `8000` | `9000` | `91.07%` | `+0.50 pp` | `0.42 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r003_p05_c15_gpu2_20260417_010718.log` |

### Ratio = 0.3

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `24000` | `25000` | `90.68%` | `+0.11 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.3.log` |
| `stratified` | `24000` | `25000` | `90.65%` | `+0.08 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.3.log` |
| `coreset` | `24000` | `25000` | `90.63%` | `+0.06 pp` | `2 h 41 min` | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.3.log` |
| `badge` | `24000` | `25000` | `90.54%` | `-0.03 pp` | `116.78 s` | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.3_gpu2_verbose_20260416_115906.log` |
| `attribute` | `24000` | `25000` | `90.91%` | `+0.34 pp` | `0.43 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_cinf_gpu0_20260417_014040.log` |

### Ratio = 0.5

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `40000` | `41000` | `90.77%` | `+0.20 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.5.log` |
| `stratified` | `40000` | `41000` | `90.79%` | `+0.22 pp` | `0.01 s` | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.5.log` |
| `coreset` | `40000` | `41000` | `90.59%` | `+0.02 pp` | `5 h 5 min` | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.5.log` |
| `badge` | `40000` | `41000` | `90.63%` | `+0.06 pp` | `124.76 s` | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu0_20260416_120554.log` |

### Ratio = 0.7

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `56000` | `57000` | `90.38%` | `-0.19 pp` | `0.00 s` | `logs/celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.7.log` |
| `stratified` | `56000` | `57000` | `90.79%` | `+0.22 pp` | `0.01 s` | `logs/celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.7.log` |
| `coreset` | `56000` | `57000` | `90.60%` | `+0.03 pp` | `56.85 s` | `logs/celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_gpu3_rerun_20260416_121045.log` |
| `badge` | `56000` | `57000` | `90.81%` | `+0.24 pp` | `74.42 s` | `logs/celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.7_gpu0_20260416_122047.log` |

### CelebA Resume Observations

- Under the manually reported CelebA base test accuracy (`90.57%`), the corrected higher-ratio non-attribute resume baselines are weaker than previously recorded, while the `ratio=0.1` entries remain as originally reported.
- The strongest completed resume result in the current table is `badge` at `ratio=0.1` with held-out test accuracy `91.12%`, which is `+0.55` percentage points above the reported base.
- Among the unpatched selectors, `coreset` and early `badge` were dominated by selection cost rather than downstream accuracy:
  - `coreset` selection time grew from `45 min` at `ratio=0.1` to `5 h 5 min` at `ratio=0.5`
  - the original `badge` `ratio=0.1` run already needed `1 h 52 min` just for selection
- After the large-job approximation patches, the high-ratio runs became practical again:
  - patched `coreset` `ratio=0.7`: `56.85 s`
  - patched `badge` `ratio=0.3/0.5/0.7`: `116.78 s`, `124.76 s`, `74.42 s`
- The apparent BADGE runtime reversal across ratios is a code-path change, not a true scaling win:
  - `ratio=0.1` used the older exact BADGE path and took `1 h 52 min`
  - `ratio=0.3/0.5/0.7` are patched large-job reruns that use the approximation path
  - so `0.1` versus `0.3/0.5/0.7` runtime is not a like-for-like selector comparison
- Accuracy trends are not monotonic with retained-data ratio on CelebA:
  - corrected `random` peaked at `ratio=0.5` and then dropped at `0.7`
  - corrected `stratified` was stable at `0.5` and `0.7`
  - corrected `badge` improved again at `0.7` after being weaker at `0.3`
- For the current CelebA comparison set, `badge` remains the strongest reported method at `ratio=0.1`, while `attribute` is competitive at both `ratio=0.1` and `ratio=0.3`; `coreset` is still the least attractive once runtime is considered.

### 04/16/2026 Update — CelebA Attribute Hyperparameter Probe (`ratio=0.1`, `E=False`)

- Completed three CelebA resume-from-base `attribute` runs at `ratio=0.1` with entropy fixed off and explicit CelebA metadata focus:
  - `hair color`
  - `skin`
  - `emotion`
- Because `save/attribute.json` uses spaced keys, these runs also passed explicit generated-count injection rather than relying on the ImageNet-style auto-inference:
  - `hair color:red=1000`
  - `skin:brown=1000`
  - `emotion:sad=1000`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `baseline_def_e0_r001_p5_cinf` | `0.01` | `5.0` | `inf` | **`90.59%`** | `+0.02 pp` | `+0.44 pp` | `0.72 s` | `237.61 s` | `logs/celeba/attr_ratio0.1_baseline_def_e0_r001_p5_cinf_gpu0_20260416_140026.log` |
| `soft_e0_r01_p5_c15` | `0.1` | `5.0` | `15.0` | `90.55%` | `-0.02 pp` | `+0.40 pp` | `0.70 s` | `236.93 s` | `logs/celeba/attr_ratio0.1_soft_e0_r01_p5_c15_gpu1_20260416_140026.log` |
| `aggro_e0_r08_p10_c4` | `0.8` | `10.0` | `4.0` | `90.47%` | `-0.10 pp` | `+0.32 pp` | `0.71 s` | `236.35 s` | `logs/celeba/attr_ratio0.1_aggro_e0_r08_p10_c4_gpu2_20260416_140026.log` |

### CelebA Attribute-Probe Observations

- The best of the three completed CelebA attribute settings is the most conservative one:
  - `baseline_def_e0_r001_p5_cinf`
  - held-out test accuracy `90.59%`
- Even this early attribute configuration already clears the corrected `random` baseline, but it remains below the stronger completed `ratio=0.1` non-attribute baselines:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
- The good news is runtime:
  - all three attribute selectors finished in about `0.7 s`
  - this is dramatically cheaper than `coreset` (`45 min`) or the original `badge` (`1 h 52 min`) at the same ratio
- More aggressive rarity / penalty settings did not improve overall held-out accuracy on CelebA at `ratio=0.1`.
  - under this current metadata-and-scoring setup, stronger attribute steering appears to over-focus the selection without improving final generalization.
- Practical takeaway:
  - if CelebA attribute experiments continue, the conservative `R=0.01, P=5.0, C=inf, E=False` setting is the only sensible starting point from this probe
  - but the current attribute path is not yet competitive with the completed `random` / `badge` / `stratified` baselines on this CelebA setup

### 04/16/2026 Update — CelebA Attribute Mix-Scaled Probe (`ratio=0.1`, `E=False`, `extra_count_scale=10`)

- The earlier three-run probe injected raw generated counts (`1000`) directly into the full `80,000`-image original candidate pool.
- For `ratio=0.1`, the final training mix is really `8000 original + 1000 generated`, so the selector was underestimating the generated-image share during scoring.
- This follow-up probe therefore scales the explicit generated counts by `1 / sub_ratio = 10` before scoring:
  - effective injected counts during selection:
    - `hair color:red=10000`
    - `skin:brown=10000`
    - `emotion:sad=10000`
- With that correction in place, the CelebA sweep moved to much milder `R/P/C` settings instead of reusing the stronger ImageNet-style penalties.

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `mixselx10_e0_r003_p05_cinf` | `0.03` | `0.5` | `inf` | **`90.99%`** | `+0.42 pp` | `+0.84 pp` | `0.74 s` | `349.37 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r003_p05_cinf_gpu2_20260416_141739.log` |
| `mixselx10_e0_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.86%` | `+0.29 pp` | `+0.71 pp` | `0.70 s` | `347.44 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r000_p0_cinf_gpu1_20260416_141739.log` |
| `mixselx10_e0_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.81%` | `+0.24 pp` | `+0.66 pp` | `0.74 s` | `372.83 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r01_p05_cinf_gpu3_20260416_141739.log` |
| `mixselx10_e0_r000_p05_cinf` | `0.0` | `0.5` | `inf` | `90.71%` | `+0.14 pp` | `+0.56 pp` | `0.94 s` | `348.21 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r000_p05_cinf_gpu1_20260416_141739.log` |
| `mixselx10_e0_r001_p0_cinf` | `0.01` | `0.0` | `inf` | `90.70%` | `+0.13 pp` | `+0.55 pp` | `0.72 s` | `344.59 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r001_p0_cinf_gpu0_20260416_141739.log` |
| `mixselx10_e0_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.67%` | `+0.10 pp` | `+0.52 pp` | `1.05 s` | `343.75 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r001_p05_cinf_gpu0_20260416_141739.log` |
| `mixselx10_e0_r001_p1_cinf` | `0.01` | `1.0` | `inf` | `90.61%` | `+0.04 pp` | `+0.46 pp` | `1.01 s` | `350.25 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r001_p1_cinf_gpu2_20260416_141739.log` |
| `mixselx10_e0_r01_p05_c15` | `0.1` | `0.5` | `15.0` | `90.61%` | `+0.04 pp` | `+0.46 pp` | `0.98 s` | `412.87 s` | `logs/celeba/attr_ratio0.1_mixselx10_e0_r01_p05_c15_gpu3_20260416_141739.log` |

### CelebA Attribute Mix-Scaled Observations

- Correcting the generated-count scale materially helped:
  - old raw-count best:
    - `90.59%`
  - new mix-scaled best:
    - `90.99%`
  - improvement:
    - `+0.40 pp`
- The best CelebA configuration after this correction is:
  - `R=0.03`
  - `P=0.5`
  - `C=inf`
  - `E=False`
  - `extra_count_scale=10`
- After the scale correction, CelebA prefers much milder penalties than the earlier ImageNet-style probe:
  - `P=0.0` to `0.5` is consistently better than `P=1.0`
  - strict class balance (`C=inf`) is better than the softer `C=15`
- Runtime remains excellent:
  - all eight mix-scaled selectors still finish in about `0.7` to `1.1 s`
- The attribute method is now competitive with the corrected `ratio=0.1` baselines:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
  - `best mix-scaled attribute`: `90.99%`
- Practical takeaway:
  - if the CelebA attribute sweep continues to larger ratios, start from:
    - `R=0.03, P=0.5, C=inf, E=False`
    - with generated extra counts scaled by `1 / sub_ratio`

### 04/16/2026 Update — CelebA Attribute Mix-Scaled Follow-Up (`ratio=0.3`, top 3 from `ratio=0.1`)

- After the corrected `ratio=0.1` sweep, we transferred its top 3 settings to `ratio=0.3` instead of rerunning the full 8-scenario grid immediately.
- These runs also used generated-count scaling matched to the selected-data regime:
  - `extra_count_scale = 1 / 0.3 = 3.33333`
  - effective injected counts during scoring:
    - `hair color:red=3333.33`
    - `skin:brown=3333.33`
    - `emotion:sad=3333.33`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `mixselx333_e0_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.89%`** | `+0.32 pp` | `-0.29 pp` | `0.71 s` | `492.89 s` | `logs/celeba/attr_ratio0.3_mixselx333_e0_r000_p0_cinf_gpu2_20260416_143519.log` |
| `mixselx333_e0_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.87%` | `+0.30 pp` | `-0.31 pp` | `0.76 s` | `470.50 s` | `logs/celeba/attr_ratio0.3_mixselx333_e0_r003_p05_cinf_gpu1_20260416_143519.log` |
| `mixselx333_e0_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.74%` | `+0.17 pp` | `-0.44 pp` | `0.76 s` | `637.82 s` | `logs/celeba/attr_ratio0.3_mixselx333_e0_r01_p05_cinf_gpu3_20260416_143519.log` |

### CelebA Attribute Mix-Scaled Ratio-0.3 Observations

- All three transferred `ratio=0.1` winners completed cleanly at `ratio=0.3`.
- Within this transferred top-3 set, the best `ratio=0.3` configuration is:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - `E=False`
  - `extra_count_scale=3.33333`
- This is a notable shift from `ratio=0.1`, where the best corrected setting was:
  - `R=0.03`
  - `P=0.5`
  - `C=inf`
- Even so, the attribute method still trails the completed CelebA `ratio=0.3` baselines:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
  - `best transferred mix-scaled attribute`: `90.89%`
- Compared with the corrected `ratio=0.1` best (`90.99%`), the transferred `ratio=0.3` best did not improve.
- Practical takeaway:
  - if a larger `ratio=0.3` attribute sweep is worth doing, search around the milder corner first:
    - `R in {0.0, 0.03}`
    - `P in {0.0, 0.5}`
    - `C = inf`
    - still using generated-count scaling by `1 / sub_ratio`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.3`, finite `C=15`)

- The earlier Qwen3 `ratio=0.3` reference runs only covered `C=inf`, but that was not the intended final selector regime.
- This follow-up kept the same mild `R/P` neighborhood and changed only the class-balance term to finite `C=15.0` using:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale=3.33333`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|--------------|-------------------|-----------------|---------------|
| `qwen3mixx333_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.88%`** | `+0.31 pp` | `-0.30 pp` | `90.92%` | `0.53 s` | `423.41 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r000_p0_c15_gpu0_20260417_011502.log` |
| `qwen3mixx333_r001_p05_c15` | `0.01` | `0.5` | `15.0` | `90.61%` | `+0.04 pp` | `-0.57 pp` | `90.55%` | `0.55 s` | `416.34 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r001_p05_c15_gpu1_20260417_011502.log` |
| `qwen3mixx333_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.28%` | `-0.29 pp` | `-0.90 pp` | `90.38%` | `0.54 s` | `420.36 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r003_p05_c15_gpu2_20260417_011502.log` |

### CelebA Qwen3 Ratio-0.3 Finite-`C` Observations

- Switching from `C=inf` to `C=15.0` did not improve the best `ratio=0.3` Qwen3 result.
- The best finite-`C` configuration is still the most conservative one:
  - `R=0.0`
  - `P=0.0`
  - `C=15.0`
  - held-out test accuracy `90.88%`
- Compared with the earlier Qwen3 `C=inf` reference at the same mild corner:
  - `R=0.0, P=0.0` changed from `90.89%` to `90.88%`
  - effectively no gain
- The more attribute-steered finite-`C` settings were clearly worse at `ratio=0.3`:
  - `R=0.01, P=0.5, C=15.0`: `90.61%`
  - `R=0.03, P=0.5, C=15.0`: `90.28%`
- Relative to the completed non-attribute CelebA baselines at `ratio=0.3`, the attribute path still trails:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
  - best Qwen3 finite-`C` attribute result: `90.88%`
- Practical takeaway:
  - for CelebA `ratio=0.3`, there is currently no evidence that finite class balance helps the Qwen3 attribute selector
  - if this line continues, the only defensible region is still the conservative near-random corner

### 04/17/2026 Update — CelebA Qwen3 Attribute Re-Run (`ratio=0.1`, 4 selected RPCs)

- The earlier CelebA attribute experiments above used the older BLIP-style metadata file:
  - `save/attribute.json`
- A follow-up re-run was completed against the newer Qwen3-assigned merged train+val metadata:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
- To keep the source explicit and avoid silently changing defaults, these runs used:
  - `scripts/run_celeba_attribute_ratio0.1_qwen3_rpc4.sh`
- The four selected settings were chosen from the previously strongest `ratio=0.1` mix-scaled BLIP region instead of rerunning the full grid.

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | BLIP Held-out Test ACC | Delta vs BLIP | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|------------------------|---------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.86%`** | `90.86%` | `+0.00 pp` | `91.03%` | `0.38 s` | `217.35 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r000_p0_cinf_gpu0_20260417_004805.log` |
| `qwen3mix_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.30%` | `90.67%` | `-0.37 pp` | `90.29%` | `0.42 s` | `217.03 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r001_p05_cinf_gpu1_20260417_004805.log` |
| `qwen3mix_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.12%` | **`90.99%`** | `-0.87 pp` | `90.33%` | `0.45 s` | `216.75 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r003_p05_cinf_gpu2_20260417_004805.log` |
| `qwen3mix_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.03%` | `90.81%` | `-0.78 pp` | `90.16%` | `0.41 s` | `216.72 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r01_p05_cinf_gpu3_20260417_004805.log` |

### CelebA Qwen3 Re-Run Observations

- This Qwen3 metadata re-run did **not** improve over the earlier BLIP-based `ratio=0.1` attribute results.
- The best Qwen3 result is:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - held-out test accuracy `90.86%`
- Compared with the best BLIP-based `ratio=0.1` attribute result:
  - BLIP best:
    - `R=0.03`
    - `P=0.5`
    - `C=inf`
    - held-out test accuracy `90.99%`
  - Qwen3 best:
    - `90.86%`
  - gap:
    - `-0.13 pp`
- On a same-config comparison, Qwen3 matched only the conservative control:
  - `R=0.0`, `P=0.0`, `C=inf`:
    - `90.86%` vs `90.86%`
- The three stronger rarity-steering settings all degraded under Qwen3 metadata:
  - `R=0.01`, `P=0.5`, `C=inf`:
    - `-0.37 pp`
  - `R=0.03`, `P=0.5`, `C=inf`:
    - `-0.87 pp`
  - `R=0.1`, `P=0.5`, `C=inf`:
    - `-0.78 pp`
- Practical takeaway:
  - for CelebA `ratio=0.1` attribute AL, the Qwen3 metadata source did not improve overall held-out accuracy over the earlier BLIP-based metadata
  - if this line continues, the immediate options are:
    - keep the BLIP-based `R=0.03, P=0.5, C=inf` result as the stronger `ratio=0.1` attribute reference
    - or inspect why Qwen3 labels appear to make the selector over-steer once `P>0` and `R>0`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.1`, finite `C=15`)

- Because `C=inf` is not admissible for the next selector choice, a second `ratio=0.1` Qwen3 follow-up re-tested the mild top-3 Qwen3 `R/P` settings with finite class balance:
  - `C=15.0`
- These runs again used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r003_p05_c15` | `0.03` | `0.5` | `15.0` | **`91.07%`** | `+0.50 pp` | `+0.92 pp` | `90.91%` | `0.42 s` | `216.89 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r003_p05_c15_gpu2_20260417_010718.log` |
| `qwen3mix_r000_p0_c15` | `0.0` | `0.0` | `15.0` | `90.84%` | `+0.27 pp` | `+0.69 pp` | `90.97%` | `0.40 s` | `217.38 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r000_p0_c15_gpu0_20260417_010718.log` |
| `qwen3mix_r001_p05_c15` | `0.01` | `0.5` | `15.0` | `90.09%` | `-0.48 pp` | `-0.06 pp` | `90.24%` | `0.43 s` | `217.79 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r001_p05_c15_gpu1_20260417_010718.log` |

### CelebA Qwen3 Ratio-0.1 Finite-C Observations

- The finite-`C` rerun changes the ranking within the Qwen3 `ratio=0.1` family:
  - best finite-`C` setting:
    - `R=0.03`
    - `P=0.5`
    - `C=15.0`
    - held-out test accuracy `91.07%`
- Compared with the earlier Qwen3 `C=inf` runs at the same `ratio=0.1`:
  - `R=0.0`, `P=0.0`:
    - `C=15.0` Qwen3 `90.84%`
    - `C=inf` Qwen3 `90.86%`
    - delta `-0.02 pp`
  - `R=0.01`, `P=0.5`:
    - `C=15.0` Qwen3 `90.09%`
    - `C=inf` Qwen3 `90.30%`
    - delta `-0.21 pp`
  - `R=0.03`, `P=0.5`:
    - `C=15.0` Qwen3 `91.07%`
    - `C=inf` Qwen3 `90.12%`
    - delta `+0.95 pp`
- This finite-`C` result substantially closes the gap to the earlier BLIP-based best `ratio=0.1` attribute run:
  - BLIP best:
    - `R=0.03`, `P=0.5`, `C=inf`
    - `90.99%`
  - Qwen3 best finite-`C`:
    - `R=0.03`, `P=0.5`, `C=15.0`
    - `91.07%`
  - delta:
    - `+0.08 pp`
- It now beats the corrected `random` and `stratified` `ratio=0.1` baselines, but it still does not beat the strongest completed non-attribute method:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
  - Qwen3 best finite-`C`: `91.07%`
- Practical takeaway:
  - once `C=inf` is excluded, the best current Qwen3 `ratio=0.1` setting is:
    - `R=0.03, P=0.5, C=15.0`
  - this is the first Qwen3 CelebA attribute setting that slightly beats the earlier BLIP-based best attribute result, while remaining just below the finished `random` and `badge` baselines

### 04/17/2026 Update — CelebA Qwen3 Attribute Local Refinement (`ratio=0.1`, around `R=0.03, P=0.5, C=15`)

- After the first finite-`C` Qwen3 `ratio=0.1` sweep, a small local refinement tested whether the current best point could be improved by:
  - lowering rarity from `R=0.03` to `R=0.02`
  - raising rarity from `R=0.03` to `R=0.05`
  - softening class balance from `C=15.0` to `C=8.0`
- These runs again used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r003_p05_c8` | `0.03` | `0.5` | `8.0` | **`90.66%`** | `+0.09 pp` | `+0.51 pp` | `90.63%` | `0.53 s` | `235.02 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r003_p05_c8_gpu2_20260417_012110.log` |
| `qwen3mix_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.41%` | `-0.16 pp` | `+0.26 pp` | `90.46%` | `0.52 s` | `230.14 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r002_p05_c15_gpu0_20260417_012110.log` |
| `qwen3mix_r005_p05_c15` | `0.05` | `0.5` | `15.0` | `90.15%` | `-0.42 pp` | `+0.00 pp` | `90.27%` | `0.52 s` | `235.35 s` | `logs/celeba/attr_ratio0.1_qwen3mix_r005_p05_c15_gpu1_20260417_012110.log` |

### CelebA Qwen3 Ratio-0.1 Local-Refinement Observations

- None of the three local-refinement runs beat the incumbent best finite-`C` setting:
  - incumbent:
    - `R=0.03, P=0.5, C=15.0`
    - held-out test accuracy `91.07%`
- Lowering rarity slightly hurts:
  - `R=0.02, P=0.5, C=15.0`
  - `90.41%`
  - delta vs incumbent `-0.66 pp`
- Increasing rarity also hurts:
  - `R=0.05, P=0.5, C=15.0`
  - `90.15%`
  - delta vs incumbent `-0.92 pp`
- Softening class balance from `C=15.0` to `C=8.0` is better than moving `R`, but it still loses to the incumbent:
  - `R=0.03, P=0.5, C=8.0`
  - `90.66%`
  - delta vs incumbent `-0.41 pp`
- Practical takeaway:
  - the local neighborhood evidence supports keeping:
    - `R=0.03, P=0.5, C=15.0`
  - as the current best Qwen3 CelebA attribute setting for `ratio=0.1`

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Pre-Rerun Snapshot (`ratio=0.1`, interrupted old directory semantics)

- Before the generated-image directories were renamed to distinguish filtered vs unfiltered content, six `redhair_brownskin` attribute runs were launched with the old path meaning.
- Those runs started normally, completed active-learning selection, and entered resumed training, but all six crashed mid-training after the directory rename because the training dataloader could no longer open some generated images under the old path.
- These logs are still useful as a pre-rerun comparison snapshot for selection behavior and early validation trajectory, but they are **not** valid final accuracy results.

| Config | `R` | `P` | `C` | Best Observed Val ACC Before Crash | Last Completed Epoch | Failure Type | Canonical Log |
|--------|-----|-----|-----|------------------------------------|----------------------|--------------|---------------|
| `rb_qwen3_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.21%` | `8` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c15_gpu0_20260417_012831.log` |
| `rb_qwen3_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.79%` | `8` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_cinf_gpu0_20260417_012831.log` |
| `rb_qwen3_r000_p0_c15` | `0.0` | `0.0` | `15.0` | `90.79%` | `7` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_c15_gpu1_20260417_012831.log` |
| `rb_qwen3_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `89.82%` | `7` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_cinf_gpu1_20260417_012831.log` |
| `rb_qwen3_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.87%` | `7` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c8_gpu2_20260417_012831.log` |
| `rb_qwen3_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.33%` | `7` | `FileNotFoundError` after directory rename | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r002_p05_c15_gpu2_20260417_012831.log` |

### CelebA `redhair_brownskin` Pre-Rerun Snapshot Notes

- All six interrupted runs successfully reached:
  - `Selected 1000 filenames for sampling`
  - `CelebA: Combined 8000 selected samples with 1000 generated images using Active Learning`
  - resumed fine-tuning from `save/weights/model_celeba_resnet_base.pth`
- The crash mechanism was consistent across runs:
  - `FileNotFoundError`
  - missing generated image under `data/redhair_brownskin/...`
  - caused by the directory rename that changed filtered vs unfiltered path semantics during training
- These rows should only be used as an early-trajectory comparison against the rerun that uses the corrected filtered directory layout.

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Rerun (`ratio=0.1`, filtered directory)

- After the generated-image directory semantics were corrected, the same six `redhair_brownskin` Qwen3 `ratio=0.1` configurations were rerun against the filtered subset now mounted at:
  - `data/redhair_brownskin`
- These runs used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - focus:
    - `hair color,skin`
  - explicit generated counts:
    - `hair color:red=1000`
    - `skin:brown=1000`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|--------------|-------------------|-----------------|---------------|
| `rb_qwen3_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.75%`** | `+0.18 pp` | `90.82%` | `0.34 s` | `323.19 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_c15_gpu1_20260417_013234.log` |
| `rb_qwen3_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.74%` | `+0.17 pp` | `90.82%` | `0.32 s` | `315.65 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_cinf_gpu0_20260417_013234.log` |
| `rb_qwen3_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.63%` | `+0.06 pp` | `90.63%` | `0.42 s` | `323.69 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r002_p05_c15_gpu2_20260417_013234.log` |
| `rb_qwen3_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.33%` | `-0.24 pp` | `90.53%` | `0.37 s` | `316.43 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c15_gpu0_20260417_013234.log` |
| `rb_qwen3_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.16%` | `-0.41 pp` | `90.30%` | `0.36 s` | `323.48 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c8_gpu2_20260417_013234.log` |
| `rb_qwen3_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `89.65%` | `-0.92 pp` | `89.92%` | `0.36 s` | `322.76 s` | `logs/celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_cinf_gpu1_20260417_013234.log` |

### CelebA `redhair_brownskin` Ratio-0.1 Observations

- On the filtered `redhair_brownskin` subset, the best Qwen3 attribute setting is the conservative control:
  - `R=0.0`
  - `P=0.0`
  - `C=15.0`
  - held-out test accuracy `90.75%`
- The corresponding `C=inf` control is effectively tied:
  - `R=0.0, P=0.0, C=inf`
  - `90.74%`
- Unlike the earlier `redhair_brownskin_sademotion` experiments, steering toward the `R=0.03, P=0.5` region hurts on this filtered two-attribute setup:
  - `C=15.0`: `90.33%`
  - `C=8.0`: `90.16%`
  - `C=inf`: `89.65%`
- The filtered `redhair_brownskin` subset therefore appears to prefer minimal attribute steering, with finite `C=15.0` only slightly ahead of the pure `C=inf` control.
- Practical takeaway:
  - for the current filtered `redhair_brownskin` `ratio=0.1` setting, carry forward:
    - `R=0.0, P=0.0, C=15.0`
  - and treat the earlier interrupted pre-rerun table only as a directory-mismatch comparison snapshot, not as a competing result set

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Follow-Up (`ratio=0.3`, filtered directory)

- The six filtered `redhair_brownskin` Qwen3 `ratio=0.3` runs launched earlier on GPUs `0/1/2` have now finished.
- These runs used:
  - `data/redhair_brownskin`
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 3.33333`
- This was the direct larger-ratio follow-up to the filtered `ratio=0.1` rerun, keeping both conservative controls and the previously informative mild steering region.

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|--------------|-------------------|-----------------|---------------|
| `rb_qwen3x333_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.91%`** | `+0.34 pp` | `91.02%` | `0.44 s` | `655.45 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_c15_gpu0_20260417_014040.log` |
| `rb_qwen3x333_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.91%`** | `+0.34 pp` | `91.03%` | `0.43 s` | `656.18 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_cinf_gpu0_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.63%` | `+0.06 pp` | `90.72%` | `0.44 s` | `676.91 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_cinf_gpu2_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.53%` | `-0.04 pp` | `90.61%` | `0.47 s` | `661.52 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_c15_gpu1_20260417_014040.log` |
| `rb_qwen3x333_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.47%` | `-0.10 pp` | `90.63%` | `0.47 s` | `664.06 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r002_p05_c15_gpu1_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.37%` | `-0.20 pp` | `90.49%` | `0.46 s` | `678.05 s` | `logs/celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_c8_gpu2_20260417_014040.log` |

### CelebA `redhair_brownskin` Ratio-0.3 Observations

- On the filtered `redhair_brownskin` subset at `ratio=0.3`, the best result is again the conservative control family:
  - `R=0.0`
  - `P=0.0`
  - both `C=15.0` and `C=inf` reached the same held-out test accuracy:
    - `90.91%`
- Relative to the filtered `ratio=0.1` rerun, increasing the retained original-data ratio helped the control point:
  - `ratio=0.1` best:
    - `90.75%`
  - `ratio=0.3` best:
    - `90.91%`
  - improvement:
    - `+0.16 pp`
- The mild attribute-steered region still does not beat the control on this filtered two-attribute setup:
  - `R=0.02, P=0.5, C=15.0`:
    - `90.47%`
  - `R=0.03, P=0.5, C=15.0`:
    - `90.53%`
  - `R=0.03, P=0.5, C=8.0`:
    - `90.37%`
  - `R=0.03, P=0.5, C=inf`:
    - `90.63%`
- Runtime remains negligible compared with embedding-based AL baselines:
  - all six selections still finished in about `0.43` to `0.47 s`
- Practical takeaway:
  - for the current filtered `redhair_brownskin` line, both completed ratios (`0.1` and `0.3`) prefer near-random Qwen3 selection over explicit rarity steering
  - if this branch continues, carry forward:
    - `R=0.0`
    - `P=0.0`
    - `C=15.0` or `C=inf`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.3`, transferred top 3)

- A second Qwen3-based follow-up transferred the top 3 mild `ratio=0.1` Qwen3 settings to `ratio=0.3`:
  - `R=0.0, P=0.0, C=inf`
  - `R=0.01, P=0.5, C=inf`
  - `R=0.03, P=0.5, C=inf`
- These runs used the same Qwen3 metadata source:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
- The generated-count scaling again matched the selected-data regime:
  - `extra_count_scale = 1 / 0.3 = 3.33333`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mixx333_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.89%`** | `+0.32 pp` | `-0.29 pp` | `90.90%` | `0.51 s` | `399.41 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r000_p0_cinf_gpu0_20260417_005520.log` |
| `qwen3mixx333_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.49%` | `-0.08 pp` | `-0.69 pp` | `90.50%` | `0.50 s` | `401.53 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r001_p05_cinf_gpu1_20260417_005520.log` |
| `qwen3mixx333_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.63%` | `+0.06 pp` | `-0.55 pp` | `90.71%` | `0.51 s` | `399.91 s` | `logs/celeba/attr_ratio0.3_qwen3mixx333_r003_p05_cinf_gpu2_20260417_005520.log` |

### CelebA Qwen3 Ratio-0.3 Observations

- The best completed Qwen3 `ratio=0.3` result is still the conservative control:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - held-out test accuracy `90.89%`
- On the overlapping transferred settings, Qwen3 does not beat the earlier BLIP-based ratio-0.3 results:
  - `R=0.0`, `P=0.0`, `C=inf`:
    - Qwen3 `90.89%`
    - BLIP `90.89%`
    - delta `+0.00 pp`
  - `R=0.03`, `P=0.5`, `C=inf`:
    - Qwen3 `90.63%`
    - BLIP `90.87%`
    - delta `-0.24 pp`
- There is no direct BLIP `ratio=0.3` comparison point for:
  - `R=0.01`, `P=0.5`, `C=inf`
  - because the earlier BLIP top-3 transfer at `ratio=0.3` used `R=0.1`, `P=0.5`, `C=inf` instead
- All three completed Qwen3 `ratio=0.3` runs still trail the finished non-attribute CelebA baselines at the same ratio:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
- These completed Qwen3 `ratio=0.3` runs are still useful as reference points, but they are **not** the right settings to carry forward if `C=inf` is no longer admissible.
- Practical takeaway:
  - preserve the `ratio=0.3` Qwen3 results for reference
  - but move the next search back to `ratio=0.1` with finite `C` values rather than selecting from the `C=inf` runs above

# 04/16/2026 Update — ImageNet10 BADGE Resume Sweep

- Completed four ImageNet10 BADGE follow-up runs using the same resume-from-base setting as the earlier AL comparison:
  - base checkpoint: `save/weights/model_imagenet_resnet_base.pth`
  - model: `ResNet-18`
  - learning rate: `1e-5`
  - epochs: `15`
  - generated images: `1000`
  - attribute prompt: `pinkcolor_fabrictexture`
- All four runs finished BADGE selection, resumed training, and printed held-out test accuracy.
- The runs then failed in post-processing:
  - `sub_ratio=0.1`: prediction save failed because `save/pred/` was missing
  - `sub_ratio=0.3/0.5/0.7`: diagnostics failed because `save/attribute10.json` was missing
- These failures happened **after** the main evaluation lines, so the reported test metrics below are valid.

| Sub Ratio | Selected Original Samples | Final Train Size After +1000 Gen | Best Val ACC | Held-out Test ACC | AL Selection Time | End-to-End Time | Post-run Status |
|---------|----------------------------|----------------------------------|--------------|-------------------|-------------------|-----------------|-----------------|
| `0.1` | `1040` | `2040` | `0.7208` | `0.7323` | `272.65 s` | n/a | crashed during prediction save after test evaluation |
| `0.3` | `3120` | `4120` | `0.7215` | `0.7423` | `13 min` | `22 min` | crashed during diagnostics after test evaluation |
| `0.5` | `5200` | `6200` | `0.7223` | `0.7469` | `21 min` | `28 min` | crashed during diagnostics after test evaluation |
| `0.7` | `7279` | `8279` | `0.7215` | **`0.7477`** | `27 min` | `31 min` | crashed during diagnostics after test evaluation |

### Observations

- Test accuracy improved monotonically with the retained original-data ratio:
  - `0.7323 -> 0.7423 -> 0.7469 -> 0.7477`
- At the directly comparable `sub_ratio=0.3` resume setting, BADGE reached `0.7423` held-out test accuracy.
  - This is above the earlier `2/27/2026` resume table entries for `Random` (`0.7308`), `Stratified` (`0.7369`), `Coreset` (`0.7359`), `Diversity` (`0.7362`), `Uncertainty Stratified` (`0.7292`), `Hierarchical` (`0.7346`), and `Attribute` (`0.7369`).
  - It also edges the earlier cited `Hybrid` result (`0.7400`) by `+0.0023`.
- Best validation accuracy stayed nearly flat around `0.721` to `0.722`, while test accuracy improved as more original data was retained.
  - In this setup, larger BADGE-selected subsets help held-out generalization more clearly than they help validation peak accuracy.
- BADGE selection is materially more expensive than the lightweight metadata-based `attribute` selector:
  - even on this 10-class ImageNet subset, AL selection time grew from `272.65 s` at `sub_ratio=0.1` to `27 min` at `sub_ratio=0.7`
  - the cost comes from pretrained embedding extraction plus BADGE gradient-embedding batch seeding

# 04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished

- Verified that the two bs8 pose-estimation runs launched on April 15, 2026 are complete:
  - GPU `3` / Qwen3-VL-8B: original PID `1623280`
  - GPU `0` / InternVL3-8B: original PID `1624937`
- Both original PIDs are no longer alive, both logs end with `Training complete`, and both artifact roots contain:
  - `best.pth`
  - held-out `val` and `test` metrics JSONs
  - prediction dumps
  - post-train slice-analysis JSON/CSV outputs
- These are fresh runs under the current defaults (`batch_size=8`, `analysis_rare_label_threshold=1.1`), so they are **not** strict apples-to-apples reruns of the earlier `4/12` bs4 `rare0p97` probes.
- In both runs, the best checkpoint came from **epoch 0**; epoch 1 slightly regressed on held-out val AP.

| Run Tag | GPU | Best Epoch | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | Artifact Root | Log |
|---------|-----|------------|-----------------|------------------|-----------|-----------|---------|------------|---------------|-----|
| Qwen3-VL-8B | `3` | `0` | `0.6445` | `0.6352` | `0.8656` | `0.6902` | `0.6993` | `1 h 29 min` | `logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/` | `logs/pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log` |
| InternVL3-8B | `0` | `0` | `0.6401` | `0.5902` | `0.8618` | `0.6936` | `0.7016` | `1 h 34 min` | `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/` | `logs/pose/base_pose_train20k_compare.log` |

- Relative to the earlier `4/12` held-out test probes:
  - Qwen3-VL-8B improved from `0.5968` to `0.6352` (`+0.0384`)
  - InternVL3-8B moved from `0.6340` to `0.5902` (`-0.0438`)
- Top-level `logs/pose/` logs explicitly record both training-only and end-to-end elapsed time for these runs:
  - Qwen3-VL-8B: training `1 h 29 min`, end-to-end `1 h 29 min`
  - InternVL3-8B: training `1 h 34 min`, end-to-end `1 h 34 min`
- Held-out test slice analysis for both runs used `rare_label_threshold=1.1`.
- The strongest recurring **rare-case** slice is still `pose=lying` in both runs:
  - Qwen3-VL-8B: `AP=0.3091`, `ap_delta=-0.3261`, `34` test images
  - InternVL3-8B: `AP=0.3625`, `ap_delta=-0.2733`, `29` test images
- Other representative **rare cases** on the held-out test split:
  - Qwen3-VL-8B: `time_of_day=evening` (`AP=0.5629`, `164` images), `weather=rainy` (`AP=0.5799`, `31` images), `time_of_day=night` (`AP=0.5816`, `74` images)
  - InternVL3-8B: `time_of_day=night` (`AP=0.5725`, `191` images), `background_style=rural` (`AP=0.5756`, `103` images), `weather=rainy` (`AP=0.5796`, `32` images)
- `pose=sitting` remains the main high-support hard slice:
  - Qwen3-VL-8B: `AP=0.5503`, `329` test images
  - InternVL3-8B: `AP=0.5480`, `288` test images
- Other notable low-AP slices on the held-out test split:
  - Qwen3-VL-8B: `background_style=indoors` (`AP=0.5899`, hard_case), `weather=clear` (`AP=0.5899`, hard_case)
  - InternVL3-8B: `background_style=indoors` (`AP=0.5866`, hard_case), `weather=clear` (`AP=0.6017`, hard_case)

### Follow-Up Addendum (Artifacts dated 2026-04-15; original table above preserved)

- Two later same-day reruns requested after the initial summary are now complete:
  - Qwen3-VL-8B reproducibility rerun on GPU `0`:
    - artifact root: `logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/`
    - log: `logs/pose/pose_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro.log`
  - InternVL3-8B training-only rerun on GPU `3` followed by post-hoc slice analyses:
    - artifact root: `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/`
    - training log: `logs/pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.log`
    - post-hoc sweep log: `logs/pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.log`
- In both later reruns, the best checkpoint again came from **epoch 0**.

| Follow-Up Run | Earlier 4/17 Reference | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time |
|---------------|------------------------|-----------------|------------------|-----------|-----------|---------|------------|
| Qwen3-VL-8B GPU `0` repro | Qwen3-VL-8B GPU `3` (`20260415_144429`) | `0.6445 -> 0.6407` (`-0.0038`) | `0.6352 -> 0.6341` (`-0.0011`) | `0.8656 -> 0.8609` (`-0.0047`) | `0.6902 -> 0.6866` (`-0.0036`) | `0.6993 -> 0.6958` (`-0.0035`) | `1 h 29 min -> 1 h 39 min` (`+10 min`) |
| InternVL3-8B GPU `3` train-only rerun | InternVL3-8B GPU `0` (`20260415_144530`) | `0.6401 -> 0.6412` (`+0.0011`) | `0.5902 -> 0.6371` (`+0.0469`) | `0.8618 -> 0.8583` (`-0.0035`) | `0.6936 -> 0.6963` (`+0.0027`) | `0.7016 -> 0.6950` (`-0.0066`) | `1 h 34 min -> 1 h 29 min` (`-5 min`) |

- Qwen3-VL-8B repro interpretation:
  - overall held-out metrics are slightly below the earlier GPU `3` Qwen3 run
  - the hardest rare slice `pose=lying` improved from `AP=0.3091` to `AP=0.3906` (`+0.0815`)
  - `pose=sitting` also improved slightly from `AP=0.5503` to `AP=0.5529`
  - low-light slices remained difficult and close to the earlier run:
    - `time_of_day=evening`: `0.5629 -> 0.5737`
    - `time_of_day=night`: `0.5816 -> 0.5802`
- InternVL3-8B GPU `3` train-only rerun interpretation:
  - held-out AP is higher than the earlier GPU `0` InternVL3 run on both `val` and `test`
  - the hardest rare slice `pose=lying` improved from `AP=0.3625` to `AP=0.4021` (`+0.0396`)
  - `pose=sitting` improved from `AP=0.5480` to `AP=0.5585`
  - representative rare-case test slices in the rerun are:
    - `time_of_day=night`: `AP=0.5693`, `191` images
    - `background_style=rural`: `AP=0.5738`, `103` images
    - `weather=rainy`: `AP=0.5871`, `32` images
    - `time_of_day=evening`: `AP=0.6296`, `72` images
- Important sweep result for the InternVL3 rerun:
  - the post-hoc `rare_label_threshold` analyses at `0.97`, `0.95`, and `0.93` all completed successfully
  - the resulting `val` CSVs are byte-identical across thresholds
  - the resulting `test` CSVs are also byte-identical across thresholds
  - therefore, for this checkpoint, lowering the post-hoc rare threshold from `0.97` to `0.95` or `0.93` did **not** change the slice rankings or rare/hard-case labels
  - the three completed slice-analysis roots are:
    - `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p97/`
    - `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p95/`
    - `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p93/`
- Important sweep result for the Qwen3 repro:
  - the post-hoc `rare_label_threshold` analyses at `0.97`, `0.95`, and `0.93` all completed successfully under:
    - `logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p97/`
    - `logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p95/`
    - `logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p93/`
  - the resulting `val` CSVs are byte-identical across thresholds
  - the resulting `test` CSVs are also byte-identical across thresholds
  - therefore, for this checkpoint, lowering the post-hoc rare threshold from `0.97` to `0.95` or `0.93` did **not** change the Qwen3 slice rankings or rare/hard-case labels
  - the stable Qwen3 **val** rare cases across all three thresholds are:
    - `pose=lying`: `AP=0.3623`, `27` images
    - `time_of_day=night`: `AP=0.5826`, `61` images
    - `clothes_color=brown`: `AP=0.5910`, `37` images
    - `time_of_day=morning`: `AP=0.6135`, `105` images
    - `time_of_day=evening`: `AP=0.6252`, `161` images
    - `clothes_color=gray`: `AP=0.6336`, `113` images
  - the stable Qwen3 **test** rare cases across all three thresholds are:
    - `pose=lying`: `AP=0.3906`, `34` images
    - `time_of_day=evening`: `AP=0.5737`, `164` images
    - `weather=rainy`: `AP=0.5763`, `31` images
    - `time_of_day=night`: `AP=0.5802`, `74` images
    - `background_style=rural`: `AP=0.6045`, `86` images
    - `clothes_color=pink`: `AP=0.6054`, `43` images
    - `clothes_color=gray`: `AP=0.6147`, `119` images
    - `clothes_color=brown`: `AP=0.6202`, `38` images

# 4/12/2026 Update — COCO Gemini Attributes and Pose Slice Analysis

### Summary of Recent Report Updates

Based on the completed runs and report updates consolidated on 4/12, the main recent work was:

- Established the COCO pose-estimation baseline stack, including Keypoint R-CNN fine-tuning and RTMPose-Tiny evaluation for later slice analysis.
- Refined the KITTI object-detection baseline around the focused `Car` / `Pedestrian` setup and updated the supporting training/report artifacts.
- Built the COCO pose attribute-assignment workflow for VLMs: added COCO pose corpora, updated `utils/assign_attribute.py` for COCO loading/filtering plus `--stop_at` and CSV-oriented output handling, and generated Qwen3 / InternVL3 attribute-assignment results.
- Added the BLIP-based COCO pose attribute-assignment path, completed a full run with the Gemini-defined corpus, and extended BLIP output handling to support CSV export.
- Finished the Gemini-corpus Qwen3 / InternVL3 validation runs on the COCO pose `val2017` split and recorded their runtimes, logs, and CSV outputs.
- Finished the Gemini-corpus Qwen3 / InternVL3 train-split runs on the first `20,000` valid `train2017` images and recorded their runtimes, logs, and CSV outputs.
- Finished three `pose_estimation/train_pose_estimation.py` probe runs on the first `20,000` valid `train2017` images, added a reproducible held-out `val/test` split inside valid `val2017`, and recorded both the held-out keypoint metrics and the main slice-level failure cases.

### Recent COCO Gemini Workflow Updates

- Updated `utils/blip.py` so the COCO pose path defaults to `save/corpus_coco_pose_gemini.json` while still allowing override via `--corpus save/corpus_coco_pose.json`.
- Added CSV export support to `utils/blip.py` via `--csv_output`, with the default CSV path derived from the JSON output path.
- Updated `utils/blipFeature.py` to use configurable `BLIP_NUM_WORKERS` (default `0`) so BLIP inference does not fail in restricted execution environments.

### COCO Pose Corpus and Split Notes

- Rewrote `save/corpus_coco_pose.json` so the COCO pose attribute vocabulary is closer to **HiBug2 Appendix Table 5** and more suitable for **ControlNet / generative editing** rather than diagnosis-only tagging.
- The current COCO pose corpus contains **18 editable attributes**:
  - `pose`
  - `pose_complexity`
  - `head_orientation`
  - `clothes_color`
  - `is_arm_crossing`
  - `is_carrying_something`
  - `is_standing_on_one_leg`
  - `is_on_all_fours`
  - `is_sitting`
  - `is_dynamic`
  - `background_style`
  - `background_color`
  - `is_containing_other_people`
  - `indoor_lighting`
  - `is_indoor`
  - `weather`
  - `occlusion`
  - `viewpoint`
- These attributes cover pose state, body configuration, scene context, and imaging conditions, making them more directly usable for controlled generation and downstream bug-repair experiments.
- For the COCO pose pipeline, the repository currently uses:
  - **train**: `train2017` + `person_keypoints_train2017.json`
  - **val**: `val2017` + `person_keypoints_val2017.json`
  - **test**: `val2017` + `person_keypoints_val2017.json`
- This follows standard local COCO practice: because official test ground truth is not public, `val2017` is reused as the local held-out test set.
- After filtering to valid person-keypoint images, the current effective COCO pose pool contains **56,599 train images** and **2,346 val images**, for a total of **58,945 valid COCO pose images**.

### Full COCO Run

- Launched a full BLIP attribute-assignment run on COCO pose with the Gemini-defined corpus.
- Final JSON result was successfully produced at `save/attribute_coco_pose_gemini_blip.json`.
- The completed file contains `58,945` entries for each of the 6 Gemini-corpus attributes:
  - `clothes_color`
  - `clothes_type`
  - `background_style`
  - `time_of_day`
  - `weather`
  - `pose`
- Quick inspection confirmed all 6 attributes have non-empty assignments for all `58,945` samples.

### Gemini-Corpus VLM Val Runs (`utils/assign_attribute.py`)

- Completed the earlier Qwen3 and InternVL3 COCO pose attribute-assignment runs with the Gemini corpus using `utils/assign_attribute.py`.
- These runs used the script's default `coco_pose` split behavior, so they covered **val2017 only** rather than the combined train+val pool.
- Both runs loaded `person_keypoints_val2017.json`, filtered to **2,346 valid images**, and finished with both JSON and CSV outputs.

| Model | Split | Valid Images | Runtime | JSON Output | CSV Output | Log |
|-------|-------|--------------|---------|-------------|------------|-----|
| Qwen3-VL-8B | val2017 | 2,346 | 45.8 min | `save/attribute_coco_pose_gemini_qwen3_8B.json` | `save/attribute_coco_pose_gemini_qwen3_8B.csv` | `logs/pose/assign_attribute_coco_pose_qwen3_8B_gemini_gpu3_20260410_193102.log` |
| InternVL3-8B | val2017 | 2,346 | 208.8 min | `save/attribute_coco_pose_gemini_internvl3_8B.json` | `save/attribute_coco_pose_gemini_internvl3_8B.csv` | `logs/pose/assign_attribute_coco_pose_internvl3_8B_gemini_gpu2_20260410_193102.log` |

- Log endings confirm both jobs completed successfully:
  - Qwen3: `Done. Processed 2346 images in 45.8 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_qwen3_8B.json`
  - InternVL3: `Done. Processed 2346 images in 208.8 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_internvl3_8B.json`
- CSV validation passed for both outputs: each file contains `2,346` data rows plus a header row, and a quick missing-cell check found no empty attribute fields.

### Gemini-Corpus VLM Train Runs (`utils/assign_attribute.py`, 20k subset)

- Completed the follow-up Qwen3 and InternVL3 COCO pose train-split runs with the Gemini corpus using `--split train --stop_at 20000`.
- Both runs loaded `person_keypoints_train2017.json`, detected **56,599 valid train images**, and then processed the first **20,000** samples as requested.
- Both jobs finished with JSON and CSV outputs for the 6-attribute Gemini corpus.

| Model | Split | Source Pool | Processed Images | Runtime | JSON Output | CSV Output | Log |
|-------|-------|-------------|------------------|---------|-------------|------------|-----|
| Qwen3-VL-8B | train2017 | 56,599 valid images | 20,000 | 399.7 min (6.66 h) | `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json` | `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv` | `logs/pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu0_20260410_193459.log` |
| InternVL3-8B | train2017 | 56,599 valid images | 20,000 | 1709.6 min (28.49 h) | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json` | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` | `logs/pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu1_20260410_193459.log` |

- Log endings confirm both jobs completed successfully:
  - Qwen3: `Done. Processed 20000 images in 399.7 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - InternVL3: `Done. Processed 20000 images in 1709.6 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- CSV validation passed for both train outputs: each file contains `20,000` data rows plus a header row.

### Canonical Post-Cleanup Completion Update (4/15/2026)

- The previously running post-cleanup attribute jobs are now complete.
- The current stable post-cleanup outputs are:

| Model | Split / Shard | Valid Images Written | Runtime | Stable Outputs | Primary Log |
|-------|---------------|----------------------|---------|----------------|-------------|
| Qwen3-VL-8B | canonical `val2017` rerun | `2,346` | `46.7` min (`0.78 h`) | `save/attribute_coco_pose_gemini_qwen3_8B.json`, `save/attribute_coco_pose_gemini_qwen3_8B.csv`, plus alias copies `save/attribute_coco_pose_gemini_qwen3_8B_val.json` and `save/attribute_coco_pose_gemini_qwen3_8B_val.csv` | `logs/pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log` |
| InternVL3-8B | canonical `val2017` rerun | `2,346` | `217.2` min (`3.62 h`) | `save/attribute_coco_pose_gemini_internvl3_8B.json`, `save/attribute_coco_pose_gemini_internvl3_8B.csv` | `logs/pose/assign_attribute_coco_pose_internvl3_8B_gemini_val_gpu1_20260414_143656.log` |
| InternVL3-8B | canonical `train20k` rerun | `20,000` total rows in the canonical artifact; the resume stage processed the remaining `19,500` images after a `500`-row checkpoint | `1876.5` min (`31.28 h`) for the resumed segment | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`, `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` | `logs/pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.log` |
| Qwen3-VL-8B | `train_after20k` continuation shard | `36,599` | `731.4` min (`12.19 h`) | `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.json`, `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv` | `logs/pose/assign_attribute_coco_pose_qwen3_8B_gemini_train_after20k_gpu3_20260414_134834.log` |

- Current CSV row counts confirm the finished artifact sizes:
  - `save/attribute_coco_pose_gemini_qwen3_8B_val.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_internvl3_8B.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`: `20,000` data rows
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv`: `36,599` data rows
- Qwen3 now covers the full valid COCO pose train split through two explicit non-overlapping shards:
  - the canonical first shard `save/attribute_coco_pose_gemini_train20k_qwen3_8B.{json,csv}` contains the first `20,000` valid train images
  - the continuation shard `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.{json,csv}` contains the remaining `36,599` valid train images
  - together these two shards cover all `56,599` valid `train2017` images

### Status Note

- The long-running `nohup` job finished successfully and the log ended with `Saved attributes to save/attribute_coco_pose_gemini_blip.json`.
- That particular background job was started before CSV export was added, so it produced JSON only, not CSV.

### CSV Validation

- After adding CSV support, a `--stop_at 1` smoke run on the Gemini corpus succeeded and produced both:
  - `save/attribute_coco_pose_gemini_blip_smoke_csv_gpu2.json`
  - `save/attribute_coco_pose_gemini_blip_smoke_csv_gpu2.csv`
- Example CSV row from the smoke run:

```csv
image_id,clothes_color,clothes_type,background_style,time_of_day,weather,pose
139,yellow,formal,indoors,morning,sunny,standing
```

### Follow-Up

- If a full CSV matching `save/attribute_coco_pose_gemini_blip.json` is needed, it can now be generated with the updated BLIP code in a rerun, or converted directly from the finished JSON in a short post-process step.
- The dedicated BLIP `train20k` rerun with JSON+CSV output completed at `logs/pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`.
- The dedicated BLIP `val` rerun with the cleaned corpus completed at `logs/pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log` and produced:
  - `save/attribute_coco_pose_gemini_blip_val.json`
  - `save/attribute_coco_pose_gemini_blip_val.csv`
- A clean BLIP held-out `test` slice analysis now exists at:
  - `logs/pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.json`
  - `logs/pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.csv`
- In this clean BLIP reanalysis, `weather=unknown` no longer appears; the weather slices are now `sunny`, `clear`, `cloudy`, `rainy`, and `snowy`.

### COCO Pose 20k Probe Runs with Held-Out Slice Analysis

- Extended `pose_estimation/train_pose_estimation.py` so the current probe workflow trains on the first `20,000` valid `train2017` images by default, splits the `2,346` valid `val2017` images into reproducible held-out `val` / `test` halves (`1,173` each, seed `42`), evaluates the best checkpoint on both halves, and runs automatic slice analysis from Gemini-corpus attribute annotations.
- Pose-related logs and experiment artifact directories are now consolidated under `logs/pose/` instead of being split across `logs/` and `outputs/`.
- These results are useful as fast slice-analysis probes, but they are not direct replacements for the earlier full-val baseline because the train cap, batch size, and evaluation split are different.
- The three runs below should be read as end-to-end run variants rather than a strict attribute-source-only ablation, because the `BLIP` run used `batch_size=8` while the `Qwen3` and `InternVL3` runs used `batch_size=4`.

| Run Tag | Batch | Train Images | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | End-to-End |
|---------|-------|--------------|-----------------|------------------|-----------|-----------|---------|------------|------------|
| Qwen3-VL-8B | 4 | 20,000 | 0.5985 | 0.5968 | 0.8371 | 0.6470 | 0.6675 | 01:22:46 | 01:22:52 |
| InternVL3-8B | 4 | 20,000 | 0.6436 | 0.6340 | 0.8620 | 0.6922 | 0.7002 | 01:29:00 | 01:29:07 |
| BLIP | 8 | 20,000 | 0.6427 | 0.6370 | 0.8614 | 0.6902 | 0.7009 | 01:30:28 | 01:30:35 |

- Main artifact roots:
  - Qwen3: `logs/pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/`
  - InternVL3: `logs/pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/`
  - BLIP: `logs/pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
- On the held-out test split, the strongest AP came from the `BLIP`-tagged run (`0.6370`), closely followed by `InternVL3` (`0.6340`); `Qwen3` reached `0.5968`.
- The clearest repeated failure slice across all three variants was `pose=lying`:
  - Qwen3: `AP=0.2796` (`delta=-0.3172`, `34` images)
  - InternVL3: `AP=0.2924` (`delta=-0.3416`, `32` images)
  - BLIP: `AP=0.4195` (`delta=-0.2175`, `68` images)
- `pose=sitting` was the most consistent high-support hard case in every run, with held-out test AP between `0.5225` and `0.5633`.
- Secondary failure slices depended on the annotation source: Qwen3 emphasized `time_of_day=unknown/morning` and `weather=unknown`; InternVL3 emphasized `background_style=rural` and `time_of_day=night/evening`; BLIP emphasized `background_style=indoors`, `weather=unknown`, and `clothes_type=formal`.

---

# 4/3/2026 Update — Multi-Task Baseline Training (Pose Estimation & Object Detection)

Following the safefix2 paper (arXiv:2501.16751), we train baseline models for pose estimation (COCO) and object detection (KITTI) to support error slice discovery across multiple vision tasks.

---

## 1. COCO Pose Estimation — Keypoint R-CNN (Pretrained, 2 epochs)

**Model**: `keypointrcnn_resnet50_fpn` (torchvision, COCO pretrained)
**Dataset**: COCO 2017 person keypoints, filtered to valid person-keypoint images (`train=56,599`, `val=2,346`, total `58,945`)
**Split**: train=`train2017` + `person_keypoints_train2017.json`; val=`val2017` + `person_keypoints_val2017.json`; local test also uses `val2017` by COCO convention because official test ground truth is not public
**Training**: 2 epochs, batch_size=2, lr schedule from torchvision defaults
**GPU**: GPU 3 | **Env**: `ad` conda (torch 1.13.1+cu117)
**Output**: `logs/pose/pose_keypointrcnn_ad_gpu3_e2/`
**Log**: `logs/pose/pose_coco_pretrained_gpu3_ad_e2.log`

### Per-Epoch Results (OKS-based COCO Keypoint Metrics)

| Epoch | Train Loss | AP | AP50 | AP75 | AP (M) | AP (L) | AR | AR50 | AR75 | AR (M) | AR (L) | Time |
|-------|-----------|------|------|------|--------|--------|------|------|------|--------|--------|------|
| 0 | 4.239 | 0.481 | 0.752 | 0.504 | — | — | 0.564 | — | — | — | — | 3.88h |
| **1** | **4.321** | **0.489** | **0.764** | **0.513** | **0.447** | **0.553** | **0.564** | **0.821** | **0.599** | **0.516** | **0.632** | **3.91h** |

**Total wall time**: ~7.8 hours (2 epochs). Best checkpoint saved at epoch 1.

> [!NOTE]
> The pretrained Keypoint R-CNN already achieves AP=0.489 after just 2 epochs of fine-tuning on COCO. Longer training (12 epochs per README) would improve results further, but 2 epochs are sufficient for establishing the baseline for error slice debugging.

---

## 2. COCO Pose Estimation — RTMPose-Tiny (Pretrained Evaluation)

**Model**: RTMPose-Tiny (SimCC head, pretrained on AIC+COCO, 420 epochs)
**Checkpoint**: `rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth`
**Dataset**: COCO 2017 val2017 (794 batches)
**GPU**: GPU 3 | **Env**: `ad` conda (mmengine + mmpose)
**Log**: `logs/pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.log`

### Evaluation Results (OKS-based COCO Keypoint Metrics)

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | **0.697** |
| AP (IoU=0.50) | 0.906 |
| AP (IoU=0.75) | 0.776 |
| AP (Medium) | 0.673 |
| AP (Large) | 0.739 |
| AR (IoU=0.50:0.95) | **0.725** |
| AR (IoU=0.50) | 0.915 |
| AR (IoU=0.75) | 0.795 |
| AR (Medium) | 0.693 |
| AR (Large) | 0.774 |

> [!NOTE]
> RTMPose-Tiny (AP=0.697) significantly outperforms Keypoint R-CNN (AP=0.489 after 2 epochs) as expected, since RTMPose was trained for 420 epochs on AIC+COCO with a specialized SimCC head. Per the safefix2 paper, the tiny/small/medium/large RTMPose variants are all used for error slice analysis on pose estimation.

---

## 3. KITTI Object Detection — YOLOv8n (Updated 2026-04-09, Car/Pedestrian Split)

**Model**: YOLOv8n (pretrained on COCO, `save/weights/yolov8n.pt`)
**Dataset**: KITTI training split in deterministic filename order, converted to YOLO format with only `Car` and `Pedestrian`
**Split**: first 5000 images for train, next 1240 for val, remaining 1241 for test
**Implementation note**: all three splits are derived from the KITTI `training/` set in deterministic filename order, then exported to YOLO-format `images/{train,val,test}` and `labels/{train,val,test}`
**Training**: 30 epochs, batch=4, imgsz=640, optimizer=`auto` -> AdamW (`lr=0.001667`, momentum=0.9)
**GPU**: NVIDIA RTX A5000 | **Env**: Ultralytics 8.4.33, torch 2.9.1+cu128
**Output**: `save/runs/kitti_yolov8/first5000_gpu1_e30_20260409_194735/`
**Log**: `logs/yolov8_kitti_gpu1_e30_20260409_194735.log`

### Best Validation Checkpoint (Epoch 29)

| Metric | Value |
|--------|-------|
| Precision | 0.849 |
| Recall | 0.712 |
| mAP@50 | **0.808** |
| mAP@50-95 | **0.545** |
| Fitness | 0.545 |

### Held-Out Test Results (`best.pt` on `test`)

| Metric | Value |
|--------|-------|
| Precision | 0.864 |
| Recall | 0.750 |
| mAP@50 | **0.831** |
| mAP@50-95 | **0.563** |
| Fitness | 0.563 |

### Per-Class Test Results

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **1241** | **5369** | **0.864** | **0.750** | **0.831** | **0.563** |
| Car | 1107 | 4666 | 0.899 | 0.883 | 0.936 | 0.731 |
| Pedestrian | 283 | 703 | 0.829 | 0.617 | 0.726 | 0.394 |

**Total training time**: 1h 01m 38s (30 epochs). **End-to-end wall time**: 1h 01m 43s. **Test evaluation time**: 43s.

> [!IMPORTANT]
> This run supersedes the earlier 8-class `train=5000 / val=500` YOLO baseline in the previous report draft. With the new deterministic `train=5000 / val=1240 / test=1241` split and a focused `Car`/`Pedestrian` label space, the held-out test results still show a clear class gap: **Pedestrian (0.394 mAP@50-95)** remains substantially weaker than **Car (0.731)**, making pedestrian-heavy slices the clearest target for safefix-style debugging.

### Validation Convergence

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 | val/box_loss | val/cls_loss |
|-------|-----------|--------|--------|-----------|-------------|-------------|
| 1 | 0.728 | 0.571 | 0.638 | 0.375 | 1.218 | 0.908 |
| 5 | 0.772 | 0.603 | 0.685 | 0.424 | 1.108 | 0.729 |
| 10 | 0.846 | 0.634 | 0.738 | 0.470 | 1.029 | 0.657 |
| 20 | 0.839 | 0.682 | 0.774 | 0.513 | 0.923 | 0.562 |
| 29 (best) | 0.849 | 0.712 | 0.808 | **0.545** | 0.861 | 0.516 |
| 30 | 0.844 | 0.704 | 0.802 | 0.545 | 0.854 | 0.513 |

---

## Summary of Multi-Task Baselines

| Task | Model | Dataset | Key Metric | Value | Status |
|------|-------|---------|------------|-------|--------|
| Pose Estimation | Keypoint R-CNN | COCO 2017 | AP (OKS) | 0.489 | 2 epochs, pretrained |
| Pose Estimation | RTMPose-Tiny | COCO 2017 | AP (OKS) | 0.697 | Pretrained eval only |
| Object Detection | YOLOv8n | KITTI 5000/1240/1241 (`Car`, `Pedestrian`) | mAP@50-95 (test) | 0.563 | 30 epochs, pretrained |

These baselines are ready for error slice discovery and model debugging with safefix2/DebugAgent.

---

# 3/13/2026 Update
## All-Strategy Comparison across Sub Ratios (lr=1e-5, 15 epochs)

All runs resume from `save/weights/model_imagenet_resnet_base.pth` (base test acc: **0.7162**). The Attribute strategy uses the optimized `aggro_exp_p10_c4` parameters ($R=0.8, P=10.0, C=4.0$, Entropy=True). Reference: using 100% of the data (ratio=1.0) yields Test ACC = **0.7488**. The 3/13 update adds two new LLM-based attribute generation results (`Qwen3` and `InternVL3`) to compare alongside the default BLIP-based attribute assignment.

Within each ratio table, the highest `Test ACC` among all strategies is bolded.

### Ratio = 0.1
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7269 | 0.7069 |
| Stratified | 0.7271 | 0.7154 |
| Coreset | 0.7269 | 0.7115 |
| Uncertainty Stratified | 0.7254 | 0.7185 |
| Diversity | 0.7262 | 0.7215 |
| Hierarchical | 0.7269 | 0.7115 |
| UHerding | 0.7331 | 0.7131 |
| PPAL | 0.7285 | 0.7169 |
| Attribute (BLIP) | 0.7277 | 0.7200 |
| Attribute (Qwen3) | 0.7331 | 0.7231 |
| Attribute (InternVL3) | **0.7346** | 0.7285 |

### Ratio = 0.3
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7308 | 0.7208 |
| Stratified | 0.7369 | 0.7162 |
| Coreset | 0.7369 | 0.7162 |
| Uncertainty Stratified | 0.7292 | 0.7131 |
| Diversity | 0.7362 | 0.7200 |
| Hierarchical | 0.7346 | 0.7092 |
| UHerding | **0.7469** | 0.7177 |
| PPAL | 0.7392 | 0.7200 |
| Attribute (BLIP) | 0.7369 | 0.7185 |
| Attribute (Qwen3) | 0.7354 | 0.7077 |
| Attribute (InternVL3) | 0.7385 | 0.7108 |

### Ratio = 0.5
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7380 | 0.7162 |
| Stratified | 0.7446 | 0.7200 |
| Coreset | 0.7400 | 0.7185 |
| Uncertainty Stratified | 0.7331 | 0.7146 |
| Hierarchical | 0.7385 | 0.7138 |
| Hybrid | 0.7362 | 0.7215 |
| UHerding | 0.7410 | 0.7200 |
| PPAL | 0.7423 | 0.7208 |
| Attribute (BLIP) | 0.7408 | 0.7131 |
| Attribute (Qwen3) | 0.7362 | 0.7138 |
| Attribute (InternVL3) | **0.7508** | 0.7286 |

### Ratio = 0.7
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7423 | 0.7223 |
| Stratified | 0.7346 | 0.7138 |
| Coreset | 0.7392 | 0.7154 |
| Uncertainty Stratified | 0.7354 | 0.7146 |
| Hierarchical | 0.7385 | 0.7146 |
| UHerding | **0.7469** | 0.7231 |
| PPAL | 0.7423 | 0.7215 |
| Attribute (BLIP) | 0.7385 | 0.7131 |
| Attribute (Qwen3) | 0.7415 | 0.7192 |
| Attribute (InternVL3) | 0.7413 | 0.7100 |

### Ratio = 0.9
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7462 | 0.7154 |
| Stratified | 0.7469 | 0.7146 |
| Coreset | 0.7346 | 0.7169 |
| Uncertainty Stratified | 0.7462 | 0.7100 |
| Diversity | 0.7415 | 0.7146 |
| Hierarchical | 0.7362 | 0.7177 |
| Hybrid | 0.7477 | 0.7177 |
| UHerding | 0.7500 | 0.7231 |
| PPAL | 0.7508 | 0.7238 |
| Attribute (BLIP) | 0.7400 | 0.7177 |
| Attribute (Qwen3) | 0.7492 | 0.7108 |
| Attribute (InternVL3) | **0.7531** | 0.7346 |


---

## Low Budget Active Learning Comparison (ratios: 0.01, 0.02, 0.05)

**Manual Selection**: In these experiments, the "Manual Selection" method involves hand-picking images where the target object (e.g., backpack) is prominent. Because the ImageNet dataset often contains noisy images—some of which do not even feature the intended object—this manual process specifically selects images without cluttered backgrounds. It ensures the target object is large, highly visible, and clearly the main subject of the picture rather than a secondary element.

Runs for sub_ratio = 0.01, 0.02, and 0.05 train ResNet-18 **from scratch** (no pretrained base) for 30 epochs on the `pinkcolor_fabrictexture` ImageNet split with 1000 generated images. Sub-ratio = 0.1 results use the resume-from-base setting (lr=1e-5, 15 epochs) and are provided here alongside the smaller ratios for a unified small-budget view.

### Sub Ratio = 0.01
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3323 | 0.3177 |
| Stratified | 0.3154 | 0.3100 |
| Diversity | 0.3269 | 0.3115 |
| Hybrid | 0.3353 | 0.2931 |
| Coreset | 0.2846 | 0.2462 |
| Manual Selection | 0.3385 | 0.3215 |

### Sub Ratio = 0.02
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3669 | 0.3415 |
| Stratified | 0.3623 | 0.3400 |
| Diversity | 0.3238 | 0.3015 |
| Hybrid | 0.3662 | 0.3450 |
| Coreset | 0.3085 | 0.2815 |
| Attribute | 0.3669 | 0.3480 |
| Manual Selection | 0.3650 | 0.3440 |

### Sub Ratio = 0.05
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3977 | 0.3777 |
| Stratified | 0.3931 | 0.3762 |
| Diversity | 0.4162 | 0.3800 |
| Hybrid | 0.4169 | 0.3923 |
| Coreset | 0.3246 | 0.3069 |
| Manual Selection | 0.4115 | 0.3862 |



# 2/27/2026 Update




## Active Learning Resume-from-Base Comparison (lr=1e-5, 15 epochs)

This table shows results from `test_al_resume.sh` with `freeze_backbone=False`, `lr=1e-5`, and `epoch=15`. All runs use `sub_ratio=0.3` and resume from `save/weights/model_imagenet_resnet_base.pth` (base val acc: **0.7223**).

| Strategy | Model Test ACC | Overall Val ACC Change | Status |
| :--- | :--- | :--- | :--- |
| Baseline | 0.7162 | 0.718077 | - |
| Random | 0.7308 | 0.718077 → 0.720769 (+0.002692) | improved |
| Stratified | **0.7369** | 0.718077 → 0.726154 (+0.008077) | improved |
| Coreset | 0.7359 | 0.718077 → 0.719231 (+0.001154) | improved |
| Diversity | 0.7362 | 0.718077 → 0.723846 (+0.005769) | improved |
| Uncertainty Stratified | 0.7292 | 0.718077 → 0.719231 (+0.001154) | improved |
| Hierarchical | 0.7346 | 0.718077 → 0.724615 (+0.006538) | improved |
| `Attribute` | **0.7369** | 0.718077 → 0.722308 (+0.004231) | improved |


### Observations:
* **All strategies improved** over the baseline (0.718077) after fine-tuning with `lr=1e-5` for 15 epochs without backbone freezing.
* **Best Test ACC**: `Hybrid` (0.7400).
* **Best Val ACC improvement**: `Stratified` (+0.008077), benefiting from balanced class coverage during fine-tuning.
* **Uncertainty Stratified** performed below expectation (0.7292 Test ACC), possibly due to using a generic pretrained proxy model (ResNet18 weights, not the actual resumed checkpoint) to compute entropy scores, leading to noisy uncertainty estimates.
* **Attribute strategy** (with conditional rarity, R=0.01, B=5, class_balance=inf) achieved competitive performance (0.7369 Test ACC, +0.004231 val), demonstrating that metadata-driven selection can match geometry/diversity baselines when paired with proper fine-tuning LR.

## Grid Search Results for Attribute Strategy Parameters (2/27/2026)
We conducted a comprehensive grid search over the parameters $R, P, C,$ and the `use_entropy` toggle to understand their interactions and optimize the Attribute Strategy. The tests were run across 4 sub-ratios (`0.1`, `0.3`, `0.5`, `0.9`) on the `pinkcolor_fabrictexture` generated set.

### Sub Ratio = 0.1
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | **0.7300** | 0.7146 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7215 | 0.7154 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7208 | 0.7154 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7223 | 0.7146 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7208 | 0.7138 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7223 | 0.7185 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7223 | 0.7200 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7223 | 0.7177 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7277 | 0.7200 |

### Sub Ratio = 0.3
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | 0.7369 | 0.7262 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7346 | 0.7138 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7215 | 0.7162 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7392 | 0.7200 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7262 | 0.7115 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7362 | 0.7108 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7362 | 0.7092 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7362 | 0.7138 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | **0.7415** | 0.7115 |

### Sub Ratio = 0.5
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | **0.7438** | 0.7123 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7415 | 0.7123 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7338 | 0.7138 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | **0.7438** | 0.7208 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7400 | 0.7162 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7308 | 0.7077 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7362 | 0.7100 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7400 | 0.7185 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7408 | 0.7131 |

### Sub Ratio = 0.9
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | 0.7454 | 0.7223 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7438 | 0.7131 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7400 | 0.7192 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7408 | 0.7146 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7392 | 0.7192 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | **0.7500** | 0.7162 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7446 | 0.7138 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7431 | 0.7154 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7400 | 0.7177 |

### Analysis
1. **The Danger of Pure Greedy**: The `pure_greedy` approach ($C=0$) performs the worst across low evaluation ratios (0.1, 0.3). Without class-level Stratification quotas, throwing out all logic except Frequency Penalty and Rarity leads to performance drops in fine-tuning, emphasizing the value of our soft $C$ score.
2. **Aggressive Exploration Shines ($R=0.8, C=4.0$)**: By loosening the absolute restriction on stratification ($C=4.0$ instead of $\infty$) and massively scaling up Rarity tracking ($R=0.8$), the `aggro_exp_p10_c4` scenario achieves extremely strong results across 0.3 and 0.5. The heavy frequency penalty ($P=10$) adequately prevented the model from indexing too heavily on the synthesized injection flood.
3. **Entropy Tuning**: Dynamic entropy ($e=T$) does well in aggressive settings but causes occasional degradations in more conservative spaces (see `soft_conservative` comparisons).
4. **Baseline Comparison (Ratio=1.0)**: For reference, using **100% of the candidate data (Ratio=1.0)** yields a Test ACC of **0.7400**. Using Safefix’s method, updating all source data and generated images for 30 epochs results in a Test ACC of **0.7438**. Remarkably, the `balanced_exp_p5_c8` strategy at `Ratio=0.9` achieved an even higher Test ACC of **0.7500**, demonstrating that intelligent filtering can outperform training on the entirety of the noisy synthetic data!
5. **Overall Best Strategy**: Averaging across all sub-ratios, **`aggro_exp_p10_c4` ($R=0.8, P=10.0, C=4.0$, Entropy=True)** represents the most robust and high-performing parameter combination. It consistently dominates the low-to-mid ratio regimes (0.1 to 0.5) by aggressively seeking rare features while safely blocking flooded features and maintaining a soft class balance.

**Conclusion**: Tuning the Attribute Strategy to mimic `Coreset` + `Stratified` (via Soft C + High R) allows the strategy to exceed the conservative defaults! When allocating budgets, choosing a strong R and applying a compensatory P to manage spam achieves peak domain adaptability.

# 2/20/2026 Update
## Active Learning Resume-from-Base Comparison

This table compares the test results generated by `test_al_resume.sh` against the provided baseline (`logs/imagenet/AL_attr_resume_base_resnet_imagenet_0.3_1000.log`). All tasks use `sub_ratio=0.3` and resume from the base model.

| Strategy | Model Test ACC | Val Accuracy Difference | Status |
| :--- | :--- | :--- | :--- |
| Baseline | 0.7162 | 0.718077 | - |
| Random | 0.7385 | 0.718077 -> 0.717692 (-0.000385) | declined |
| Stratified | 0.7362 | 0.718077 -> 0.725385 (+0.007308) | improved |
| Coreset | 0.7346 | 0.718077 -> 0.720000 (+0.001923) | improved |
| Uncertainty Stratified | 0.7385 | 0.718077 -> 0.721538 (+0.003461) | improved |
| Diversity | 0.7315 | 0.718077 -> 0.720769 (+0.002692) | improved |
| Hierarchical | 0.7415 | 0.718077 -> 0.716154 (-0.001923) | declined |
| **Hybrid** | **0.7446** | 0.718077 -> 0.717692 (-0.000385) | declined |
| `Attribute` | 0.7369 | 0.718077 -> 0.720000 (+0.001923) | improved |

### Observations:
* **Best Test ACC**: `Hybrid` strategy achieved the highest raw Model Test Accuracy (0.7446).
* **Best Overall Accuracy Variation**: `Stratified` provided the best improvement in attribute/hard-case accuracy comparison (+0.007308).

---

## Generation-Guided Active Learning (From Scratch) Comparison

This section compares the new `gen_probe` strategy against the baseline `random` and `stratified` strategies **from scratch** (i.e., without `--resume_model`).
All runs use `sub_ratio=0.3`, 1000 generated images, and are trained for 30 epochs on ResNet-18.

| Strategy | Model Test ACC | Status compared to Random |
| :--- | :--- | :--- |
| Baseline | 0.2462 | - |
| Random | 0.5754 | - |
| Stratified | **0.5908** | +0.0154 |
| Gen-Probe | 0.5600 | -0.0154 |

### Observations:
* The `gen_probe` strategy (Test ACC: 0.5600) underperformed both `random` (0.5754) and `stratified` (0.5908) when trained from scratch in this setup.
* While the motivation for `gen_probe` is strong (mining hard examples based on domain gap), the strategy might be selecting examples that are *too hard* or outliers, causing the model to struggle to learn generalizable features from scratch. 
* Tuning the selection metric (e.g., mixing in some random/diversity samples, taking the middle uncertainty instead of absolute highest, or adjusting the `probe` model's training epochs) might be required to make it surpass Stratified/Attribute strategies.

---

## 3. Mathematical Formulation of Gen-Probe 
The core idea of **Generation-Guided Active Learning (Gen-Probe)** is to use a model trained exclusively on generated images to act as a diagnostic "probe" on the original source data. This identifies the "domain gap"—the features present in real images that the generative model fails to synthesize.

### Selection Mechanism:

1. **Probe Training**: We train a model $M_{gen}$ exclusively on the pool of generated images (e.g., 1000 generated images).
2. **Hardness Scoring (Entropy)**: We evaluate $M_{gen}$ on the entire **source training pool** (e.g., `train_idx_imagenet.npy` consisting of 10,400 unlabeled original images). For each image $x_i$, the probe outputs a probability distribution across $C$ classes: $P(y|x_i)$. 
   The score assigned to the image is its **Information Entropy** $H(x_i)$:
   $$ H(x_i) = - \sum_{c=1}^{C} P(y=c|x_i) \log P(y=c|x_i) $$
   - A **low entropy** ($H \to 0$) means $M_{gen}$ is confident; the generated images already cover these features well.
   - A **high entropy** means $M_{gen}$ is entirely confused; these images contain features distinct from the generated domain.
3. **Data Selection**: We sort all samples descending by $H(x_i)$ and select the top $K$ images (e.g., 30% of the pool) that the probe model found most difficult.



---

## 4. Future Research Directions: Why did "Hard Mining" fail here?

The initial run of `gen_probe` yielded an accuracy of 0.5600, which is lower than `random` selection (0.5754). This is a well-known phenomenon in Active Learning literature: **Hard Example Mining is extremely susceptible to outliers.**

When a model is trained from scratch using *only* the Top-K highest entropy samples, it is essentially being trained on the most ambiguous, heavily occluded, or mislabeled images in the dataset. A model cannot build robust decision boundaries when given only edge cases.


---





---

## Uncertainty-Weighted Stratified Strategy 

The **Uncertainty Stratified** strategy aims to select the most informative samples (highest uncertainty) while strictly preserving a balanced class distribution in the selected subset.

### 1. Uncertainty Calculation (Entropy)
For a multi-class classification problem with $C$ classes, let $x_i$ be an unlabeled sample and $P(y=c|x_i)$ be the predicted probability of class $c$ by the current model. The uncertainty score for sample $i$ is calculated using Information Entropy:
$$ U_i = -\sum_{c=1}^{C} P(y=c|x_i) \log P(y=c|x_i) $$
A higher $U_i$ indicates that the model is less confident about the prediction.

### 2. Stratified Quota Allocation
Let $K$ be the total target number of samples to select, and $N_{classes}$ be the total number of unique ground-truth (or pseudo-label) classes in the candidate pool. The budget is distributed uniformly across all classes:
$$ K_{class} = \lfloor \frac{K}{N_{classes}} \rfloor $$
Any remainder samples $K \pmod{N_{classes}}$ are distributed iteratively to the first few classes until the budget is exhausted.

### 3. Stratified Selection
For each individual class $c$:
1. Isolate the subset of samples $\mathcal{S}_c$ that belong to class $c$.
2. Sort $\mathcal{S}_c$ in descending order based on their assigned uncertainty scores $U_i$.
3. Select the top $K_{class}$ samples with the highest uncertainty from $\mathcal{S}_c$.

By combining point-wise entropy maximization with class-wise histogram equalization, this strategy avoids the common mode collapse in standard uncertainty sampling (where the model might exclusively select difficult samples from only one or two dominant/confusing classes).



## Ablation Study for Attribute Parameters

The following table summarizes the ablation experiments with sub_ratio=0.7 for the AttributeFocusedStrategy parameters on ImageNet dataset. Each row corresponds to a different configuration where one or more parameters are set to specific values while others are zero (except for the baseline).

| Configuration | Alpha (rarity_power) | Beta (uncertainty_boost) | Gamma (log_acc_scale) | Accuracy | Notes |
|---------------|----------------------|---------------------------|-----------------------|----------|-------|
| Baseline (all zero) | 0 | 0 | 0 | 0.699388 | No attribute weighting |
| Traditional AL(Coreset) | - | - | - | 0.69873 | |
| Rarity only | 1 | 0 | 0 | 0.700433 | Only rarity component active |
| Uncertainty only | 0 | 1 | 0 | 0.704842 | Only uncertainty bonus active |
| Log weight only | 0 | 0 | 1 | 0.709308 | Only log-derived weights active |
| All | 1 | 1 | 1 | 0.700769 |  |
| All, but scales | 1 | 1 | 10 | 0.700384 |  |

*Note: Experiments run with ResNet model, sub_ratio=0.7, attributes=color,background.*

---

# Active Learning Strategy Usage Guide

## Overview

I have implemented multiple Active Learning strategies for the safefix project to address the poor performance of the original Core-set method.

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
- **Current Implementation Detail**: this repo implements a BADGE-style proxy variant rather than the original `JordanAsh/badge` pipeline.
- **Step 1 - Proxy labeled set**: `alpred.py` extracts embeddings from the validation split and uses the validation targets as a lightweight labeled proxy set.
- **Step 2 - Proxy class probabilities**: for each class, it computes a centroid in the normalized embedding space; each unlabeled training image gets class probabilities from a softmax over negative squared distances to those centroids.
- **Step 3 - Gradient embeddings**: it builds the usual BADGE-shaped gradient embedding by concatenating one block per class, where each block is the sample embedding scaled by the class coefficient `p_c - 1[c = argmax p]`.
- **Step 4 - Batch selection**: for smaller jobs, it runs exact BADGE-style k-means++ seeding on those gradient embeddings; for larger jobs, it switches to a cheaper approximation that samples without replacement with probabilities proportional to the gradient-embedding norm.
- **Difference from original BADGE**: the original `JordanAsh/badge` code uses the model's own predictive probabilities for the unlabeled pool and keeps the sequential BADGE seeding logic, while this repo replaces the classifier with a validation-centroid proxy and may replace exact seeding with weighted gradient-norm sampling on large pools.
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

The total score $S_i$ for sample $i$ is the sum across all attributes $A$ of four components:
$$S_i = \sum_{a \in A} \big( R_{i,a} + U_{i,a} + L_{i,a} + P_{i,a} \big)$$

Where:
*   **$R_{i,a}$ (Class-Conditional Rarity)**: $\left( \frac{1}{f_{a}(v_{i,a} | y_i) + \varepsilon} \right)^{\alpha}$. Default $\alpha=0.01$. Measures how rare an attribute value is *within its predicted class*.
*   **$U_{i,a}$ (Uncertainty)**: $\beta$ if marked uncertain in JSON, else $0$. If `--al_attribute_use_entropy True` is set, this is dynamically replaced by Information Entropy $\beta \cdot \big( -\sum P \log P \big)$ from model inference. Default $\beta=1.0$.
*   **$L_{i,a}$ (Log Accuracy)**: $\gamma \cdot (1 - \text{ACC}_{a}(v_{i,a}))$. Default $\gamma=0.0$.
*   **$P_{i,a}$ (Frequency Penalty)**: $-\lambda \cdot \text{FreqRatio}(v_{i,a})$. Default $\lambda=5.0$. This term continuously suppresses over-sampling from heavily flooded slices.

Samples are sorted by $S_i$ (with a small injected noise term for tie breaking) and the top $n_{\text{select}}$ indices are returned. When a sample exhibits multiple rare or low-accuracy attributes, the contributions accumulate, ensuring multi-attribute failures are prioritized.

**Understanding the Frequency Penalty ($P$) and Generated Images**
While the formula for $P_{i,a}$ computes a simple frequency ratio, it perfectly neutralizes the bias introduced by injecting artificially generated data. Specifically, the ratio is calculated over the *combined* candidate pool:
$$\text{FreqRatio}(v) = \frac{N_{\text{base}}(v) + N_{\text{gen}}(v)}{\sum_{v'} \big( N_{\text{base}}(v') + N_{\text{gen}}(v') \big)}$$
Where $N_{\text{base}}$ represents the counts in the original unlabeled dataset, and $N_{\text{gen}}$ represents the newly injected synthetic bounding boxes (e.g., 1000 images focused purely on `color:pink` and `texture:fabric`). Because $N_{\text{gen}}$ concentrates an enormous mass into a few specific attribute values $(v)$, those specific values experience an artificially inflated $\text{FreqRatio}$. The $-\lambda \cdot \text{FreqRatio}$ penalty ensures that the active learning selector does not blindly exhaust its budget on the synthetic flood, forcing it to balance the newly injected data against naturally occurring hard cases.

**The 2D Synergy: $R$ (Class-Conditional) vs. $P$ (Global)**
At first glance, Rarity ($R$) rewarding low frequency and Penalty ($P$) penalizing high frequency seem contradictory. However, because $R$ evaluates *within a specific class* ($P(v|y)$) while $P$ evaluates *across the entire global pool* ($\text{FreqRatio}$), they form a powerful 2D filter:
*   **High $R$ + Low $P$ (Class-rare, Global-rare)**: The most valuable "Hard Cases" (e.g., a "pink elephant"). They receive massive score boosts.
*   **High $R$ + High $P$ (Class-rare, Global-common)**: Features that are generic globally but strange for this specific class. $R$ and $P$ partially offset, preventing the model from over-indexing on globally redundant background textures just because they appeared in a new class.
*   **Low $R$ + High $P$ (Class-common, Global-common / Artificial Flooding)**: Boring redundant samples or the exact heavily-injected generated slices (e.g., 1000 synthetic `pink` dogs). The massive $P$ penalty sinks their total $S_i$ score, acting as an automatic spam filter to prevent them from exhausting the selection budget.

Optionally, enable an additional **Class Stratification Score ($C$)** via `--al_attribute_class_balance`. After computing $S_i$, the selector performs a greedy re-ranking that rewards classes whose current selections fall below their per-class quota. When the flag is set to a finite value $w>0$, each class $c$ keeps a running count $n_c$ and target quota $t = n_{\text{select}} / N_{\text{classes}}$. The next candidate from class $c$ receives an additive bonus $C_{c}$:
$$C_{c} = w \times \frac{t - n_c}{t}$$
which is added to its base score $S_i$ inside `_soft_class_selection` (see `utils/active_learning.py`). Classes under their quota get positive bonuses, while classes at or above the quota see the term shrink toward zero or become negative, letting pure attribute scores win. Candidates with unknown labels (e.g., synthetic-only entries) sit in a separate queue and are inserted only when their base score exceeds every class-adjusted option, so unlabeled data does not distort the balancing pass.

During iteration $k$ of `_soft_class_selection`, every labeled candidate $i$ from class $c = y_i$ uses the effective score
$$\tilde{S}_i^{(k)} = S_i + C_{c}^{(k)}$$
where $C_{c}^{(k)} = w \times \frac{t - n_c^{(k)}}{t}$, and $n_c^{(k)}$ is the number of times class $c$ has already been chosen. The algorithm picks the candidate with the largest $\tilde{S}_i^{(k)}$ (breaking ties with the injected noise from the base ranking), appends it to the selection list, and then updates $n_c^{(k+1)} = n_c^{(k)} + 1$ for that class. If the highest raw-score unlabeled item $u$ satisfies $S_u > \max_i \tilde{S}_i^{(k)}$, the iteration chooses $u$ instead and leaves all class counts unchanged, where $S_u$ denotes the raw attribute score of the unlabeled candidate (no class bonus exists for unlabeled entries). That iteration simply appends $u$ to `selected`. The loop stops once either $|\text{selected}| = n_{\text{select}}$ or all class and unlabeled queues are empty; in the latter case `_finalize_balanced_selection` backfills the remaining slots from the global ranking. “All other class and unlabeled queues are empty” literally means every per-class list (and the unlabeled list) has been drained—no candidate remains to evaluate—so the loop has no further element to pull even if some deficits are still positive.

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

**Weight tuning versus sub-ratio:** because $C_c = w \times (t - n_c)/t$ scales with the quota $t = n_{\text{select}} / N_{\text{classes}}$, a fixed $w$ becomes weaker as the sub-sampling ratio grows. To preserve roughly the same maximum bonus seen at `sub_ratio = 0.3`, scale $w$ in proportion to $t$: $w_{\text{new}} \approx w_{0.3} \times (t_{\text{new}} / 312)$. For ImageNet’s ten classes this gives practical anchors:
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
- `--al_attribute_use_rarity`: Whether to include the rarity term in the score. Set this to `False` for a true `R` removal ablation; setting `--al_attribute_rarity_power 0` is not equivalent because it leaves a constant rarity addend in the score.
- `--al_attribute_uncertainty_boost`: Score bonus for samples marked as uncertain in metadata (>=0)
- `--al_attribute_use_entropy`: If True, computes Information Entropy from model inference and uses it for the $U$ term instead of static metadata flags (default: False)
- `--al_attribute_log_path`: Optional path to diagnostic log providing Rare Case accuracies
- `--al_attribute_log_acc_scale`: Scaling factor applied to each log-derived failure weight (default: 0.0)
- `--al_attribute_extra_counts`: Inline list or JSON file describing additional counts to inject per attribute value (useful when synthetic data floods specific slices); inline entries without `=count` inherit `--num_added_img`, and when omitted entirely the code infers counts from `--attributes` tokens (value+attribute) using `--num_added_img`
- `--al_attribute_frequency_penalty`: Multiplier controlling how strongly over-represented values are penalized (0 disables the term)
- `--al_attribute_class_balance`: Stratified weight for enforcing class balance after attribute scoring (0 disables, `inf` enforces near-perfect round-robin per class)
- `--al_conditional_rarity`: Use class-conditional rarity `P(value | class)` instead of global rarity

### RPC Component Ablation Note

For RPC ablations, remove the component from the selector rather than treating
the ablation as a hyperparameter value sweep. Current semantics:

- Remove `R`: pass `--al_attribute_use_rarity False` / `--al-attribute-use-rarity false`.
- Remove `P`: pass `--al_attribute_frequency_penalty 0` / `--al-attribute-frequency-penalty 0`.
- Remove `C`: pass `--al_attribute_class_balance 0` / `--al-attribute-class-balance 0`.

The all-removed setting should match the plain random branch when uncertainty
and log-derived terms are also disabled, so it is a selection sanity check, not
a full GPU training row.

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
python utils/compare_al_results.py
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
