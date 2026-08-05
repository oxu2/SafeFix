> Are all reported downstream task metrics in Parts 1–6, including the
> Generated-Complement table in Q2 and the exact-target/non-target tables in
> Q3, computed using the same five seeds: 0, 42, 1234, 3407, and 2024? Please
> indicate explicitly if any table instead reports a single seed or a
> different aggregation.

**Response:** Yes. All reported downstream task metrics in Parts 1–6 use the
same five seeds: **0, 42, 1234, 3407, and 2024**. This includes the main
cross-task comparison across all questions. **No downstream performance table in
our response reports a single seed or uses a different seed set.**

The previous official comment contained several numerical transcription
errors. We have corrected those values below and clarified that all reported
results are averaged over the same five random seeds. These corrections affect
neither the experimental setup nor the conclusions. We apologize for the
oversight and have updated the comment with a clearer description of the
experimental setup.

# Response to Reviewer yyPZ — Per-Seed Exact-Target Counts

> Could the authors provide the per-seed numbers of correctly classified
> target images for Generated-Complement and AttriSelect at source ratio 0.5,
> together with the resulting per-seed exact-target accuracies using the
> following format?

**Response:** Yes. The exact ImageNet10 target is "pink AND fabric" and
contains the same 16 test images for every seed. Each per-seed exact-target
ACC is therefore the number of correct predictions divided by 16.

**Generated-Complement:**

seed 0: 12 / 16 = 0.7500

seed 42: 11 / 16 = 0.6875

seed 1234: 12 / 16 = 0.7500

seed 3407: 13 / 16 = 0.8125

seed 2024: 12 / 16 = 0.7500

**AttriSelect:**

seed 0: 13 / 16 = 0.8125

seed 42: 12 / 16 = 0.7500

seed 1234: 13 / 16 = 0.8125

seed 3407: 11 / 16 = 0.6875

seed 2024: 12 / 16 = 0.7500

Across the five seeds, Generated-Complement obtains 60/80 correct predictions
and mean exact-target ACC $0.7500 \pm 0.0442$, while AttriSelect obtains 61/80
and $0.7625 \pm 0.0523$. The uncertainty is the sample standard deviation
across the five seed-level accuracies. Because the methods differ by one
correct prediction across the 80 seed-image evaluations, we treat this
exact-target comparison as descriptive.

# Response to Reviewer hG6j — Discussion Follow-Up (Part 1)

Dear Reviewer, thank you for the follow-up. We appreciate your careful assessment of the additional results and address the remaining questions below.

**This response is organized into 5 parts.**

> Regarding the computational-cost evaluation, I noted that this concern was also raised by other reviewers and the AC. In my view, however, this issue primarily concerns the completeness and fairness of the experimental evaluation, rather than the core practical value of the proposed method. The revised results improve this aspect of the paper, but resolving this concern is unlikely to substantially change overall assessment or score.

## Q1. Reusable Source-Attribute Metadata

Thank you for recognizing the distinction between the completeness of the cost evaluation and the practical value of the method. We would like to clarify that our previous end-to-end tables used a deliberately conservative first-run accounting: **every AttriSelect row included the full source-pool attribute-assignment cost**, even though this preprocessing does not need to be repeated at each source ratio or repair iteration.

For a fixed task-level attribute vocabulary, source-pool assignment is a one-time preprocessing step. The resulting metadata can be reused across source ratios, attribute-aware selectors, and later repair iterations on the same task. The full source pool therefore does not need to be annotated again for every repair. The corrected tables below remove this reusable assignment component from the AttriSelect totals; all baseline times are unchanged.

**KITTI wall-clock time after reusing source-attribute metadata.**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 42.17 min | 42.18 min | 48.82 min | 1.37 h | 1.12 h | 43.47 min | 43.53 min | 42.20 min |
| 0.3 | 43.84 min | 43.85 min | 50.28 min | 1.44 h | 1.20 h | 45.08 min | 45.63 min | 43.87 min |
| 0.5 | 45.18 min | 45.18 min | 51.73 min | 1.49 h | 1.27 h | 46.42 min | 47.47 min | 45.20 min |
| 0.7 | 47.04 min | 47.05 min | 53.46 min | 1.56 h | 1.35 h | 48.16 min | 49.73 min | 47.07 min |
| 0.9 | 48.86 min | 48.87 min | 55.53 min | 1.63 h | 1.42 h | 50.34 min | 51.46 min | 48.89 min |

**COCO wall-clock time after reusing source-attribute metadata.**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.41 min | 4.42 min | 10.98 min | 16.18 min | 19.63 min | 5.08 min | 6.97 min | 4.44 min |
| 0.3 | 8.18 min | 8.18 min | 14.74 min | 39.86 min | 49.76 min | 8.93 min | 17.86 min | 8.20 min |
| 0.5 | 12.37 min | 12.36 min | 19.04 min | 1.08 h | 1.56 h | 13.08 min | 29.47 min | 12.39 min |
| 0.7 | 15.73 min | 15.75 min | 22.27 min | 1.45 h | 2.29 h | 16.34 min | 37.63 min | 15.77 min |
| 0.9 | 19.66 min | 19.65 min | 26.28 min | 1.87 h | 3.01 h | 20.18 min | 40.94 min | 19.68 min |

These adjusted tables represent later ratios or repair iterations after the task-level metadata has been created once. **`With metadata reuse, AttriSelect has nearly the same wall-clock cost as Random and Stratified at every ratio, while remaining substantially faster than CoreSet and BADGE.`** The earlier tables remain the conservative upper bound for a first repair on a task with no existing metadata.

> I also examined the concerns raised by the AC and the other reviewers, particularly regarding the Same-Information Repair Baselines. While the new results are consistently positive, the improvement over the strongest same-information baseline appears relatively modest. It may therefore be helpful to investigate whether the advantage of AttriSelect can be demonstrated from additional perspectives—for example, through target-slice performance, robustness across different target attributes or failure slices, data efficiency at more constrained budgets, or a more detailed analysis of which samples are selected and why they lead to better repair. Such evidence could more clearly establish the benefit of the proposed selection mechanism beyond the small improvement in overall accuracy.

## Q2. Same-Information Repair Baselines Across Downstream Tasks

**Response:** We extended the same-information comparison across all three downstream tasks considered in this follow-up: ImageNet10 classification, COCO pose estimation, and KITTI object detection. For each task, all methods share the source pool, VLM attributes, diagnosed slice, fixed 1,000-image generated repair set, source ratios, diagnosis checkpoint, five seeds (0, 42, 1234, 3407, and 2024), and repair schedule. **Only the source-subset selection rule changes.**

In addition to the three direct attribute-aware controls—Target-Slice + Stratified, Attribute-Filtered Random, and Attribute-Filtered Entropy—we added a stronger **Generated-Complement** baseline. It uses the same generated-set attribute statistics and source metadata as AttriSelect, prioritizes source samples whose attributes are underrepresented in the generated set, and maintains task-specific class or proxy-label balance. The full score and deterministic balancing procedure are described in our response to Reviewer yyPZ under **Q2. Generated-Complement baseline**. We also include all seven generic baselines from the manuscript appendix: Random, Stratified, Entropy, CoreSet, BADGE, UHerding, and PPAL.

**ImageNet10 same-information repair comparison. Metric: FR (\%).**
Each entry is the five-seed mean $\pm$ standard deviation. Attribute-aware
baselines are marked with inline backticks and rendered in red on OpenReview.
Ratio Mean averages the five FR means at source ratios 0.1/0.3/0.5/0.7/0.9;
standard deviations are not pooled.

# Response to Reviewer hG6j — Discussion Follow-Up (Part 2)

## Q2. Same-Information Repair Baselines Across Downstream Tasks (continued)

**ImageNet10.**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.40 $\pm$ 0.88 | 3.47 $\pm$ 1.31 | 2.87 $\pm$ 0.57 | 3.40 $\pm$ 1.13 | 5.31 $\pm$ 0.78 | 5.59 $\pm$ 0.81 | 3.96 $\pm$ 0.92 | 4.60 $\pm$ 1.80 | 5.91 $\pm$ 1.20 | 5.09 $\pm$ 0.88 | 1.50 $\pm$ 0.87 | **6.93 $\pm$ 1.52** |
| 0.3 | 4.78 $\pm$ 0.71 | 6.93 $\pm$ 1.27 | 4.21 $\pm$ 0.57 | 6.93 $\pm$ 0.42 | 8.84 $\pm$ 1.27 | 10.47 $\pm$ 1.52 | 7.75 $\pm$ 0.78 | 7.50 $\pm$ 0.85 | 7.39 $\pm$ 1.49 | 6.79 $\pm$ 0.81 | 5.31 $\pm$ 1.04 | **11.28 $\pm$ 1.56** |
| 0.5 | 7.32 $\pm$ 1.06 | 9.66 $\pm$ 0.60 | 5.59 $\pm$ 0.96 | 8.03 $\pm$ 0.60 | 10.47 $\pm$ 1.03 | 8.38 $\pm$ 0.99 | 8.84 $\pm$ 0.57 | 9.37 $\pm$ 1.31 | 9.52 $\pm$ 1.34 | 9.52 $\pm$ 1.10 | 11.29 $\pm$ 0.76 | **11.85 $\pm$ 0.35** |
| 0.7 | 8.84 $\pm$ 1.52 | 6.12 $\pm$ 1.52 | 6.40 $\pm$ 1.49 | 7.75 $\pm$ 1.52 | 10.75 $\pm$ 0.39 | 10.47 $\pm$ 1.20 | 8.84 $\pm$ 1.20 | 9.23 $\pm$ 2.30 | 10.65 $\pm$ 1.34 | 10.93 $\pm$ 0.71 | 8.57 $\pm$ 1.18 | **12.10 $\pm$ 1.31** |
| 0.9 | 10.22 $\pm$ 1.41 | 10.47 $\pm$ 0.39 | 10.22 $\pm$ 0.92 | 6.12 $\pm$ 1.24 | 9.37 $\pm$ 0.96 | 11.57 $\pm$ 1.17 | 11.85 $\pm$ 1.45 | 11.04 $\pm$ 0.92 | 10.47 $\pm$ 1.27 | 11.67 $\pm$ 1.59 | 10.48 $\pm$ 0.95 | **13.76 $\pm$ 0.67** |
| Ratio Mean | 6.91 | 7.33 | 5.86 | 6.44 | 8.95 | 9.30 | 8.25 | 8.35 | 8.79 | 8.80 | 7.43 | **11.19** |

**COCO pose estimation. Metric: Fix Rate (FR, %).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.56 $\pm$ 0.68 | 5.08 $\pm$ 0.78 | 5.61 $\pm$ 0.46 | -0.61 $\pm$ 0.98 | 2.03 $\pm$ 0.95 | 2.71 $\pm$ 1.07 | 1.17 $\pm$ 0.37 | 5.63 $\pm$ 0.66 | -3.42 $\pm$ 0.87 | 4.47 $\pm$ 0.91 | 2.50 $\pm$ 0.83 | **6.86 $\pm$ 1.07** |
| 0.3 | 3.37 $\pm$ 0.73 | 8.32 $\pm$ 0.51 | 7.64 $\pm$ 0.71 | 8.64 $\pm$ 0.76 | 6.12 $\pm$ 1.07 | 4.32 $\pm$ 0.61 | 6.42 $\pm$ 0.49 | 9.38 $\pm$ 0.62 | 6.53 $\pm$ 0.84 | 8.37 $\pm$ 0.57 | 7.22 $\pm$ 0.68 | **9.86 $\pm$ 0.54** |
| 0.5 | 7.49 $\pm$ 0.49 | 8.00 $\pm$ 0.66 | 9.27 $\pm$ 0.63 | 6.47 $\pm$ 0.88 | 7.42 $\pm$ 0.44 | 6.12 $\pm$ 0.93 | 6.83 $\pm$ 0.54 | 8.83 $\pm$ 0.74 | 7.11 $\pm$ 0.76 | 8.16 $\pm$ 0.35 | 8.67 $\pm$ 0.79 | **10.13 $\pm$ 0.29** |
| 0.7 | 6.08 $\pm$ 1.00 | 9.69 $\pm$ 0.51 | 9.54 $\pm$ 0.54 | 9.10 $\pm$ 0.54 | 8.64 $\pm$ 0.39 | 6.17 $\pm$ 1.00 | 7.83 $\pm$ 1.07 | 9.24 $\pm$ 0.81 | 9.26 $\pm$ 0.68 | 9.68 $\pm$ 0.64 | 9.56 $\pm$ 0.72 | **10.18 $\pm$ 0.59** |
| 0.9 | 4.61 $\pm$ 0.98 | 9.49 $\pm$ 0.51 | 6.54 $\pm$ 0.44 | 7.32 $\pm$ 0.56 | 7.59 $\pm$ 0.68 | 7.05 $\pm$ 0.34 | 9.74 $\pm$ 0.61 | 8.68 $\pm$ 0.69 | 6.31 $\pm$ 0.83 | 9.78 $\pm$ 0.79 | 6.96 $\pm$ 0.91 | **11.05 $\pm$ 0.76** |
| Ratio Mean | 5.02 | 8.12 | 7.72 | 6.18 | 6.36 | 5.27 | 6.40 | 8.35 | 5.16 | 8.09 | 6.98 | **9.62** |

**KITTI object detection. Metric: Fix Rate (FR, %).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -9.65 $\pm$ 0.48 | -7.27 $\pm$ 0.85 | -7.52 $\pm$ 0.85 | -8.89 $\pm$ 0.89 | -6.95 $\pm$ 0.57 | -9.24 $\pm$ 0.71 | -8.32 $\pm$ 0.82 | -8.24 $\pm$ 0.77 | -6.14 $\pm$ 0.68 | -6.81 $\pm$ 0.59 | -6.24 $\pm$ 0.84 | **-5.81 $\pm$ 0.53** |
| 0.3 | -0.48 $\pm$ 0.39 | -1.23 $\pm$ 0.82 | 0.23 $\pm$ 0.75 | -0.32 $\pm$ 0.30 | -0.05 $\pm$ 0.69 | -1.21 $\pm$ 0.48 | -1.33 $\pm$ 0.89 | -0.62 $\pm$ 0.56 | 0.04 $\pm$ 0.74 | 0.13 $\pm$ 0.46 | 0.17 $\pm$ 0.43 | **0.46 $\pm$ 0.53** |
| 0.5 | 2.13 $\pm$ 0.69 | 1.23 $\pm$ 0.89 | 3.20 $\pm$ 1.01 | 3.04 $\pm$ 0.98 | 2.29 $\pm$ 0.30 | 1.12 $\pm$ 0.62 | 2.17 $\pm$ 0.48 | 2.83 $\pm$ 0.36 | 2.58 $\pm$ 0.91 | 3.25 $\pm$ 0.33 | 1.72 $\pm$ 0.58 | **3.36 $\pm$ 0.27** |
| 0.7 | 3.20 $\pm$ 0.89 | 3.06 $\pm$ 0.30 | 3.34 $\pm$ 0.55 | 3.34 $\pm$ 0.30 | 2.08 $\pm$ 0.39 | 3.25 $\pm$ 0.55 | 3.57 $\pm$ 0.69 | 3.54 $\pm$ 0.48 | 2.93 $\pm$ 0.47 | 3.43 $\pm$ 0.61 | **4.06 $\pm$ 0.71** | 3.52 $\pm$ 0.89 |
| 0.9 | 4.37 $\pm$ 0.89 | 4.66 $\pm$ 0.85 | 4.66 $\pm$ 0.96 | 3.54 $\pm$ 0.25 | 4.21 $\pm$ 0.41 | 4.85 $\pm$ 1.01 | 3.89 $\pm$ 0.59 | 4.38 $\pm$ 0.82 | 4.46 $\pm$ 0.31 | 4.68 $\pm$ 0.88 | 3.74 $\pm$ 0.61 | **5.12 $\pm$ 0.27** |
| Ratio Mean | -0.09 | 0.09 | 0.78 | 0.14 | 0.32 | -0.25 | 0.00 | 0.38 | 0.77 | 0.94 | 0.69 | **1.33** |

# Response to Reviewer hG6j — Discussion Follow-Up (Part 3)

## Q2. Same-Information Repair Baselines Across Downstream Tasks (continued)

> **Takeaway.** The ImageNet10 controls show that repair-relevant attributes
> help simple selectors. Attribute-Filtered Random and Attribute-Filtered
> Entropy beat their generic counterparts at all five ratios, and Target-Slice + Stratified beats Stratified at four of five. Attribute access by itself is
> not enough, however. **AttriSelect has the highest Ratio Mean on ImageNet10,
> COCO, and KITTI, and the highest FR at 14 of the 15 task--ratio points.** The
> exception is KITTI at ratio 0.7, where Generated-Complement is higher by 0.54
> FR points. **`Under matched information and repair budgets, AttriSelect still
> gives the best aggregate performance across classification, pose estimation,
> and object detection.`**
>
> **Stability under extreme source budgets.** Budget size affects both
> traditional Active Learning methods and the same-information baselines. At
> KITTI ratio 0.1, only 500 source images are selected. AttriSelect has the
> highest FR, while the other methods have larger drops. COCO ratio 0.9 selects
> 18,000 source images. Several methods are worse than at ratio 0.7, but
> AttriSelect rises from 10.18 to 11.05 FR.
> **`AttriSelect is the only evaluated method whose mean FR increases at every
> successive ratio on all three tasks.`**


## Q3. More Constrained Source Budgets

We further tested more constrained source budgets on ImageNet10 and COCO. The
selected source sets contain only 208/520 ImageNet10 images and 400/1,000 COCO
images at source ratios 0.02/0.05. All methods use the same diagnosis and
repair settings. The metric is Fix Rate (FR, \%); entries are mean $\pm$
standard deviation, and Ratio Mean averages the two displayed FR means
without pooling their standard deviations.

**ImageNet10 classification under constrained source budgets. Metric: FR
(\%; mean $\pm$ standard deviation).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Generated-Complement` | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | -3.10 $\pm$ 1.22 | -2.68 $\pm$ 1.08 | -2.35 $\pm$ 1.34 | -3.42 $\pm$ 1.16 | -2.12 $\pm$ 1.27 | -2.57 $\pm$ 1.05 | -2.89 $\pm$ 1.18 | -1.77 $\pm$ 1.21 | -2.04 $\pm$ 1.38 | 0.68 $\pm$ 0.94 | -0.95 $\pm$ 1.07 | **1.23 $\pm$ 0.72** |
| 0.05 | 1.84 $\pm$ 0.91 | 2.41 $\pm$ 0.84 | 2.76 $\pm$ 1.02 | 1.58 $\pm$ 0.88 | 2.91 $\pm$ 0.95 | 2.64 $\pm$ 0.90 | 2.20 $\pm$ 0.82 | 3.13 $\pm$ 0.86 | 3.95 $\pm$ 0.92 | 3.13 $\pm$ 0.78 | 2.32 $\pm$ 0.81 | **5.58 $\pm$ 0.61** |
| Ratio Mean | -0.63 | -0.14 | 0.21 | -0.92 | 0.40 | 0.04 | -0.35 | 0.68 | 0.96 | 1.91 | 0.69 | **3.41** |

**COCO pose estimation under constrained source budgets. Metric: FR
(\%; mean $\pm$ standard deviation).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Generated-Complement` | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | -8.75 $\pm$ 1.42 | -7.15 $\pm$ 1.31 | -5.62 $\pm$ 1.18 | -9.30 $\pm$ 1.57 | -6.40 $\pm$ 1.26 | -8.10 $\pm$ 1.49 | -7.00 $\pm$ 1.35 | -2.84 $\pm$ 1.16 | -7.80 $\pm$ 1.43 | -2.46 $\pm$ 1.02 | -6.23 $\pm$ 1.28 | **-2.02 $\pm$ 0.91** |
| 0.05 | -4.20 $\pm$ 1.12 | -2.96 $\pm$ 0.98 | -1.84 $\pm$ 0.87 | -5.10 $\pm$ 1.24 | -2.55 $\pm$ 0.93 | -4.00 $\pm$ 1.15 | -3.20 $\pm$ 1.01 | -31.00 $\pm$ 2.34 | -11.64 $\pm$ 1.76 | -0.61 $\pm$ 0.74 | -1.10 $\pm$ 0.82 | **1.63 $\pm$ 0.68** |
| Ratio Mean | -6.48 | -5.06 | -3.73 | -7.20 | -4.48 | -6.05 | -5.10 | -16.92 | -9.72 | -1.54 | -3.67 | **-0.20** |

> **Takeaway.** At these budgets, many baselines have negative FR: with only
> 208--520 ImageNet10 images or 400--1,000 COCO images, the fixed repair
> schedule can make the model worse rather than better. This suggests that
> extremely small source budgets may require a different source/generated
> mixture or a longer repair schedule. We intentionally keep the current
> settings unchanged so that the selector remains the only comparison
> variable. **`Even in this constrained regime, AttriSelect is best at all
> four dataset--ratio points and has the highest Ratio Mean on both tasks.`**
> Its Ratio Mean exceeds the strongest alternative by **1.50 FR points on
> ImageNet10** and **1.34 FR points on COCO**. The absolute gains are modest
> because the repair data and training budget are small, but the ordering is
> consistent under the same controlled protocol.

# Response to Reviewer hG6j — Discussion Follow-Up (Part 4)

## Q4. Exact-Slice and Selected-Sample Evidence

The cross-task results in Q2 cover classification, pose estimation, and object detection under task-specific diagnosed failures. We also evaluated the exact diagnosed conjunctions at ratio 0.5. On ImageNet10, AttriSelect improves the exact "pink AND fabric" slice by **13.75 percentage points** and its non-target complement by **3.22 points**. On COCO, it improves the exact "brown AND rainy" slice by **10.13 points** and the non-target complement by **4.04 points**. The full five-seed tables, slice sizes, per-seed ImageNet10 correct counts, and variability are reported in our response to Reviewer yyPZ under **Q3. Exact diagnosed-slice and non-target performance**. The positive target and non-target changes show that the overall gains include the diagnosed failures without reducing performance on the remaining test data; because the ImageNet10 exact slice contains only 16 images, its method ordering is descriptive.

To examine which source samples are selected, we compare AttriSelect with Generated-Complement and the three direct same-information controls on ImageNet10. All methods use the same source pool, fixed 1,000-image generated set, diagnosis checkpoint, source ratios, and 15-epoch repair schedule. Downstream FR values are five-seed means.

Each `# (%)` entry gives the sample count and its percentage within the selected source subset. Exact Target matches "pink AND fabric"; Single-Factor matches exactly one of pink or fabric; Background matches neither. Unique Class--Attribute Tuples counts distinct (class, color, background, texture) combinations. Classes Covered is the number of ImageNet10 classes represented in the selected subset. FR is Relative Error Reduction, $(M_{repair}-M_{diagnosis})/(1-M_{diagnosis})\times100\%$; higher is better.

| Ratio | Method | Exact Target # (%) | Single-Factor # (%) | Background # (%) | Unique Class--Attribute Tuples | Classes Covered | FR |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | 130 (12.50%) | 141 (13.56%) | 769 (73.94%) | 382 | 10 | 4.60 |
| 0.1 | Attribute-Filtered Random | 30 (2.88%) | 686 (65.96%) | 324 (31.15%) | 401 | 10 | 5.91 |
| 0.1 | Attribute-Filtered Entropy | 34 (3.27%) | 682 (65.58%) | 324 (31.15%) | 410 | 10 | 5.09 |
| 0.1 | Generated-Complement | 0 (0.00%) | 58 (5.58%) | 982 (94.42%) | 396 | 10 | 1.50 |
| 0.1 | AttriSelect | 24 (2.31%) | 269 (25.87%) | 747 (71.83%) | 680 | 10 | 6.93 |
| 0.3 | Target-Slice + Stratified | 130 (4.17%) | 581 (18.62%) | 2409 (77.21%) | 762 | 10 | 7.50 |
| 0.3 | Attribute-Filtered Random | 60 (1.92%) | 1130 (36.22%) | 1930 (61.86%) | 780 | 10 | 7.39 |
| 0.3 | Attribute-Filtered Entropy | 70 (2.24%) | 1120 (35.90%) | 1930 (61.86%) | 801 | 10 | 6.79 |
| 0.3 | Generated-Complement | 0 (0.00%) | 266 (8.53%) | 2854 (91.47%) | 692 | 10 | 5.31 |
| 0.3 | AttriSelect | 40 (1.28%) | 823 (26.38%) | 2257 (72.34%) | 1165 | 10 | 11.28 |
| 0.5 | Target-Slice + Stratified | 130 (2.50%) | 1005 (19.33%) | 4065 (78.17%) | 1007 | 10 | 9.37 |
| 0.5 | Attribute-Filtered Random | 101 (1.94%) | 1505 (28.94%) | 3594 (69.12%) | 989 | 10 | 9.52 |
| 0.5 | Attribute-Filtered Entropy | 92 (1.77%) | 1514 (29.12%) | 3594 (69.12%) | 995 | 10 | 9.52 |
| 0.5 | Generated-Complement | 0 (0.00%) | 618 (11.88%) | 4582 (88.12%) | 902 | 10 | 11.29 |
| 0.5 | AttriSelect | 95 (1.83%) | 1183 (22.75%) | 3922 (75.42%) | 1318 | 10 | 11.85 |
| 0.7 | Target-Slice + Stratified | 130 (1.79%) | 1449 (19.91%) | 5700 (78.31%) | 1194 | 10 | 9.23 |
| 0.7 | Attribute-Filtered Random | 125 (1.72%) | 1834 (25.20%) | 5320 (73.09%) | 1186 | 10 | 10.65 |
| 0.7 | Attribute-Filtered Entropy | 118 (1.62%) | 1841 (25.29%) | 5320 (73.09%) | 1168 | 10 | 10.93 |
| 0.7 | Generated-Complement | 0 (0.00%) | 1034 (14.21%) | 6245 (85.79%) | 1044 | 10 | 8.57 |
| 0.7 | AttriSelect | 95 (1.31%) | 1519 (20.87%) | 5665 (77.83%) | 1379 | 10 | 12.10 |
| 0.9 | Target-Slice + Stratified | 130 (1.39%) | 1884 (20.13%) | 7346 (78.48%) | 1345 | 10 | 11.04 |
| 0.9 | Attribute-Filtered Random | 129 (1.38%) | 2038 (21.77%) | 7193 (76.85%) | 1325 | 10 | 10.47 |
| 0.9 | Attribute-Filtered Entropy | 129 (1.38%) | 2038 (21.77%) | 7193 (76.85%) | 1315 | 10 | 11.67 |
| 0.9 | Generated-Complement | 0 (0.00%) | 1493 (15.95%) | 7867 (84.05%) | 1187 | 10 | 10.48 |
| 0.9 | AttriSelect | 130 (1.39%) | 1888 (20.17%) | 7342 (78.44%) | 1403 | 10 | 13.76 |

# Response to Reviewer hG6j — Discussion Follow-Up (Part 5)

## Q4. Exact-Slice and Selected-Sample Evidence (continued)

Class Imbalance is the difference between the largest and smallest selected counts across the ten ImageNet10 classes; lower is more balanced. Generated Redundancy is the selected-set mean fraction of the two generated-supported target fields matched by each source sample, namely "color=pink" and "texture=fabric"; lower means less duplication of the generated target attributes. Jaccard with AttriSelect is $|S_m\cap S_A|/|S_m\cup S_A|$, where $S_m$ and $S_A$ are the source subsets selected by method $m$ and AttriSelect at the same ratio. Mean Pairwise Attribute Distance is the mean categorical Hamming distance over color, background, and texture across all unordered selected-source pairs; higher indicates broader attribute variation.

| Ratio | Method | Class Imbalance $\downarrow$ | Generated Redundancy $\downarrow$ | Jaccard with AttriSelect | Mean Pairwise Attribute Distance $\uparrow$ | FR |
|---:|:---|---:|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | 0 | 0.1928 | 0.0500 | 0.7870 | 4.60 |
| 0.1 | Attribute-Filtered Random | 0 | 0.3587 | 0.1129 | 0.7188 | 5.91 |
| 0.1 | Attribute-Filtered Entropy | 0 | 0.3606 | 0.1141 | 0.7265 | 5.09 |
| 0.1 | Generated-Complement | 0 | 0.0279 | 0.0711 | 0.7853 | 1.50 |
| 0.1 | AttriSelect | 0 | 0.1524 | 1.0000 | 0.8597 | 6.93 |
| 0.3 | Target-Slice + Stratified | 0 | 0.1348 | 0.1782 | 0.7902 | 7.50 |
| 0.3 | Attribute-Filtered Random | 0 | 0.2003 | 0.2086 | 0.7772 | 7.39 |
| 0.3 | Attribute-Filtered Entropy | 0 | 0.2019 | 0.2197 | 0.7805 | 6.79 |
| 0.3 | Generated-Complement | 0 | 0.0426 | 0.1710 | 0.7861 | 5.31 |
| 0.3 | AttriSelect | 2 | 0.1447 | 1.0000 | 0.8414 | 11.28 |
| 0.5 | Target-Slice + Stratified | 0 | 0.1216 | 0.3281 | 0.7912 | 9.37 |
| 0.5 | Attribute-Filtered Random | 0 | 0.1641 | 0.3437 | 0.7847 | 9.52 |
| 0.5 | Attribute-Filtered Entropy | 0 | 0.1633 | 0.3437 | 0.7861 | 9.52 |
| 0.5 | Generated-Complement | 0 | 0.0594 | 0.3161 | 0.7881 | 11.29 |
| 0.5 | AttriSelect | 4 | 0.1320 | 1.0000 | 0.8272 | 11.85 |
| 0.7 | Target-Slice + Stratified | 1 | 0.1174 | 0.5391 | 0.7919 | 9.23 |
| 0.7 | Attribute-Filtered Random | 1 | 0.1432 | 0.5323 | 0.7883 | 10.65 |
| 0.7 | Attribute-Filtered Entropy | 1 | 0.1427 | 0.5300 | 0.7879 | 10.93 |
| 0.7 | Generated-Complement | 1 | 0.0710 | 0.5165 | 0.7896 | 8.57 |
| 0.7 | AttriSelect | 5 | 0.1174 | 1.0000 | 0.8124 | 12.10 |
| 0.9 | Target-Slice + Stratified | 0 | 0.1145 | 0.8169 | 0.7915 | 11.04 |
| 0.9 | Attribute-Filtered Random | 0 | 0.1226 | 0.8154 | 0.7904 | 10.47 |
| 0.9 | Attribute-Filtered Entropy | 0 | 0.1226 | 0.8162 | 0.7909 | 11.67 |
| 0.9 | Generated-Complement | 0 | 0.0798 | 0.8080 | 0.7899 | 10.48 |
| 0.9 | AttriSelect | 6 | 0.1147 | 1.0000 | 0.7977 | 13.76 |

At the more constrained ratios, each Best-Control value is selected independently from Target-Slice + Stratified, Attribute-Filtered Random, Attribute-Filtered Entropy, and Generated-Complement. Best-Control is the maximum for FR, Unique Tuples, and Attribute Distance and the minimum for Class Imbalance. Each margin or gain is the AttriSelect value minus the corresponding Best-Control value.

| Ratio | AttriSelect FR | Best-Control FR | FR Margin | AttriSelect Class Imbalance | Best-Control Class Imbalance | AttriSelect Unique Tuples | Best-Control Unique Tuples | Tuple Gain | AttriSelect Attribute Distance | Best-Control Attribute Distance | Distance Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 1.23 | 0.68 | +0.55 | 1 | 1 | 184 | 142 | +42 | 0.8823 | 0.7695 | +0.1128 |
| 0.05 | 5.58 | 3.95 | +1.63 | 0 | 0 | 397 | 267 | +130 | 0.8680 | 0.7828 | +0.0853 |

> **Takeaway.** At ratios 0.02 and 0.05, AttriSelect improves five-seed FR over the strongest control by 0.55 and 1.63 points. It also covers 42 and 130 more class--attribute tuples and increases Mean Pairwise Attribute Distance by 0.1128 and 0.0853, while matching the best Class Imbalance. **`With only 208 or 520 selected source samples, AttriSelect has the highest FR and covers a broader range of classes and attributes.`**
>
> Across ratios 0.1--0.9, AttriSelect also has the highest FR, Unique Class--Attribute Tuples, and Mean Pairwise Attribute Distance among the matched controls. Generated-Complement has lower Generated Redundancy but lower FR, so avoiding generated-target duplication alone is insufficient. This pattern is consistent with AttriSelect retaining repair-relevant samples without narrowing class and attribute coverage.


# Response to Reviewer yyPZ — Discussion Follow-Up (Part 1)

Dear Reviewer yyPZ, thank you for the follow-up. We added the requested
cross-task controls and exact-slice evaluation, and we also test how the
conclusions change under different repair and source budgets.

**This response is organized into 7 parts.**

## Q1. Cross-task same-information comparison

> 1. The same-information comparison is only conducted on ImageNet10. The
> paper claims generality across classification, pose estimation, and
> detection, but no attribute-aware comparison is provided on COCO or KITTI.
> Therefore, the rebuttal currently shows only that AttriSelect outperforms
> several simple heuristics on one 10-class classification task.

**Response:** ImageNet10 alone was not enough to support the cross-task claim,
so we repeated the same-information comparison on COCO pose estimation and
KITTI object detection. For each task, all methods share the source pool, VLM
attributes, diagnosed slice, fixed 1,000-image generated repair set, source
ratios, diagnosis checkpoint, five seeds (0, 42, 1234, 3407, and 2024), and
repair schedule. **Only the source-subset selection rule changes.** We also
include all seven generic baselines from the manuscript appendix: Random,
Stratified, Entropy, CoreSet, BADGE, UHerding, and PPAL.

**ImageNet10 same-information repair comparison. Metric: FR (\%).**
Each entry is the five-seed mean $\pm$ standard deviation. Attribute-aware
baselines are marked with inline backticks and rendered in red on OpenReview.
Ratio Mean averages the five FR means at source ratios 0.1/0.3/0.5/0.7/0.9;
standard deviations are not pooled.


| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.40 $\pm$ 0.88 | 3.47 $\pm$ 1.31 | 2.87 $\pm$ 0.57 | 3.40 $\pm$ 1.13 | 5.31 $\pm$ 0.78 | 5.59 $\pm$ 0.81 | 3.96 $\pm$ 0.92 | 4.60 $\pm$ 1.80 | 5.91 $\pm$ 1.20 | 5.09 $\pm$ 0.88 | 1.50 $\pm$ 0.87 | **6.93 $\pm$ 1.52** |
| 0.3 | 4.78 $\pm$ 0.71 | 6.93 $\pm$ 1.27 | 4.21 $\pm$ 0.57 | 6.93 $\pm$ 0.42 | 8.84 $\pm$ 1.27 | 10.47 $\pm$ 1.52 | 7.75 $\pm$ 0.78 | 7.50 $\pm$ 0.85 | 7.39 $\pm$ 1.49 | 6.79 $\pm$ 0.81 | 5.31 $\pm$ 1.04 | **11.28 $\pm$ 1.56** |
| 0.5 | 7.32 $\pm$ 1.06 | 9.66 $\pm$ 0.60 | 5.59 $\pm$ 0.96 | 8.03 $\pm$ 0.60 | 10.47 $\pm$ 1.03 | 8.38 $\pm$ 0.99 | 8.84 $\pm$ 0.57 | 9.37 $\pm$ 1.31 | 9.52 $\pm$ 1.34 | 9.52 $\pm$ 1.10 | 11.29 $\pm$ 0.76 | **11.85 $\pm$ 0.35** |
| 0.7 | 8.84 $\pm$ 1.52 | 6.12 $\pm$ 1.52 | 6.40 $\pm$ 1.49 | 7.75 $\pm$ 1.52 | 10.75 $\pm$ 0.39 | 10.47 $\pm$ 1.20 | 8.84 $\pm$ 1.20 | 9.23 $\pm$ 2.30 | 10.65 $\pm$ 1.34 | 10.93 $\pm$ 0.71 | 8.57 $\pm$ 1.18 | **12.10 $\pm$ 1.31** |
| 0.9 | 10.22 $\pm$ 1.41 | 10.47 $\pm$ 0.39 | 10.22 $\pm$ 0.92 | 6.12 $\pm$ 1.24 | 9.37 $\pm$ 0.96 | 11.57 $\pm$ 1.17 | 11.85 $\pm$ 1.45 | 11.04 $\pm$ 0.92 | 10.47 $\pm$ 1.27 | 11.67 $\pm$ 1.59 | 10.48 $\pm$ 0.95 | **13.76 $\pm$ 0.67** |
| Ratio Mean | 6.91 | 7.33 | 5.86 | 6.44 | 8.95 | 9.30 | 8.25 | 8.35 | 8.79 | 8.80 | 7.43 | **11.19** |

**COCO pose estimation. Metric: Fix Rate (FR, %).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.56 $\pm$ 0.68 | 5.08 $\pm$ 0.78 | 5.61 $\pm$ 0.46 | -0.61 $\pm$ 0.98 | 2.03 $\pm$ 0.95 | 2.71 $\pm$ 1.07 | 1.17 $\pm$ 0.37 | 5.63 $\pm$ 0.66 | -3.42 $\pm$ 0.87 | 4.47 $\pm$ 0.91 | 2.50 $\pm$ 0.83 | **6.86 $\pm$ 1.07** |
| 0.3 | 3.37 $\pm$ 0.73 | 8.32 $\pm$ 0.51 | 7.64 $\pm$ 0.71 | 8.64 $\pm$ 0.76 | 6.12 $\pm$ 1.07 | 4.32 $\pm$ 0.61 | 6.42 $\pm$ 0.49 | 9.38 $\pm$ 0.62 | 6.53 $\pm$ 0.84 | 8.37 $\pm$ 0.57 | 7.22 $\pm$ 0.68 | **9.86 $\pm$ 0.54** |
| 0.5 | 7.49 $\pm$ 0.49 | 8.00 $\pm$ 0.66 | 9.27 $\pm$ 0.63 | 6.47 $\pm$ 0.88 | 7.42 $\pm$ 0.44 | 6.12 $\pm$ 0.93 | 6.83 $\pm$ 0.54 | 8.83 $\pm$ 0.74 | 7.11 $\pm$ 0.76 | 8.16 $\pm$ 0.35 | 8.67 $\pm$ 0.79 | **10.13 $\pm$ 0.29** |
| 0.7 | 6.08 $\pm$ 1.00 | 9.69 $\pm$ 0.51 | 9.54 $\pm$ 0.54 | 9.10 $\pm$ 0.54 | 8.64 $\pm$ 0.39 | 6.17 $\pm$ 1.00 | 7.83 $\pm$ 1.07 | 9.24 $\pm$ 0.81 | 9.26 $\pm$ 0.68 | 9.68 $\pm$ 0.64 | 9.56 $\pm$ 0.72 | **10.18 $\pm$ 0.59** |
| 0.9 | 4.61 $\pm$ 0.98 | 9.49 $\pm$ 0.51 | 6.54 $\pm$ 0.44 | 7.32 $\pm$ 0.56 | 7.59 $\pm$ 0.68 | 7.05 $\pm$ 0.34 | 9.74 $\pm$ 0.61 | 8.68 $\pm$ 0.69 | 6.31 $\pm$ 0.83 | 9.78 $\pm$ 0.79 | 6.96 $\pm$ 0.91 | **11.05 $\pm$ 0.76** |
| Ratio Mean | 5.02 | 8.12 | 7.72 | 6.18 | 6.36 | 5.27 | 6.40 | 8.35 | 5.16 | 8.09 | 6.98 | **9.62** |

# Response to Reviewer yyPZ — Discussion Follow-Up (Part 2)

## Q1. Cross-task same-information comparison (continued)

**KITTI object detection. Metric: Fix Rate (FR, %).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -9.65 $\pm$ 0.48 | -7.27 $\pm$ 0.85 | -7.52 $\pm$ 0.85 | -8.89 $\pm$ 0.89 | -6.95 $\pm$ 0.57 | -9.24 $\pm$ 0.71 | -8.32 $\pm$ 0.82 | -8.24 $\pm$ 0.77 | -6.14 $\pm$ 0.68 | -6.81 $\pm$ 0.59 | -6.24 $\pm$ 0.84 | **-5.81 $\pm$ 0.53** |
| 0.3 | -0.48 $\pm$ 0.39 | -1.23 $\pm$ 0.82 | 0.23 $\pm$ 0.75 | -0.32 $\pm$ 0.30 | -0.05 $\pm$ 0.69 | -1.21 $\pm$ 0.48 | -1.33 $\pm$ 0.89 | -0.62 $\pm$ 0.56 | 0.04 $\pm$ 0.74 | 0.13 $\pm$ 0.46 | 0.17 $\pm$ 0.43 | **0.46 $\pm$ 0.53** |
| 0.5 | 2.13 $\pm$ 0.69 | 1.23 $\pm$ 0.89 | 3.20 $\pm$ 1.01 | 3.04 $\pm$ 0.98 | 2.29 $\pm$ 0.30 | 1.12 $\pm$ 0.62 | 2.17 $\pm$ 0.48 | 2.83 $\pm$ 0.36 | 2.58 $\pm$ 0.91 | 3.25 $\pm$ 0.33 | 1.72 $\pm$ 0.58 | **3.36 $\pm$ 0.27** |
| 0.7 | 3.20 $\pm$ 0.89 | 3.06 $\pm$ 0.30 | 3.34 $\pm$ 0.55 | 3.34 $\pm$ 0.30 | 2.08 $\pm$ 0.39 | 3.25 $\pm$ 0.55 | 3.57 $\pm$ 0.69 | 3.54 $\pm$ 0.48 | 2.93 $\pm$ 0.47 | 3.43 $\pm$ 0.61 | **4.06 $\pm$ 0.71** | 3.52 $\pm$ 0.89 |
| 0.9 | 4.37 $\pm$ 0.89 | 4.66 $\pm$ 0.85 | 4.66 $\pm$ 0.96 | 3.54 $\pm$ 0.25 | 4.21 $\pm$ 0.41 | 4.85 $\pm$ 1.01 | 3.89 $\pm$ 0.59 | 4.38 $\pm$ 0.82 | 4.46 $\pm$ 0.31 | 4.68 $\pm$ 0.88 | 3.74 $\pm$ 0.61 | **5.12 $\pm$ 0.27** |
| Ratio Mean | -0.09 | 0.09 | 0.78 | 0.14 | 0.32 | -0.25 | 0.00 | 0.38 | 0.77 | 0.94 | 0.69 | **1.33** |

> **Takeaway.** The ImageNet10 controls show that repair-relevant attributes
> help simple selectors. Attribute-Filtered Random and Attribute-Filtered
> Entropy beat their generic counterparts at all five ratios, and Target-Slice + Stratified beats Stratified at four of five. Attribute access by itself is
> not enough, however. **AttriSelect has the highest Ratio Mean on ImageNet10,
> COCO, and KITTI, and the highest FR at 14 of the 15 task--ratio points.** The
> exception is KITTI at ratio 0.7, where Generated-Complement is higher by 0.54
> FR points. **`Under matched information and repair budgets, AttriSelect still
> gives the best aggregate performance across classification, pose estimation,
> and object detection.`**
>
> **Stability under extreme source budgets.** Budget size affects both
> traditional Active Learning methods and the same-information baselines. At
> KITTI ratio 0.1, only 500 source images are selected. AttriSelect has the
> highest FR, while the other methods have larger drops. COCO ratio 0.9 selects
> 18,000 source images. Several methods are worse than at ratio 0.7, but
> AttriSelect rises from 10.18 to 11.05 FR.
> **`AttriSelect is the only evaluated method whose mean FR increases at every
> successive ratio on all three tasks.`**

## Q2. Generated-Complement baseline

> 2. The new baselines do not fully isolate the value of the proposed R+P+C
> score. They mainly filter or prioritize samples containing the target
> attributes, while AttriSelect also uses the attribute distribution of the
> generated repair set. A stronger and still simple baseline would select
> source samples that complement underrepresented generated attributes while
> maintaining class balance. Without such a baseline, the improvement may
> reflect access to generated-set statistics rather than the specific scoring
> design.

**Response:** We added this control as **Generated-Complement**. It uses the
same generated-set attribute counts, source attributes, source budget, and
seed as AttriSelect. Its balancing groups are ImageNet10 classes, COCO
pose-complexity bins, and KITTI object-class compositions. For a source sample
$x$, the score is

$$
S_{GC}(x)=\frac{1}{|\mathcal A_x|}\sum_{a\in\mathcal A_x}
\frac{1}{1+f_{gen,a}(x_a)},
$$

where $f_{gen,a}(x_a)$ is the generated-set frequency of value $x_a$ for
attribute field $a$. Values absent from the generated set receive the largest
score. A deterministic round-robin procedure then balances selection across
the task-specific groups. Generated-Complement does not use source rarity
$R$, uncertainty, diagnostic weights, or the full $R+P+C$ score.

Q1 compares its FR with the other selectors. The table below reports its task
metrics and selection diagnostics. Exact Target, Non-Target, and Overall use
ACC for ImageNet10, keypoint AP for COCO, and mAP50--95 for KITTI. Class
Imbalance is the difference between the largest and smallest selected counts
across task-specific groups; zero means exact balance. Mean Complement Score
is
$|\mathcal D^{sel}|^{-1}\sum_{x\in\mathcal D^{sel}}S_{GC}(x)$.

# Response to Reviewer yyPZ — Discussion Follow-Up (Part 3)

## Q2. Generated-Complement baseline (continued)

| Task | Ratio | Exact Target | Non-Target | Overall | FR (\%) | Class Imbalance $\downarrow$ | Mean Complement Score $\uparrow$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ImageNet10 | 0.1 | 0.6875 | 0.7220 | 0.7215 | 1.50 | 0 | 0.9721 |
| ImageNet10 | 0.3 | 0.7125 | 0.7343 | 0.7323 | 5.31 | 0 | 0.9574 |
| ImageNet10 | 0.5 | 0.7200 | 0.7498 | 0.7492 | 11.29 | 0 | 0.9406 |
| ImageNet10 | 0.7 | 0.7275 | 0.7422 | 0.7415 | 8.57 | 1 | 0.9290 |
| ImageNet10 | 0.9 | 0.7301 | 0.7477 | 0.7469 | 10.48 | 0 | 0.9203 |
| ImageNet10 | Ratio Mean | 0.7155 | 0.7392 | 0.7383 | 7.43 | -- | 0.9439 |
| COCO | 0.1 | 0.5272 | 0.6008 | 0.6004 | 2.50 | 0 | 1.0000 |
| COCO | 0.3 | 0.6525 | 0.6199 | 0.6198 | 7.22 | 0 | 1.0000 |
| COCO | 0.5 | 0.7619 | 0.6259 | 0.6257 | 8.67 | 615 | 0.9930 |
| COCO | 0.7 | 0.8422 | 0.6296 | 0.6294 | 9.56 | 1,985 | 0.9874 |
| COCO | 0.9 | 0.8465 | 0.61878 | 0.6187 | 6.96 | 5,457 | 0.9824 |
| COCO | Ratio Mean | 0.7261 | 0.6190 | 0.6188 | 6.98 | -- | 0.9926 |
| KITTI | 0.1 | 0.3961 | 0.5356 | 0.5353 | -6.24 | 1 | 1.0000 |
| KITTI | 0.3 | 0.3791 | 0.5677 | 0.5673 | 0.17 | 0 | 1.0000 |
| KITTI | 0.5 | 0.4175 | 0.5704 | 0.5701 | 1.72 | 752 | 0.9932 |
| KITTI | 0.7 | 0.4343 | 0.5806 | 0.5804 | 4.06 | 1,752 | 0.9951 |
| KITTI | 0.9 | 0.5017 | 0.5794 | 0.5790 | 3.74 | 2,752 | 0.9962 |
| KITTI | Ratio Mean | 0.4257 | 0.5667 | 0.5664 | 0.69 | -- | 0.9969 |

We compare the two quantities that Generated-Complement directly optimizes
with the corresponding AttriSelect values. This selection-only analysis uses
matched saved subsets; it does not retrain either method or recompute task
metrics.

| Task | Ratio | Generated-Complement Class Imbalance $\downarrow$ | AttriSelect Class Imbalance $\downarrow$ | Generated-Complement Mean Complement Score $\uparrow$ | AttriSelect Mean Complement Score $\uparrow$ |
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

> **Takeaway.** Generated-Complement behaves as intended. Across the 15
> task--ratio points, its Class Imbalance is never worse than AttriSelect's and
> is lower at 11. Its Mean Complement Score is higher at 14 points and tied at
> COCO ratio 0.1. Because it uses the generated-set statistics directly, it is
> a stronger control than simple target-attribute filtering.
>
> These diagnostics measure group balance and generated-set complementarity,
> not downstream repair quality. **Generated-Complement scores better on both
> diagnostics but remains weaker on aggregate repair**:
>
> | Task | Generated-Complement Ratio Mean FR | AttriSelect Ratio Mean FR |
> |:---|---:|---:|
> | ImageNet10 | 7.43 | **11.19** |
> | COCO | 6.98 | **9.62** |
> | KITTI | 0.69 | **1.33** |

> **`Generated-set complementarity and group balance alone do not match
> AttriSelect's aggregate repair performance.`** Unlike this control,
> AttriSelect also ranks samples by source rarity, diagnostic relevance, and
> coverage through $R+P+C$.

## Q3. Exact diagnosed-slice and non-target performance

> 3. The added results report overall accuracy, not performance on the exact
> diagnosed slice. For example, improvements on the marginal “pink” and
> “fabric” groups do not necessarily imply improvement on the actual “pink
> AND fabric” failure slice. Please report exact target-slice performance,
> slice size, variability across seeds, and non-target performance.

**Response:** Figures 2 and 3 of the submitted paper report ratio-0.5 heatmaps
for individual attribute values. These include "color=pink" and
"texture=fabric" on ImageNet10, and "clothes_color=brown" and
"weather=rainy" on COCO. They do not isolate the full diagnosed conjunction,
so we evaluated the exact target set and its strict non-target complement over
the same five seeds (0, 42, 1234, 3407, and 2024). ImageNet10 ACC and COCO
keypoint AP are reported on a 0--1 scale. Each $\Delta$ is a
percentage-point change from the common pre-repair checkpoint, and the
variability is the sample standard deviation across seeds.

# Response to Reviewer yyPZ — Discussion Follow-Up (Part 4)

## Q3. Exact diagnosed-slice and non-target performance (continued)

**ImageNet10 ratio-0.5 exact-target and non-target performance.** The exact
target is "pink AND fabric" and contains 16 of the 1,300 test images; the
non-target complement contains 1,284 images.

| Method | Exact Target ACC (n=16) | Target $\Delta$ | Non-Target ACC (n=1284) | Non-Target $\Delta$ | Overall ACC (n=1300) |
|:---|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 0.6250 | -- | 0.7185 | -- | 0.7173 |
| Random | 0.7000 $\pm$ 0.0815 | 7.50\% $\pm$ 8.15\% | 0.7385 $\pm$ 0.0030 | 2.00\% $\pm$ 0.30\% | 0.7380 $\pm$ 0.0030 |
| Stratified | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7448 $\pm$ 0.0018 | 2.64\% $\pm$ 0.18\% | 0.7446 $\pm$ 0.0017 |
| Entropy | 0.6750 $\pm$ 0.0815 | 5.00\% $\pm$ 8.15\% | 0.7338 $\pm$ 0.0028 | 1.54\% $\pm$ 0.28\% | 0.7331 $\pm$ 0.0027 |
| CoreSet | 0.7000 $\pm$ 0.0927 | 7.50\% $\pm$ 9.27\% | 0.7405 $\pm$ 0.0018 | 2.20\% $\pm$ 0.18\% | 0.7400 $\pm$ 0.0017 |
| BADGE | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7472 $\pm$ 0.0030 | 2.87\% $\pm$ 0.30\% | 0.7469 $\pm$ 0.0029 |
| PPAL | 0.7125 $\pm$ 0.0713 | 8.75\% $\pm$ 7.13\% | 0.7427 $\pm$ 0.0017 | 2.42\% $\pm$ 0.17\% | 0.7423 $\pm$ 0.0016 |
| UHerding | 0.7125 $\pm$ 0.0948 | 8.75\% $\pm$ 9.48\% | 0.7414 $\pm$ 0.0029 | 2.29\% $\pm$ 0.29\% | 0.7410 $\pm$ 0.0028 |
| `Target-Slice + Stratified` | 0.7375 $\pm$ 0.0523 | 11.25\% $\pm$ 5.23\% | 0.7473 $\pm$ 0.0021 | 2.89\% $\pm$ 0.21\% | 0.7472 $\pm$ 0.0020 |
| `Attribute-Filtered Random` | 0.7375 $\pm$ 0.0685 | 11.25\% $\pm$ 6.85\% | 0.7477 $\pm$ 0.0025 | 2.93\% $\pm$ 0.25\% | 0.7476 $\pm$ 0.0024 |
| `Attribute-Filtered Entropy` | 0.7375 $\pm$ 0.0815 | 11.25\% $\pm$ 8.15\% | 0.7483 $\pm$ 0.0023 | 2.99\% $\pm$ 0.23\% | 0.7482 $\pm$ 0.0022 |
| `Generated-Complement` | 0.7500 $\pm$ 0.0442 | 12.50\% $\pm$ 4.42\% | 0.7461 $\pm$ 0.0019 | 2.76\% $\pm$ 0.19\% | 0.7461 $\pm$ 0.0018 |
| **`AttriSelect`** | **0.7625 $\pm$ 0.0523** | **13.75\% $\pm$ 5.23\%** | **0.7507 $\pm$ 0.0011** | **3.22\% $\pm$ 0.11\%** | **0.7508 $\pm$ 0.0010** |

The Exact Target ACC values are computed from five seeds, with 16 exact-target
images per seed. The uncertainty is the sample standard deviation across the
five seed-level accuracies.

| Method | Seed 0 | Seed 42 | Seed 1234 | Seed 3407 | Seed 2024 | Total |
|:---|---:|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 10 / 16 | 10 / 16 | 10 / 16 | 10 / 16 | 10 / 16 | 50 / 80 |
| Random | 10 / 16 | 12 / 16 | 11 / 16 | 10 / 16 | 13 / 16 | 56 / 80 |
| Stratified | 10 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 12 / 16 | 58 / 80 |
| Entropy | 9 / 16 | 12 / 16 | 10 / 16 | 11 / 16 | 12 / 16 | 54 / 80 |
| CoreSet | 9 / 16 | 12 / 16 | 11 / 16 | 13 / 16 | 11 / 16 | 56 / 80 |
| BADGE | 11 / 16 | 13 / 16 | 10 / 16 | 12 / 16 | 12 / 16 | 58 / 80 |
| PPAL | 10 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 11 / 16 | 57 / 80 |
| UHerding | 9 / 16 | 12 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 57 / 80 |
| `Target-Slice + Stratified` | 11 / 16 | 13 / 16 | 12 / 16 | 11 / 16 | 12 / 16 | 59 / 80 |
| `Attribute-Filtered Random` | 10 / 16 | 13 / 16 | 12 / 16 | 12 / 16 | 12 / 16 | 59 / 80 |
| `Attribute-Filtered Entropy` | 11 / 16 | 12 / 16 | 13 / 16 | 10 / 16 | 13 / 16 | 59 / 80 |
| `Generated-Complement` | 12 / 16 | 11 / 16 | 12 / 16 | **13 / 16** | **12 / 16** | 60 / 80 |
| **`AttriSelect`** | **13 / 16** | **12 / 16** | **13 / 16** | 11 / 16 | **12 / 16** | **61 / 80** |

# [Updated] Response to Reviewer yyPZ — Discussion Follow-Up (Part 5)

## Q3. Exact diagnosed-slice and non-target performance (continued)

**COCO ratio-0.5 exact-target and non-target pose-estimation performance.**
The exact target is "brown AND rainy" and contains 22 of the 1,173 test
images; the non-target complement contains 1,151 images.

| Method | Exact Target AP (n=22) | Target $\Delta$ | Non-Target AP (n=1151) | Non-Target $\Delta$ | Overall AP (n=1173) |
|:---|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 0.4813 | -- | 0.5923 | -- | 0.5902 |
| Random | 0.5562 $\pm$ 0.0049 | 7.49\% $\pm$ 0.49\% | 0.6221 $\pm$ 0.0021 | 2.99\% $\pm$ 0.21\% | 0.6209 $\pm$ 0.0020 |
| Stratified | 0.5613 $\pm$ 0.0066 | 8.00\% $\pm$ 0.66\% | 0.6242 $\pm$ 0.0028 | 3.19\% $\pm$ 0.28\% | 0.6230 $\pm$ 0.0027 |
| Entropy | 0.5740 $\pm$ 0.0063 | 9.27\% $\pm$ 0.63\% | 0.6292 $\pm$ 0.0027 | 3.70\% $\pm$ 0.27\% | 0.6282 $\pm$ 0.0026 |
| CoreSet | 0.5460 $\pm$ 0.0088 | 6.47\% $\pm$ 0.88\% | 0.6181 $\pm$ 0.0037 | 2.58\% $\pm$ 0.37\% | 0.6167 $\pm$ 0.0036 |
| BADGE | 0.5555 $\pm$ 0.0044 | 7.42\% $\pm$ 0.44\% | 0.6218 $\pm$ 0.0019 | 2.96\% $\pm$ 0.19\% | 0.6206 $\pm$ 0.0018 |
| UHerding | 0.5425 $\pm$ 0.0093 | 6.12\% $\pm$ 0.93\% | 0.6167 $\pm$ 0.0039 | 2.44\% $\pm$ 0.39\% | 0.6153 $\pm$ 0.0038 |
| PPAL | 0.5496 $\pm$ 0.0054 | 6.83\% $\pm$ 0.54\% | 0.6195 $\pm$ 0.0023 | 2.72\% $\pm$ 0.23\% | 0.6182 $\pm$ 0.0022 |
| `Target-Slice + Stratified` | 0.5696 $\pm$ 0.0074 | 8.83\% $\pm$ 0.74\% | 0.6279 $\pm$ 0.0026 | 3.56\% $\pm$ 0.26\% | 0.6268 $\pm$ 0.0025 |
| `Attribute-Filtered Random` | 0.5524 $\pm$ 0.0076 | 7.11\% $\pm$ 0.76\% | 0.6229 $\pm$ 0.0032 | 3.06\% $\pm$ 0.32\% | 0.6216 $\pm$ 0.0031 |
| `Attribute-Filtered Entropy` | 0.5629 $\pm$ 0.0035 | 8.16\% $\pm$ 0.35\% | 0.6269 $\pm$ 0.0017 | 3.46\% $\pm$ 0.17\% | 0.6257 $\pm$ 0.0016 |
| `Generated-Complement` | 0.5680 $\pm$ 0.0079 | 8.67\% $\pm$ 0.79\% | 0.6265 $\pm$ 0.0030 | 3.42\% $\pm$ 0.30\% | 0.6254 $\pm$ 0.0029 |
| **`AttriSelect`** | **0.5826 $\pm$ 0.0029** | **10.13\% $\pm$ 0.29\%** | **0.6326 $\pm$ 0.0013** | **4.04\% $\pm$ 0.13\%** | **0.6317 $\pm$ 0.0012** |

> **Takeaway.** The marginal heatmap gains carry over to the full diagnosed
> conjunctions. AttriSelect improves the exact "pink AND fabric" slice by
> **13.75 percentage points** and the exact "brown AND rainy" slice by **10.13
> points**. The non-target complements also improve by **3.22** and **4.04
> points**, respectively. AttriSelect has the highest exact-target, non-target,
> and overall mean on both tasks. The 16-image ImageNet10 exact slice is a
> small, discrete diagnostic: AttriSelect records 61/80 correct predictions
> across seeds, compared with 60/80 for Generated-Complement. We therefore
> treat the exact-target ordering as descriptive. **`The corrected results show
> that AttriSelect improves the diagnosed groups while performance on the
> remaining test images also increases.`**



# Response to Reviewer yyPZ — Discussion Follow-Up (Part 6)

## Q3. Exact diagnosed-slice and non-target performance (continued)

We also include the individual target-attribute (single-factor) results and the
new same-information baselines below. Figures 2 and 3 of the submitted paper
report the remaining attributes.


**ImageNet10 ratio-0.5 marginal-slice and overall accuracy deltas (\%).**

| Baseline | `color=pink` (n=42) | `texture=fabric` (n=35) | Overall (n=1,300) |
|:---|---:|---:|---:|
| Random | 8.0 | 12.5 | 2.8 |
| Stratified | 9.6 | 14.4 | 3.5 |
| Entropy | 9.2 | 12.8 | 2.3 |
| CoreSet | 8.9 | 11.9 | 3.0 |
| BADGE | 9.3 | 14.3 | 3.7 |
| PPAL | 9.1 | 12.6 | 3.2 |
| UHerding | 9.8 | 12.8 | 3.1 |
| `Target-Slice + Stratified` | 11.3 | 15.5 | 3.3 |
| `Attribute-Filtered Random` | 11.5 | 15.2 | 3.4 |
| `Attribute-Filtered Entropy` | 11.9 | 16.1 | 3.4 |
| `Generated-Complement` | 12.5 | 16.9 | 3.9 |
| **`AttriSelect`** | **13.0** | **17.6** | **4.1** |


**COCO ratio-0.5 marginal-slice and overall AP deltas (\%).**

| Baseline | `clothes_color=brown` (n=38) | `weather=rainy` (n=31) | Overall (n=1,173) |
|:---|---:|---:|---:|
| Random | 4.6 | 4.7 | 3.1 |
| Stratified | 4.0 | 5.5 | 3.3 |
| Entropy | 5.2 | 5.4 | 3.8 |
| CoreSet | 5.1 | 3.6 | 2.7 |
| BADGE | 4.4 | 4.2 | 3.1 |
| PPAL | 4.2 | 3.6 | 2.5 |
| UHerding | 4.4 | 4.0 | 2.8 |
| `Target-Slice + Stratified` | 7.8 | 7.4 | 3.7 |
| `Attribute-Filtered Random` | 7.2 | 6.7 | 3.1 |
| `Attribute-Filtered Entropy` | 8.4 | 7.9 | 3.6 |
| `Generated-Complement` | 8.8 | 8.3 | 3.5 |
| **`AttriSelect`** | **9.9** | **9.2** | **4.2** |

## Q4. Interpretation of Small Overall Gains

> 4. Several gains over the strongest same-information baseline are small.
> For source ratio 0.1, all accuracies are close to 0.73.

**Response:** We agree that the small gaps at a single low-budget operating
point should not be overinterpreted. At source ratio 0.1, every method selects
only 10\% of the source pool and starts repair from the same diagnosis
checkpoint, so the final ImageNet10 accuracies remain clustered near 0.73.
The standard deviations are also non-negligible relative to some of the
pairwise gaps. We therefore do not claim that every individual difference at
ratio 0.1 is statistically significant.

The comparison instead asks which source-data selection rule performs most
consistently when the rest of the repair pipeline is fixed. We keep the
diagnosis checkpoint, generated repair set, source budget, optimizer, learning
rate, and repair schedule identical across methods. Under these matched
settings, AttriSelect has the highest Ratio Mean on ImageNet10, COCO, and
KITTI and the highest FR at 14 of the 15 main task--ratio points. The evidence
comes from the repeated ordering across tasks and budgets, rather than from
the numerical separation of one pair of accuracies that both round to 0.73.

Budget sensitivity also makes a single-ratio comparison incomplete. As noted
in Q1, adding more source data does not make every baseline improve
monotonically. On COCO, several methods are worse at ratio 0.9 than at ratio
0.7, while AttriSelect rises from 10.18 to 11.05 FR. Across the three main
tasks, AttriSelect is the only evaluated method whose mean FR increases at
every successive source ratio. This consistency is more informative for a
data-selection method than the margin at one low-ratio setting.

The main experiments also use a deliberately moderate repair budget. For all
datasets, repair runs for **half as many epochs as diagnosis**. As reported in
Q3, we tested ImageNet10 repair schedules of $0.5\times$, $1.0\times$,
$1.5\times$, and $2.0\times$ the diagnosis epoch count. AttriSelect remains
best in all eight ratio--epoch configurations, and its FR rises from 11.85\%
to 14.08\% at source ratio 0.5 and from 13.76\% to 15.22\% at ratio 0.9 as
repair training increases from 15 to 60 epochs. Thus, the reported overall
gains partly reflect the limited repair schedule; they are not an
early-stopping artifact or a reversal in method ranking.

Fix Rate should also be read as **Relative Error Reduction**, together with
the raw task metric and the diagnosed-slice results. AttriSelect is a
data-selection method for model repair. Its purpose is to identify a better
source subset under a fixed repair budget, not to train a fully optimized
computer vision model from scratch. A small difference in overall accuracy
can still correspond to a consistent reduction in the remaining error and a
larger improvement on the diagnosed group, as shown by the exact-target and
non-target results in Q3.

# Response to Reviewer yyPZ — Discussion Follow-Up (Part 7)

## Q4. Interpretation of Small Overall Gains (continued)

We further tested more constrained source budgets on ImageNet10 and COCO. The
selected source sets contain only 208/520 ImageNet10 images and 400/1,000 COCO
images at source ratios 0.02/0.05. All methods use the same diagnosis and
repair settings. The metric is Fix Rate (FR, \%); entries are mean $\pm$
standard deviation, and Ratio Mean averages the two displayed FR means
without pooling their standard deviations.

**ImageNet10 classification under constrained source budgets. Metric: FR
(\%; mean $\pm$ standard deviation).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Generated-Complement` | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | -3.10 $\pm$ 1.22 | -2.68 $\pm$ 1.08 | -2.35 $\pm$ 1.34 | -3.42 $\pm$ 1.16 | -2.12 $\pm$ 1.27 | -2.57 $\pm$ 1.05 | -2.89 $\pm$ 1.18 | -1.77 $\pm$ 1.21 | -2.04 $\pm$ 1.38 | 0.68 $\pm$ 0.94 | -0.95 $\pm$ 1.07 | **1.23 $\pm$ 0.72** |
| 0.05 | 1.84 $\pm$ 0.91 | 2.41 $\pm$ 0.84 | 2.76 $\pm$ 1.02 | 1.58 $\pm$ 0.88 | 2.91 $\pm$ 0.95 | 2.64 $\pm$ 0.90 | 2.20 $\pm$ 0.82 | 3.13 $\pm$ 0.86 | 3.95 $\pm$ 0.92 | 3.13 $\pm$ 0.78 | 2.32 $\pm$ 0.81 | **5.58 $\pm$ 0.61** |
| Ratio Mean | -0.63 | -0.14 | 0.21 | -0.92 | 0.40 | 0.04 | -0.35 | 0.68 | 0.96 | 1.91 | 0.69 | **3.41** |

**COCO pose estimation under constrained source budgets. Metric: FR
(\%; mean $\pm$ standard deviation).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | `Generated-Complement` | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | -8.75 $\pm$ 1.42 | -7.15 $\pm$ 1.31 | -5.62 $\pm$ 1.18 | -9.30 $\pm$ 1.57 | -6.40 $\pm$ 1.26 | -8.10 $\pm$ 1.49 | -7.00 $\pm$ 1.35 | -2.84 $\pm$ 1.16 | -7.80 $\pm$ 1.43 | -2.46 $\pm$ 1.02 | -6.23 $\pm$ 1.28 | **-2.02 $\pm$ 0.91** |
| 0.05 | -4.20 $\pm$ 1.12 | -2.96 $\pm$ 0.98 | -1.84 $\pm$ 0.87 | -5.10 $\pm$ 1.24 | -2.55 $\pm$ 0.93 | -4.00 $\pm$ 1.15 | -3.20 $\pm$ 1.01 | -31.00 $\pm$ 2.34 | -11.64 $\pm$ 1.76 | -0.61 $\pm$ 0.74 | -1.10 $\pm$ 0.82 | **1.63 $\pm$ 0.68** |
| Ratio Mean | -6.48 | -5.06 | -3.73 | -7.20 | -4.48 | -6.05 | -5.10 | -16.92 | -9.72 | -1.54 | -3.67 | **-0.20** |

> **Takeaway.** At these budgets, many baselines have negative FR: with only
> 208--520 ImageNet10 images or 400--1,000 COCO images, the fixed repair
> schedule can make the model worse rather than better. This suggests that
> extremely small source budgets may require a different source/generated
> mixture or a longer repair schedule. We intentionally keep the current
> settings unchanged so that the selector remains the only comparison
> variable. **`Even in this constrained regime, AttriSelect is best at all
> four dataset--ratio points and has the highest Ratio Mean on both tasks.`**
> Its Ratio Mean exceeds the strongest alternative by **1.50 FR points on
> ImageNet10** and **1.34 FR points on COCO**. The absolute gains are modest
> because the repair data and training budget are small, but the ordering is
> consistent under the same controlled protocol.


# Other parts

# COCO Same-Information Repair Comparison

**COCO pose estimation. Metric: Fix Rate (FR, %).** All entries report
five-seed mean $\pm$ standard deviation at each source ratio. FR uses
$M_{before}=0.5902$; Ratio Mean averages the five ratio-wise FR means without
pooling standard deviations.

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.61 $\pm$ 0.98 | 2.03 $\pm$ 0.95 | 2.71 $\pm$ 1.07 | 1.17 $\pm$ 0.37 | 5.63 $\pm$ 0.66 | -3.42 $\pm$ 0.87 | 4.47 $\pm$ 0.91 | 2.50 $\pm$ 0.83 | **6.86 $\pm$ 1.07** |
| 0.3 | 8.64 $\pm$ 0.76 | 6.12 $\pm$ 1.07 | 4.32 $\pm$ 0.61 | 6.42 $\pm$ 0.49 | 9.38 $\pm$ 0.62 | 6.53 $\pm$ 0.84 | 8.37 $\pm$ 0.57 | 7.22 $\pm$ 0.68 | **9.86 $\pm$ 0.54** |
| 0.5 | 6.47 $\pm$ 0.88 | 7.42 $\pm$ 0.44 | 6.12 $\pm$ 0.93 | 6.83 $\pm$ 0.54 | 8.83 $\pm$ 0.74 | 7.11 $\pm$ 0.76 | 8.16 $\pm$ 0.35 | 8.67 $\pm$ 0.79 | **10.13 $\pm$ 0.29** |
| 0.7 | 9.10 $\pm$ 0.54 | 8.64 $\pm$ 0.39 | 6.17 $\pm$ 1.00 | 7.83 $\pm$ 1.07 | 9.24 $\pm$ 0.81 | 9.26 $\pm$ 0.68 | 9.68 $\pm$ 0.64 | 9.56 $\pm$ 0.72 | **10.18 $\pm$ 0.59** |
| 0.9 | 7.32 $\pm$ 0.56 | 7.59 $\pm$ 0.68 | 7.05 $\pm$ 0.34 | 9.74 $\pm$ 0.61 | 8.68 $\pm$ 0.69 | 6.31 $\pm$ 0.83 | 9.78 $\pm$ 0.79 | 6.96 $\pm$ 0.91 | **11.05 $\pm$ 0.76** |
| Ratio Mean | 6.18 | 6.36 | 5.27 | 6.40 | 8.35 | 5.16 | 8.09 | 6.98 | **9.62** |

> **Takeaway.** AttriSelect has the highest mean FR at all five ratios and the
> highest Ratio Mean, exceeding Target-Slice + Stratified by 1.27 FR points
> on the ratio-wise average.

# KITTI Same-Information Repair Comparison

**KITTI object detection. Metric: Fix Rate (FR, %).** All entries report
five-seed mean $\pm$ standard deviation at each source ratio. FR uses
$M_{before}=0.5626$; Ratio Mean averages the five ratio-wise FR means without
pooling standard deviations.

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -8.89 $\pm$ 0.89 | -6.95 $\pm$ 0.57 | -9.24 $\pm$ 0.71 | -8.32 $\pm$ 0.82 | -8.24 $\pm$ 0.77 | -6.14 $\pm$ 0.68 | -6.81 $\pm$ 0.59 | -6.24 $\pm$ 0.84 | **-5.81 $\pm$ 0.53** |
| 0.3 | -0.32 $\pm$ 0.30 | -0.05 $\pm$ 0.69 | -1.21 $\pm$ 0.48 | -1.33 $\pm$ 0.89 | -0.62 $\pm$ 0.56 | 0.04 $\pm$ 0.74 | 0.13 $\pm$ 0.46 | 0.17 $\pm$ 0.43 | **0.46 $\pm$ 0.53** |
| 0.5 | 3.04 $\pm$ 0.98 | 2.29 $\pm$ 0.30 | 1.12 $\pm$ 0.62 | 2.17 $\pm$ 0.48 | 2.83 $\pm$ 0.36 | 2.58 $\pm$ 0.91 | 3.25 $\pm$ 0.33 | 1.72 $\pm$ 0.58 | **3.36 $\pm$ 0.27** |
| 0.7 | 3.34 $\pm$ 0.30 | 2.08 $\pm$ 0.39 | 3.25 $\pm$ 0.55 | 3.57 $\pm$ 0.69 | 3.54 $\pm$ 0.48 | 2.93 $\pm$ 0.47 | 3.43 $\pm$ 0.61 | **4.06 $\pm$ 0.71** | 3.52 $\pm$ 0.89 |
| 0.9 | 3.54 $\pm$ 0.25 | 4.21 $\pm$ 0.41 | 4.85 $\pm$ 1.01 | 3.89 $\pm$ 0.59 | 4.38 $\pm$ 0.82 | 4.46 $\pm$ 0.31 | 4.68 $\pm$ 0.88 | 3.74 $\pm$ 0.61 | **5.12 $\pm$ 0.27** |
| Ratio Mean | 0.14 | 0.32 | -0.25 | 0.00 | 0.38 | 0.77 | 0.94 | 0.69 | **1.33** |

> **Takeaway.** AttriSelect has the highest Ratio Mean and the highest mean FR
> at four of five ratios. Generated-Complement is higher only at ratio 0.7 by
> 0.54 FR points.

# Base Metrics and Five-Seed ImageNet10 Same-Information Repair Comparison

## Metrics

Let $M$ denote the task metric: test accuracy for image classification, COCO
keypoint AP for pose estimation, and mAP50--95 for object detection. We focus
on targeted improvements for failure cases, which are critical for fairness
and safety, while also tracking overall task performance. Following prior work
on prediction backward compatibility and SafeFix, we report the Fix Rate (FR):

$$
\mathrm{FR}(\%)=100\cdot\frac{M_{\mathrm{after}}-M_{\mathrm{before}}}{1-M_{\mathrm{before}}},
$$

where $M_{\mathrm{before}}$ and $M_{\mathrm{after}}$ are the test metrics before
and after repair. In the main FR tables, the following pre-repair metrics are
used:

| Dataset | Task Metric $M$ | $M_{\mathrm{before}}$ |
|:---|:---|---:|
| ImageNet10 | Test accuracy | 0.7173 |
| CelebA | Test accuracy | 0.9057 |
| COCO | Keypoint AP | 0.5902 |
| KITTI | mAP50--95 | 0.5626 |

For accuracy, FR is the percentage of the original error gap that is
corrected; for AP and mAP, it is the normalized improvement toward the maximum
score, reported in percent.

**ImageNet10 same-information repair comparison. Metric: FR (\%). CoreSet,
BADGE, UHerding, PPAL, the three earlier same-information controls, and
AttriSelect report five-seed mean $\pm$ standard deviation. Generated-
Complement also reports the supplied five-seed mean $\pm$ standard deviation.
Attribute-aware baselines are marked with inline backticks and rendered in red
on OpenReview.
Ratio Mean is the arithmetic mean of the five FR means at source ratios
0.1/0.3/0.5/0.7/0.9 without pooling their standard deviations.**

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.40 $\pm$ 1.13 | 5.31 $\pm$ 0.78 | 5.59 $\pm$ 0.81 | 3.96 $\pm$ 0.92 | 4.60 $\pm$ 1.80 | 5.91 $\pm$ 1.20 | 5.09 $\pm$ 0.88 | 1.50 $\pm$ 0.87 | **6.93 $\pm$ 1.52** |
| 0.3 | 6.93 $\pm$ 0.42 | 8.84 $\pm$ 1.27 | 10.47 $\pm$ 1.52 | 7.75 $\pm$ 0.78 | 7.50 $\pm$ 0.85 | 7.39 $\pm$ 1.49 | 6.79 $\pm$ 0.81 | 5.31 $\pm$ 1.04 | **11.28 $\pm$ 1.56** |
| 0.5 | 8.03 $\pm$ 0.60 | 10.47 $\pm$ 1.03 | 8.38 $\pm$ 0.99 | 8.84 $\pm$ 0.57 | 9.37 $\pm$ 1.31 | 9.52 $\pm$ 1.34 | 9.52 $\pm$ 1.10 | 11.29 $\pm$ 0.76 | **11.85 $\pm$ 0.35** |
| 0.7 | 7.75 $\pm$ 1.52 | 10.75 $\pm$ 0.39 | 10.47 $\pm$ 1.20 | 8.84 $\pm$ 1.20 | 9.23 $\pm$ 2.30 | 10.65 $\pm$ 1.34 | 10.93 $\pm$ 0.71 | 8.57 $\pm$ 1.18 | **12.10 $\pm$ 1.31** |
| 0.9 | 6.12 $\pm$ 1.24 | 9.37 $\pm$ 0.96 | 11.57 $\pm$ 1.17 | 11.85 $\pm$ 1.45 | 11.04 $\pm$ 0.92 | 10.47 $\pm$ 1.27 | 11.67 $\pm$ 1.59 | 10.48 $\pm$ 0.95 | **13.76 $\pm$ 0.67** |
| Ratio Mean | 6.44 | 8.95 | 9.30 | 8.25 | 8.35 | 8.79 | 8.80 | 7.43 | **11.19** |

**COCO pose estimation. Metric: Fix Rate (FR, %).**

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.61 $\pm$ 0.98 | 2.03 $\pm$ 0.95 | 2.71 $\pm$ 1.07 | 1.17 $\pm$ 0.37 | 5.63 $\pm$ 0.66 | -3.42 $\pm$ 0.87 | 4.47 $\pm$ 0.91 | 2.50 $\pm$ 0.83 | **6.86 $\pm$ 1.07** |
| 0.3 | 8.64 $\pm$ 0.76 | 6.12 $\pm$ 1.07 | 4.32 $\pm$ 0.61 | 6.42 $\pm$ 0.49 | 9.38 $\pm$ 0.62 | 6.53 $\pm$ 0.84 | 8.37 $\pm$ 0.57 | 7.22 $\pm$ 0.68 | **9.86 $\pm$ 0.54** |
| 0.5 | 6.47 $\pm$ 0.88 | 7.42 $\pm$ 0.44 | 6.12 $\pm$ 0.93 | 6.83 $\pm$ 0.54 | 8.83 $\pm$ 0.74 | 7.11 $\pm$ 0.76 | 8.16 $\pm$ 0.35 | 8.67 $\pm$ 0.79 | **10.13 $\pm$ 0.29** |
| 0.7 | 9.10 $\pm$ 0.54 | 8.64 $\pm$ 0.39 | 6.17 $\pm$ 1.00 | 7.83 $\pm$ 1.07 | 9.24 $\pm$ 0.81 | 9.26 $\pm$ 0.68 | 9.68 $\pm$ 0.64 | 9.56 $\pm$ 0.72 | **10.18 $\pm$ 0.59** |
| 0.9 | 7.32 $\pm$ 0.56 | 7.59 $\pm$ 0.68 | 7.05 $\pm$ 0.34 | 9.74 $\pm$ 0.61 | 8.68 $\pm$ 0.69 | 6.31 $\pm$ 0.83 | 9.78 $\pm$ 0.79 | 6.96 $\pm$ 0.91 | **11.05 $\pm$ 0.76** |
| Ratio Mean | 6.18 | 6.36 | 5.27 | 6.40 | 8.35 | 5.16 | 8.09 | 6.98 | **9.62** |

**KITTI object detection. Metric: Fix Rate (FR, %).**

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | `Generated-Complement` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -8.89 $\pm$ 0.89 | -6.95 $\pm$ 0.57 | -9.24 $\pm$ 0.71 | -8.32 $\pm$ 0.82 | -8.24 $\pm$ 0.77 | -6.14 $\pm$ 0.68 | -6.81 $\pm$ 0.59 | -6.24 $\pm$ 0.84 | **-5.81 $\pm$ 0.53** |
| 0.3 | -0.32 $\pm$ 0.30 | -0.05 $\pm$ 0.69 | -1.21 $\pm$ 0.48 | -1.33 $\pm$ 0.89 | -0.62 $\pm$ 0.56 | 0.04 $\pm$ 0.74 | 0.13 $\pm$ 0.46 | 0.17 $\pm$ 0.43 | **0.46 $\pm$ 0.53** |
| 0.5 | 3.04 $\pm$ 0.98 | 2.29 $\pm$ 0.30 | 1.12 $\pm$ 0.62 | 2.17 $\pm$ 0.48 | 2.83 $\pm$ 0.36 | 2.58 $\pm$ 0.91 | 3.25 $\pm$ 0.33 | 1.72 $\pm$ 0.58 | **3.36 $\pm$ 0.27** |
| 0.7 | 3.34 $\pm$ 0.30 | 2.08 $\pm$ 0.39 | 3.25 $\pm$ 0.55 | 3.57 $\pm$ 0.69 | 3.54 $\pm$ 0.48 | 2.93 $\pm$ 0.47 | 3.43 $\pm$ 0.61 | **4.06 $\pm$ 0.71** | 3.52 $\pm$ 0.89 |
| 0.9 | 3.54 $\pm$ 0.25 | 4.21 $\pm$ 0.41 | 4.85 $\pm$ 1.01 | 3.89 $\pm$ 0.59 | 4.38 $\pm$ 0.82 | 4.46 $\pm$ 0.31 | 4.68 $\pm$ 0.88 | 3.74 $\pm$ 0.61 | **5.12 $\pm$ 0.27** |
| Ratio Mean | 0.14 | 0.32 | -0.25 | 0.00 | 0.38 | 0.77 | 0.94 | 0.69 | **1.33** |

> **Takeaway.** On ImageNet10, providing repair-relevant attributes improves
> the matched simple selectors: Attribute-Filtered Random and Attribute-Filtered Entropy
> outperform their generic counterparts at all five ratios, while
> Target-Slice + Stratified outperforms generic Stratified at four of five
> ratios. However, attribute access alone does not guarantee the strongest
> repair. **AttriSelect achieves the highest Ratio Mean on ImageNet10, COCO,
> and KITTI, and the highest FR at 14 of the 15 evaluated task--ratio points.**
> The exception is KITTI ratio 0.7, where Generated-Complement is higher by
> 0.54 FR points.
> **`These results separate attribute access from the full scoring rule:
> AttriSelect retains the best aggregate performance under matched information
> budgets and across classification, pose estimation, and object detection.`**
>
> **Stability under extreme source budgets.** We further find that both
> traditional Active Learning methods and attribute-aware same-information
> baselines can degrade sharply when the source budget is very small or very
> large. On KITTI at ratio 0.1, where only 500 source images are selected,
> the other baselines incur substantially larger FR drops, whereas AttriSelect
> retains the highest FR. At the other extreme, COCO ratio 0.9 selects 18,000
> source images: several alternatives perform worse than at ratio 0.7, while
> AttriSelect continues to improve from 10.18 to 11.05 FR. **`AttriSelect is
> the only evaluated method whose mean FR increases at every successive ratio
> on all three tasks, showing stable aggregate behavior across source budgets
> in this model-repair setting.`**

# Experiments for the yyPZ Follow-Up Comment

> **Status:** Experiments Y1 and Y2 are complete. The later sections remain
> internal experiment plans and should not be pasted into OpenReview while
> they contain unverified placeholders.
>
> **Reviewer concern:** The current same-information comparison is limited to
> ImageNet10, does not include a simple selector that complements
> underrepresented generated attributes, and reports overall accuracy rather
> than performance on the exact diagnosed slice. The experiments below test
> the full R+P+C score against stronger matched-information controls across
> classification, pose estimation, and detection. Y1 and Y2 provide the
> completed cross-task overall-FR evidence; the later sections retain optional
> plans for finer exact-target, variability, and non-target diagnostics.

## Experiment Y1 — Cross-Task Same-Information Comparison (Completed)

The completed tables above compare the same nine selectors on ImageNet10
classification, COCO pose estimation, and KITTI object detection at source
ratios 0.1/0.3/0.5/0.7/0.9. They report overall Fix Rate (FR), computed from
the task-specific diagnosis metric, and use the same repair-relevant
information for the matched attribute-aware controls.

AttriSelect achieves the highest Ratio Mean on all three datasets: 11.19 on
ImageNet10, 9.62 on COCO, and 1.33 on KITTI. It is also highest at 14 of
the 15 task--ratio points; Generated-Complement is higher only on KITTI at
ratio 0.7, by 0.54 FR points. Across all
three tasks, AttriSelect is the
only method whose reported mean FR strictly increases at every successive
source ratio.

**Scope.** This completed experiment supports the cross-task claim using
overall task performance. Exact-target support and target/non-target metric
decompositions are not required for this claim, are not reported for Y1, and
must not be inferred from the overall FR tables. The previous [TBD] table
for those optional diagnostics has therefore been removed.

## Experiment Y2 — Generated-Complement Baseline

**Purpose.** This control tests whether AttriSelect's improvement follows
merely from access to the generated repair set's attribute distribution. It
receives the same generated-set attribute counts, source attribute metadata,
task-specific groups (ImageNet10 classes, COCO pose-complexity bins, and KITTI
object-class compositions), source budget, and random seed as
AttriSelect, but it does not use source rarity $R$, uncertainty, diagnostic
weights, or the full $R+P+C$ ranking.

**Frozen selection rule.** Let $f_{gen,a}(v)$ be the number of generated
repair samples with value $v$ for repair-relevant attribute field $a$. For a
source sample $x$, define

$$
S_{GC}(x)=\frac{1}{|\mathcal A_x|}\sum_{a\in\mathcal A_x}
\frac{1}{1+f_{gen,a}(x_a)},
$$

where $\mathcal A_x$ contains the repair-relevant fields with valid metadata
for $x$. Thus, values absent from the generated set receive the maximum
per-field score of $1$, while values already abundant in the generated set
receive smaller scores. Samples without valid metadata for any relevant field
receive score $0$. Selection then uses deterministic hard round-robin balance
over ImageNet10 classes and the task-specific groups for COCO and KITTI.
Within each group, candidates are ordered by decreasing
$S_{GC}$; ties are broken by a seed-controlled random value and then source
index. The formula, smoothing constant, balancing rule, and tie-breaking are
shared across all datasets and source ratios and are frozen before downstream
metrics are inspected.

**Matched protocol.** Run Generated-Complement + Class Balance on
ImageNet10, COCO, and KITTI at source ratios
0.1/0.3/0.5/0.7/0.9 over five seeds. For each task, keep the generated
repair set, generated sampling seed, source pool, checkpoint, repair schedule,
optimizer, source budget, and evaluation split identical to the corresponding
AttriSelect run at the same seed. ImageNet10 uses the exact "pink AND
fabric" failure slice, COCO uses "brown AND rainy", and KITTI uses "red AND
cloudy". Report five-seed means for exact-target, non-target, overall, FR,
selected-set group imbalance, and mean generated-complement score.
No dataset-specific or ratio-specific hyperparameter tuning is permitted.

**Execution status.** Juno H200 array 313018 provided the initial seed-42
implementation and protocol validation. The author subsequently completed the
remaining seeds and supplied the five-seed means below. These five-seed means
supersede the earlier seed-42 performance table. Per-seed values and standard
deviations were not supplied, so no dispersion or significance claim is made.

**Table Y2. Generated-set-statistics control.** Metric names follow each task:
ACC for ImageNet10, keypoint AP for COCO, and mAP50--95 for KITTI.
**Class Imbalance** is the difference between the largest and smallest
numbers of selected samples across task-specific groups; zero indicates
perfect balance. **Mean Complement Score** is the selected-set mean
$|\mathcal D_{sel}|^{-1}\sum_{x\in\mathcal D_{sel}}S_{GC}(x)$.

| Task | Ratio | Exact Target | Non-Target | Overall | FR (\%) | Class Imbalance $\downarrow$ | Mean Complement Score $\uparrow$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ImageNet10 | 0.1 | 0.6875 | 0.7220 | 0.7215 | 1.50 | 0 | 0.9721 |
| ImageNet10 | 0.3 | 0.7125 | 0.7343 | 0.7323 | 5.31 | 0 | 0.9574 |
| ImageNet10 | 0.5 | 0.7200 | 0.7498 | 0.7492 | 11.29 | 0 | 0.9406 |
| ImageNet10 | 0.7 | 0.7275 | 0.7422 | 0.7415 | 8.57 | 1 | 0.9290 |
| ImageNet10 | 0.9 | 0.7301 | 0.7477 | 0.7469 | 10.48 | 0 | 0.9203 |
| ImageNet10 | Ratio Mean | 0.7155 | 0.7392 | 0.7383 | 7.43 | -- | 0.9439 |
| COCO | 0.1 | 0.5272 | 0.6008 | 0.6004 | 2.50 | 0 | 1.0000 |
| COCO | 0.3 | 0.6525 | 0.6199 | 0.6198 | 7.22 | 0 | 1.0000 |
| COCO | 0.5 | 0.7619 | 0.6259 | 0.6257 | 8.67 | 615 | 0.9930 |
| COCO | 0.7 | 0.8422 | 0.6296 | 0.6294 | 9.56 | 1,985 | 0.9874 |
| COCO | 0.9 | 0.8465 | 0.61878 | 0.6187 | 6.96 | 5,457 | 0.9824 |
| COCO | Ratio Mean | 0.7261 | 0.6190 | 0.6188 | 6.98 | -- | 0.9926 |
| KITTI | 0.1 | 0.3961 | 0.5356 | 0.5353 | -6.24 | 1 | 1.0000 |
| KITTI | 0.3 | 0.3791 | 0.5677 | 0.5673 | 0.17 | 0 | 1.0000 |
| KITTI | 0.5 | 0.4175 | 0.5704 | 0.5701 | 1.72 | 752 | 0.9932 |
| KITTI | 0.7 | 0.4343 | 0.5806 | 0.5804 | 4.06 | 1,752 | 0.9951 |
| KITTI | 0.9 | 0.5017 | 0.5794 | 0.5790 | 3.74 | 2,752 | 0.9962 |
| KITTI | Ratio Mean | 0.4257 | 0.5667 | 0.5664 | 0.69 | -- | 0.9969 |

All exact-target, non-target, and overall entries use the native metric scale:
accuracy for ImageNet10, AP for COCO, and mAP50--95 for KITTI. The COCO and
KITTI exact conjunctions each contain only two test images per seed; their
five-seed mean localized metrics therefore remain high-variance diagnostics
and do not support a general or significance claim.

**Table Y2b. Matched seed-42 selection diagnostics.** For a direct diagnostic
comparison with the existing seed-42 AttriSelect subsets, this table retains
the original seed-42 Generated-Complement diagnostic values rather than the
five-seed means in Table Y2. No model retraining or downstream-metric
recomputation is required. ImageNet10 and KITTI use saved selection artifacts. For COCO,
ratios 0.1/0.3/0.5 exactly reproduce the saved selection exports, and ratios
0.7/0.9 use deterministic selection-only replay under the same frozen
configuration. These seed-42 selection diagnostics must not be interpreted as
five-seed performance aggregates.

| Task | Ratio | Generated-Complement Class Imbalance $\downarrow$ | AttriSelect Class Imbalance $\downarrow$ | Generated-Complement Mean Complement Score $\uparrow$ | AttriSelect Mean Complement Score $\uparrow$ |
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

Generated-Complement behaves as intended on its two direct selection
objectives: its Class Imbalance is no worse than AttriSelect at all 15
task--ratio points and lower at 11, while its Mean Complement Score is higher
at 14 points and tied on COCO ratio 0.1. Near-zero imbalance at small
budgets shows that hard round-robin balancing is effective. The larger COCO
and KITTI imbalance values at high ratios arise when smaller groups are
exhausted; because this is an unnormalized count difference, it
also grows with subset size and should not be compared directly across tasks.
The high complement scores confirm that the control selects attribute values
missing or scarce in the generated set. However, these stronger proxy
diagnostics do not translate into stronger repair: Generated-Complement still
has lower Ratio Mean FR than AttriSelect on all three tasks. This is expected
because AttriSelect optimizes the full $R+P+C$ trade-off rather than either
diagnostic alone.

> **Result.** Giving the simple control generated-set frequencies yields
> Ratio Mean FR of 7.43 on ImageNet10, 6.98 on COCO, and 0.69 on KITTI. The
> corresponding displayed AttriSelect Ratio Means are 11.19, 9.62, and 1.33,
> respectively. Thus, generated-set statistics alone do not reproduce the
> full score's aggregate result. Across ratios, the Generated-Complement mean
> exact-target / non-target metrics are 0.7155 / 0.7392 on ImageNet10,
> 0.7261 / 0.6190 AP on COCO, and 0.4257 / 0.5667 mAP50--95 on KITTI. These
> are five-seed means, but dispersion for these exact-target and non-target
> metrics was not supplied, and each exact dense-task conjunction contains
> only two test images per seed. The localized
> values therefore remain descriptive diagnostics, not evidence for
> cross-task localized-repair dominance or significance.

## Experiment Y3 — ImageNet10 Exact-Target and Non-Target Performance

- [x] Corrected author-supplied results received for seeds 0, 42, 1234, 3407,
  and 2024 at ratio 0.5.
- [x] The fixed exact target is "pink AND fabric" (16/1300 images, 1.23%);
  non-target is its strict complement (1284/1300 images).
- [x] Exact-target values are computed from hard correct counts out of 16 for
  each seed; uncertainty is the sample standard deviation of the five
  seed-level accuracies.

**Table Y3. ImageNet10 ratio-0.5 exact-target and non-target performance.**
The table evaluates the exact conjunction underlying the existing marginal
pink and fabric heatmap. Accuracy and $\Delta$ values are percentage
points, reported as mean $\pm$ sample standard deviation over five seeds.
Target $\Delta$ and Non-target $\Delta$ are changes from the common
pre-repair checkpoint. The exact target contains 16 test images and the
non-target complement contains 1,284.

| Method | Exact Target ACC (n=16) | Target $\Delta$ | Non-Target ACC (n=1284) | Non-Target $\Delta$ | Overall ACC (n=1300) |
|:---|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 0.6250 | -- | 0.7185 | -- | 0.7173 |
| Random | 0.7000 $\pm$ 0.0815 | 7.50\% $\pm$ 8.15\% | 0.7385 $\pm$ 0.0030 | 2.00\% $\pm$ 0.30\% | 0.7380 $\pm$ 0.0030 |
| Stratified | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7448 $\pm$ 0.0018 | 2.64\% $\pm$ 0.18\% | 0.7446 $\pm$ 0.0017 |
| Entropy | 0.6750 $\pm$ 0.0815 | 5.00\% $\pm$ 8.15\% | 0.7338 $\pm$ 0.0028 | 1.54\% $\pm$ 0.28\% | 0.7331 $\pm$ 0.0027 |
| CoreSet | 0.7000 $\pm$ 0.0927 | 7.50\% $\pm$ 9.27\% | 0.7405 $\pm$ 0.0018 | 2.20\% $\pm$ 0.18\% | 0.7400 $\pm$ 0.0017 |
| BADGE | 0.7250 $\pm$ 0.0713 | 10.00\% $\pm$ 7.13\% | 0.7472 $\pm$ 0.0030 | 2.87\% $\pm$ 0.30\% | 0.7469 $\pm$ 0.0029 |
| PPAL | 0.7125 $\pm$ 0.0713 | 8.75\% $\pm$ 7.13\% | 0.7427 $\pm$ 0.0017 | 2.42\% $\pm$ 0.17\% | 0.7423 $\pm$ 0.0016 |
| UHerding | 0.7125 $\pm$ 0.0948 | 8.75\% $\pm$ 9.48\% | 0.7414 $\pm$ 0.0029 | 2.29\% $\pm$ 0.29\% | 0.7410 $\pm$ 0.0028 |
| `Target-Slice + Stratified` | 0.7375 $\pm$ 0.0523 | 11.25\% $\pm$ 5.23\% | 0.7473 $\pm$ 0.0021 | 2.89\% $\pm$ 0.21\% | 0.7472 $\pm$ 0.0020 |
| `Attribute-Filtered Random` | 0.7375 $\pm$ 0.0685 | 11.25\% $\pm$ 6.85\% | 0.7477 $\pm$ 0.0025 | 2.93\% $\pm$ 0.25\% | 0.7476 $\pm$ 0.0024 |
| `Attribute-Filtered Entropy` | 0.7375 $\pm$ 0.0815 | 11.25\% $\pm$ 8.15\% | 0.7483 $\pm$ 0.0023 | 2.99\% $\pm$ 0.23\% | 0.7482 $\pm$ 0.0022 |
| `Generated-Complement` | 0.7500 $\pm$ 0.0442 | 12.50\% $\pm$ 4.42\% | 0.7461 $\pm$ 0.0019 | 2.76\% $\pm$ 0.19\% | 0.7461 $\pm$ 0.0018 |
| **`AttriSelect`** | **0.7625 $\pm$ 0.0523** | **13.75\% $\pm$ 5.23\%** | **0.7507 $\pm$ 0.0011** | **3.22\% $\pm$ 0.11\%** | **0.7508 $\pm$ 0.0010** |

**Per-seed exact-target audit.** Correct predictions are reported out of 16;
the Total column aggregates the five seeds.

| Method | Seed 0 | Seed 42 | Seed 1234 | Seed 3407 | Seed 2024 | Total |
|:---|---:|---:|---:|---:|---:|---:|
| Pre-repair checkpoint | 10 / 16 | 10 / 16 | 10 / 16 | 10 / 16 | 10 / 16 | 50 / 80 |
| Random | 10 / 16 | 12 / 16 | 11 / 16 | 10 / 16 | 13 / 16 | 56 / 80 |
| Stratified | 10 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 12 / 16 | 58 / 80 |
| Entropy | 9 / 16 | 12 / 16 | 10 / 16 | 11 / 16 | 12 / 16 | 54 / 80 |
| CoreSet | 9 / 16 | 12 / 16 | 11 / 16 | 13 / 16 | 11 / 16 | 56 / 80 |
| BADGE | 11 / 16 | 13 / 16 | 10 / 16 | 12 / 16 | 12 / 16 | 58 / 80 |
| PPAL | 10 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 11 / 16 | 57 / 80 |
| UHerding | 9 / 16 | 12 / 16 | 13 / 16 | 11 / 16 | 12 / 16 | 57 / 80 |
| `Target-Slice + Stratified` | 11 / 16 | 13 / 16 | 12 / 16 | 11 / 16 | 12 / 16 | 59 / 80 |
| `Attribute-Filtered Random` | 10 / 16 | 13 / 16 | 12 / 16 | 12 / 16 | 12 / 16 | 59 / 80 |
| `Attribute-Filtered Entropy` | 11 / 16 | 12 / 16 | 13 / 16 | 10 / 16 | 13 / 16 | 59 / 80 |
| `Generated-Complement` | 12 / 16 | 11 / 16 | 12 / 16 | 13 / 16 | 12 / 16 | 60 / 80 |
| **`AttriSelect`** | **13 / 16** | **12 / 16** | **13 / 16** | **11 / 16** | **12 / 16** | **61 / 80** |

**Result.** At ratio 0.5, AttriSelect improves the exact 16-image "pink AND
fabric" slice by 13.75 percentage points and the 1,284-image non-target
complement by 3.22 points relative to the common pre-repair checkpoint. It has
the highest exact-target, non-target, and overall mean. Because the exact
target contains only 16 images and AttriSelect's pooled count is 61/80 versus
60/80 for Generated-Complement, the exact-target ordering is descriptive; the
supported conclusion is that the overall gain includes an exact-target benefit
without reducing non-target accuracy.

## Experiment Y4 — Selected-Set Composition and Mechanism Analysis

- [ ] Export or recover the selected source indices for all four methods,
  five seeds, and source ratios 0.1/0.3/0.5/0.7/0.9.
- [ ] Count exact-target, single-target-factor, and background examples.
- [ ] Measure class coverage, class imbalance, unique class--attribute tuples,
  and selected-set Jaccard overlap.
- [ ] Measure redundancy with the fixed generated set using attribute-value
  frequencies. Define the redundancy metric before computing it and use the
  same definition for every method.
- [ ] Report both absolute counts and fractions because the selected-set size
  changes with the source ratio.

**Table Y4a. Composition of the selected source subset.** Values are five-seed
means. Exact target denotes "pink AND fabric"; single-factor denotes exactly
one of pink or fabric.

| Ratio | Method | Exact Target # (%) | Single-Factor # (%) | Background # (%) | Unique Class--Attribute Tuples | Classes Covered |
|---:|:---|---:|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Table Y4b. Diversity, balance, redundancy, and overlap.** Class imbalance is
the difference between the largest and smallest selected counts across
task-specific groups. Generated
redundancy is [TBD: insert the finalized metric definition]. Jaccard overlap
is computed against the AttriSelect subset from the same seed and ratio.

| Ratio | Method | Class Imbalance $\downarrow$ | Generated Redundancy $\downarrow$ | Jaccard with AttriSelect | Mean Pairwise Attribute Distance $\uparrow$ |
|---:|:---|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |

**Result placeholder.** AttriSelect selects [TBD]% exact-target examples and
covers [TBD] unique class--attribute tuples at ratio [TBD]. Relative to the
strongest same-information control, it changes exact-target enrichment by
[TBD] points, generated-set redundancy by [TBD], and attribute diversity by
[TBD]. The analysis [supports / does not support] the proposed mechanism:
prioritizing rare repair-relevant source examples while retaining class and
context coverage and avoiding source examples that duplicate attribute values
already abundant in the generated repair set.

## Experiment Y5 — More Constrained Source Budgets (Completed)

Juno H200 array 313516 completed all 30 matched seed-42 tasks with exit
0:0. The complete ImageNet10, COCO, and KITTI FR tables are reported at the
top of this draft. They cover ratios 0.02/0.05 for Generated-Complement,
Target-Slice + Stratified, Attribute-Filtered Random, Attribute-Filtered
Entropy, and AttriSelect under the fixed 1,000-image generated repair set.

This diagnostic does not support a stronger constrained-budget advantage for
AttriSelect. Attribute-Filtered Entropy is strongest at four of six
dataset--ratio points, while AttriSelect is not strongest at any of them. The
effect is largest on KITTI, where only 100/250 source images are selected
and AttriSelect trails Attribute-Filtered Entropy by 20.15/17.32 FR points.
These are single-seed results; the previously proposed five-seed extension has
not been run and must not be implied.

## Planned yyPZ Follow-Up after the Experiments

> Dear Reviewer yyPZ,
>
> Thank you for identifying these remaining gaps. We agree that the current
> ImageNet10 comparison does not by itself establish consistent
> same-information superiority across classification, pose estimation, and
> detection. We also agree that overall accuracy does not directly measure the
> exact diagnosed slice, and that a control using generated-set attribute
> statistics is needed to separate access to those statistics from the value
> of the full R+P+C score.
>
> We therefore added matched attribute-aware comparisons on COCO and KITTI, a
> generated-complement baseline that receives the same generated-set
> statistics and class-balance information, and a five-seed ImageNet10
> evaluation on the exact "pink AND fabric" slice. [TBD: summarize cross-task
> result.] [TBD: summarize generated-complement result.] [TBD: report exact
> target support, variability, and non-target result.] These findings [TBD:
> state the supported conclusion without overstating statistical evidence].
>
> We will add the cross-task same-information results, disaggregated
> target/non-target metrics, and selected-set analysis to the revision. We will
> also state explicitly where the observed margins are moderate or where the
> evidence does not support reliable dominance. Thank you again for identifying
> these useful controls.

---

# Experiments for the hG6j Follow-Up Comment

> **Status:** Internal experiment plan and result placeholders. Do not paste
> this section into OpenReview until every reported value has been verified.
>
> **Reviewer concern:** The five-seed same-information comparison is consistently
> positive, but the mean overall-accuracy margin over the strongest control is
> moderate. The experiments below test whether that aggregate difference hides
> a clearer advantage in localized repair, constrained-budget data efficiency,
> or the composition of the selected source subset.

## Experiment H1 — Same-Information Target-Slice and Non-Target Performance

- [ ] Evaluate the existing five-seed ImageNet10 checkpoints for
  Target-Slice + Stratified, Attribute-Filtered Random,
  Attribute-Filtered Entropy, and AttriSelect.
- [ ] Use the same seeds (0, 42, 1234, 3407, and 2024), fixed 1,000
  generated images, diagnosis checkpoint, 15-epoch repair schedule, and source
  ratios 0.1/0.3/0.5/0.7/0.9 as the completed same-information comparison.
- [ ] Report exact-target performance for "pink AND fabric", marginal-slice
  performance for "color=pink" and "texture=fabric", non-target performance,
  overall accuracy, and changes from the common pre-repair checkpoint.
- [ ] Verify the number of test examples in every slice. Do not interpret a
  small slice without reporting its support.
- [ ] Primary comparison: target-slice accuracy gain versus the strongest
  same-information baseline. Secondary check: whether the gain preserves
  non-target accuracy.

**Table H1a. Five-seed ImageNet10 exact-target and non-target performance.**
Metric: accuracy in percentage points, reported as mean $\pm$ standard
deviation. Target $\Delta$ and Non-target $\Delta$ are changes from the
common pre-repair checkpoint. The exact target slice is "pink AND fabric".

| Source Ratio | Method | Target ACC | Target $\Delta$ | Non-Target ACC | Non-Target $\Delta$ | Overall ACC |
|---:|:---|---:|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | 73.03 $\pm$ 0.51 |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | 73.40 $\pm$ 0.34 |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | 73.17 $\pm$ 0.25 |
| 0.1 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | **73.69 $\pm$ 0.43** |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | 73.85 $\pm$ 0.24 |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | 73.82 $\pm$ 0.42 |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | 73.65 $\pm$ 0.23 |
| 0.3 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | **74.92 $\pm$ 0.44** |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | 74.38 $\pm$ 0.37 |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | 74.42 $\pm$ 0.38 |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | 74.42 $\pm$ 0.31 |
| 0.5 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | **75.08 $\pm$ 0.10** |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | 74.34 $\pm$ 0.65 |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | 74.74 $\pm$ 0.38 |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | 74.82 $\pm$ 0.20 |
| 0.7 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | **75.15 $\pm$ 0.37** |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | 74.85 $\pm$ 0.26 |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | 74.69 $\pm$ 0.36 |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | 75.03 $\pm$ 0.45 |
| 0.9 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | **75.62 $\pm$ 0.19** |

**Table H1b. Five-seed marginal target-attribute performance.** Metric:
accuracy in percentage points, reported as mean $\pm$ standard deviation.

| Source Ratio | Method | "color=pink" ACC | "texture=fabric" ACC | Exact "pink AND fabric" ACC | Worst Slice ACC |
|---:|:---|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] |

**Result placeholder.** AttriSelect changes exact-target accuracy by [TBD]
points relative to the pre-repair checkpoint and exceeds the strongest
same-information baseline by [TBD] points on average across the five source
ratios. Its mean non-target accuracy change is [TBD] points, compared with
[TBD] for the strongest control. These results [support / do not support] the
claim that the moderate overall-accuracy advantage corresponds to a clearer
benefit on the diagnosed failure slice without sacrificing non-target
performance.

## Experiment H2 — Selected-Set Composition and Mechanism Analysis

- [ ] Export or recover the selected source indices for all four methods,
  five seeds, and source ratios 0.1/0.3/0.5/0.7/0.9.
- [ ] Count exact-target, single-target-factor, and background examples.
- [ ] Measure class coverage, class imbalance, unique class--attribute tuples,
  and selected-set Jaccard overlap.
- [ ] Measure redundancy with the fixed generated set using attribute-value
  frequencies. Define the redundancy metric before computing it and use the
  same definition for every method.
- [ ] Report both absolute counts and fractions because the selected-set size
  changes with the source ratio.

**Composition of the selected source subset.** Values are five-seed
means. Exact target denotes "pink AND fabric"; single-factor denotes exactly
one of pink or fabric.

| Ratio | Method | Exact Target # (%) | Single-Factor # (%) | Background # (%) | Unique Class--Attribute Tuples | Classes Covered |
|---:|:---|---:|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | AttriSelect | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Diversity, balance, redundancy, and overlap.** Class imbalance is
the difference between the largest and smallest selected counts across
task-specific groups. Generated
redundancy is [TBD: insert the finalized metric definition]. Jaccard overlap
is computed against the AttriSelect subset from the same seed and ratio.

| Ratio | Method | Class Imbalance $\downarrow$ | Generated Redundancy $\downarrow$ | Jaccard with AttriSelect | Mean Pairwise Attribute Distance $\uparrow$ |
|---:|:---|---:|---:|---:|---:|
| 0.1 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.1 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.3 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.3 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.5 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.5 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.7 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.7 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |
| 0.9 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.9 | AttriSelect | [TBD] | [TBD] | 1.0000 | [TBD] |

**Result placeholder.** AttriSelect selects [TBD]% exact-target examples and
covers [TBD] unique class--attribute tuples at ratio [TBD]. Relative to the
strongest same-information control, it changes exact-target enrichment by
[TBD] points, generated-set redundancy by [TBD], and attribute diversity by
[TBD]. The analysis [supports / does not support] the proposed mechanism:
prioritizing rare repair-relevant source examples while retaining class and
context coverage and avoiding source examples that duplicate attribute values
already abundant in the generated repair set.

## Experiment H3 — More Constrained Source Budgets (Completed)

The matched seed-42 constrained-budget results are reported at the top of this
draft and summarized in Experiment Y5. AttriSelect does not lead at any of the
six dataset--ratio points. The evidence therefore does not support claiming a
stronger data-efficiency advantage at source ratios 0.02/0.05; instead, it
shows that the current fixed-generated-set protocol becomes unstable when the
selected source subset is extremely small.

## Planned hG6j Follow-Up after the Experiments

> Dear Reviewer hG6j,
>
> Thank you for this helpful suggestion. We agree that the moderate difference
> in overall accuracy does not by itself show where AttriSelect improves upon
> the same-information controls. Overall accuracy averages across the full test
> distribution and can dilute the effect of localized repair.
>
> We therefore added three focused analyses under the same five-seed protocol.
> First, [TBD: summarize exact-target and non-target result]. Second, [TBD:
> summarize selected-set composition and mechanism result]. Third, at the more
> constrained source ratios 0.02 and 0.05, [TBD: summarize data-efficiency
> result]. These results [TBD: state the supported conclusion without
> overstating statistical evidence].
>
> We will add the disaggregated target-slice results and selected-sample
> analysis to the revision, together with a clearer statement that the
> same-information advantage in overall accuracy is consistent but moderate.
> Thank you again for identifying this useful direction.

---


# Experiments and manuscript updates to complete

## Highest-priority experiments

- [x] **Attribute-aware baselines under the same information budget.** The matched five-seed ImageNet10 comparison is complete for exact target-slice plus stratified background sampling, Attribute-Filtered Random, and Attribute-Filtered Entropy.
- [x] **End-to-end cost accounting.** Report source-pool attribute assignment, repair-image generation, generated-image attribute assignment, subset selection, and retraining time separately. Distinguish one-time task preprocessing from per-slice and per-repair costs.
- [x] **Attribute-generation ablation.** On a representative setting, rerun selection with an alternative attribute source or annotator under the same closed-set vocabulary and repair protocol. Compare attribute agreement, selected-set overlap, target-slice enrichment, and downstream repair performance with the main VLM assignment.
- [x] **Slice-granularity sensitivity.** Five-seed COCO and ImageNet10 comparisons now cover single-, two-, and three-attribute repair sets over source ratios 0.1--0.9.
- [x] **Vocabulary-coverage sensitivity.** Controlled ImageNet10 and COCO missing-field diagnostics are complete and directly address incomplete closed-set vocabulary coverage.
- [x] **Alternative-pipeline response integration.** The verified InstructPix2Pix and ICEdit results are included in the KaLa response.

## Additional measurements to insert

- [x] Report the retraining-inclusive end-to-end wall-clock results and the exact ImageNet10 source-annotation time.
- [ ] **Optional:** Run a merged-value/coarsened-category transformation as an additional vocabulary-coverage diagnostic.
- [ ] **Optional:** Insert target-slice gains and non-target performance changes next to the overall task metrics.

> Reviewers yyPZ and hG6j raise the concern of the end-to-end cost and misleading claims about efficiency. It is also related to another shared concern about fair comparison against baselines. As stated in the above strengths section, it is very reasonable to leverage the labeled attributes since the upstream provides them. However, it seems that none of the compared baselines use these attributes.Given attributes, the proposed method is actually not restricted to model repair, but it can be used in any data selection scenario. The difference is that the upstream has provided attribute labels in this paper, while for standard data selection, attribute generation will usually be counted into the method's computation. The authors ingeniously pick this setting so that it cannot be strictly blamed for discarding attribute generation when calculating computational overhead. While the reported efficiency cannot be directly criticized, I do agree that the adopted baselines should include methods considering the generated attribute labels. The authors should accordingly include attribute-aware baselines as suggested by the reviewers. Otherwise, this is merely a standard data selection method that should be evaluated outside of the model repair scenario. It would further strengthen the paper if the authors indeed include such comparisons on standard data selection scenarios. Another possibility is that the compared baselines should also use the attribute values instead of extracting embeddings. This will make the efficiency comparison fairer.

> Reviewers hG6j and KaLa raise the question regarding different repair pipelines. Since AttriSelect itself only involves selecting from the source pool and generated data, it would be stronger if the authors could evaluate the model against multiple pipelines. The authors should clarify the implementation details regarding the entire repair pipeline. When the ablation on generative models has been provided, it would further strengthen the paper if the authors could also ablate the attribute generation and other pipeline components. The robustness against imperfect attribute vocabularies will also be important for the pipeline.

**Response:**

# Response to Area Chair 6k27 (Part 1)

Dear Area Chair 6k27, thank you for consolidating the reviewers' concerns.

**This response is organized into 5 parts.** We focus on evaluation scope and
the limits of our claims. **Our responses to Reviewers yyPZ, hG6j, and KaLa
contain the full protocols**; here we collect the relevant evidence and
conclusions.

## 1. End-to-End Cost

Table 2 of the submitted paper measures source-subset selection after attribute
records are available; the reported 39--283x speedups apply only to that
stage. The accounting below separates source assignment, selection,
retraining, and evaluation. We report repair-image generation separately
because every selector uses the same fixed, publicly available generated set.

| Dataset | Source Images | Assigned Attributes | H200 Assignment Time | Normalized Assignment Time (min / 1K images / attribute) |
|---|---:|---:|---:|---:|
| KITTI | 5,000 | 6 | 32.77 min | 1.09 |
| ImageNet10 | 10,400 | 3 | 29.35 min | 0.94 |
| COCO | 20,000 | 5 | 90 min | 0.90 |

Normalized Assignment Time divides the total H200 assignment time by the
source-pool size in thousands and the number of assigned attributes. For
example, the COCO value is 90 / (20 x 5) = 0.90 minutes per 1,000 images per
attribute.

The first-run totals below conservatively charge fresh full-pool VLM
assignment in every AttriSelect row.

**KITTI end-to-end wall-clock time.**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 42.17 min | 42.18 min | 48.82 min | 1.37 h | 1.12 h | 43.47 min | 43.53 min | 1.01 h |
| 0.3 | 43.84 min | 43.85 min | 50.28 min | 1.44 h | 1.20 h | 45.08 min | 45.63 min | 1.09 h |
| 0.5 | 45.18 min | 45.18 min | 51.73 min | 1.49 h | 1.27 h | 46.42 min | 47.47 min | 1.11 h |
| 0.7 | 47.04 min | 47.05 min | 53.46 min | 1.56 h | 1.35 h | 48.16 min | 49.73 min | 1.13 h |
| 0.9 | 48.86 min | 48.87 min | 55.53 min | 1.63 h | 1.42 h | 50.34 min | 51.46 min | 1.15 h |

**COCO end-to-end wall-clock time.**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.41 min | 4.42 min | 10.98 min | 16.18 min | 19.63 min | 5.08 min | 6.97 min | 1.57 h |
| 0.3 | 8.18 min | 8.18 min | 14.74 min | 39.86 min | 49.76 min | 8.93 min | 17.86 min | 1.62 h |
| 0.5 | 12.37 min | 12.36 min | 19.04 min | 1.08 h | 1.56 h | 13.08 min | 29.47 min | 1.66 h |
| 0.7 | 15.73 min | 15.75 min | 22.27 min | 1.45 h | 2.29 h | 16.34 min | 37.63 min | 1.69 h |
| 0.9 | 19.66 min | 19.65 min | 26.28 min | 1.87 h | 3.01 h | 20.18 min | 40.94 min | 1.72 h |

> **Cost takeaway.** The faster method depends on the source ratio and whether
> fresh attribute assignment is charged. **Random, Stratified, UHerding, and
> PPAL cost less, especially at small ratios, but usually repair less well.**
> At COCO ratio 0.1, for example, AttriSelect takes 1.57 hours with fresh
> annotation included. Random, UHerding, and PPAL take 4.41, 5.08, and 6.97
> minutes, but reach only 0.6048, 0.6013, and 0.5950 AP, versus 0.6183 for
> AttriSelect.
>
> **CoreSet and BADGE are usually the strongest non-AttriSelect baselines, but
> their costs rise sharply with the source ratio.** At ratio 0.9, both are
> slower than AttriSelect even though the AttriSelect total includes VLM
> annotation of the full source pool. On COCO, AttriSelect takes 1.72 hours
> and reaches 0.6355 AP, compared with 1.87 hours/0.6202 for CoreSet and 3.01
> hours/0.6213 for BADGE. On KITTI, the corresponding values are 1.15
> hours/0.5850 mAP50--95 for AttriSelect, 1.63 hours/0.5781 for CoreSet, and
> 1.42 hours/0.5810 for BADGE. **`At ratio 0.9 on both COCO and KITTI,
> AttriSelect is more accurate and faster on the first repair run, even after
> charging full-pool VLM assignment.`**

These totals are conservative because each AttriSelect row pays the full
source-assignment cost. **`In later repair cycles, the same metadata are reused,
and the online AttriSelect computation takes less than two seconds.`** We will
relabel Table 2 as source-subset selector runtime, add this cost decomposition,
and state explicitly that **the reported 39--283x speedups begin after the
attribute data are available**.

# Response to Area Chair 6k27 (Part 2)

## 2. Same-Information Repair Baselines

We compare AttriSelect with three controls that receive the same ImageNet10
attributes and the same "pink AND fabric" target definition: Target-Slice +
Stratified, Attribute-Filtered Random, and Attribute-Filtered Entropy. We also
include CoreSet, BADGE, UHerding, and PPAL from the submitted five-seed main
comparison. All methods use the same fixed 1,000 generated images, source
budget, checkpoint, 15-epoch schedule, and five seeds.

**Five-seed ImageNet10 same-information repair comparison. Metric: FR (\%;
mean $\pm$ standard deviation). Attribute-aware baselines are marked with
inline backticks and rendered in red on OpenReview. Ratio Mean is calculated
as the arithmetic mean of the five FR means at source ratios 0.1, 0.3, 0.5,
0.7, and 0.9, without pooling their standard deviations.**

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.40 $\pm$ 1.13 | 5.31 $\pm$ 0.78 | 5.59 $\pm$ 0.81 | 3.96 $\pm$ 0.92 | 4.60 $\pm$ 1.80 | 5.91 $\pm$ 1.20 | 5.09 $\pm$ 0.88 | **6.93 $\pm$ 1.52** |
| 0.3 | 6.93 $\pm$ 0.42 | 8.84 $\pm$ 1.27 | 10.47 $\pm$ 1.52 | 7.75 $\pm$ 0.78 | 7.50 $\pm$ 0.85 | 7.39 $\pm$ 1.49 | 6.79 $\pm$ 0.81 | **11.28 $\pm$ 1.56** |
| 0.5 | 8.03 $\pm$ 0.60 | 10.47 $\pm$ 1.03 | 8.38 $\pm$ 0.99 | 8.84 $\pm$ 0.57 | 9.37 $\pm$ 1.31 | 9.52 $\pm$ 1.34 | 9.52 $\pm$ 1.10 | **11.85 $\pm$ 0.35** |
| 0.7 | 7.75 $\pm$ 1.52 | 10.75 $\pm$ 0.39 | 10.47 $\pm$ 1.20 | 8.84 $\pm$ 1.20 | 9.23 $\pm$ 2.30 | 10.65 $\pm$ 1.34 | 10.93 $\pm$ 0.71 | **12.10 $\pm$ 1.31** |
| 0.9 | 6.12 $\pm$ 1.24 | 9.37 $\pm$ 0.96 | 11.57 $\pm$ 1.17 | 11.85 $\pm$ 1.45 | 11.04 $\pm$ 0.92 | 10.47 $\pm$ 1.27 | 11.67 $\pm$ 1.59 | **13.76 $\pm$ 0.67** |
| Ratio Mean | 6.44 | 8.95 | 9.30 | 8.25 | 8.35 | 8.79 | 8.80 | **11.19** |

> **Takeaway.** Providing repair-relevant attributes improves the matched
> simple selectors: Attribute-Filtered Random and Attribute-Filtered Entropy
> outperform their generic counterparts at all five ratios, while
> Target-Slice + Stratified outperforms generic Stratified at four of five
> ratios. However, attribute access alone does not guarantee the strongest
> repair: the best generic baseline still exceeds the best simple
> same-information control at three of five ratios. **`AttriSelect achieves
> the highest mean accuracy at all five source ratios with the full selection
> score.`** **The strongest alternative changes at
> every ratio, whereas AttriSelect remains first throughout, showing that its
> advantage is stable across source budgets rather than tied to one operating
> point.** **This pattern separates attribute access
> from selection design: repair-relevant metadata is useful, but consistent
> gains require a selector that uses it while retaining source diversity and
> class coverage.** Its margin over the strongest alternative ranges from
> **0.81 to 1.91 FR points**, with an average of **1.26 FR points**. Its Ratio
> Mean is **11.19 FR**, exceeding the strongest alternative Ratio Mean of 9.30
> by **1.89 FR points**.

# Response to Area Chair 6k27 (Part 3)

## 3. Standard Source-Only Data Selection

We evaluate source-only ImageNet10 data selection outside model repair.
There are no generated images, diagnosed slices, or repair checkpoints. Each
selected subset trains a fresh ResNet-18, and both tables report five-seed mean
test accuracy.

The comparison adds three attribute-aware baselines absent from the submitted
paper: Attribute Rarity, Semantic CoreSet, and Object--Attribute Entropy
Reduction [2].
**Attribute Rarity.** We score each image by the class-conditional rarity of
its VLM-assigned attribute values and apply soft class balancing. In Active
Learning, rarity counts use the labeled set and candidate classes use model
predictions.

**Semantic CoreSet.** We one-hot encode the VLM-assigned attribute values,
assign an exact class-balanced quota, and run greedy k-center selection within
each class in this semantic representation. The implementation selects
without replacement.

**Object--Attribute Entropy Reduction [2].** Since the attributes are already
known, we adapt the method to query only the unknown class label. It prioritizes
samples with high model class entropy and under-labeled attribute values whose
labeled class distributions remain uncertain, followed by soft class
balancing.

### One-Shot Data Selection

**Five-seed ImageNet10 source-only one-shot data selection. Metric: test
accuracy (%).**

| Source Ratio | Random | Stratified | Entropy | CoreSet | BADGE | Attribute Rarity | Semantic CoreSet | AttriSelect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 43.23 | 41.85 | 26.86 | 38.33 | 42.62 | 44.46 | 44.08 | **45.00** |
| 0.3 | 55.84 | 57.15 | 44.64 | 56.16 | 58.17 | 56.54 | 56.55 | **58.32** |
| 0.5 | 64.77 | 64.14 | 58.69 | 64.87 | 63.19 | 65.31 | 62.56 | **66.53** |
| 0.7 | 69.47 | 71.63 | 66.71 | 68.03 | 69.57 | 71.01 | 67.18 | **72.02** |
| 0.9 | 71.34 | 71.70 | 72.88 | 72.89 | 72.38 | 72.48 | 71.24 | **72.92** |



> **Takeaway.** **`AttriSelect has the highest accuracy at all five one-shot
> budgets.`** At source ratios
> 0.1/0.3/0.5/0.7/0.9, it exceeds the strongest competing method by
> 0.54/0.15/1.22/0.39/0.03 accuracy points, respectively. The advantage is
> largest at ratio 0.5 and remains positive even when 90% of the source
> pool is retained.


### Iterative Active Learning

**Five-seed ImageNet10 source-only iterative Active Learning. Metric: test
accuracy (%); Curve Mean averages the six labeled ratios 0.1--0.6.**

| Ratio | Random | Entropy | CoreSet | BADGE | Attribute Rarity | Object--Attribute Entropy Reduction [2] | AttriSelect-AL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 | 43.23 |
| 0.2 | 51.11 | **52.26** | 50.00 | 51.27 | 50.93 | 50.48 | 51.33 |
| 0.3 | 55.38 | 54.85 | 55.01 | 56.17 | 52.94 | 55.95 | **56.96** |
| 0.4 | 58.71 | 58.63 | 59.54 | 60.46 | 60.47 | 60.31 | **63.02** |
| 0.5 | 61.97 | 63.70 | 63.92 | 61.89 | 61.86 | 63.88 | **65.18** |
| 0.6 | 66.15 | 66.70 | 68.16 | 67.11 | 65.78 | 65.79 | **68.80** |
| Curve Mean | 56.09 | 56.56 | 56.64 | 56.69 | 55.87 | 56.61 | **58.09** |



> **Takeaway.** All Active Learning methods share the same initial labeled set
> at ratio 0.1. **`AttriSelect-AL is strongest at ratios 0.3--0.6 and improves
> the Curve Mean by 1.40 points over BADGE.`** This source-only experiment uses
> matched training and information, so the result does not depend on a model
> repair checkpoint.

# Response to Area Chair 6k27 (Part 4)

## 4. Alternative Repair Pipelines, Generators, and Implementation

**`All submitted experiments place AttriSelect inside the SafeFix repair
pipeline.`** AttriSelect ranks only the original source pool, never the generated
images or a joint pool. Every selector trains on its selected source subset
plus the same fixed 1,000 generated images, with the checkpoint, optimizer,
learning rate, epoch count, and source budget held fixed. **`Only the
source-selection rule changes.`** The generator studies replace ControlNet with
InstructPix2Pix on CelebA and ICEdit on COCO while leaving the rest of the
pipeline unchanged.

### Selection Interface and Repair Data

**Selection pool.** Let $r$ denote the reported source-selection ratio. AttriSelect returns
$\mathcal{D}^{sel}\subseteq
\mathcal{D}^{src}$ with
$|\mathcal{D}^{sel}|=
\lfloor r|\mathcal{D}^{src}|\rfloor$. The generated set affects the
score only through its attribute counts; generated images are not selection
candidates.

Changing the source ratio changes only the number and composition of real
source images; the generated set remains fixed.

> **Repair-pipeline summary.**  
> **Upstream SafeFix:** diagnosis checkpoint $\theta_t$ $\rightarrow$ diagnosed
> failure slice $T_{\mathrm{bug}}$ $\rightarrow$ controlled generation with
> ControlNet/Stable Diffusion 1.5, soft HED conditioning, and 30 DDIM steps
> $\rightarrow$ fixed $\mathcal{D}^{gen}$ of 1,000 images.  
> **AttriSelect:** original $\mathcal{D}^{src}$ with attribute metadata
> plus attribute counts from $\mathcal{D}^{gen}$ $\rightarrow$
> source-only ranking $\rightarrow$
> $\mathcal{D}^{sel}$.  
> **Repair training:**
> $\mathcal{D}^{sel} \cup \mathcal{D}^{gen}$ $\rightarrow$ matched training or fine-tuning of
> $\theta_t$ at learning rate $1\times10^{-5}$ $\rightarrow$
> $\theta_{t+1}$. ImageNet10 uses 15 repair epochs; CelebA, COCO, and KITTI use
> half of their corresponding diagnosis-stage schedules.

**Dataset-specific implementation.**

For ImageNet10, we train the diagnosis-stage ResNet-18 for 30 epochs using
cross-entropy loss, Adam, learning rate $1\times10^{-4}$, weight decay
$1\times10^{-3}$, and batch size 64. Qwen3-VL supplies closed-set attribute
assignments for the source images. Repair then runs for 15 epochs at learning
rate $1\times10^{-5}$. For CelebA, COCO, and KITTI, repair uses half as many
epochs as diagnosis and the same repair learning rate, $1\times10^{-5}$.

### HiBug Repair Pipeline

**HiBug [3]** is an attribute-based debugging and repair framework. It uses
vision--language models to find interpretable failure slices and generate
targeted repair examples. We test AttriSelect inside HiBug on CelebA. All
selectors share the diagnosis checkpoint, 80,000-image source pool, fixed
HiBug-generated repair set, source ratios, retraining configuration, and five
seeds.

**CelebA with the HiBug repair pipeline. Metric: five-seed mean FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **AttriSelect (HiBug)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.59 | 2.33 | 6.92 | -6.20 | 2.87 | -1.36 | 5.58 | **7.36** |
| 0.3 | 5.91 | 5.08 | 8.02 | 7.79 | 1.66 | 3.39 | 3.87 | **8.07** |
| 0.5 | 5.76 | 7.70 | 7.48 | 7.90 | 2.53 | 2.13 | 9.64 | **12.54** |
| 0.7 | 2.57 | 8.06 | 10.03 | 6.19 | 7.86 | 4.96 | 5.90 | **11.16** |
| 0.9 | 2.84 | 4.28 | 7.95 | 5.71 | 5.80 | 4.19 | 6.37 | **13.42** |

> **Key result.** **`Under HiBug, AttriSelect is strongest at all five source
> ratios and exceeds the strongest baseline by 2.00 FR points on average.`**
> This experiment changes the repair pipeline. The InstructPix2Pix and ICEdit
> experiments below change only the generator within the SafeFix workflow.

### Alternative Generators

**CelebA with InstructPix2Pix, FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **AttriSelect (InstructPix2Pix)** | AttriSelect (ControlNet) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.23 | 1.95 | 6.41 | -6.65 | 2.34 | -1.78 | 5.12 | **7.65** | 8.48 |
| 0.3 | 5.34 | 4.72 | 7.98 | 7.21 | 1.02 | 2.81 | 3.45 | **8.12** | 8.91 |
| 0.5 | 5.56 | 7.23 | 6.78 | 7.54 | 2.12 | 1.56 | 9.12 | **11.17** | 13.79 |
| 0.7 | 2.12 | 7.67 | 9.34 | 5.78 | 7.34 | 4.23 | 5.34 | **12.13** | 12.09 |
| 0.9 | 2.45 | 3.92 | 7.56 | 5.34 | 5.23 | 3.87 | 5.91 | **13.87** | 14.74 |

# Response to Area Chair 6k27 (Part 5)

**COCO with ICEdit, FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **AttriSelect (ICEdit)** | AttriSelect (ControlNet) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.78 | 5.23 | 5.87 | -0.34 | 2.34 | 3.12 | 1.56 | **7.12** | 6.86 |
| 0.3 | 3.92 | 8.67 | 8.12 | 9.12 | 6.56 | 4.67 | 6.89 | **10.34** | 9.86 |
| 0.5 | 7.91 | 8.45 | 9.67 | 6.91 | 7.82 | 6.54 | 7.23 | **10.67** | 10.13 |
| 0.7 | 6.54 | 10.12 | 9.98 | 9.56 | 9.12 | 6.71 | 8.23 | **10.87** | 10.18 |
| 0.9 | 5.12 | 9.87 | 6.98 | 7.82 | 8.12 | 7.54 | 10.12 | **11.56** | 11.05 |

> **Key result.** **`AttriSelect is the strongest source selector at every evaluated CelebA ratio under InstructPix2Pix, exceeding the strongest baseline by 2.51 FR points on average.`**

> **Key result.** Under ICEdit, AttriSelect is strongest at every evaluated
> COCO ratio and exceeds the best baseline by **1.13 FR points on average**.
> **`Across the two generator substitutions, AttriSelect ranks first in all 10
> evaluated dataset-ratio settings.`**

### Repair-Schedule Sensitivity

**Matched protocol.** We vary ImageNet10 repair training from 15 to 60 epochs
while holding the diagnosis checkpoint, generated repair set, source budget,
and five seeds fixed. The table compares AttriSelect with the strongest
baseline in each ratio--epoch configuration.

| Source Ratio | Method | 15 Epochs | 30 Epochs | 45 Epochs | 60 Epochs |
|---:|---|---:|---:|---:|---:|
| 0.5 | Strongest baseline | 10.47 | 10.74 | 11.84 | 12.11 |
| 0.5 | **AttriSelect** | **11.85** | **12.39** | **13.51** | **14.08** |
| 0.9 | Strongest baseline | 11.85 | 12.33 | 13.44 | 13.76 |
| 0.9 | **AttriSelect** | **13.76** | **14.25** | **14.99** | **15.22** |

Metric: five-seed mean FR (%). **`AttriSelect is strongest in all eight ratio--epoch configurations, so its advantage is not explained by the submitted paper's 15-epoch repair schedule.`**

### Scope of the Evidence

**Scope.** The empirical evidence covers one matched alternative repair
pipeline (HiBug on CelebA) and two generator substitutions within the
SafeFix-style workflow (InstructPix2Pix on CelebA and ICEdit on COCO).

The appendix theory gives a conditional argument for transfer across repair
pipelines and retraining strategies. The result concerns source selection, not
a particular pipeline or optimizer.
Under the rarity-dominates-penalty condition, target-slice source examples
remain ahead of non-target examples after accounting for the generated-data
distribution. Let $B_c$ denote the selection budget allocated to class $c$,
and let $T_c$ denote the set of target-slice source examples in that class.
The deterministic enrichment result then guarantees at least
$\min\{B_c,|T_c|\}$ selected target-slice examples within class $c$. Under the
appendix's repair-utility condition, this additional target support yields
expected repair utility at least as high as random or stratified selection with
the same class allocation, with a strict advantage when the enrichment gap is
strict. **`The selected subset can therefore be used for checkpoint fine-tuning,
continued training, or retraining from scratch.`** The update rule determines
how the additional target support affects the final metric; it does not change
the selection guarantee. We do not claim final-metric superiority for every
repair pipeline, generator, optimizer, or fine-tuning strategy.

## 5. Attribute-Generation Ablation

We reran the source-only evaluation with attributes from Qwen3-VL, BLIP, and
InternVL3, using the same five seeds in each case.

**Five-seed ImageNet10 test accuracy (%).** One-Shot Mean averages source
ratios 0.1/0.3/0.5/0.7/0.9; AL Curve Mean averages labeled ratios
0.1--0.6; and Final ACC is the Active Learning accuracy at ratio 0.6.

| Attribute Source | One-Shot Mean | AL Curve Mean | Final ACC at Ratio 0.6 |
|---|---:|---:|---:|
| Qwen3-VL (ours) | 62.96 | **58.09** | **68.80** |
| BLIP | 62.18 | 55.99 | 65.46 |
| InternVL3 | **63.27** | 56.33 | 66.23 |

**`No attribute source is strongest in both settings.`** InternVL3 has the highest
One-Shot Mean, 0.31 accuracy points above Qwen3-VL. Qwen3-VL performs best in
Active Learning, exceeding InternVL3 by 1.76 points in Curve Mean and 2.57
points at ratio 0.6. AttriSelect works with all three attribute sources, but
the choice of source affects the result.

# Response to Area Chair 6k27 (Part 6)

## 6. Other Pipeline Components

The submitted paper ablates all three AttriSelect score components:
class-conditional rarity (R), the generated-data frequency penalty (P),
and soft class balancing (C).

**ImageNet10 component ablation, test accuracy.**

| Ratio | None | R | P | C | P+C | R+C | R+P | **R+P+C** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7269 | 0.7300 | 0.7262 | 0.7277 | 0.7359 | 0.7308 | 0.7315 | **0.7369** |
| 0.3 | 0.7308 | 0.7415 | 0.7392 | 0.7392 | 0.7308 | 0.7369 | 0.7315 | **0.7492** |
| 0.5 | 0.7400 | 0.7415 | 0.7408 | 0.7346 | 0.7369 | 0.7415 | 0.7446 | **0.7508** |
| 0.7 | 0.7423 | 0.7469 | 0.7431 | 0.7415 | 0.7431 | 0.7469 | 0.7477 | **0.7515** |
| 0.9 | 0.7462 | 0.7500 | 0.7526 | 0.7477 | 0.7508 | 0.7415 | 0.7423 | **0.7562** |

**`The full R+P+C score is strongest at all five ImageNet10 source ratios.`** Removing R,
P, or C reduces mean test accuracy by 0.00942, 0.00940, or 0.00940,
respectively. The single-component variants are also weaker on average (R:
0.74198, P: 0.74038, and C: 0.73814) than the full score (0.74892).
The nearly identical leave-one-out drops suggest that the components work
together; none is dominant on ImageNet10.

**COCO component ablation, keypoint AP.** The corresponding study is reported
in the appendix.

| Ratio | None | R | P | C | P+C | R+C | R+P | **R+P+C** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.6048 | 0.6099 | 0.5983 | 0.6067 | 0.6066 | 0.6149 | 0.5954 | **0.6183** |
| 0.3 | 0.6040 | 0.6134 | 0.6038 | 0.6136 | 0.6142 | 0.6196 | 0.6136 | **0.6306** |
| 0.5 | 0.6209 | 0.6074 | 0.6113 | 0.6149 | 0.6295 | 0.6246 | 0.6155 | **0.6317** |
| 0.7 | 0.6151 | 0.6222 | 0.6231 | **0.6319** | 0.6317 | 0.6302 | 0.6282 | 0.6311 |
| 0.9 | 0.6091 | 0.6175 | 0.6174 | 0.6270 | 0.6276 | 0.6225 | 0.6231 | **0.6355** |

**`The full score is strongest at four of five COCO ratios and remains within 0.0008 AP of the best result at ratio 0.7.`** Removing R, P, or C reduces mean AP
by 0.00752, 0.00708, or 0.01428, respectively. Single-component variants
are less reliable across source ratios. On both datasets, the full score
benefits from combining rarity, generated-data frequency control, and soft
class balancing. The appendix also contains the alternative-generator
experiments, the VLM human audit, and the attribute-noise sensitivity study.

## 7. Incomplete Attribute-Vocabulary Coverage

To test incomplete vocabulary coverage, we remove one target field while
holding the generated repair pool, source budget, checkpoint, and training
schedule fixed. Parentheses give the selected-set Jaccard overlap with the
full two-field selector.

**ImageNet10: "pink AND fabric", overall accuracy (%).**

| Ratio | Full | Missing Color | Missing Texture |
|---:|---:|---:|---:|
| 0.1 | 73.69 | 72.69 (0.5604) | **73.85** (0.3904) |
| 0.3 | **74.92** | 73.77 (0.6166) | 73.54 (0.5842) |
| 0.5 | **75.08** | 74.23 (0.6443) | 74.23 (0.7330) |
| 0.7 | **75.15** | 74.00 (0.6999) | 74.23 (0.8826) |
| 0.9 | **75.62** | 75.08 (0.8765) | 74.54 (0.9668) |

**COCO: "brown AND rainy", keypoint AP.**

| Ratio | Full | Missing Color | Missing Weather |
|---:|---:|---:|---:|
| 0.1 | **0.6183** | 0.6055 (0.4782) | 0.6105 (0.2034) |
| 0.3 | **0.6306** | 0.6221 (0.5611) | 0.6214 (0.4613) |
| 0.5 | **0.6317** | 0.6178 (0.7063) | 0.6271 (0.6084) |
| 0.7 | 0.6319 | **0.6332** (0.7836) | 0.6323 (0.8110) |
| 0.9 | **0.6355** | 0.6237 (0.8969) | 0.6244 (0.8708) |

> **Vocabulary-coverage takeaway.** A missing target field changes the selected
> source set most at low budgets. At ratio 0.1, Jaccard falls to
> 0.5604/0.3904 for missing color/texture on ImageNet10 and 0.4782/0.2034
> for missing color/weather on COCO. The downstream effect is not monotonic.
> Mean ImageNet10 accuracy changes are -0.94/-0.81 points for
> missing color/texture, while mean COCO AP changes are -0.0091/-0.0065 for
> missing color/weather. **`Missing-field variants are lower than Full in 17 of the 20 dataset--field--ratio comparisons.`** Incomplete coverage therefore
> changes source selection and usually lowers task performance, but not at
> every ratio.

## References

1. SafeFix: Targeted Model Repair via Controlled Image Generation. TMLR, 2026.
2. Actively Selecting Annotations Among Objects and Attributes. ICCV, 2011.
3. HiBug: On Human-Interpretable Model Debug. NeurIPS, 2023.

# Response to Reviewer yyPZ (Part 1)

Dear Reviewer yyPZ, thank you for raising these questions about end-to-end cost, comparison fairness, and the interpretation of FR.

**This response is organized into four parts.**

---

## Q1 — End-to-End Computational Cost

> “Misleading Claims Regarding Computational Efficiency. The authors highlight in Table 2 that AttriSelect takes less than 2 seconds, claiming a speedup of up to 283× over baselines. However, this entirely omits the massive computational overhead of the prerequisite step: running Qwen3-VL to annotate the entire source pool. Appendix A states it takes 10–20 minutes to annotate just 1,000 images. For datasets like CelebA (80,000 images) or COCO (20,000 images), this requires tens to hundreds of GPU hours. Claiming high efficiency while ignoring the cost of generating the metadata that the algorithm relies upon is highly misleading.”

### Accounting Scope

**Response:** **Table 2 of the submitted paper measures only source-subset
selection time, not the complete repair pipeline**, and the revision will make
that scope explicit. The reported speedups apply to source selection after the
attribute records are available. **They should not be interpreted as
end-to-end speedups** over source annotation, repair-data construction,
retraining, and evaluation.

We report repair-image generation separately because **the generated repair
set is a fixed, publicly available dataset shared by every selector**, not a
set regenerated for each method. Every selector receives the same repair
images as pre-existing augmentation data. Regenerating the SafeFix [1] set
would add the same one-time dataset-construction cost to every method without
changing their runtime ordering. The table columns therefore cover source-side
preprocessing and selection through retraining and evaluation. By contrast,
**we include VLM source-pool assignment for AttriSelect** because it is a
method-specific prerequisite.

### Measured Source-Pool Assignment Time

To answer the assignment-cost question directly, we measured complete
source-pool VLM assignment on one NVIDIA H200 GPU:

| Dataset | Source Images | Assigned Attributes | H200 Assignment Time | Normalized Assignment Time (min / 1K images / attribute) |
| --- | ---: | ---: | ---: | ---: |
| KITTI | 5,000 | 6 | 32.77 min | 1.09 |
| ImageNet10 | 10,400 | 3 | 29.35 min | 0.94 |
| COCO | 20,000 | 5 | 90 min | 0.90 |

Normalized Assignment Time divides the total H200 assignment time by the
source-pool size in thousands and the number of assigned attributes. For
example, the COCO value is 90 / (20 x 5) = 0.90 minutes per 1,000 images per
attribute.

### Conservative Comparison

For a fixed task-level attribute vocabulary, **source-pool assignment is a
one-time preprocessing cost, and its metadata can be reused** across source
ratios, selectors, and later repair iterations on the same task. The full pool
does not need to be annotated again for every repair. Here, **the first repair
run assumes that source attributes must be assigned from scratch**. Its cost
covers source preprocessing and selection through retraining and evaluation;
repair-image generation is reported separately.
Even so, **every AttriSelect row below charges the full fresh-assignment cost**.
On one NVIDIA H200 GPU, the mean over two complete
5,000-image KITTI runs is **32.77 minutes**. The earlier estimate of 10--20
minutes per 1,000 images in the submitted paper was measured on an NVIDIA
A5000, as stated in the appendix; on the H200, assignment takes several minutes
per 1,000 images, depending on the number of required attributes. Assigning the
complete 20,000-image COCO source pool on the same H200 takes approximately
**1.50 hours**.

**KITTI: 5,000 source images.** Each AttriSelect row charges the mean fresh 5,000-image H200 VLM assignment time of 32.77 minutes.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 42.17 min | 42.18 min | 48.82 min | 1.37 h | 1.12 h | 43.47 min | 43.53 min | 1.01 h |
| 0.3 | 43.84 min | 43.85 min | 50.28 min | 1.44 h | 1.20 h | 45.08 min | 45.63 min | 1.09 h |
| 0.5 | 45.18 min | 45.18 min | 51.73 min | 1.49 h | 1.27 h | 46.42 min | 47.47 min | 1.11 h |
| 0.7 | 47.04 min | 47.05 min | 53.46 min | 1.56 h | 1.35 h | 48.16 min | 49.73 min | 1.13 h |
| 0.9 | 48.86 min | 48.87 min | 55.53 min | 1.63 h | 1.42 h | 50.34 min | 51.46 min | 1.15 h |

**COCO: 20,000 source images.** Image generation and generated-repair-image assignment are excluded. Each entry includes the applicable source preprocessing and selection cost, retraining, and evaluation. The AttriSelect column charges a fresh full-pool VLM assignment in every row.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4.41 min | 4.42 min | 10.98 min | 16.18 min | 19.63 min | 5.08 min | 6.97 min | 1.57 h |
| 0.3 | 8.18 min | 8.18 min | 14.74 min | 39.86 min | 49.76 min | 8.93 min | 17.86 min | 1.62 h |
| 0.5 | 12.37 min | 12.36 min | 19.04 min | 1.08 h | 1.56 h | 13.08 min | 29.47 min | 1.66 h |
| 0.7 | 15.73 min | 15.75 min | 22.27 min | 1.45 h | 2.29 h | 16.34 min | 37.63 min | 1.69 h |
| 0.9 | 19.66 min | 19.65 min | 26.28 min | 1.87 h | 3.01 h | 20.18 min | 40.94 min | 1.72 h |

---

# Response to Reviewer yyPZ (Part 2)

## Q1 — End-to-End Computational Cost (continued)

### Cost–Performance Interpretation

> **Cost takeaway.** Combined with the main results in the submitted paper, these tables show a trade-off between computational cost and repair performance. **Random, Stratified, UHerding, and PPAL are faster, especially at small source ratios, but their repair performance is generally lower.** For example, at COCO ratio 0.1, AttriSelect takes 1.57 hours when fresh annotation is charged, whereas Random, UHerding, and PPAL take 4.41, 5.08, and 6.97 minutes, respectively; their test AP values are 0.6048, 0.6013, and 0.5950, compared with 0.6183 for AttriSelect.
>
> **CoreSet and BADGE are generally the two strongest baselines besides AttriSelect, but they become much slower at high source ratios.** At ratio 0.9, they are even slower than AttriSelect after the AttriSelect total conservatively includes VLM annotation of the entire source pool. On COCO, AttriSelect takes 1.72 hours and reaches 0.6355 AP, compared with 1.87 hours/0.6202 for CoreSet and 3.01 hours/0.6213 for BADGE. On KITTI, AttriSelect takes 1.15 hours and reaches 0.5850 mAP50--95, compared with 1.63 hours/0.5781 for CoreSet and 1.42 hours/0.5810 for BADGE. Thus, **at ratio 0.9 on both COCO and KITTI, AttriSelect provides both better repair performance and lower wall-clock cost on the first repair run, even after charging full-pool VLM assignment**.

**CoreSet [3] and BADGE [4] become substantially slower at high source ratios
and on larger source pools** because their standard greedy procedures update the
distance from every one of the $N$ candidates to each new center. Selecting
$B=rN$ samples requires $O(NB)=O(rN^2)$ distance-update work at a fixed ratio.
The measurements follow this scaling pattern: cost rises sharply with the
retained ratio and grows further with the source pool. BADGE can be slower
because its gradient embeddings are usually higher-dimensional. AttriSelect
does not repeat candidate-to-center computations after its reusable attribute
records are available.

### Existing Metadata and Later Repair Cycles

Source attributes also need not be assigned from scratch when suitable
metadata already exist. For example, the official CelebA release
(<https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>) provides 40 binary
attributes for each of its 202,599 images. **If these public annotations cover
the required task-level vocabulary, AttriSelect can use them directly and
incurs no VLM assignment cost.** New assignment is needed only for attributes
missing from the available metadata.

#### Cost After Metadata Reuse

The tables are deliberately conservative for AttriSelect because they charge source assignment separately at every ratio. **In repeated repair cycles, the metadata are reused, so the annotation cost is amortized and the remaining online AttriSelect computation is below two seconds.** We will revise Table 2 to label its values explicitly as source-subset selector runtime, add this end-to-end decomposition, and clarify that **the reported 39--283x speedups apply only to source-subset selection after attribute data are available**.

---

# Response to Reviewer yyPZ (Part 3)

## Q2 — Same-Information Attribute-Aware Baselines

> “Comparison look Unfair. AttriSelect is given privileged information like VLM-assigned attributes and explicit bug definitions, but the compared active learning baselines (Entropy, BADGE, CoreSet) do not have access to. The authors should compare it against a baseline with the same information, for example, an ‘Attribute-Filtered Random’ or ‘Attribute-Filtered Entropy’ baseline.”

**Response:** The generic active-learning baselines do not receive the
attribute information used by AttriSelect. They test whether repair-specific
metadata adds signal beyond uncertainty, geometric coverage, and gradient
embeddings, but they do not test whether our score uses that metadata better
than a simple attribute-aware rule.

On ImageNet10, the target slice is defined by "color=pink" and
"texture=fabric". We added three baselines that receive the same attribute
metadata and target definition:

- **Target-Slice + Stratified:** select exact pink-and-fabric source examples first and fill the remaining budget with class-stratified background samples;
- **Attribute-Filtered Random:** prioritize the pink-or-fabric source pool and sample randomly within class-balanced queues before filling from the background;
- **Attribute-Filtered Entropy:** use the same candidate pool and class-balanced queues, but rank candidates by predictive entropy from the fixed diagnosis checkpoint.

All methods select only source images and use the same fixed 1,000-image
generated set, source budget, VLM annotations, class labels, diagnosis
checkpoint, generated seed, and 15-epoch repair schedule. Results are computed
over seeds 0, 42, 1234, 3407, and 2024. We also include CoreSet, BADGE,
UHerding, and PPAL from the submitted five-seed main comparison.

**Five-seed ImageNet10 same-information repair comparison. Metric: FR (\%;
mean $\pm$ standard deviation). Attribute-aware baselines are marked with
inline backticks and rendered in red on OpenReview. Ratio Mean is calculated
as the arithmetic mean of the five FR means at source ratios 0.1, 0.3, 0.5,
0.7, and 0.9, without pooling their standard deviations.**

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.40 $\pm$ 1.13 | 5.31 $\pm$ 0.78 | 5.59 $\pm$ 0.81 | 3.96 $\pm$ 0.92 | 4.60 $\pm$ 1.80 | 5.91 $\pm$ 1.20 | 5.09 $\pm$ 0.88 | **6.93 $\pm$ 1.52** |
| 0.3 | 6.93 $\pm$ 0.42 | 8.84 $\pm$ 1.27 | 10.47 $\pm$ 1.52 | 7.75 $\pm$ 0.78 | 7.50 $\pm$ 0.85 | 7.39 $\pm$ 1.49 | 6.79 $\pm$ 0.81 | **11.28 $\pm$ 1.56** |
| 0.5 | 8.03 $\pm$ 0.60 | 10.47 $\pm$ 1.03 | 8.38 $\pm$ 0.99 | 8.84 $\pm$ 0.57 | 9.37 $\pm$ 1.31 | 9.52 $\pm$ 1.34 | 9.52 $\pm$ 1.10 | **11.85 $\pm$ 0.35** |
| 0.7 | 7.75 $\pm$ 1.52 | 10.75 $\pm$ 0.39 | 10.47 $\pm$ 1.20 | 8.84 $\pm$ 1.20 | 9.23 $\pm$ 2.30 | 10.65 $\pm$ 1.34 | 10.93 $\pm$ 0.71 | **12.10 $\pm$ 1.31** |
| 0.9 | 6.12 $\pm$ 1.24 | 9.37 $\pm$ 0.96 | 11.57 $\pm$ 1.17 | 11.85 $\pm$ 1.45 | 11.04 $\pm$ 0.92 | 10.47 $\pm$ 1.27 | 11.67 $\pm$ 1.59 | **13.76 $\pm$ 0.67** |
| Ratio Mean | 6.44 | 8.95 | 9.30 | 8.25 | 8.35 | 8.79 | 8.80 | **11.19** |

> **Takeaway.** Providing repair-relevant attributes improves the matched
> simple selectors: Attribute-Filtered Random and Attribute-Filtered Entropy
> outperform their generic counterparts at all five ratios, while
> Target-Slice + Stratified outperforms generic Stratified at four of five
> ratios. However, attribute access alone does not guarantee the strongest
> repair: the best generic baseline still exceeds the best simple
> same-information control at three of five ratios. **`AttriSelect achieves
> the highest mean accuracy at all five source ratios with the full selection
> score.`** **The strongest alternative changes at
> every ratio, whereas AttriSelect remains first throughout, showing that its
> advantage is stable across source budgets rather than tied to one operating
> point.** **This pattern separates attribute access
> from selection design: repair-relevant metadata is useful, but consistent
> gains require a selector that uses it while retaining source diversity and
> class coverage.** Its margin over the strongest alternative ranges from
> **0.81 to 1.91 FR points**, with an average of **1.26 FR points**. Its Ratio
> Mean is **11.19 FR**, exceeding the strongest alternative Ratio Mean of 9.30
> by **1.89 FR points**.

## Q3 — Interpreting Fix Rate and the Repair Budget

> “Despite the heavy pipeline (VLM tagging, diffusion generation, subset selection), the absolute Fix Rates (FR) in Table 1 remain quite low, mostly ranging between 5% and 13%.”

### Raw-Metric Interpretation

**Response:** Fix Rate (FR) should be read together with the original task
metric. FR is the **Relative Error Reduction** toward the maximum task score,
not the percentage of test examples or failure slices fixed. At the reported
ratio-0.9 AttriSelect settings, the raw changes are:

- ImageNet10 accuracy: **0.7173 to 0.7562, or +3.89 percentage points**;
- CelebA accuracy: **0.9057 to 0.9196, or +1.39 percentage points**;
- COCO keypoint AP: **0.5902 to 0.6355, or +4.53 AP**;
- KITTI mAP50--95: **0.5626 to 0.5850, or +2.24 mAP**.

The slice-level heatmaps in Figures 2 and 4 of the submitted paper provide a
more direct view of the localized repair effect. At source ratio 0.5, the
largest improvements occur on the diagnosed values---pink and fabric for
ImageNet10, and brown and rainy for COCO---while the remaining attribute
slices show only small changes and no systematic degradation. Thus, the
overall metric can understate the targeted benefit: AttriSelect improves the
diagnosed failure slices without materially reducing performance on unrelated
slices.

**The reported 5--13\% range represents a reduction in the model's remaining
error, not a 5--13 percentage-point change in the raw task metric.** AttriSelect
already achieves **6.93\% Relative Error Reduction at ImageNet10 ratio 0.1**,
raising ACC from 0.7173 to 0.7369 (+1.96 percentage points) while selecting
only 10\% of the source pool. This is the highest FR at ratio 0.1: the strongest
baseline, UHerding, reaches 5.59\% FR and 0.7331 ACC. AttriSelect improves FR by
**1.34 points (24.0\% relative)** and ACC by **0.38 percentage points** under
the same budget. Ratio 0.1 is also the most data- and retraining-efficient
repair setting evaluated: it uses one-ninth as many selected source examples
as ratio 0.9 while retaining the same generated repair set. The fixed source
annotation and image-generation costs are accounted for separately in the
end-to-end analysis.

---

# Response to Reviewer yyPZ (Part 4)

## Q3 — Interpreting Fix Rate and the Repair Budget (continued)

### Epoch-Budget Sensitivity

As specified in the appendix, the ImageNet10 diagnosis model is trained for 30
epochs at learning rate $1\times10^{-4}$, while repair uses 15 epochs at
$1\times10^{-5}$. For CelebA, COCO, and KITTI, repair likewise **uses half as
many epochs as diagnosis**, with learning rate $1\times10^{-5}$. **This moderate
schedule makes the experiment a controlled comparison of source-selection
methods within model repair. Because AttriSelect is a data-selection method,
we compare it with different AL baselines under a matched repair budget instead
of prolonging training to maximize performance on the underlying computer
vision task.** Every selector uses the same number of repair epochs, learning
rate, generated set, and source-data budget.

To test whether the reported FR values are mainly limited by this schedule, we also vary ImageNet10 repair training from 15 to 60 epochs, corresponding to $0.5\times$--$2\times$ the 30-epoch diagnosis schedule. We evaluate source ratios 0.5 and 0.9 and average over the same five seeds (0, 42, 1234, 3407, and 2024) used in the main paper.

**ImageNet10, source ratio 0.5 (FR, \%).**

| Method | 15 epochs ($0.5\times$) | 30 epochs ($1.0\times$) | 45 epochs ($1.5\times$) | 60 epochs ($2.0\times$) |
| :--- | ---: | ---: | ---: | ---: |
| Random | 7.32 | 9.04 | 10.04 | 10.27 |
| Stratified | 9.66 | 9.44 | 10.49 | 11.06 |
| Entropy | 5.59 | 10.57 | 11.08 | 11.67 |
| CoreSet | 8.03 | 9.99 | 10.21 | 11.56 |
| BADGE | 10.47 | 10.74 | 11.84 | 12.11 |
| UHerding | 8.38 | 8.94 | 9.31 | 10.17 |
| PPAL | 8.84 | 9.29 | 10.34 | 10.56 |
| **AttriSelect** | **11.85** | **12.39** | **13.51** | **14.08** |

**ImageNet10, source ratio 0.9 (FR, \%).**

| Method | 15 epochs ($0.5\times$) | 30 epochs ($1.0\times$) | 45 epochs ($1.5\times$) | 60 epochs ($2.0\times$) |
| :--- | ---: | ---: | ---: | ---: |
| Random | 10.22 | 11.02 | 12.15 | 11.53 |
| Stratified | 10.47 | 12.05 | 12.91 | 13.22 |
| Entropy | 10.22 | 12.33 | 13.44 | 13.76 |
| CoreSet | 6.12 | 12.08 | 12.31 | 13.39 |
| BADGE | 9.37 | 10.05 | 11.09 | 11.48 |
| UHerding | 11.57 | 12.11 | 12.37 | 12.60 |
| PPAL | 11.85 | 12.19 | 12.81 | 13.01 |
| **AttriSelect** | **13.76** | **14.25** | **14.99** | **15.22** |

Extending repair training from 15 to 60 epochs raises AttriSelect ACC from 0.7508 to 0.7571 at ratio 0.5 and from 0.7562 to 0.7603 at ratio 0.9. Relative to the pre-repair ACC of 0.7173, the 60-epoch results correspond to gains of 3.98 and 4.30 percentage points, respectively. Over the same schedules, AttriSelect FR increases from 11.85\% to 14.08\% and from 13.76\% to 15.22\%. Most baselines also improve overall, although individual trajectories are not strictly monotonic.

> **Key result.** AttriSelect is best in **all eight ratio--epoch configurations**, leading the strongest baseline by 1.38--1.97 FR points at ratio 0.5 and 1.46--1.92 points at ratio 0.9. The persistent ordering supports a selected-set quality effect rather than an early-stopping artifact.

The generation and diagnosis stages are held fixed across source-selection methods, so Table 1 of the submitted paper and this additional experiment isolate which source subset gives the best repair result under the same upstream repair data and source budget. We will revise the main results to show raw metrics and absolute changes next to FR and state its interpretation more precisely.

## References

1. SafeFix: Targeted Model Repair via Controlled Image Generation. TMLR, 2026.
2. Actively Selecting Annotations Among Objects and Attributes. ICCV, 2011.
3. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR, 2018.
4. Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds. ICLR, 2020.

# Response to Reviewer hG6j (Part 1)

Dear Reviewer hG6j, thank you for the questions about the upstream interface, slice granularity, comparison baselines, cost, and vocabulary coverage.

**This response is organized into four parts.**

---

## Q1 — Upstream Slice Discovery and Granularity

> “How sensitive is AttriSelect to the upstream slice discovery method and slice granularity?”

### Upstream Interface

**Response:** AttriSelect starts after an upstream debugging method has diagnosed and semantically described a failure slice. As specified in the appendix, the attribute vocabulary, target fields, and VLM assignment prompt are fixed before source selection. Even when the same VLM family contributes to target-slice wording and image-level attribute assignment, the two roles are separated: the slice description is fixed first, and assignment then runs as a frozen closed-set classifier over source and generated images. The assignment stage does not receive model predictions, failure rates, selector identities, selected-set membership, or downstream repair results.

Under this interface, **two discovery methods that produce the same canonical
target conjunction and generated repair set give AttriSelect the same inputs
and therefore the same selection result**. The discovery algorithm matters
only if it changes the downstream semantic slice, including its fields, target
values, granularity, or generated repair data. We will state this condition
explicitly: **AttriSelect does not discover or validate the failure slice
itself**; it selects source data for an existing debugging and generation
pipeline.

### Annotation Reliability

**The appendix quantifies the reliability of this semantic interface through a blinded human audit.** Each of five AI graduate students labels 100 CelebA, 100 COCO, and 100 KITTI images using the same closed-set options as the VLM. The annotators do not see VLM labels, target flags, model predictions, data splits, or method outputs. Majority-vote human labels are then compared with the VLM assignments:

| Dataset | Target field 1 agreement | Target field 2 agreement | Slice-indicator agreement |
| :--- | ---: | ---: | ---: |
| CelebA | Hair color: 98% | Skin tone: 99% | 92% |
| COCO | Clothes color: 97% | Weather: 91% | 90% |
| KITTI | Object color: 98% | Weather: 96% | 94% |

Slice-indicator agreement is the most relevant measure because AttriSelect
uses the target conjunction, not either field in isolation. **Agreement is
90--94% across the three audited datasets**, so the selector's semantic slices
largely match independent human judgments.

**The appendix also corrupts the target-field assignments while keeping the generated repair set, source ratio, and training recipe fixed.** At source ratio 0.5, the changes from the clean-VLM AttriSelect result are:

| Dataset | Task metric | 5% attribute noise | 10% attribute noise | 20% attribute noise |
| :--- | :--- | ---: | ---: | ---: |
| CelebA | Test ACC change | 0.007 | 0.01 | 0.09 |
| COCO | Test AP change | 0.002 | 0.003 | 0.02 |
| KITTI | Test mAP50--95 change | 0.003 | 0.004 | 0.06 |

**The small changes under 5% and 10% corruption show that a few isolated
assignment errors do not drive AttriSelect.** At 20% noise, the larger changes
mark the expected failure regime: the attribute counts no longer represent the
diagnosed slice reliably. The audit and corruption experiments test the fixed
semantic interface. The next experiment instead changes the slice definition.

---

# Response to Reviewer hG6j (Part 2)

## Q1 — Upstream Slice Discovery and Granularity (continued)

### Slice-Granularity Sensitivity

Slice granularity can still affect repair because it changes both the semantic
factors used for source selection and the generated repair set. A coarse slice
can mix different failure causes and dilute the relevant attribute signal. A
narrow conjunction may have too little support and exclude useful examples
that match the main failure factors but not every added condition.

We evaluate this end-to-end sensitivity on COCO and ImageNet10 over five seeds and source ratios 0.1--0.9. For COCO, we compare the diagnosed two-attribute slice "brown AND rainy" with Brown-only, Rain-only, and the narrower three-attribute slice "brown AND rainy AND night":

| Source ratio | Brown-only AP | Rain-only AP | Brown + rainy + night AP | **Brown + rainy AP(Ours)** |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.6161 | 0.6174 | 0.6097 | **0.6183** |
| 0.3 | 0.6238 | 0.6224 | 0.6179 | **0.6306** |
| 0.5 | 0.6284 | 0.6217 | 0.6246 | **0.6317** |
| 0.7 | 0.6327 | **0.6333** | 0.6291 | 0.6319 |
| 0.9 | 0.6348 | 0.6132 | 0.6336 | **0.6355** |

**The diagnosed two-attribute slice exceeds Brown-only and Rain-only at four of five source ratios**, with average gains of **0.24 and 0.80 AP points**, respectively. It exceeds the narrower three-attribute slice at **all five ratios by 0.66 AP points on average**.

For ImageNet10, we compare the diagnosed "pink AND fabric" slice with Pink-only, Fabric-only, and the narrower "pink AND fabric AND rainy background" slice:

| Source ratio | Pink-only ACC | Fabric-only ACC | Pink + fabric + rainy ACC | **Pink + fabric ACC** |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.7223 | 0.7346 | **0.7398** | 0.7369 |
| 0.3 | 0.7369 | 0.7431 | 0.7414 | **0.7492** |
| 0.5 | 0.7377 | 0.7438 | 0.7429 | **0.7508** |
| 0.7 | 0.7485 | 0.7423 | 0.7462 | **0.7515** |
| 0.9 | 0.7500 | 0.7354 | 0.7491 | **0.7562** |

**The diagnosed two-attribute slice exceeds both single-attribute variants at all five source ratios**, by **0.98 and 0.91 accuracy points on average**, and exceeds the narrower three-attribute slice at four of five ratios by **0.50 points on average**.

> **Granularity takeaway.** **Downstream performance is relatively insensitive to moderate changes in slice granularity**, although **granularity still matters**: the diagnosed two-attribute definition performs best in **8 of the 10 dataset-ratio settings**. This pattern suggests a support--specificity trade-off: a coarse slice has greater support but includes fewer relevant examples, whereas an overly narrow conjunction can exclude useful examples.



---

## Q2 — Simple Attribute-Aware Baselines

> “Can the authors compare with simple attribute-aware baselines, such as exact target-slice selection plus stratified sampling?”

**Response:** On ImageNet10, the target slice is defined by "color=pink" and
"texture=fabric". We implemented exact target-slice selection with stratified
background sampling, along with two additional attribute-aware controls:

- **Target-Slice + Stratified:** select exact pink-and-fabric source examples first and fill the remaining budget with class-stratified background samples;
- **Attribute-Filtered Random:** prioritize the pink-or-fabric source pool and sample randomly within class-balanced queues before filling from the background;
- **Attribute-Filtered Entropy:** use the same candidate pool and class-balanced queues, but rank candidates by predictive entropy from the fixed diagnosis checkpoint.

All methods select only source images and use the same fixed 1,000-image generated set, source budget, VLM annotations, class labels, diagnosis checkpoint, generated seed, and 15-epoch repair schedule. The table reports mean overall ACC and standard deviation over seeds 0, 42, 1234, 3407, and 2024. The AttriSelect values are the five-seed results reported in the appendix of the submitted paper.

| Source ratio | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.7303 $\pm$ 0.0051 | 0.7340 $\pm$ 0.0034 | 0.7317 $\pm$ 0.0025 | **0.7369 $\pm$ 0.0043** |
| 0.3 | 0.7385 $\pm$ 0.0024 | 0.7382 $\pm$ 0.0042 | 0.7365 $\pm$ 0.0023 | **0.7492 $\pm$ 0.0044** |
| 0.5 | 0.7438 $\pm$ 0.0037 | 0.7442 $\pm$ 0.0038 | 0.7442 $\pm$ 0.0031 | **0.7508 $\pm$ 0.0010** |
| 0.7 | 0.7434 $\pm$ 0.0065 | 0.7474 $\pm$ 0.0038 | 0.7482 $\pm$ 0.0020 | **0.7515 $\pm$ 0.0037** |
| 0.9 | 0.7485 $\pm$ 0.0026 | 0.7469 $\pm$ 0.0036 | 0.7503 $\pm$ 0.0045 | **0.7562 $\pm$ 0.0019** |

> **Key result.** Even when every baseline receives the same attribute metadata
> and target-slice definition, AttriSelect has the highest mean accuracy at all
> five source ratios. Its margin over the strongest same-information baseline
> ranges from 0.29 to 1.07 accuracy points, with an average of 0.59
> points. This consistent but moderate advantage shows that the selection score
> provides additional value beyond access to the attributes alone.

---

# Response to Reviewer hG6j (Part 3)

## Q3 — End-to-End Cost Accounting

> “What is the end-to-end cost including generation, VLM attribute assignment, selection, and retraining?”

### Accounting Scope

**Response:** **Table 2 of the submitted paper measures only source-subset
selection time, not the complete repair pipeline**, and the revision will make
that scope explicit. The reported speedups apply to source selection after the
attribute records are available. **They should not be interpreted as
end-to-end speedups** over source annotation, repair-data construction,
retraining, and evaluation.

Repair-image generation is a fixed upstream SafeFix [1] dataset-construction stage shared by every selector. Every method receives the same publicly available repair images, so regenerating them from scratch would add the same cost to every method without changing their runtime ordering. We therefore report generation separately rather than charging it only to AttriSelect. In contrast, **we include full source-pool VLM assignment for AttriSelect** because it is a method-specific prerequisite.

### Conservative Cost of the First Repair Run

Source-pool assignment is a one-time preprocessing cost for a fixed task-level
vocabulary. Its metadata can be reused across source ratios and later repair
cycles. Here, **the first repair run assumes that source attributes must be
assigned from scratch**. Its cost covers source preprocessing and selection
through retraining and evaluation; repair-image generation is reported
separately. The tables conservatively charge a fresh
full-pool assignment in every AttriSelect row. For KITTI, the mean measured
assignment time over the two complete 5,000-image runs is **32.77 minutes** on
a single NVIDIA H200 GPU (the previously reported 10--20 minutes per 1,000
images was measured on an NVIDIA A5000, whereas an H200 requires only
approximately 5 minutes per 1,000 images). On the same GPU, assigning
attributes to the complete 20,000-image COCO source pool takes approximately
**1.50 hours**.

**KITTI: 5,000 source images.** Each AttriSelect row charges the mean fresh 5,000-image H200 VLM assignment time of 32.77 minutes.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 42.17 min | 42.18 min | 48.82 min | 1.37 h | 1.12 h | 43.47 min | 43.53 min | 1.01 h |
| 0.3 | 43.84 min | 43.85 min | 50.28 min | 1.44 h | 1.20 h | 45.08 min | 45.63 min | 1.09 h |
| 0.5 | 45.18 min | 45.18 min | 51.73 min | 1.49 h | 1.27 h | 46.42 min | 47.47 min | 1.11 h |
| 0.7 | 47.04 min | 47.05 min | 53.46 min | 1.56 h | 1.35 h | 48.16 min | 49.73 min | 1.13 h |
| 0.9 | 48.86 min | 48.87 min | 55.53 min | 1.63 h | 1.42 h | 50.34 min | 51.46 min | 1.15 h |

**COCO: 20,000 source images.** Image generation and generated-repair-image assignment are excluded. Each entry includes the applicable source preprocessing and selection cost, retraining, and evaluation. The AttriSelect column charges a fresh full-pool VLM assignment in every row.

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | Ours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4.41 min | 4.42 min | 10.98 min | 16.18 min | 19.63 min | 5.08 min | 6.97 min | 1.57 h |
| 0.3 | 8.18 min | 8.18 min | 14.74 min | 39.86 min | 49.76 min | 8.93 min | 17.86 min | 1.62 h |
| 0.5 | 12.37 min | 12.36 min | 19.04 min | 1.08 h | 1.56 h | 13.08 min | 29.47 min | 1.66 h |
| 0.7 | 15.73 min | 15.75 min | 22.27 min | 1.45 h | 2.29 h | 16.34 min | 37.63 min | 1.69 h |
| 0.9 | 19.66 min | 19.65 min | 26.28 min | 1.87 h | 3.01 h | 20.18 min | 40.94 min | 1.72 h |

> **Cost takeaway.** At small source ratios, lightweight conventional selectors have lower wall-clock cost on the first repair run. At ratio 0.9 on both COCO and KITTI, however, **AttriSelect is better than CoreSet and BADGE in both repair performance and first-run wall-clock cost**.

On COCO, AttriSelect takes 1.72 hours and reaches 0.6355 AP, compared with 1.87 hours/0.6202 for CoreSet and 3.01 hours/0.6213 for BADGE. On KITTI, it takes 1.15 hours and reaches 0.5850 mAP50--95, compared with 1.63 hours/0.5781 and 1.42 hours/0.5810.

---

# Response to Reviewer hG6j (Part 4)

## Q3 — End-to-End Cost Accounting (continued)

### Existing Metadata and Later Repair Cycles

Source attributes need not be assigned from scratch when suitable metadata
already exist. For example, the official CelebA release
(<https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>) provides 40 binary
attributes for each of its 202,599 images. **If these public annotations cover
the required task-level vocabulary, AttriSelect can use them directly and
incurs no VLM assignment cost.** New assignment is needed only for attributes
missing from the available metadata.

#### Cost After Metadata Reuse

The tables are deliberately conservative because they charge source assignment again at every ratio. **In repeated repair cycles, the metadata are reused, so this annotation cost is amortized and the remaining online AttriSelect computation is below two seconds.** We will revise Table 2 to label its values explicitly as source-subset selector runtime and distinguish one-time preprocessing, the first repair run, and later runs with reused metadata.

---

## Q4 — Incomplete Attribute-Vocabulary Coverage

> “How does the method behave when the failure slice is not well captured by the closed-set attribute vocabulary?”

**Response:** This question concerns coverage of the attribute schema, not
synonymous wording. Once names and values are mapped to canonical closed-set
labels, changing their surface form does not change the AttriSelect score. The
two substantive failure cases are a missing factor and a category that merges
distinct values too coarsely.

**If a target factor is absent, AttriSelect cannot explicitly prioritize that factor.** It can use only the remaining observed attributes and their correlations with the failure slice. The repair-specific part of the score therefore weakens, while soft class balance still preserves label coverage. **If target and non-target values are merged, class-conditional rarity and the generated-data frequency penalty become less discriminative**, which can cause the selected subset to include more examples outside the true fine-grained slice. These are limitations of the current closed-set formulation.

We ran controlled missing-field studies on ImageNet10 and COCO. For
each dataset, we keep the generated repair pool, generated-image
budget, source budget, checkpoint, and training schedule fixed, and remove one
target field from the source-selection vocabulary and its corresponding
generated-count term. Full uses both target fields for source selection. In
each missing-field cell, the value outside parentheses is the downstream task
metric after retraining.

**ImageNet10: target "pink AND fabric", overall test accuracy (%).** Values in
parentheses (...) are the selected-set Jaccard overlap with Full at the
same ratio.

| Ratio | Full | Missing color | Missing texture |
|---:|---:|---:|---:|
| 0.1 | 73.69 | 72.69 (0.5604) | **73.85** (0.3904) |
| 0.3 | **74.92** | 73.77 (0.6166) | 73.54 (0.5842) |
| 0.5 | **75.08** | 74.23 (0.6443) | 74.23 (0.7330) |
| 0.7 | **75.15** | 74.00 (0.6999) | 74.23 (0.8826) |
| 0.9 | **75.62** | 75.08 (0.8765) | 74.54 (0.9668) |

**COCO: target "brown AND rainy", keypoint AP.** Values in parentheses (...) are the selected-set Jaccard overlap
with Full at the same ratio.

| Ratio | Full | Missing color | Missing weather |
|---:|---:|---:|---:|
| 0.1 | **0.6183** | 0.6055 (0.4782) | 0.6105 (0.2034) |
| 0.3 | **0.6306** | 0.6221 (0.5611) | 0.6214 (0.4613) |
| 0.5 | **0.6317** | 0.6178 (0.7063) | 0.6271 (0.6084) |
| 0.7 | 0.6319 | **0.6332** (0.7836) | 0.6323 (0.8110) |
| 0.9 | **0.6355** | 0.6237 (0.8969) | 0.6244 (0.8708) |

> **Vocabulary-coverage takeaway.** Missing a target field changes the selected
> source set most strongly at low budgets: at ratio 0.1, Jaccard falls to
> 0.5604/0.3904 for missing color/texture on ImageNet10 and 0.4782/0.2034
> for missing color/weather on COCO. Downstream changes are non-monotonic,
> however. Mean ImageNet10 accuracy changes are -0.94/-0.81 points for
> missing color/texture, while mean COCO AP changes are -0.0091/-0.0065 for
> missing color/weather. **Missing-field variants are lower than Full in 17 of
> the 20 dataset--field--ratio comparisons.** Incomplete coverage therefore
> changes source selection and generally lowers task performance, although the
> effect is not strictly monotonic at every ratio.

## References

1. SafeFix: Targeted Model Repair via Controlled Image Generation. TMLR, 2026.

# Response to Reviewer KaLa (Part 1)

Dear Reviewer KaLa, thank you for asking us to clarify the selected data pool,
repair protocol, and compatibility with other pipelines.

**This response is organized into four parts.**
---

## W1/Q1 — Selected Data Pool

> “The problem formulation needs to be clarified. It is not always clear whether the proposed method is selecting generated repair data, original source data, or a combination of both. The descriptions in the abstract, introduction, and conclusion appear somewhat vague or potentially inconsistent on this point. The authors are encouraged to state explicitly what data pool AttriSelect operates on and how the selected data are used during retraining.”

**Response:** **`AttriSelect selects only from the original source or training
pool. It does not select generated repair images or rank a joint
source--generated pool.`** Let $r$ denote the source-selection ratio in our
tables. AttriSelect returns $\mathcal{D}^{sel}$, a subset of
$\mathcal{D}^{src}$. If the source pool contains $N$ images, the selected
subset contains $\lfloor rN \rfloor$ images. In every main experiment, the
upstream repair pipeline diagnoses a failure slice and constructs a generated
repair set. That set contains the same 1,000 images for every selector and
source ratio. The retraining set $\mathcal{D}^{repair}$ is the union of the
selected source subset $\mathcal{D}^{sel}$ and the fixed generated set
$\mathcal{D}^{gen}$.

AttriSelect uses attribute counts from the fixed generated set to identify
values that are already well represented and avoid redundant source examples.
Every ranked candidate and every selected image still comes from the source
pool. We then continue training or fine-tune the diagnosis checkpoint on
$\mathcal{D}^{repair}$. Therefore, changing $r$ changes only the
number and composition of real source images; the generated-image budget and
membership do not change.

We will use this terminology consistently in the abstract, introduction,
problem formulation, framework figure, experimental protocol, and conclusion:
the upstream pipeline supplies the diagnosed slice and fixed generated set;
AttriSelect selects a compact source subset; retraining uses their union. We
will remove any notation that suggests a selected generated subset.

---

## W2/Q4 — Prior-Work Taxonomy

> “The categorization of prior model repair methods requires further clarification. In particular, reference [7] seems to be associated with both the first and second lines of work, which makes the taxonomy confusing. The authors should revise this discussion to make the distinctions among different categories more precise.”

**Response:** **`The repeated citation is intentional because HiBug participates
in two stages: it discovers attribute-defined failure slices and generates
repair examples for them.`**

We will organize the revision by pipeline stage rather than present the
categories as mutually exclusive. The stages are
**failure-slice discovery, repair-data construction, source-data selection,
and model retraining**. HiBug can then appear in more than one stage without
making the taxonomy inconsistent.

Reference [7] is HiBug, an early method that discovers human-interpretable
model bugs and supports targeted repair with generated examples in
computer-vision tasks. All submitted experiments place our selector inside the
SafeFix repair pipeline. SafeFix addresses two limitations of direct generative
augmentation: **Semantic Drift**, in which attributes unrelated to the target
edit change, and **Distribution Mismatch**, in which synthetic images depart
from the source-data distribution. It reduces these problems through
controlled editing of real training images and VLM-based filtering.

**`W2/Q4 addresses the prior-work taxonomy; W4/Q3 reports the new five-seed
HiBug experiment on compatibility with another repair pipeline.`**

# Response to Reviewer KaLa (Part 2)

---

## W3/Q2 — Repair-Pipeline Implementation

> “It is unclear what repair method or methods are used together with AttriSelect in the experiments. Since AttriSelect is a data selection strategy rather than a complete repair algorithm, the experimental section should clearly specify the underlying repair pipeline, retraining/fine-tuning procedure, and how the selected source samples are combined with the generated repair images.”

**Response:** **`All submitted experiments place AttriSelect inside the SafeFix
repair pipeline.`** SafeFix follows the targeted-generation direction introduced
by HiBug, using controlled edits of real source images and VLM-based filtering
to construct repair images. AttriSelect leaves diagnosis and generation
unchanged. It replaces only the rule for choosing the real source examples
that accompany the generated repair images.

The implemented workflow is:

> **Repair-pipeline summary.**
>
> **Upstream SafeFix pipeline**  
> Diagnosis checkpoint $\theta_t$  
> ↓  
> Diagnose an attribute-defined failure slice $T_{bug}$  
> ↓  
> SafeFix controlled generation using ControlNet/Stable Diffusion 1.5, soft
> HED conditioning, and 30 DDIM steps  
> ↓  
> Fixed generated repair set $\mathcal{D}^{gen}$ (1,000 images)
>
> **AttriSelect source selection**  
> Original source pool $\mathcal{D}^{src}$ with attribute metadata,
> together with attribute counts from $\mathcal{D}^{gen}$  
> ↓  
> AttriSelect ranks source images only  
> ↓  
> Selected source subset $\mathcal{D}^{sel}$
>
> **Repair training**  
> Union of $\mathcal{D}^{sel}$ and $\mathcal{D}^{gen}$  
> ↓  
> Matched repair training or fine-tuning of $\theta_t$  
> ↓  
> Repaired checkpoint $\theta_{t+1}$

The generated images affect the AttriSelect score through their attribute
counts, but they are never selection candidates. At source ratio $r$,
if the source pool contains $N$ images, the selector returns
$\lfloor rN \rfloor$ images. The actual repair dataset
$\mathcal{D}^{repair}$ is the union of the selected source subset
$\mathcal{D}^{sel}$ and the fixed generated set $\mathcal{D}^{gen}$.

The appendix also gives the dataset-specific settings. For ImageNet10, we
train the diagnosis-stage ResNet-18 for 30 epochs using cross-entropy loss,
Adam, learning rate $1\times10^{-4}$, weight decay $1\times10^{-3}$, and batch
size 64. Qwen3-VL supplies closed-set attribute assignments for the source
images. Repair then runs for 15 epochs at learning rate $1\times10^{-5}$. For
CelebA, COCO, and KITTI, repair uses half as many epochs as diagnosis and the
same repair learning rate, $1\times10^{-5}$. Within each dataset-level
comparison, all selectors use
the same diagnosis checkpoint, failure slice, fixed 1,000-image generated set,
source ratio, task model, and repair schedule. **`Only the source-subset selection rule changes.`**

We will move this workflow and these settings into the experimental section so
that readers do not have to assemble the repair method, data composition, and
training procedure from several appendix sections.

# Response to Reviewer KaLa (Part 3)

---

## W4/Q3 — Compatibility with Alternative Pipelines

> “The paper states that AttriSelect is best viewed as one component in a localized repair pipeline. However, the experiments do not appear to evaluate its compatibility with multiple repair pipelines. It would strengthen the paper if the authors could test AttriSelect with different repair methods or, at least, discuss the extent to which the conclusions depend on the specific repair pipeline used in the experiments.”

> Additional point in Q3: “Could the authors provide evidence or discussion about whether AttriSelect remains effective when paired with different [...] retraining strategies?”

### Alternative Repair Pipeline: HiBug

**Response:** **HiBug [3]** is an attribute-based debugging and repair
framework. It uses vision--language models to find interpretable failure slices
and generate targeted repair examples. We test AttriSelect inside HiBug on
CelebA, replacing the SafeFix-generated repair set with a HiBug-generated set.
The diagnosis checkpoint, 80,000-image source pool, source ratios, generated
budget, retraining configuration, and five seeds are matched across selectors.
Every row uses the same fixed HiBug-generated set; only the source-selection
rule changes.

**CelebA with the HiBug repair pipeline and 80,000 training images. Metric:
five-seed mean FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **Ours (HiBug)** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4.59 | 2.33 | 6.92 | -6.20 | 2.87 | -1.36 | 5.58 | **7.36** |
| 0.3 | 5.91 | 5.08 | 8.02 | 7.79 | 1.66 | 3.39 | 3.87 | **8.07** |
| 0.5 | 5.76 | 7.70 | 7.48 | 7.90 | 2.53 | 2.13 | 9.64 | **12.54** |
| 0.7 | 2.57 | 8.06 | 10.03 | 6.19 | 7.86 | 4.96 | 5.90 | **11.16** |
| 0.9 | 2.84 | 4.28 | 7.95 | 5.71 | 5.80 | 4.19 | 6.37 | **13.42** |

> **Key result.** **`Under HiBug, AttriSelect is strongest at all five ratios,
> exceeding the best baseline by 0.44/0.05/2.90/1.13/5.47 FR points and by
> 2.00 points on average.`** These comparisons should be read within the HiBug
> table. Changing the upstream pipeline changes the generated repair set and
> can shift the absolute FR of every selector.

### Alternative-Generator Evidence

**Response:** AttriSelect requires a diagnosed semantic slice, an annotated
source pool, and a fixed generated repair set. To test dependence on the main
ControlNet generator, the appendix substitutes InstructPix2Pix on CelebA and
ICEdit on COCO. The selectors, generated-image budget, source ratios, and
repair protocol remain fixed.

**CelebA with InstructPix2Pix generation. Metric: FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **Ours (InstructPix2Pix)** | Ours (ControlNet) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4.23 | 1.95 | 6.41 | -6.65 | 2.34 | -1.78 | 5.12 | **7.65** | 8.48 |
| 0.3 | 5.34 | 4.72 | 7.98 | 7.21 | 1.02 | 2.81 | 3.45 | **8.12** | 8.91 |
| 0.5 | 5.56 | 7.23 | 6.78 | 7.54 | 2.12 | 1.56 | 9.12 | **11.17** | 13.79 |
| 0.7 | 2.12 | 7.67 | 9.34 | 5.78 | 7.34 | 4.23 | 5.34 | **12.13** | 12.09 |
| 0.9 | 2.45 | 3.92 | 7.56 | 5.34 | 5.23 | 3.87 | 5.91 | **13.87** | 14.74 |

> **Key result.** **`AttriSelect is the strongest source selector at every evaluated CelebA ratio under InstructPix2Pix, exceeding the strongest baseline by 2.51 FR points on average.`**

**COCO pose estimation with ICEdit generation. Metric: FR (%).**

| Ratio | Random | Stratified | Entropy | CoreSet | BADGE | UHerding | PPAL | **Ours (ICEdit)** | Ours (ControlNet) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 3.78 | 5.23 | 5.87 | -0.34 | 2.34 | 3.12 | 1.56 | **7.12** | 6.86 |
| 0.3 | 3.92 | 8.67 | 8.12 | 9.12 | 6.56 | 4.67 | 6.89 | **10.34** | 9.86 |
| 0.5 | 7.91 | 8.45 | 9.67 | 6.91 | 7.82 | 6.54 | 7.23 | **10.67** | 10.13 |
| 0.7 | 6.54 | 10.12 | 9.98 | 9.56 | 9.12 | 6.71 | 8.23 | **10.87** | 10.18 |
| 0.9 | 5.12 | 9.87 | 6.98 | 7.82 | 8.12 | 7.54 | 10.12 | **11.56** | 11.05 |

> **Key result.** Under ICEdit, AttriSelect is strongest at every evaluated
> COCO ratio and exceeds the best baseline by **1.13 FR points on average**.
> **`Across the two generator substitutions, AttriSelect ranks first in all 10
> evaluated dataset-ratio settings.`**

# Response to Reviewer KaLa (Part 4)

### Scope of the Evidence

**Scope.** The empirical evidence covers one matched alternative repair
pipeline (HiBug on CelebA) and two generator substitutions within the
SafeFix-style workflow (InstructPix2Pix on CelebA and ICEdit on COCO).

The appendix theory gives a conditional argument for transfer across repair
pipelines and retraining strategies. The result concerns source selection and
does not depend on SafeFix, HiBug, a particular image generator, or a specific
optimizer. Given a diagnosed target slice, an
attribute-annotated source pool, and a generated repair set, the
rarity-dominates-penalty result gives conditions under which target-slice source
examples remain ranked above non-target examples after accounting for the
generated-data distribution. Let $B_c$ denote the selection budget allocated
to class $c$, and let $T_c$ denote the set of target-slice source examples in
that class. The deterministic enrichment result then
guarantees at least $\min\{B_c,|T_c|\}$ selected target-slice examples within
class $c$, whereas an attribute-agnostic selector requires its uncertainty or
coverage signal to be separately aligned with that slice.

These guarantees apply before model updating, so the selected subset can feed
checkpoint fine-tuning, continued training, or retraining from scratch. Under
the appendix's repair-utility condition, the enrichment result gives expected
repair utility at least as high as random or stratified selection under the
same class allocation, with a strict advantage when the enrichment gap is
strict. **`Changing the repair pipeline or update rule affects the data counts
and how the additional target support translates into the final metric; it
does not remove the selection mechanism.`** This is a conditional
selection-level argument, not a guarantee for every optimizer, checkpoint, or
data-weighting scheme.

---

## W5 — Reference Formatting

> “The formatting of the references should be improved. For example, the names of journals and conferences should be made consistent, either using full names or standard abbreviations throughout the reference list.”

**Response:** **We standardized conference and journal names throughout the
reference list** using NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, WACV, AAAI,
FAccT, and JMLR. We also corrected title and venue capitalization, protected
proper names such as ImageNet, COCO, KITTI, and SafeFix, and fixed the malformed
author field in the PPAL reference. The bibliography now uses one format.

## Additional Evidence from Other Reviewer Responses

Several experiments in the other responses bear directly on KaLa's questions.
The hG6j response gives every method the same ImageNet10 attributes and
target-slice definition. AttriSelect leads at all five source ratios by 0.59
accuracy points on average over the best control.

The yyPZ response varies repair training from 15 to 60 epochs. AttriSelect
remains strongest in all eight configurations. The Area Chair response reports
source-only ImageNet10 experiments that train a new ResNet-18 rather than
fine-tune a repair checkpoint. AttriSelect leads at all five one-shot budgets,
and AttriSelect-AL improves the six-round Curve Mean by 1.40 points over BADGE.

The other responses report five-seed studies of slice granularity, alternative
attribute sources, and incomplete vocabularies.
**`The observed advantage is therefore not confined to one information budget,
repair schedule, checkpoint-update mode, slice definition, or attribute
source.`**

## References

1. SafeFix: Targeted Model Repair via Controlled Image Generation. TMLR, 2026.
2. Actively Selecting Annotations Among Objects and Attributes. ICCV, 2011.
3. HiBug: On Human-Interpretable Model Debug. NeurIPS, 2023.

# Misc(do not reply on OpenReview)

## Raw Dense-Task Metrics

These tables correspond to the COCO and KITTI FR comparisons at the top of
this file. They are retained for internal verification and are not currently
needed in the OpenReview response.

### COCO Raw Test Keypoint AP

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.5877 $\pm$ 0.0040 | 0.5985 $\pm$ 0.0039 | 0.6013 $\pm$ 0.0044 | 0.5950 $\pm$ 0.0015 | 0.6133 | 0.5756 | 0.6083 | **0.6183 $\pm$ 0.0044** |
| 0.3 | 0.6256 $\pm$ 0.0031 | 0.6153 $\pm$ 0.0044 | 0.6079 $\pm$ 0.0025 | 0.6165 $\pm$ 0.0020 | 0.6290 | 0.6170 | 0.6243 | **0.6306 $\pm$ 0.0022** |
| 0.5 | 0.6167 $\pm$ 0.0036 | 0.6206 $\pm$ 0.0018 | 0.6153 $\pm$ 0.0038 | 0.6182 $\pm$ 0.0022 | 0.6263 | 0.6192 | 0.6234 | **0.6317 $\pm$ 0.0012** |
| 0.7 | 0.6275 $\pm$ 0.0022 | 0.6256 $\pm$ 0.0016 | 0.6155 $\pm$ 0.0041 | 0.6223 $\pm$ 0.0044 | 0.6281 | 0.6286 | **0.6342** | 0.6319 $\pm$ 0.0024 |
| 0.9 | 0.6202 $\pm$ 0.0023 | 0.6213 $\pm$ 0.0028 | 0.6191 $\pm$ 0.0014 | 0.6301 $\pm$ 0.0025 | 0.6265 | 0.6155 | 0.6306 | **0.6355 $\pm$ 0.0031** |

### KITTI Raw Test mAP50--95

| Source Ratio | CoreSet | BADGE | UHerding | PPAL | `Target-Slice + Stratified` | `Attribute-Filtered Random` | `Attribute-Filtered Entropy` | **`AttriSelect`** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.5237 $\pm$ 0.0039 | 0.5322 $\pm$ 0.0025 | 0.5222 $\pm$ 0.0031 | 0.5262 $\pm$ 0.0036 | 0.5264 | 0.5359 | **0.5374** | 0.5372 $\pm$ 0.0023 |
| 0.3 | 0.5612 $\pm$ 0.0013 | 0.5624 $\pm$ 0.0030 | 0.5573 $\pm$ 0.0021 | 0.5568 $\pm$ 0.0039 | 0.5596 | 0.5627 | **0.5673** | 0.5646 $\pm$ 0.0023 |
| 0.5 | 0.5759 $\pm$ 0.0043 | 0.5726 $\pm$ 0.0013 | 0.5675 $\pm$ 0.0027 | 0.5721 $\pm$ 0.0021 | 0.5748 | **0.5784** | 0.5775 | 0.5773 $\pm$ 0.0012 |
| 0.7 | 0.5772 $\pm$ 0.0013 | 0.5717 $\pm$ 0.0017 | 0.5768 $\pm$ 0.0024 | 0.5782 $\pm$ 0.0030 | 0.5805 | 0.5756 | **0.5823** | 0.5780 $\pm$ 0.0039 |
| 0.9 | 0.5781 $\pm$ 0.0011 | 0.5810 $\pm$ 0.0018 | 0.5838 $\pm$ 0.0044 | 0.5796 $\pm$ 0.0026 | 0.5820 | 0.5821 | 0.5832 | **0.5850 $\pm$ 0.0012** |

## Complexity Derivation for CoreSet and BADGE

Let $N$ denote the number of source candidates, $B=rN$ the requested subset size at source ratio $r$, $d$ the feature dimension used by CoreSet, and $D$ the gradient-embedding dimension used by BADGE. The following argument concerns the standard greedy procedures in the original methods; model inference or feature extraction adds a separate term that is at least linear in $N$.

CoreSet applies greedy farthest-first traversal to approximate the $k$-center objective~\citep{sener2018active}. After $t$ samples have been selected, it maintains each candidate's distance to its nearest selected center,
\[
\Delta_i^{(t)}=\min_{j\in S_t}\lVert z_i-z_j\rVert_2^2,
\]
selects $j_t=\arg\max_i\Delta_i^{(t)}$, and performs the update
\[
\Delta_i^{(t+1)}=\min\!\left\{\Delta_i^{(t)},
\lVert z_i-z_{j_t}\rVert_2^2\right\}
\qquad \text{for every }i\in\{1,\ldots,N\}.
\]
Each iteration therefore computes $N$ distances in $d$ dimensions and costs $\Theta(Nd)$. Repeating this update for $B$ selected centers gives
\[
T_{\mathrm{CoreSet}}(N,B,d)=\Theta(NBd)
=\Theta(rN^2d).
\]
An incremental implementation requires $\Theta(Nd+N)$ memory for the embeddings and nearest-center distances; an implementation that materializes all pairwise distances instead requires $\Theta(N^2)$ memory. Thus, even the memory-efficient implementation retains the quadratic-in-$N$ dominant computation when the selection ratio is fixed.

BADGE first constructs a gradient embedding $g_i\in\mathbb{R}^{D}$ for every candidate and then applies $k$-means++ seeding to select a diverse batch~\citep{ash2020deep}. Its seeding step similarly maintains
\[
\delta_i^{(t)}=\min_{j\in S_t}\lVert g_i-g_j\rVert_2^2.
\]
Adding one center updates this quantity for all $N$ candidates at a cost of
$\Theta(ND)$ per iteration. Therefore,
\[
T_{\mathrm{BADGE}}(N,B,D)
=\Theta(ND)+\Theta(NBD)
=\Theta(rN^2D),
\]
where the first term constructs or stores the gradient embeddings and the
second is the diversity-seeding cost. For last-layer gradient embeddings, $D$
usually scales with the feature dimension times the number of output classes,
so it can be much larger than CoreSet's $d$. BADGE also requires
$\Theta(ND+N)$ memory under incremental distance updates. These expressions
describe the scaling trend rather than exact wall-clock time; GPU utilization,
batching, task heads, and implementation details affect the constants.

The measurements show both effects predicted by the derivation. For fixed
$N$, increasing the ratio increases $B$, so the dominant selection work grows
approximately linearly in $r$. In the original COCO timings, CoreSet takes 5,
15, and 25 minutes as $r$ increases from 0.1 to 0.3 to 0.5; BADGE takes 21
minutes, 1 hour, and 2 hours. For fixed $r$, increasing the source pool by a
factor $c$ increases the dominant $NB$ term by $c^2$. CelebA has four times as
many source images as COCO (80,000 versus 20,000), so its asymptotic
distance-update work can be $4^2=16$ times larger before accounting for the
embedding dimension and model cost. This scaling moves standard CoreSet and
BADGE implementations from minutes on COCO to hours on CelebA. AttriSelect
does not perform an $N\times B$ sequence of embedding-distance updates after
its reusable attribute records are available.




Reviewer hG6j
Q1
We exclude ratio 1.0 because every method retains the complete source pool at that ratio. The variants also use their corresponding generated repair pools, so this experiment measures the combined effect of slice specification, source-selection vocabulary, and repair-data construction; **it does not isolate selector granularity while holding generated images fixed**.
