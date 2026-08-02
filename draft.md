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
- [ ] Use the same seeds (`0`, `42`, `1234`, `3407`, and `2024`), fixed 1,000
  generated images, diagnosis checkpoint, 15-epoch repair schedule, and source
  ratios `0.1/0.3/0.5/0.7/0.9` as the completed same-information comparison.
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
deviation. `Target $\Delta$` and `Non-target $\Delta$` are changes from the
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
  five seeds, and source ratios `0.1/0.3/0.5/0.7/0.9`.
- [ ] Count exact-target, single-target-factor, and background examples.
- [ ] Measure class coverage, class imbalance, unique class--attribute tuples,
  and selected-set Jaccard overlap.
- [ ] Measure redundancy with the fixed generated set using attribute-value
  frequencies. Define the redundancy metric before computing it and use the
  same definition for every method.
- [ ] Report both absolute counts and fractions because the selected-set size
  changes with the source ratio.

**Table H2a. Composition of the selected source subset.** Values are five-seed
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

**Table H2b. Diversity, balance, redundancy, and overlap.** Class imbalance is
the maximum-minus-minimum selected fraction across the ten classes. Generated
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

## Experiment H3 — More Constrained Source Budgets

- [ ] Add ImageNet10 source ratios `0.02` and `0.05` for the same four methods
  and five seeds. Retain ratio `0.1` as the existing reference.
- [ ] Keep the fixed generated set at 1,000 images and report the corresponding
  selected-source counts: 208, 520, and 1,040 images at ratios `0.02`, `0.05`,
  and `0.1` from the 10,400-image source pool.
- [ ] Use the same diagnosis checkpoint and 15-epoch repair schedule.
- [ ] Report overall, exact-target, and non-target accuracy. Explicitly note
  that generated images outnumber selected source images at ratios `0.02` and
  `0.05`, which changes the source/generated mixture.
- [ ] Use one SLURM array for all new training runs because the tasks share the
  same code, environment, model, and resource shape.

**Table H3. Five-seed constrained-budget ImageNet10 repair.** Metric: accuracy
in percentage points, reported as mean $\pm$ standard deviation.

| Source Ratio | Selected Source Images | Method | Target ACC | Non-Target ACC | Overall ACC |
|---:|---:|:---|---:|---:|---:|
| 0.02 | 208 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] |
| 0.02 | 208 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] |
| 0.02 | 208 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] |
| 0.02 | 208 | AttriSelect | [TBD] | [TBD] | [TBD] |
| 0.05 | 520 | Target-Slice + Stratified | [TBD] | [TBD] | [TBD] |
| 0.05 | 520 | Attribute-Filtered Random | [TBD] | [TBD] | [TBD] |
| 0.05 | 520 | Attribute-Filtered Entropy | [TBD] | [TBD] | [TBD] |
| 0.05 | 520 | AttriSelect | [TBD] | [TBD] | [TBD] |
| 0.10 | 1,040 | Target-Slice + Stratified | [TBD] | [TBD] | 73.03 $\pm$ 0.51 |
| 0.10 | 1,040 | Attribute-Filtered Random | [TBD] | [TBD] | 73.40 $\pm$ 0.34 |
| 0.10 | 1,040 | Attribute-Filtered Entropy | [TBD] | [TBD] | 73.17 $\pm$ 0.25 |
| 0.10 | 1,040 | AttriSelect | [TBD] | [TBD] | **73.69 $\pm$ 0.43** |

**Result placeholder.** At the most constrained source ratio of `0.02`,
AttriSelect reaches [TBD] overall accuracy and [TBD] exact-target accuracy,
exceeding the strongest same-information control by [TBD] and [TBD] points,
respectively. Across ratios `0.02/0.05/0.10`, its average overall and
exact-target margins are [TBD] and [TBD] points. These results [support / do
not support] a stronger data-efficiency advantage under constrained source
budgets.

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
> constrained source ratios `0.02` and `0.05`, [TBD: summarize data-efficiency
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
attributes and the same "pink AND fabric" target definition:
Target-Slice + Stratified, Attribute-Filtered Random, and Attribute-Filtered
Entropy. All methods use the same fixed 1,000 generated images, source budget,
checkpoint, 15-epoch schedule, and five seeds.


**Five-seed ImageNet10 same-information repair comparison. Metric: test
accuracy (fraction; mean $\pm$ standard deviation).**

| Source Ratio | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | **AttriSelect** |
|---:|---:|---:|---:|---:|
| 0.1 | 0.7303 $\pm$ 0.0051 | 0.7340 $\pm$ 0.0034 | 0.7317 $\pm$ 0.0025 | **0.7369 $\pm$ 0.0043** |
| 0.3 | 0.7385 $\pm$ 0.0024 | 0.7382 $\pm$ 0.0042 | 0.7365 $\pm$ 0.0023 | **0.7492 $\pm$ 0.0044** |
| 0.5 | 0.7438 $\pm$ 0.0037 | 0.7442 $\pm$ 0.0038 | 0.7442 $\pm$ 0.0031 | **0.7508 $\pm$ 0.0010** |
| 0.7 | 0.7434 $\pm$ 0.0065 | 0.7474 $\pm$ 0.0038 | 0.7482 $\pm$ 0.0020 | **0.7515 $\pm$ 0.0037** |
| 0.9 | 0.7485 $\pm$ 0.0026 | 0.7469 $\pm$ 0.0036 | 0.7503 $\pm$ 0.0045 | **0.7562 $\pm$ 0.0019** |

> **Key result.** **`With attribute metadata and the target-slice definition
> matched across methods, AttriSelect has the highest mean accuracy at all
> five source ratios.`** Its margin over the strongest control ranges from 0.29
> to 1.07 accuracy points and averages 0.59 points. The attributes alone do
> not account for the difference.

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

# Response to Area Chair 6k27 (Part 3)

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

`Changing the source ratio changes only the number and composition of real
source images; the generated set remains fixed.`

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

# Response to Area Chair 6k27 (Part 4)

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

# Response to Area Chair 6k27 (Part 5)

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

All methods select only source images and use the same fixed 1,000-image generated set, source budget, VLM annotations, class labels, diagnosis checkpoint, generated seed, and 15-epoch repair schedule. The table reports mean overall ACC and standard deviation over seeds 0, 42, 1234, 3407, and 2024. The AttriSelect values are the five-seed results reported in the appendix of the submitted paper.

**Five-seed ImageNet10 same-information repair comparison. Metric: test
accuracy (fraction; mean $\pm$ standard deviation).**

| Source ratio | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | **AttriSelect** |
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

| Source ratio | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | **AttriSelect** |
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
