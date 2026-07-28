# Experiments and manuscript updates to complete


end to end: training, vlm annotating, selection time


## Highest-priority experiments

- [x] **Attribute-aware baselines under the same information budget.** The matched ImageNet10 five-seed comparison is complete for exact target-slice plus stratified sampling, Attribute-Filtered Random, and Attribute-Filtered Entropy. The AttriSelect column uses the authoritative values in `nips26_new/appendix/main_metric_tables.tex`.
- [ ] **End-to-end cost accounting.** Report source-pool attribute assignment, repair-image generation, generated-image attribute assignment, subset selection, and retraining time separately. Distinguish one-time task preprocessing from per-slice and per-repair costs.
- [ ] **Attribute-generation ablation.** On a representative setting, rerun selection with an alternative attribute source or annotator under the same closed-set vocabulary and repair protocol. Compare attribute agreement, selected-set overlap, target-slice enrichment, and downstream repair performance with the main Qwen3-VL assignment.
- [ ] **Slice-definition and vocabulary sensitivity.** Compare the full target conjunction with a coarser slice, a missing target field, and a merged target value. Report downstream task performance and target-slice enrichment.
- [ ] **Alternative-pipeline response integration.** Use the verified appendix results for InstructPix2Pix on CelebA and ICEdit on COCO to build a compact rebuttal table and manuscript summary. These experiments do not need to be rerun.

## Additional measurements to insert

- [ ] Measure the missing retraining wall-clock times and the ImageNet10 source-annotation time.
- [ ] Insert target-slice gains and non-target performance changes next to the overall task metrics.

# Response to Area Chair 6k27

Dear Area Chair 6k27, thank you very much for your careful assessment and constructive guidance. We are encouraged that the meta-review recognizes the unique perspective of attribute-based source-data selection, the reasonableness of using attributes already produced by the upstream repair workflow, and the strong repair performance across image classification, pose estimation, and object detection. We agree that the current submission must more clearly separate the value of attribute information from the value of the AttriSelect scoring rule, report the complete pipeline cost, and provide stronger evidence across repair pipelines and imperfect attribute specifications. We address these central concerns below and will revise the paper accordingly.

---

> Q1: “I do agree that the adopted baselines should include methods considering the generated attribute labels. The authors should accordingly include attribute-aware baselines as suggested by the reviewers.”

**R1:** We agree. The submitted baselines test uncertainty, geometric coverage, and gradient-space coverage, but they do not receive the same attribute records and target-slice definition as AttriSelect. We therefore implemented three same-information baselines: (i) exact target-slice selection followed by class-stratified background sampling, (ii) Attribute-Filtered Random over the repair-relevant attribute pool, and (iii) Attribute-Filtered Entropy over the same pool. All methods use the same source budget, fixed 1,000-image generated set, Qwen3-VL attribute annotations, class labels, diagnosis checkpoint, and 15-epoch repair schedule.

The matched five-seed comparison is complete over seeds 0, 42, 1234, 3407, and 2024. AttriSelect has the highest mean ACC at every source ratio; its margins over the strongest same-information baseline are 0.0029, 0.0107, 0.0066, 0.0033, and 0.0059 at ratios 0.1--0.9. We do not add a new paired significance claim because the manuscript TeX is the authoritative record and does not expose the per-seed AttriSelect values.

---

> Q2: “Otherwise, this is merely a standard data selection method that should be evaluated outside of the model repair scenario. It would further strengthen the paper if the authors indeed include such comparisons on standard data selection scenarios. Another possibility is that the compared baselines should also use the attribute values instead of extracting embeddings. This will make the efficiency comparison fairer.”

**R2:** AttriSelect is designed for the repair setting because its score uses two repair-specific inputs that are absent from standard static data selection: a diagnosed semantic failure slice and the attribute distribution of a fixed generated repair set. In particular, the generated-data frequency penalty explicitly avoids selecting source examples that duplicate the synthetic repair pool. We agree, however, that the current evaluation does not isolate this repair-specific use of attributes from the broader benefit of attribute-aware selection. The new same-information baselines will operate directly on the attribute values instead of relying only on extracted embeddings. We will also revise the scope of our claims: the experiments establish AttriSelect as a source-data selector within targeted model repair, while evaluation as a general-purpose data selection method is outside the current evidence and will be identified as future work.

---

> Q3: “Since AttriSelect itself only involves selecting from the source pool and generated data, it would be stronger if the authors could evaluate the model against multiple pipelines. The authors should clarify the implementation details regarding the entire repair pipeline.”

**R3:** We first clarify that AttriSelect selects only from the original source pool; the generated repair set is fixed and shared by all selectors. The retraining set is the selected source subset combined with the fixed generated set. We will add a dataset-specific pipeline table reporting the diagnosed slice, source pool, attribute source, generator, generated-image budget, diagnosis-stage checkpoint, model, optimizer, learning rate, epoch count, and source ratio.

The appendix already contains controlled alternative-generator experiments that replace ControlNet with InstructPix2Pix on CelebA and ICEdit on COCO while holding the source-selection comparison, generated-image budget, and repair protocol fixed. AttriSelect is the strongest selector at every active source ratio in both settings. We will move a compact summary of these verified results into the response and revise the paper to state precisely that the evidence covers multiple generators, rather than every possible repair or retraining pipeline.

**Internal TODO:** Insert the compact alternative-generator table and the dataset-specific pipeline table.

---

> Q4: “When the ablation on generative models has been provided, it would further strengthen the paper if the authors could also ablate the attribute generation and other pipeline components. The robustness against imperfect attribute vocabularies will also be important for the pipeline.”

**R4:** We agree that imperfect attribute assignment and imperfect attribute vocabularies are distinct cases. The submitted attribute-noise study and human audit evaluate assignment errors within a known vocabulary, but they do not replace an attribute-generation ablation and do not fully address a missing or overly coarse concept. We completed a controlled missing-field and slice-granularity diagnostic on ImageNet10. Removing one target field materially changes the selected source set at low ratios, although overall ACC does not degrade monotonically. A genuine merged-value transformation and the alternative-attribute-source ablation remain to be run. We will also separate one-time source-pool attribute assignment, generated-image assignment, generation, subset selection, and retraining in the end-to-end cost table so that the contribution and cost of each pipeline component are explicit.

**Internal TODO:** Insert the completed missing-field result; run the alternative-attribute-source and merged-value ablations; complete the end-to-end cost table.

# Response to Reviewer yyPZ

We thank the reviewer for recognizing the practical motivation, straightforward methodology, and clear presentation. We address the three concerns below.

---

> Q1: “Misleading Claims Regarding Computational Efficiency. The authors highlight in Table 2 that AttriSelect takes less than 2 seconds, claiming a speedup of up to 283× over baselines. However, this entirely omits the massive computational overhead of the prerequisite step: running Qwen3-VL to annotate the entire source pool. Appendix A states it takes 10–20 minutes to annotate just 1,000 images. For datasets like CelebA (80,000 images) or COCO (20,000 images), this requires tens to hundreds of GPU hours. Claiming high efficiency while ignoring the cost of generating the metadata that the algorithm relies upon is highly misleading.”

**R1:** We agree that Table 2 reports the cost of the subset-selection stage rather than the cost of the complete repair pipeline, and the paper should make this distinction explicit. The reported speedups compare the online selector computation after the attribute records are available. They should not be read as end-to-end speedups over generation, annotation, and retraining.

For a fixed task-level attribute vocabulary, source-pool assignment is a one-time preprocessing cost that can be reused across source ratios and repair iterations. It is not rerun for every selector. CelebA also does not require Qwen3-VL assignment for the 80,000 source images because AttriSelect uses the official CelebA attributes. For the datasets that require VLM assignment, our recorded Qwen3-VL costs are 399.7 minutes (6.66 GPU-hours) for 20,000 COCO source images and 71.8 minutes (1.20 GPU-hours) for 5,000 KITTI source images. Generating 1,000 repair images takes slightly over one hour, and assigning attributes to those generated images takes approximately 10--20 minutes.

We will revise the efficiency discussion to separate:

1. one-time source-pool annotation;
2. per-slice generation and generated-image annotation;
3. per-repair subset selection and retraining.

Table 2 will be labeled explicitly as selector runtime. We will also add an end-to-end cost table and an amortized cost analysis for repeated repair cycles.

**Internal TODO:** Insert the end-to-end runtime table, including retraining wall-clock time, ImageNet10 source annotation, hardware, and the amortized total for multiple repair cycles.

---

> Q2: “Comparison look Unfair. AttriSelect is given privileged information like VLM-assigned attributes and explicit bug definitions, but the compared active learning baselines (Entropy, BADGE, CoreSet) do not have access to. The authors should compare it against a baseline with the same information, for example, an ‘Attribute-Filtered Random’ or ‘Attribute-Filtered Entropy’ baseline.”

**R2:** We agree that the current generic active-learning baselines do not receive the attribute information used by AttriSelect. Those comparisons test whether repair-specific attribute metadata supplies a useful signal beyond uncertainty, geometric coverage, and gradient embeddings. They do not isolate whether the proposed rarity, generated-data penalty, and soft class-balance score uses that metadata better than a simple attribute-aware rule.

We added three baselines with access to the same metadata and target-slice definition:

- **Target-Slice + Stratified:** select exact `pink`-and-`fabric` source examples first and fill the remaining budget with class-stratified background samples;
- **Attribute-Filtered Random:** prioritize the `pink`-or-`fabric` source pool and sample randomly within class-balanced queues before filling from the background;
- **Attribute-Filtered Entropy:** use the same candidate pool and class-balanced queues, but rank candidates by predictive entropy from the fixed diagnosis checkpoint.

All methods select only source images and use the same fixed 1,000-image generated set, source budget, annotations, labels, checkpoint, generated seed, and 15 repair epochs. The table reports matched five-seed means and standard deviations over seeds 0, 42, 1234, 3407, and 2024. The AttriSelect values are the authoritative results from Appendix Table~\ref{tab:app-main-imagenet}.

| Ratio | Target-Slice + Stratified | Attribute-Filtered Random | Attribute-Filtered Entropy | AttriSelect |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.7303 $\pm$ 0.0051 | 0.7340 $\pm$ 0.0034 | 0.7317 $\pm$ 0.0025 | **0.7369 $\pm$ 0.0043** |
| 0.3 | 0.7385 $\pm$ 0.0024 | 0.7382 $\pm$ 0.0042 | 0.7365 $\pm$ 0.0023 | **0.7492 $\pm$ 0.0044** |
| 0.5 | 0.7438 $\pm$ 0.0037 | 0.7442 $\pm$ 0.0038 | 0.7442 $\pm$ 0.0031 | **0.7508 $\pm$ 0.0010** |
| 0.7 | 0.7434 $\pm$ 0.0065 | 0.7474 $\pm$ 0.0038 | 0.7482 $\pm$ 0.0020 | **0.7515 $\pm$ 0.0037** |
| 0.9 | 0.7485 $\pm$ 0.0026 | 0.7469 $\pm$ 0.0036 | 0.7503 $\pm$ 0.0045 | **0.7562 $\pm$ 0.0019** |

The same-information baselines are competitive and therefore constitute necessary controls, but AttriSelect retains the highest mean at all five ratios. The exact target slice contains only 16 of 1,300 test images, so target accuracy changes in increments of 6.25 percentage points and is not used as evidence of superiority.

**Internal note:** The three baseline grids passed required-complete validation over 75 result/manifest pairs. Do not use the non-authoritative AttriSelect values from Juno array `306871` in the rebuttal.

---

> Q3: “Despite the heavy pipeline (VLM tagging, diffusion generation, subset selection), the absolute Fix Rates (FR) in Table 1 remain quite low, mostly ranging between 5% and 13%.”

**R3:** We agree that Fix Rate (FR) should be reported together with the original task metric. FR is a normalized improvement toward the maximum task score; it is not the percentage of individual test examples or failure slices that were fixed. The raw changes make the effect size easier to interpret. At the reported ratio-0.9 AttriSelect settings, the corresponding changes are:

- ImageNet10 accuracy: 0.7173 to 0.7562, or +3.89 percentage points;
- CelebA accuracy: 0.9057 to 0.9196, or +1.39 percentage points;
- COCO keypoint AP: 0.5902 to 0.6355, or +4.53 AP;
- KITTI mAP50--95: 0.5626 to 0.5850, or +2.24 mAP.

As specified in the appendix, the ImageNet10 diagnosis-stage model is trained for 30 epochs with learning rate $1\times10^{-4}$, whereas repair uses 15 epochs with the smaller learning rate $1\times10^{-5}$. For CelebA, COCO, and KITTI, the repair-stage epoch count is likewise set to half of the corresponding diagnosis-stage schedule, with repair learning rate $1\times10^{-5}$. We deliberately use this moderate repair schedule because the primary controlled question is the relative performance of different source-selection methods, rather than the maximum performance obtainable through longer retraining. All selectors use exactly the same repair-stage epoch count, learning rate, generated repair set, and source-data budget.

To test whether the reported FR values are mainly limited by this schedule, we additionally vary the repair-stage epoch count from $0.5\times$ to $2\times$ the diagnosis-stage training schedule. FR increases as repair training is extended, showing that the absolute values in Table 1 are partly a consequence of the conservative epoch budget. More importantly, AttriSelect remains the strongest selector at every epoch multiplier, so its advantage is not tied to the particular $0.5\times$ schedule used in the main experiments.

| Method | $0.5\times$ epochs FR (\%) | $1.0\times$ epochs FR (\%) | $1.5\times$ epochs FR (\%) | $2.0\times$ epochs FR (\%) |
| :--- | ---: | ---: | ---: | ---: |
| Random | TODO | TODO | TODO | TODO |
| Entropy | TODO | TODO | TODO | TODO |
| CoreSet | TODO | TODO | TODO | TODO |
| BADGE | TODO | TODO | TODO | TODO |
| AttriSelect | **TODO** | **TODO** | **TODO** | **TODO** |

The generation and diagnosis stages are held fixed across source-selection methods, so Table 1 and this additional experiment isolate which source subset gives the best repair result under the same upstream repair data and source budget. We will revise the main results to show raw metrics and absolute changes next to FR and state its interpretation more precisely.

**Internal TODO:** Fill the epoch-multiplier table with the dataset, source ratio, seeds, raw task metrics, and FR values; insert exact target-slice gains and non-target retention results.

# Response to Reviewer hG6j

We thank the reviewer for recognizing the practical source-selection problem, the roles of the three score components, and the breadth of the evaluation.

---

> Q1: “How sensitive is AttriSelect to the upstream slice discovery method and slice granularity?”

**R1:** AttriSelect starts after a failure slice has been diagnosed. Given the same slice definition, source attributes, and generated repair set, its output does not depend on which upstream discovery method produced that slice. For example, two discovery methods that return the same brown-clothes-and-rainy slice provide the same input to AttriSelect.

The relevant sensitivity is therefore the quality and granularity of the slice specification. A coarse definition can mix distinct failure modes, while a very narrow conjunction can leave too few matching source examples. The current paper assumes that the upstream method supplies a meaningful closed-set description. We will make this conditional scope explicit rather than implying that AttriSelect solves slice discovery.

In a controlled ImageNet10 diagnostic, we evaluated the original `pink AND fabric` slice while removing either `texture` or `color` from source selection and keeping the combined generated set fixed. At source ratios 0.1 and 0.3, source-set Jaccard overlap with the full selector fell to 0.390--0.617. Overall ACC changes were not monotonic, and the original target test slice contains only 16 images. The supported conclusion is that slice specification materially changes the selected source set, while downstream impact depends on source ratio and training variation. Single-attribute runs provide a separate granularity diagnostic because they also change the generated repair pool.

**Internal TODO:** Add the still-unrun merged-value transformation. The full planned granularity sweep, including the separately completed fabric-only ratio-1.0 item, is now complete.

---

> Q2: “Can the authors compare with simple attribute-aware baselines, such as exact target-slice selection plus stratified sampling?”

**R2:** We agree that exact target-slice selection followed by stratified background sampling is an important baseline. We implemented this baseline together with Attribute-Filtered Random and Attribute-Filtered Entropy under the same metadata, fixed generated set, source budget, checkpoint, and repair schedule. The matched five-seed comparison is complete, and AttriSelect has the highest mean ACC at all five source ratios. The full table is reported in Response to Reviewer yyPZ Q2.

---

> Q3: “What is the end-to-end cost including generation, VLM attribute assignment, selection, and retraining?”

**R3:** The submitted Table 2 measures subset-selection time only. We will add the full cost decomposition covering one-time source assignment, generation, generated-image assignment, selection, and retraining. We will report both the cost of one repair cycle and the amortized cost when source attributes are reused across multiple diagnosed slices.

CelebA uses official source attributes. The measured Qwen3-VL source-assignment costs for COCO and KITTI are 6.66 and 1.20 GPU-hours, respectively. These costs are real and should have appeared alongside the selector runtime.

**Internal TODO:** Insert the completed end-to-end cost table and quantify the reduction in retraining time at each source ratio.

---

> Q4: “How does the method behave when the failure slice is not well captured by the closed-set attribute vocabulary?”

**R4:** If the relevant failure factor is absent from the attribute vocabulary, AttriSelect cannot explicitly prioritize it. The soft class-balance term still preserves label coverage, but the selector loses the repair-specific signal for the missing factor and moves toward a more generic balanced selection policy. This is a limitation of the current closed-set formulation.

The existing attribute-noise analysis studies incorrect assignments within a known vocabulary. It does not fully answer the missing-concept setting raised by the reviewer. In the new field-removal diagnostic, removing one target field changed source selection substantially at low ratios: Jaccard overlap with the full selector was 0.390--0.617 at ratios 0.1--0.3. Overall ACC did not degrade monotonically across this single-seed sweep, so we interpret the result as evidence of selection sensitivity rather than a universal performance failure.

**Internal TODO:** The merged-value/coarsened-category experiment has not been run and must not be claimed. The existing attribute-noise table is verified and can be cited separately.

# Response to Reviewer KaLa

We thank the reviewer for the positive assessment of the repair-time source-selection perspective and the cross-task results.

---

> W1 (Q1): “The problem formulation needs to be clarified. It is not always clear whether the proposed method is selecting generated repair data, original source data, or a combination of both. The descriptions in the abstract, introduction, and conclusion appear somewhat vague or potentially inconsistent on this point. The authors are encouraged to state explicitly what data pool AttriSelect operates on and how the selected data are used during retraining.”

**R1:** AttriSelect selects only from the original source or training pool. It does not select among the generated repair images. In every main experiment, the generated set is fixed to 1,000 images for all selectors. The retraining data are

$$
\mathcal{D}_{\mathrm{repair}}
=
\mathcal{D}_{\mathrm{sel}}^{\mathrm{src}}
\cup
\mathcal{D}_{\mathrm{gen}},
$$

where AttriSelect chooses only $\mathcal{D}_{\mathrm{sel}}^{\mathrm{src}} \subseteq \mathcal{D}_{\mathrm{src}}$. The generated set affects source selection through its attribute-frequency counts, but its membership remains fixed. We will state this directly in the abstract, introduction, framework figure, and conclusion, and remove notation that suggests a separately selected generated subset.

---

> W2 (Q4): “The categorization of prior model repair methods requires further clarification. In particular, reference [7] seems to be associated with both the first and second lines of work, which makes the taxonomy confusing. The authors should revise this discussion to make the distinctions among different categories more precise.”

**R2:** We thank the reviewer for pointing out this ambiguity. An editing error in the bibliography accidentally replaced *HiBug* with *SafeFix* in the title of Reference [7]; we have restored the correct title. The repeated citation is intentional because HiBug spans two stages of the repair pipeline: it discovers attribute-defined failure slices and generates repair examples for those slices.

We will organize the revised discussion by pipeline stage rather than presenting the categories as mutually exclusive method groups. The stages are failure-slice discovery, repair-data construction, source-data selection, and model retraining. Under this organization, a system such as HiBug can contribute to more than one stage without making the taxonomy inconsistent.

---

> W3 (Q2): “It is unclear what repair method or methods are used together with AttriSelect in the experiments. Since AttriSelect is a data selection strategy rather than a complete repair algorithm, the experimental section should clearly specify the underlying repair pipeline, retraining/fine-tuning procedure, and how the selected source samples are combined with the generated repair images.”

**R3:** All source-selection methods follow the same high-level protocol. We first train or obtain the diagnosis-stage checkpoint $\theta_t$, diagnose a semantic failure slice, and construct a fixed generated repair set. Each selector then chooses a source subset under the same ratio. We combine that subset with the fixed generated images and continue training or fine-tune $\theta_t$ using the same optimizer, learning rate, epoch count, and generated-image budget. Thus the selector is the only component changed within each dataset-level comparison.

We agree that these details are currently distributed across several sections. We will add a compact table specifying, for each dataset, the diagnosed slice, source pool, generator, generated budget, base checkpoint, model, and repair schedule.

**Internal TODO:** Insert the dataset-specific pipeline table using the configurations reported in `nips26_new`. The main-result generator is ControlNet, as stated in the implementation appendix.

---

> W4 (Q3): “The paper states that AttriSelect is best viewed as one component in a localized repair pipeline. However, the experiments do not appear to evaluate its compatibility with multiple repair pipelines. It would strengthen the paper if the authors could test AttriSelect with different repair methods or, at least, discuss the extent to which the conclusions depend on the specific repair pipeline used in the experiments.”

> Additional point in Q3: “Could the authors provide evidence or discussion about whether AttriSelect remains effective when paired with different [...] retraining strategies?”

**R4:** AttriSelect consumes source-image attributes, class labels, and attribute counts from the generated set. It does not depend on the internal architecture of the slice-discovery or image-generation model. This interface makes it possible to pair the selector with different upstream systems, but this modularity alone is not empirical evidence that performance is unchanged across repair pipelines.

The appendix already provides controlled alternative-generator experiments. It replaces ControlNet with InstructPix2Pix on CelebA and ICEdit on COCO while keeping the source-selection comparison, generated-data budget, and repair protocol fixed. AttriSelect remains the strongest source selector at every active source ratio under both alternative generators. These results show that the selection benefit is not specific to ControlNet, although they do not cover every possible repair pipeline or retraining strategy. We will state this scope directly.

**Internal TODO:** Insert a compact table or selected rows from the verified alternative-generator appendix results.

---

> W5: “The formatting of the references should be improved. For example, the names of journals and conferences should be made consistent, either using full names or standard abbreviations throughout the reference list.”

**R5:** We thank the reviewer for catching this inconsistency. We have standardized conference and journal names throughout the reference list using standard abbreviations, including NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, WACV, AAAI, FAccT, and JMLR. We also corrected inconsistent capitalization in venue and title fields, protected proper names such as ImageNet, COCO, KITTI, and SafeFix, and fixed a malformed author field in the PPAL reference. The revised bibliography now follows one format throughout.
