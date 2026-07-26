# Experiments and manuscript updates to complete


end to end: training, vlm annotating, selection time


## Highest-priority experiments

- [ ] **Attribute-aware baselines under the same information budget.** Add (i) exact target-slice selection followed by stratified background sampling, (ii) Attribute-Filtered Random, and (iii) Attribute-Filtered Entropy. Use the same source budget, fixed generated set, attribute annotations, class labels, training schedule, and seeds as AttriSelect.
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

**R1:** We agree. The submitted baselines test uncertainty, geometric coverage, and gradient-space coverage, but they do not receive the same attribute records and target-slice definition as AttriSelect. We will add three same-information baselines: (i) exact target-slice selection followed by stratified background sampling, (ii) Attribute-Filtered Random, and (iii) Attribute-Filtered Entropy. Every method will use the same source budget, fixed generated set, attribute annotations, class labels, repair schedule, and random seeds. This comparison directly tests whether AttriSelect's rarity, generated-data frequency penalty, and soft class-balance terms improve over simpler rules given the same information.

**Internal TODO:** Insert the same-information baseline table with the raw task metric, target-slice metric, non-target change, and variance.

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

**R4:** We agree that imperfect attribute assignment and imperfect attribute vocabularies are distinct cases. The submitted attribute-noise study and human audit evaluate assignment errors within a known vocabulary, but they do not replace an attribute-generation ablation and do not fully address a missing or overly coarse concept. We will therefore add two controlled tests. First, on a representative setting, we will replace the main Qwen3-VL attribute assignment with an alternative attribute source or annotator while keeping the closed-set vocabulary and repair protocol fixed; we will compare attribute agreement, selected-set overlap, target-slice enrichment, and downstream repair performance. Second, we will compare the full target conjunction with a coarser slice, a missing target field, and a merged target value. We will also separate one-time source-pool attribute assignment, generated-image assignment, generation, subset selection, and retraining in the end-to-end cost table so that the contribution and cost of each pipeline component are explicit.

**Internal TODO:** Insert the alternative-attribute-source ablation, vocabulary-sensitivity results, and complete end-to-end cost table.

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

We will add three baselines with access to the same metadata and target-slice definition:

- **Target-Slice + Stratified:** select exact target-slice examples first and fill the remaining budget with stratified source samples;
- **Attribute-Filtered Random:** filter or prioritize repair-relevant attribute examples, then sample randomly under the same budget;
- **Attribute-Filtered Entropy:** use the same attribute-aware candidate pool, then rank candidates by predictive entropy.

All methods will use the same fixed generated set and retraining schedule. This comparison separates the value of attribute metadata from the value of the AttriSelect score.

**Internal TODO:** Insert the same-information baseline table. Report the raw task metric, target-slice metric, non-target change, and variance for each baseline and AttriSelect.

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

We will also add a controlled sensitivity experiment that removes a target field, merges a target value into a broader category, and compares single-attribute and conjunctive slice definitions.

**Internal TODO:** Insert the slice-granularity results and specify which dataset, source ratio, target definitions, and seeds are used.

---

> Q2: “Can the authors compare with simple attribute-aware baselines, such as exact target-slice selection plus stratified sampling?”

**R2:** We agree that exact target-slice selection followed by stratified background sampling is an important baseline. We will compare it with AttriSelect, Attribute-Filtered Random, and Attribute-Filtered Entropy under the same metadata and data budget. This test directly asks whether modeling rarity, overlap with the generated pool, and soft class balance improves over a binary target-slice rule.

**Internal TODO:** Reuse the same-information baseline table from the response to Reviewer yyPZ and summarize the main result here.

---

> Q3: “What is the end-to-end cost including generation, VLM attribute assignment, selection, and retraining?”

**R3:** The submitted Table 2 measures subset-selection time only. We will add the full cost decomposition covering one-time source assignment, generation, generated-image assignment, selection, and retraining. We will report both the cost of one repair cycle and the amortized cost when source attributes are reused across multiple diagnosed slices.

CelebA uses official source attributes. The measured Qwen3-VL source-assignment costs for COCO and KITTI are 6.66 and 1.20 GPU-hours, respectively. These costs are real and should have appeared alongside the selector runtime.

**Internal TODO:** Insert the completed end-to-end cost table and quantify the reduction in retraining time at each source ratio.

---

> Q4: “How does the method behave when the failure slice is not well captured by the closed-set attribute vocabulary?”

**R4:** If the relevant failure factor is absent from the attribute vocabulary, AttriSelect cannot explicitly prioritize it. The soft class-balance term still preserves label coverage, but the selector loses the repair-specific signal for the missing factor and moves toward a more generic balanced selection policy. This is a limitation of the current closed-set formulation.

The existing attribute-noise analysis studies incorrect assignments within a known vocabulary. It does not fully answer the missing-concept setting raised by the reviewer. We will separate these cases in the revision and add a field-removal and value-merging experiment.

**Internal TODO:** Insert the missing-field and coarsened-vocabulary results. The existing attribute-noise table is verified and can be cited separately.

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
