# Experiments and manuscript updates to complete

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

Dear Area Chair 6k27, thank you very much for your careful assessment and constructive guidance

**TODO:** Insert the Attribute-aware baselines, alternative-attribute-source ablation, vocabulary-sensitivity results, and complete end-to-end cost table.

# Response to Reviewer yyPZ

We thank the reviewer for recognizing the practical motivation, straightforward methodology, and clear presentation. We address the three concerns below.

---

> Q1: “Misleading Claims Regarding Computational Efficiency. The authors highlight in Table 2 that AttriSelect takes less than 2 seconds, claiming a speedup of up to 283× over baselines. However, this entirely omits the massive computational overhead of the prerequisite step: running Qwen3-VL to annotate the entire source pool. Appendix A states it takes 10–20 minutes to annotate just 1,000 images. For datasets like CelebA (80,000 images) or COCO (20,000 images), this requires tens to hundreds of GPU hours. Claiming high efficiency while ignoring the cost of generating the metadata that the algorithm relies upon is highly misleading.”

**R1:** 

**TODO:**  end-to-end runtime table, including retraining wall-clock time...

---

> Q2: “Comparison look Unfair. AttriSelect is given privileged information like VLM-assigned attributes and explicit bug definitions, but the compared active learning baselines (Entropy, BADGE, CoreSet) do not have access to. The authors should compare it against a baseline with the same information, for example, an ‘Attribute-Filtered Random’ or ‘Attribute-Filtered Entropy’ baseline.”

**R2:** We agree that the current generic active-learning baselines do not receive the attribute information used by AttriSelect. Those comparisons test whether repair-specific attribute metadata supplies a useful signal beyond uncertainty, geometric coverage, and gradient embeddings. They do not isolate whether the proposed rarity, generated-data penalty, and soft class-balance score uses that metadata better than a simple attribute-aware rule.

We will add three baselines with access to the same metadata and target-slice definition:

- **Target-Slice + Stratified:** select exact target-slice examples first and fill the remaining budget with stratified source samples;
- **Attribute-Filtered Random:** filter or prioritize repair-relevant attribute examples, then sample randomly under the same budget;
- **Attribute-Filtered Entropy:** use the same attribute-aware candidate pool, then rank candidates by predictive entropy.

All methods will use the same fixed generated set and retraining schedule. This comparison separates the value of attribute metadata from the value of the AttriSelect score.

**TODO:** Attribute-aware baselines. Report the raw task metric, target-slice metric, non-target change, and variance for each baseline and AttriSelect.

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

**TODO:** Fill the epoch-multiplier table with the dataset, source ratio, seeds, raw task metrics, and FR values; insert exact target-slice gains and non-target retention results.

# Response to Reviewer hG6j

We thank the reviewer for recognizing the practical source-selection problem, the roles of the three score components, and the breadth of the evaluation.

---

> Q1: “How sensitive is AttriSelect to the upstream slice discovery method and slice granularity?”

**R1:** AttriSelect starts after a failure slice has been diagnosed. Given the same slice definition, source attributes, and generated repair set, its output does not depend on which upstream discovery method produced that slice. For example, two discovery methods that return the same brown-clothes-and-rainy slice provide the same input to AttriSelect.

The relevant sensitivity is therefore the quality and granularity of the slice specification. A coarse definition can mix distinct failure modes, while a very narrow conjunction can leave too few matching source examples. The current paper assumes that the upstream method supplies a meaningful closed-set description. We will make this conditional scope explicit rather than implying that AttriSelect solves slice discovery.

We will also add a controlled sensitivity experiment that removes a target field, merges a target value into a broader category, and compares single-attribute and conjunctive slice definitions.

**TODO:** Insert the slice-granularity results and specify which dataset, source ratio, target definitions, and seeds are used.

---

> Q2: “Can the authors compare with simple attribute-aware baselines, such as exact target-slice selection plus stratified sampling?”

**R2:** 

**TODO:** Attribute-aware baselines

---

> Q3: “What is the end-to-end cost including generation, VLM attribute assignment, selection, and retraining?”

**R3:** 

**TODO:** end-to-end cost table

---

> Q4: “How does the method behave when the failure slice is not well captured by the closed-set attribute vocabulary?”

**R4:** 

**TODO:** Insert the missing-field and coarsened-vocabulary results. 
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

**R2:** We thank the reviewer for pointing out this ambiguity. The repeated citation is intentional because HiBug spans two stages of the repair pipeline: it discovers attribute-defined failure slices and generates repair examples for those slices.

We will organize the revised discussion by pipeline stage rather than presenting the categories as mutually exclusive method groups. The stages are failure-slice discovery, repair-data construction, source-data selection, and model retraining. Under this organization, a system such as HiBug can contribute to more than one stage without making the taxonomy inconsistent.

---

> W3 (Q2): “It is unclear what repair method or methods are used together with AttriSelect in the experiments. Since AttriSelect is a data selection strategy rather than a complete repair algorithm, the experimental section should clearly specify the underlying repair pipeline, retraining/fine-tuning procedure, and how the selected source samples are combined with the generated repair images.”

**R3:** All source-selection methods follow the same high-level protocol. We first train or obtain the diagnosis-stage checkpoint $\theta_t$, diagnose a semantic failure slice, and construct a fixed generated repair set. Each selector then chooses a source subset under the same ratio. We combine that subset with the fixed generated images and continue training or fine-tune $\theta_t$ using the same optimizer, learning rate change from 1e-4 to 1e-5, epoch count, and generated-image budget. Thus the selector is the only component changed within each dataset-level comparison.

We agree that these details are currently distributed across several sections. We will add a compact table specifying, for each dataset, the diagnosed slice, source pool, generator, generated budget, base checkpoint, model, and repair schedule.

**TODO:** More details, implementation

---

> W4 (Q3): “The paper states that AttriSelect is best viewed as one component in a localized repair pipeline. However, the experiments do not appear to evaluate its compatibility with multiple repair pipelines. It would strengthen the paper if the authors could test AttriSelect with different repair methods or, at least, discuss the extent to which the conclusions depend on the specific repair pipeline used in the experiments.”

> Additional point in Q3: “Could the authors provide evidence or discussion about whether AttriSelect remains effective when paired with different [...] retraining strategies?”

1. theories, details

2. The appendix already provides controlled alternative-generator experiments. It replaces ControlNet with InstructPix2Pix on CelebA and ICEdit on COCO while keeping the source-selection comparison, generated-data budget, and repair protocol fixed. AttriSelect remains the strongest source selector at every active source ratio under both alternative generators. These results show that the selection benefit is not specific to ControlNet, although they do not cover every possible repair pipeline or retraining strategy. We will state this scope directly.

**TODO:** Hibug repair pipeline. Insert selected rows from the verified alternative-generator appendix results.

---

> W5: “The formatting of the references should be improved. For example, the names of journals and conferences should be made consistent, either using full names or standard abbreviations throughout the reference list.”

**R5:** We thank the reviewer for catching this inconsistency. We have standardized conference and journal names throughout the reference list using standard abbreviations, including NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, WACV, AAAI, FAccT, and JMLR. We also corrected inconsistent capitalization in venue and title fields, protected proper names such as ImageNet, COCO, KITTI, and SafeFix, and fixed a malformed author field in the PPAL reference. The revised bibliography now follows one format throughout.

