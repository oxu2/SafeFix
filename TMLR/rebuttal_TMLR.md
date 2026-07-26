## Note on Camera Ready Revision

Dear reviewers and Action Editor,

We have uploaded a Camera Ready Revision to incorporate the latest discussion-driven clarifications. The main updates are:

1. We further clarified the contribution relative to HiBug in the Introduction, Baselines, Main Results analysis, and Ablation Study. In particular, we now state that SafeFix and HiBug share the same high-level diagnose-generate-retrain structure, but differ in the repair-data construction step: HiBug generates new images from attribute prompts, whereas SafeFix edits real training images and verifies the edited outputs with LVLMs.
2. We clarified that the performance gap is not due to a stronger diffusion backbone, since both SafeFix and HiBug use Stable Diffusion 1.5. The key difference is source-image-conditioned editing plus LVLM filtering.
3. We updated the denser visual prediction task table caption to specify the metrics: COCO keypoint AP for pose estimation and mAP50-95 for KITTI object detection.
4. We adjusted the font and size of Figures 4 and 5 to match the rest of the text.

These updates are intended to make the revised manuscript more precise and easier to verify against the reviewer discussion. The revision does not change the core method or experimental data.


## Note on Revised PDF Upload

Dear reviewers and Action Editor,

We have uploaded a revised PDF during the discussion period to incorporate the clarifications and additional evidence discussed in the rebuttal. The revision does not change the core method. The main updates are:

1. We renamed the previous "Fix Rate" metric to **Relative Error Reduction (RER)**, clarified its aggregate error-reduction definition, and report it together with absolute accuracy.
2. We clarified the contribution relative to HiBug: SafeFix addresses semantic drift and distribution mismatch by **conditioning generation on real training images**, rather than generating images from attribute prompts alone.
3. We added additional experiments requested or motivated by the discussion, including **COCO pose estimation and KITTI object detection** in **Section 5.3 of the main text**, **several more targeted repair baselines and long-tail learning baselines** in **Appendix H**, and **cross-model LVLM validation** in **Appendix D.3**.
4. We expanded the limitations discussion.

We hope this revised PDF makes the response easier to verify before final recommendations are submitted.

## Response to Reviewer KtnS Follow-up

We thank the reviewer for confirming that Question 3 is adequately answered. We address the remaining questions below:

### 1. Contribution Relative to HiBug (Question 1)
We agree that SafeFix follows the same high-level repair structure as HiBug: diagnose a failure slice, generate additional data, and retrain. However, our contribution is in the **repair-data construction step**. Specifically, SafeFix addresses two failure modes of prompt-only generation: **semantic drift** and **distribution mismatch**.
* **HiBug** proposes attribute text and generates a completely new image from the prompt. Even when the target attribute appears, the generated image may no longer match the original training distribution (e.g., changes in eye size, face contour, background, pose, and lighting).
* **SafeFix** instead **conditions generation on real training images**, editing diagnosed attributes while preserving all other visual factors of the source image.

### 2. Performance of HiBug + LVLM Filtering
The reviewer asks why LVLM filtering does not resolve the issue of bad generated samples in HiBug (e.g., Fig. 3(b)), and why HiBug+LVLM performance remains low in Table 6.
* **Same Backbone:** Both SafeFix and HiBug use the same underlying diffusion backbone (**Stable Diffusion 1.5**). The performance difference is not due to the generative models.
* **Filtering vs. Conditional Generation:** LVLM filtering can remove visibly invalid HiBug samples, but it **cannot enforce consistency with a specific original training image**. For example, even if the filter keeps an image with red hair, that image is not an edit of the source image. Non-target features (e.g., face shape, background) may all differ from the original, which are exactly the factors that prompt-only HiBug generation ignores.
* **Why SafeFix Succeeds:** SafeFix performs conditional image editing from real training images and then filters the edited outputs. Thus, the LVLM operates on candidates that are already tied to the source distribution. Filtering alone cannot replace conditional generation from the original training data.

### 3. Metric Renaming and Clarification
We agree that the term "Fix Rate" can be misleading. We have renamed it to **Relative Error Reduction (RER)**. The metric is computed from aggregate accuracy before and after repair:

$$
RER =
\frac{Err_{\text{before}} - Err_{\text{after}}}
{Err_{\text{before}}}
= \frac{(1 - Acc_{\text{before}}) - (1 - Acc_{\text{after}})}
{1 - Acc_{\text{before}}}
= \frac{Acc_{\text{after}} - Acc_{\text{before}}}
{1 - Acc_{\text{before}}}.
$$
Here, $Err = 1 - Acc$. This measures the fraction of the original error rate removed by repair. We will also report it together with absolute accuracy so the effect size is clear.
Since the formula uses aggregate accuracy values rather than per-sample correctness changes, we will avoid saying that it directly counts previously misclassified samples that become correct.


# Response to Reviewer KtnS

## 1. Contribution and Novelty

**Reviewer KtnS Comment:**
> The claim is supported, but it is not a contribution. I am also not sure about its novelty. [...] the overall pipeline feels like a mix of existing works.

**Response:**
We thank the reviewer for raising this point. SafeFix does use existing modules such as ControlNet and large vision-language models (LVLMs). The contribution is not that these modules are new in isolation; it is the repair procedure that combines them to target underrepresented semantic subpopulations while controlling distribution shift and semantic drift.

Previous generation-based methods (such as HiBug) rely solely on language prompts, which often fail to preserve attributes unrelated to the intended edit and introduce a mismatch with the original data distribution. Previous retrieval-based methods (such as AdaVision and DCD) rely on external datasets, which often introduce a domain gap.

SafeFix addresses these limitations in two concrete ways:
1. It anchors generation on real training instances using structure-conditioned control, so the edited samples stay close to the source distribution.
2. It uses LVLM filtering to check that the generated images contain the intended attributes and preserve the label.

We will add the following table to make this distinction explicit:

| Feature | AdaVision / DCD | HiBug | SafeFix (Ours) |
| :--- | :--- | :--- | :--- |
| **Strategy** | Retrieval from external datasets | Generation via language prompts | Generation conditioned on training images |
| **Distribution Alignment** | Low (Domain gap from external data) | Low (Generative artifacts and drift) | High (Preserves original image structure) |
| **Attribute Preservation** | N/A | Low (Prompt-only control) | High (Structure conditioning) |
| **Output Verification** | Implicit | None | Explicit (LVLM filtering) |

## 2. Generation-based vs. Retrieval-based Approaches

**Reviewer KtnS Comment:**
> This is a question, not a requested change. You mentioned that works like AdaVision and DCD retrieve samples from another dataset. If you have a sufficiently rich dataset, how do such methods compare to the proposed generation-based approach?

**Response:**
This is a useful question. If an infinitely rich external dataset existed, retrieval would be attractive because it avoids generative artifacts. In practice, controlled generation has several advantages:

1. **Targeting Rare Combinations:** Even in extremely large datasets, specific combinations of attributes (for example, a specific object in a highly unusual background or lighting condition) may simply not exist or remain statistically rare. Generation allows us to create these exact rare compositions on demand.
2. **Distribution consistency:** Retrieved samples from an external dataset can carry domain shifts such as different sensors, resolutions, or object framings. SafeFix conditions on the original training images, so non-targeted attributes and image structure remain tied to the target dataset.
3. **Search efficiency:** Finding rare attribute combinations in a large external pool can require scanning millions of images. Conditional generation creates the requested combination directly.

# Response to Reviewer NMkE

## 1. Fix Rate Interpretation and Absolute Accuracy

**Reviewer NMkE Comment:**
> Fix Rate is the dominant metric but is hard to compare across rows, because it normalizes by the initial error rate, which varies a lot between settings. The most striking numbers correspond to roughly a one-point absolute accuracy gain, a much smaller real effect than the headline suggests.

**Response:**
We agree that the metric now renamed **Relative Error Reduction (RER)** should not be read as the only effect-size measure. The revised definition makes the normalization explicit. Let $Err = 1 - Acc$. Then:

$$
RER =
\frac{Err_{\text{before}} - Err_{\text{after}}}
{Err_{\text{before}}}
= \frac{(1 - Acc_{\text{before}}) - (1 - Acc_{\text{after}})}
{1 - Acc_{\text{before}}}
= \frac{Acc_{\text{after}} - Acc_{\text{before}}}
{1 - Acc_{\text{before}}}.
$$

This measures the fraction of the original error rate removed by repair, which clarifies what the metric captures. It is useful for model repair under the same model, dataset, and repair budget, but it should be paired with absolute test accuracy. **Appendix Tables 7 and 8** already report those accuracy values for every method, model, and augmentation budget. The gains are not just a normalization artifact: on CelebA with 1,000 added images, SafeFix improves test accuracy from 90.57% to 91.92% for ResNet (+1.35%), from 85.02% to 87.91% for ViT (+2.89%), and from 88.32% to 90.98% for CLIP (+2.66%). On ImageNet10, SafeFix improves ResNet from 71.73% to 74.31% with 100 added images (+2.58%) and to 74.88% with 1,000 added images (+3.15%). For the high-accuracy ViT/ImageNet10 setting noted by the reviewer, the absolute gain is smaller because the base accuracy is already 97.42%, but SafeFix still reaches 98.41% with 1,000 added images (+0.99%), while the strongest baseline in the same setting reaches 97.78% or lower depending on the augmentation budget.

SafeFix is also not meant to be an unrestricted recipe for state-of-the-art classification accuracy. The question is narrower: given the same original model, the same diagnosed failure slice, and the same added-data budget, can the method repair underrepresented slices without hurting overall performance? Under that controlled repair setting, SafeFix gives the strongest absolute accuracy and RER across the compared methods. In the revision, we will move the appendix accuracy tables closer to the main results or cross-reference them directly when discussing RER.

## 2. Threshold-Conditioned "No New Bugs" Claim

**Reviewer NMkE Comment:**
> The "no new bugs" claim is defined entirely through the thresholds $\rho$ and $\epsilon$, whose sensitivity the authors themselves note, so the claim as stated in the main text is stronger than the threshold-dependent evidence supports.

**Response:**
We agree that the claim needs a clear evaluation scope. The current evidence does show no newly introduced bugs under the evaluated attribute sets. **Figure 4** reports attribute-level validation accuracy before and after repair: targeted attributes improve, while non-targeted attributes do not form new failure slices. On ImageNet10, for example, SafeFix improves `pink color` from 69.90% to 74.76% and `fabric texture` from 65.85% to 75.61%; non-targeted attributes such as the `rocks` background remain stable. The attribute-variant analysis also shows that detected bugs drop from 10 to 4 on CelebA and from 13 to 7 on ImageNet10, with no additional bug slices appearing among the evaluated attributes.

We will revise the wording to state the scope directly: within the attribute sets and bug definitions used in our evaluation, SafeFix does not introduce additional detected failure slices. We will also move the threshold-sensitivity discussion closer to this claim so readers can see that the bug count is measured under specified $\rho$ and $\epsilon$ values.

## 3. Separation Between Diagnosis, Attribute Assignment, and Filtering Models

**Reviewer NMkE Comment:**
> There seems to be a circular evaluation within the method: The same family of LVLMs proposes attributes, assigns slice membership, filters generated data, and (indirectly) defines the slices on which the "fix" is measured.

**Response:**
We will clarify that these stages are not performed by the same LVLM. Candidate attributes are proposed by **GPT-4 with vision** following the HiBug-style diagnosis setting. Slice membership is assigned using **BLIP-based VQA** under the same attribute-assignment setting as HiBug. Filtering then uses **Qwen2.5-VL-7B** and **LLaVA-v1.5-7B** to check the generated images. The models used for attribute proposal, slice assignment, and filtering are therefore separated.

Correlated LVLM errors are still a fair concern, so we add a cross-model validation check. Samples accepted by **Qwen2.5-VL-7B** are re-evaluated by **LLaVA-v1.5-7B**, and vice versa. This does not require retraining, and it directly tests whether the retained samples are accepted only by the original filter or also by an independent LVLM. We will also make clear that the final repair effect is measured by downstream classifier accuracy and bug counts on the fixed validation/test split, not by the LVLM filter itself.

We will add the following cross-model validation table:

| Dataset | Target attributes | Original filter | Independent verifier | Accepted samples | Attribute pass rate | Label-preservation pass rate | All-condition pass rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CelebA | red hair + brown skin + sad emotion | Qwen2.5-VL-7B | LLaVA-v1.5-7B | 1,000 | 91.8% | 98.4% | 87.6% |
| CelebA | red hair + brown skin + sad emotion | LLaVA-v1.5-7B | Qwen2.5-VL-7B | 1,053 | 78.6% | 97.1% | 72.4% |
| ImageNet10 | pink color + fabric texture | Qwen2.5-VL-7B | LLaVA-v1.5-7B | 1,000 | 94.2% | 99.0% | 91.5% |
| ImageNet10 | pink color + fabric texture | LLaVA-v1.5-7B | Qwen2.5-VL-7B | 1,129 | 84.7% | 98.2% | 80.3% |

The results show that LLaVA-v1.5-7B is less conservative and accepts more generated samples, while Qwen2.5-VL-7B gives a stricter verification signal. Samples accepted by one LVLM still achieve high all-condition pass rates under the independent verifier, so the filtering results are not just artifacts of a single LVLM's decision boundary.
This observation is consistent with our Appendix LVLM analysis, where Qwen2.5-VL is described as the stricter default verifier and LLaVA-v1.5 as having a higher pass rate, especially for emotion-related attributes.

## 4. Comparison to Dataset Interfaces and Latent-space Failure Directions

**Reviewer NMkE Comment:**
> An empirical comparison against at least one closely related targeted-repair-via-generation method (Vendrow et al. 2023, Dataset Interfaces; or Jain et al. 2023, latent-space failure directions) can be added.

**Response:**
We agree that Dataset Interfaces (DI) [Vendrow et al., 2023] and Latent-space Failure Directions (LFD) [Jain et al., 2023] are closer targeted-repair baselines than generic augmentation methods, and we will add both in the revision. We adapt DI from counterfactual diagnosis to repair by using its generated counterfactual samples as augmentation data and retraining the downstream classifier under the same data budget as SafeFix. For example, on ImageNet10, DI learns one class-specific textual inversion token for each class and generates attribute-conditioned counterfactual samples using prompts such as:

```text
"a photo of <class-token> with pink color and fabric texture"
```

We adapt LFD in the same spirit, using failure-directed synthetic samples as augmentation data. For each class, LFD trains a linear SVM in CLIP image-embedding space to separate correctly classified validation samples from misclassified ones. The SVM normal vector is treated as a class-specific failure direction and matched with text descriptions in CLIP text space to construct failure-conditioned generation prompts.

We will report the two new baselines in the following comparison table:

| Method | CelebA Test ACC | CelebA Fixed Bugs (out of 13) | ImageNet10 Test ACC | ImageNet10 Fixed Bugs (out of 17) |
| :--- | :--- | :--- | :--- | :--- |
| Base | 0.9057 | - | 0.7173 | - |
| Dataset Interfaces (DI) | 0.9132 | 8 | 0.7297 | 13 |
| Latent-space Failure Directions (LFD) | 0.9188 | **11** | 0.7454 | **15** |
| SafeFix | **0.9192** | 10 | **0.7488** | **15** |

These adapted baselines are strong, especially LFD, which fixes 11 CelebA bugs and 15 ImageNet10 bugs. LFD can fix slightly more bugs because it explicitly learns directions that separate misclassified from correctly classified validation samples, so its generations concentrate near the model's failure modes. SafeFix still achieves the highest test accuracy on both datasets and matches the best ImageNet10 bug-fixing count, which suggests a better repair-accuracy tradeoff.

## 5. Large-scale and Pretrained Classifier Validity

**Reviewer NMkE Comment:**
> For the validity of the method against large-scale datasets and pretrained classifiers, the authors should either utilize their method on classifiers that are trained on larger scale data (such as the complete ImageNet dataset) or perform their training accordingly (on larger datasets).

**Response:**
We agree that large-scale and pretrained settings should be discussed more explicitly. We did not run full ImageNet-scale repair because the repair step requires conditional generation that changes only a small number of target attributes while preserving the class label and unrelated content. For many ImageNet classes, those edits are hard to define and validate without changing the object identity or creating ambiguous labels. We therefore use CelebA and ImageNet10 in the classification experiments to isolate the repair mechanism under controlled attribute annotations, fixed validation/test splits, and matched augmentation budgets.

We also did not use pretrained classifiers in the main classification experiments for the reason stated in Appendix E: "This design allows us to isolate the effectiveness of SafeFix from pre-existing biases inherent in pre-trained weights, such as the latent knowledge of ImageNet. This approach is also necessary because when generated images are added to a source dataset, the specific large-scale datasets used for subsequent model training are often unknown, and retraining on such data is not feasible." This choice makes the repair-data comparison controlled; it does not mean SafeFix only applies to models trained from scratch.

We will also bring the larger visual-task results into the main paper. On COCO person keypoints with Keypoint R-CNN and a ResNet-50-FPN backbone, SafeFix improves test AP from 0.5902 to 0.6280, while the best competing baseline reaches 0.6145. On KITTI object detection with YOLOv8n, SafeFix improves test mAP50-95 from 0.5626 to 0.5790, compared with 0.5713 for the best competing baseline. These results do not replace a full ImageNet-scale pretrained-classifier study, but they show that the diagnose-generate-filter-retrain pipeline works on larger datasets and denser visual prediction tasks, including settings with standard pretrained visual backbones. We will revise the claims accordingly: the current evidence supports SafeFix as a controlled model-repair framework across classification, pose estimation, and detection.

# Response to Reviewer 1vpz

## 1. Contribution Clarity

**Reviewer 1vpz Comment:**
> The contribution is not clearly separated from prior work [...] The actual difference is in Stages 2–3: a stronger generation backbone and the addition of LVLM filtering. The delta over HiBug is primarily "use stronger off-the-shelf tools", which is an incremental upgrade enabled by today's foundation models.

**Response:**
We agree that Stage 1 builds on HiBug-style diagnosis and that Stage 4 is standard retraining. The contribution is the repair mechanism in Stages 2-3: SafeFix combines structure-conditioned editing and LVLM verification to address a failure mode of prior generation-based repair pipelines.

HiBug's text-prompt generation can change non-target attributes, backgrounds, and image structure, which creates semantic drift and distribution mismatch. SafeFix anchors generation on real training images through ControlNet, edits the diagnosed failure attributes, and preserves source-image structure and unrelated visual factors. The LVLM filter then rejects generations that miss the intended attributes or change the label. We will revise the introduction and related-work discussion to state this contribution more explicitly and add a comparison table separating SafeFix from HiBug, retrieval-based repair methods, and generic augmentation methods.

## 2. Connection to Long-Tail Learning (LTL) and Distributional Safety

**Reviewer 1vpz Comment:**
> The paper does not engage with the long-tail learning field. [...] the paper does not adequately analyze whether injecting synthetic data for rare slices shifts the effective training distribution and degrades performance on common classes.

**Response:**
We agree that the long-tail learning connection should be discussed. The key distinction is between *class-level* long-tail learning and *attribute-level* subgroup failures. Standard LTL methods reweight or resample based on class frequency, but they do not see latent attribute failures such as "red hair + brown skin" in a "lipstick" classifier; they only see the main class label.

To test this, we evaluated four standard LTL baselines using only the original data:

| Method | CelebA Test ACC | CelebA Fixed Bugs (out of 13) | ImageNet Test ACC | ImageNet Fixed Bugs (out of 17) |
| :--- | :--- | :--- | :--- | :--- |
| **Base** | 0.9057 | - | 0.7173 | - |
| Focal Loss | 0.9134 | 8 | 0.6880 | 11 |
| Class-Balanced Loss | 0.9122 | 8 | 0.7270 | 13 |
| Oversampling | 0.9101 | 7 | 0.7290 | 14 |
| LDAM | 0.9141 | 8 | 0.7120 | 12 |
| **SafeFix (Ours)** | **0.9192** | **10** | **0.7488** | **15** |

SafeFix improves overall accuracy while fixing more discovered attribute-level bugs. We will add this LTL comparison and discussion to the revised manuscript.

## 3. Scalability and Dataset Complexity

**Reviewer 1vpz Comment:**
> both datasets are not large enough to support the paper's generality and scalability: CelebA is reduced to a single binary task with a small attribute vocabulary, and ImageNet10 contains only 10 hand-picked classes.

**Response:**
We understand the concern about evaluation scale. To test richer attribute spaces and denser prediction tasks, we also evaluated SafeFix on object detection and pose estimation.

Specifically, we follow HiBug2 on COCO person keypoints, using Keypoint R-CNN with a ResNet-50-FPN backbone, 20,000 source training images, and 1,173-image held-out validation and test splits, evaluated by COCO keypoint AP. For object detection, we use KITTI Car and Pedestrian instances converted to YOLO format with 5,000/1,240/1,241 train/validation/test images, and train YOLOv8n. Implementation details are given in Appendix~\ref{app:implementation-details}.

| Task | Dataset / Model | Source Train | Val / Test | Metric | Base | Best Baseline | SafeFix | Gain over Best Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pose estimation | COCO person keypoints / Keypoint R-CNN ResNet-50-FPN | 20,000 | 1,173 / 1,173 | Test AP | 0.5902 | 0.6145 | **0.6280** | +0.0135 |
| Object detection | KITTI Car+Pedestrian / YOLOv8n | 5,000 | 1,240 / 1,241 | Test mAP50-95 | 0.5626 | 0.5713 | **0.5790** | +0.0077 |

These results show that SafeFix is not limited to the small classification setting: it improves over both the base model and the best competing baseline on denser prediction tasks with larger training sets and richer attribute spaces. We will move these results from the appendix into the main text.

## 4. Broader Impact Concerns

**Reviewer 1vpz Comment:**
> if the LVLM rejects brown-skin images at a higher rate than white-skin images (plausible given known biases in vision-language models), the "fairness repair" pipeline may itself be biased.

**Response:**
We agree that LVLM bias is a real risk. Appendix D.2 reports that the pass rate of Qwen2.5-VL for emotion filtering is about 80%. We therefore use a conservative filtering prompt. For emotion filtering, for example, the prompt asks the LVLM: *"If you see any hint of [target] emotion in this picture, answer 1; only answer 0 if you are completely sure there is no such emotion."* This reduces unwarranted rejections. In our experiments, using this conservative prompt instead of a stricter one does not materially change the final repaired-model accuracy.

If rejection rates are higher for a subgroup, the pipeline can generate a larger initial pool to reach the target number of accepted samples; this increases compute, but it does not require accepting lower-quality images. We will add this limitation to the Broader Impact section and discuss debiased LVLMs and adaptive thresholds as future work.
    
