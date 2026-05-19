# Rebuttal to Reviewer

## 1. Contribution and Novelty

**Reviewer Comment:**
> The claim is supported, but it is not a contribution. I am also not sure about its novelty. [...] the overall pipeline feels like a mix of existing works.

**Response:**
We thank the reviewer for the opportunity to clarify our contributions. We agree that our pipeline builds upon established modules such as ControlNet and large vision-language models (LVLMs). However, our main contribution is not the invention of these individual modules, but rather the specific way they are integrated to solve a challenging problem: targeted model repair for underrepresented semantic subpopulations without introducing distribution shift or semantic drift.

Previous generation-based methods (such as HiBug) rely solely on language prompts, which often fail to preserve attributes unrelated to the intended edit and introduce a mismatch with the original data distribution. Previous retrieval-based methods (such as AdaVision and DCD) rely on external datasets, which often introduce a domain gap.

SafeFix addresses these limitations through a carefully designed pipeline:
1. It anchors generation on real training instances using structure-conditioned control, ensuring distribution alignment.
2. It introduces a verification mechanism using LVLMs to ensure precise semantic editing of sensitive attributes.

We will add the following table to the manuscript to explicitly highlight how SafeFix differs from prior works:

| Feature | AdaVision / DCD | HiBug | SafeFix (Ours) |
| :--- | :--- | :--- | :--- |
| **Strategy** | Retrieval from external datasets | Generation via language prompts | Generation conditioned on training images |
| **Distribution Alignment** | Low (Domain gap from external data) | Low (Generative artifacts and drift) | High (Preserves original image structure) |
| **Attribute Preservation** | N/A | Low (Prompt-only control) | High (Structure conditioning) |
| **Output Verification** | Implicit | None | Explicit (LVLM filtering) |

## 2. Metric Definition (Fix Rate)

**Reviewer Comment:**
> According to the paper, the metric Fix Rate "measures the fraction of previously misclassified samples that are corrected." However, Eq. (6) does not compute that. It looks only at the aggregate accuracy, not per-sample accuracy.

**Response:**
We appreciate this precise observation. If we expand the accuracy terms as $Acc_{\text{before}} = \frac{C_{\text{before}}}{N}$ and $Acc_{\text{after}} = \frac{C_{\text{after}}}{N}$ (where $C$ is the number of correct predictions and $N$ is the total number of samples), the derivation of the equation unfolds as follows:
$$ FR = \frac{Acc_{\text{after}} - Acc_{\text{before}}}{1 - Acc_{\text{before}}} = \frac{\frac{C_{\text{after}}}{N} - \frac{C_{\text{before}}}{N}}{1 - \frac{C_{\text{before}}}{N}} = \frac{\frac{C_{\text{after}} - C_{\text{before}}}{N}}{\frac{N - C_{\text{before}}}{N}} = \frac{C_{\text{after}} - C_{\text{before}}}{N - C_{\text{before}}} $$
Here, $N - C_{\text{before}}$ is exactly the total number of previously misclassified samples, and $C_{\text{after}} - C_{\text{before}}$ represents the net increase in correct predictions. Therefore, in a net aggregate sense, $FR$ does functionally compute the fraction of previously misclassified samples that are corrected. 

However, we fully agree with the reviewer that because this formula relies on aggregate counts rather than tracking the exact per-sample transitions (i.e., verifying that sample $i$ specifically flipped from incorrect to correct), describing it as the "relative reduction in the error rate" is mathematically more rigorous and avoids any ambiguity. We will update the text in the revised manuscript to adopt this improved phrasing. 

## 3. Generation-based vs. Retrieval-based Approaches

**Reviewer Comment:**
> This is a question, not a requested change. You mentioned that works like AdaVision and DCD retrieve samples from another dataset. If you have a sufficiently rich dataset, how do such methods compare to the proposed generation-based approach?

**Response:**
This is an insightful question. If an infinitely rich and diverse dataset were available, retrieval-based methods would perform very well because they avoid generative artifacts. However, in practice, generation-based approaches offer three distinct advantages over retrieval:

1. **Targeting Rare Combinations:** Even in extremely large datasets, specific combinations of attributes (for example, a specific object in a highly unusual background or lighting condition) may simply not exist or remain statistically rare. Generation allows us to create these exact rare compositions on demand.
2. **Distribution Consistency:** Retrieved samples from an external dataset often carry unintended domain shifts (for example, differences in camera sensors, resolutions, or typical object framings). SafeFix generates samples by conditioning directly on the original training images. This ensures that all non-targeted attributes and the general image distribution match the target task exactly, isolating the repair to only the intended failure attributes.
3. **Search Efficiency:** Retrieval from massive datasets, especially when searching for highly specific attribute combinations, often requires iterating through millions of samples. This process can be exceptionally time-consuming. In contrast, conditional generation directly synthesizes the required samples, which is often more efficient than an exhaustive search over a sufficiently rich dataset.

# Rebuttal to Reviewer 2

## 1. Contribution Clarity

**Reviewer Comment:**
> The contribution is not clearly separated from prior work [...] The actual difference is in Stages 2–3: a stronger generation backbone and the addition of LVLM filtering. The delta over HiBug is primarily "use stronger off-the-shelf tools", which is an incremental upgrade enabled by today's foundation models.

**Response:**
Please refer to our response to Reviewer 1 (**Section 1: Contribution and Novelty**) for our primary discussion on how our pipeline distinguishes itself from prior works like HiBug. 

To further clarify why the delta over HiBug is more than just an incremental swap of stronger off-the-shelf components: while the individual modules are off-the-shelf, their architectural integration solves a fundamental limitation in generation-based debugging. HiBug's reliance on unconditioned text-to-image models causes semantic drift—altering background contexts and non-targeted attributes. By anchoring the generation on the original training images via ControlNet, SafeFix guarantees that the generated images perfectly align with the target dataset distribution. Furthermore, because generative models frequently fail to render sensitive demographic attributes correctly, adding an LVLM as a closed-loop verification gate is structurally necessary to prevent bias and ensure precision. We will clarify this architectural motivation in the introduction.

## 2. Connection to Long-Tail Learning (LTL) and Distributional Safety

**Reviewer Comment:**
> The paper does not engage with the long-tail learning field. [...] the paper does not adequately analyze whether injecting synthetic data for rare slices shifts the effective training distribution and degrades performance on common classes.

**Response:**
We thank the reviewer for pointing out this connection. It is important to distinguish between *class-level* long-tail learning and *attribute-level* subgroup failures. Traditional LTL methods assume labels are known and reweight or resample based on class frequency. However, they cannot fix latent, attribute-level bugs (for example, "red hair + brown skin" failing in a "lipstick" classifier) because these methods only see the main class label ("lipstick").

To empirically demonstrate this, we evaluated four standard LTL baselines on our datasets (operating strictly on the original data). As shown below, class-level LTL methods provide minor overall improvements but fail to resolve the attribute-level bugs discovered by our pipeline:

| Method | CelebA Test ACC | CelebA Fixed Bugs (out of 13) | ImageNet Test ACC | ImageNet Fixed Bugs (out of 17) |
| :--- | :--- | :--- | :--- | :--- |
| **Base** | 0.9057 | - | 0.7173 | - |
| Focal Loss | 0.9134 | 8 | 0.6880 | 11 |
| Class-Balanced Loss | 0.9122 | 8 | 0.7270 | 13 |
| Oversampling | 0.9101 | 7 | 0.7290 | 14 |
| LDAM | 0.9141 | 8 | 0.7120 | 12 |
| **SafeFix (Ours)** | **0.9192** | **10** | **0.7488** | **15** |

Crucially, SafeFix explicitly ensures distributional safety. Our results show that overall accuracy improves alongside the targeted subgroups, demonstrating that augmenting underrepresented slices with our structure-conditioned generation does not degrade majority-class performance. We will add this LTL comparison and discussion to the revised manuscript.

## 3. Scalability and Dataset Complexity

**Reviewer Comment:**
> both datasets are not large enough to support the paper's generality and scalability: CelebA is reduced to a single binary task with a small attribute vocabulary, and ImageNet10 contains only 10 hand-picked classes.

**Response:**
We understand the reviewer's concern regarding the complexity of the evaluation. To demonstrate scalability to much richer attribute spaces and denser prediction tasks, we have also evaluated SafeFix on complex object detection and pose estimation settings.

For example, on the COCO Pose Estimation task (identifying people in "brown" clothing under "rainy" conditions) and the KITTI Object Detection task (detecting cars under "red-cloudy" conditions), SafeFix continues to outperform data augmentation and previous generative baselines. On KITTI YOLOv8, SafeFix improves the Test mAP50-95 to 0.5790 (compared to the baseline 0.5626), and on COCO Pose it improves the Test AP to 0.6280 (compared to the baseline 0.5902). We will bring these results from the appendix into the main text to strengthen the claims of scalability.

## 4. Broader Impact Concerns

**Reviewer Comment:**
> if the LVLM rejects brown-skin images at a higher rate than white-skin images (plausible given known biases in vision-language models), the "fairness repair" pipeline may itself be biased.

**Response:**
We acknowledge that relying on an LVLM can introduce the risk of inheriting its inherent demographic biases. As noted in Appendix D.2, "Empirically, the pass rate of Qwen2.5-VL for emotion filtering is approximately 80%." In our experiments, we deliberately adopted a conservative filtering strategy to mitigate unwarranted rejections. For instance, our prompt for emotion filtering explicitly instructs the LVLM: *"If you see any hint of [target] emotion in this picture, answer 1; only answer 0 if you are completely sure there is no such emotion."* Empirically, we observed that using this conservative filtering approach, as opposed to a highly strict one, does not significantly impact the final accuracy of the repaired model. Therefore, we believe that any residual bias from the LVLM does not excessively impact the SafeFix pipeline.

Furthermore, even if the rejection rate is high for certain subgroups, it does not negatively affect the pipeline's overall functionality. Because the generation process is scalable, we can simply generate a larger initial pool of synthetic images to obtain the required number of filtered samples (for example, generating 2,000 images with a 50% rejection rate still yields the needed 1,000 valid images). This only requires additional computation time rather than degrading the quality of the repair. Nevertheless, we fully agree that the underlying bias in LVLMs remains a valid concern, and addressing it through debiased LVLMs or adaptive thresholding will be an important direction for future work. We will add this discussion to the Broader Impact section.
