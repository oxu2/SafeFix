


## AttributeFocusedStrategy (Attribute-Driven Sampling)
<!-- ```bash
python alpred.py --al_strategy attribute --attribute_file save/attribute10.json \
  --al_attribute_focus color,background --al_attribute_rarity_power 1.5 --al_attribute_uncertainty_boost 1.0
```
- **Principle**: Builds a score for every unlabeled sample by combining attribute rarity, metadata-level uncertainty, and optional diagnostic weights derived from logs.
- **Use Cases**: Scenarios where human annotations describe attribute slices (color, background, hairstyle, etc.) and the model underperforms on specific combinations.
- **Advantages**: Lightweight (no feature extraction), controllable focus on high-value slices, deterministic once metadata is fixed. -->

Let $i$ index a candidate sample and $A$ denote the set of attributes contained in the metadata (e.g., $A = \{\text{color}, \text{background}\}$). For attribute $a \in A$, let $v_{i,a}$ be the attribute value of sample $i$, and $f_{a}(v)$ be the frequency of value $v$ among the current pool. The strategy computes three additive components:

1. **Rarity term** (controlled by `--al_attribute_rarity_power`):
  $$R_{i,a} = \left(\frac{1}{f_{a}(v_{i,a}) + \varepsilon}\right)^{\alpha}$$
  where $\alpha \geq 0$ is the rarity power and $\varepsilon$ avoids division by zero. Higher $\alpha$ exaggerates the contrast between common and rare attributes.

2. **Uncertainty bonus** :
  $$U_{i,a} = \begin{cases}
  \beta, & \text{if sample $i$ is marked uncertain for attribute $a$}\\
  0, & \text{otherwise}
  \end{cases}$$
  with $\beta \geq 0$ set by `--al_attribute_uncertainty_boost`.

3. **Log-derived weight**: diagnostic logs record rare cases along with their classification accuracy. The parser converts each entry into a failure weight $w_{a}(v) = 1 - \text{ACC}$. For attribute $a$ and value $v$, the contribution becomes
  $$L_{i,a} = \gamma \cdot w_{a}(v_{i,a})$$
  where $\gamma$ is the scaling factor. Lower accuracies therefore produce larger boosts, while missing entries default to zero.

The total score for sample $i$ is the sum across all attributes:
$$S_i = \sum_{a \in A} \big(R_{i,a} + U_{i,a} + L_{i,a}\big).$$

Debug attribute breakdown for 'color':
  value=maroon, freq=121, indices=[32,61,205,280,408,521,573,620,691,777,...], R=0.0083, U=1.0000, L=0.0000
  value=purple, freq=154, indices=[8,24,26,103,159,178,251,292,313,327,...], R=0.0065, U=1.0000, L=3.1579
  value=tan, freq=186, indices=[71,127,131,195,220,221,234,266,272,282,...], R=0.0054, U=1.0000, L=3.0000
  value=yellow, freq=327, indices=[18,27,47,66,123,176,179,209,322,324,...], R=0.0031, U=1.0000, L=3.1111
  value=gray, freq=330, indices=[33,39,50,51,52,97,129,167,194,199,...], R=0.0030, U=1.0000, L=0.0000
  value=pink, freq=407, indices=[12,44,113,162,192,204,241,244,258,263,...], R=0.0025, U=1.0000, L=3.0097
  value=silver, freq=529, indices=[41,76,172,193,257,273,283,414,438,603,...], R=0.0019, U=1.0000, L=0.0000
  value=orange, freq=539, indices=[2,10,20,57,62,88,91,95,144,146,...], R=0.0019, U=1.0000, L=3.0909
  value=green, freq=554, indices=[17,36,45,46,53,65,75,81,87,90,...], R=0.0018, U=1.0000, L=3.2558
  value=brown, freq=607, indices=[101,120,150,151,156,175,264,287,370,377,...], R=0.0016, U=1.0000, L=0.0000
  value=red, freq=615, indices=[79,121,124,139,142,177,188,191,245,256,...], R=0.0016, U=1.0000, L=0.0000
  value=blue, freq=870, indices=[14,22,25,28,42,58,70,84,108,112,...], R=0.0011, U=1.0000, L=0.0000
  value=white, freq=1287, indices=[37,106,109,128,138,166,169,183,228,286,...], R=0.0008, U=1.0000, L=0.0000
  value=black, freq=1398, indices=[0,15,23,30,35,48,49,80,96,100,...], R=0.0007, U=1.0000, L=0.0000
  value=multicolored, freq=2476, indices=[3,6,7,19,29,31,34,40,43,56,...], R=0.0004, U=1.0000, L=0.0000



ACC: 0.674419 	Rare Case: attribute "color", description "green"
ACC: 0.699029 	Rare Case: attribute "color", description "pink"
ACC: 0.700000 	Rare Case: attribute "color", description "tan"
ACC: 0.690909 	Rare Case: attribute "color", description "orange"
ACC: 0.688889 	Rare Case: attribute "color", description "yellow"
ACC: 0.684211 	Rare Case: attribute "color", description "purple"
ACC: 0.665289 	Hard case: attribute "color", description "black"

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