# Qwen Human Audit Samples

This folder contains random image samples for a human audit of the Qwen VLM
attribute assignments used by the SafeFix COCO and KITTI experiments.

## Contents

- `coco_images/`: 100 sampled COCO train images.
- `kitti_images/`: 100 sampled KITTI train images.
- `coco_qwen_random100_assignments.csv`: COCO image ids and Qwen assignments.
- `kitti_qwen_random100_assignments.csv`: KITTI image ids and Qwen assignments.
- `coco_qwen_random100_image_ids.txt`: COCO image ids only.
- `kitti_qwen_random100_image_ids.txt`: KITTI image ids only.
- `human_response_template.csv`: blinded response template for annotators.
- `qwen_random100_summary.json`: sample generation metadata.

The assignment CSVs include the Qwen labels. Do not show those label columns to
human annotators before they answer.

## Study Goal

The goal is to estimate whether human judgments agree with the Qwen
assignments used for targeted data repair:

- COCO target attributes: `clothes_color=brown`, `weather=rainy`.
- KITTI target attributes: `color=red`, `weather=cloudy`.

Agreement should be computed after annotation, separately for each target
attribute and each dataset.

## Annotation Protocol

For each image, show only:

- the image,
- the dataset name,
- the image id or file name.

Do not show:

- Qwen labels,
- target flags such as `qwen_target_*`,
- any model prediction or expected answer.

Ask annotators to choose from the same fixed label set that Qwen used.

### COCO Questions

For each COCO image, ask:

1. What is the dominant color of the main person's clothes?
   Options: `red`, `blue`, `green`, `yellow`, `black`, `white`, `gray`,
   `brown`, `pink`.
2. What is the weather condition like in the image?
   Options: `sunny`, `cloudy`, `rainy`, `snowy`, `clear`.

Optional extra COCO questions, if you want full assignment agreement:

- Clothes type: `casual`, `formal`, `sportswear`, `swimwear`, `winterwear`,
  `uniform`.
- Background style: `urban`, `rural`, `natural`, `artificial`, `indoors`.
- Time of day: `morning`, `afternoon`, `evening`, `night`.
- Pose: `sitting`, `jumping`, `lying`, `standing`, `walking`, `crouching`,
  `running`, `kneeling`.

### KITTI Questions

For each KITTI image, ask:

1. What is the dominant visible color of the most prominent target object or
   person?
   Options: `white`, `black`, `gray`, `silver`, `red`, `blue`, `green`,
   `yellow`, `brown`.
2. What is the weather condition like in the scene?
   Options: `sunny`, `cloudy`, `rainy`, `foggy`, `clear`, `snowy`.

Optional extra KITTI questions, if you want full assignment agreement:

- Background style: `urban city`, `rural`, `natural`, `highway`,
  `residential`.
- Traffic density: `empty`, `light`, `moderate`, `heavy`.
- Lighting condition: `bright`, `dim`, `strong shadows`, `overcast`, `glare`.
- Road condition: `dry`, `wet`, `snowy`, `muddy`.

## Suggested Response Format

Use one response CSV per annotator. Start from
`human_response_template.csv`, duplicate it for each annotator, and fill
`annotator_id` plus the relevant human-label columns. Keep the Qwen assignment
CSVs separate.

For COCO rows, fill:

```csv
annotator_id,human_clothes_color,human_coco_weather
```

For KITTI rows, fill:

```csv
annotator_id,human_kitti_color,human_kitti_weather
```

Leave the non-relevant columns blank. For example, on a COCO row, leave
`human_kitti_color` and `human_kitti_weather` blank.

If multiple humans label the same image, keep all rows. Majority vote can be
computed later.

## Agreement Analysis

After collecting responses, join human labels with the Qwen assignment CSV by
`dataset` and `image_id`.

Report at least:

- COCO `clothes_color`: exact agreement with Qwen.
- COCO `weather`: exact agreement with Qwen.
- COCO binary target agreement for `clothes_color=brown`.
- COCO binary target agreement for `weather=rainy`.
- KITTI `color`: exact agreement with Qwen.
- KITTI `weather`: exact agreement with Qwen.
- KITTI binary target agreement for `color=red`.
- KITTI binary target agreement for `weather=cloudy`.

For the paper or appendix, the most relevant numbers are the binary target
agreements, because they validate the metadata used by the repair selector.

## Important Caveat

This folder uses a purely random sample of 100 images per dataset. Target
positives are sparse in this draw:

- COCO: 2 images labeled `clothes_color=brown` by Qwen.
- COCO: 5 images labeled `weather=rainy` by Qwen.
- KITTI: 10 images labeled `color=red` by Qwen.
- KITTI: 0 images labeled `weather=cloudy` by Qwen.

This random sample is useful for measuring overall agreement. It is not ideal
for estimating target-positive precision for rare targets, especially KITTI
`weather=cloudy`. For a stronger target audit, create an additional stratified
sample that intentionally includes enough Qwen-positive examples for each
target attribute.

## Reproducibility

The sample was generated with:

```bash
python3 scripts/export_qwen_human_audit_samples.py
```

Default settings:

- random seed: `20260506`
- COCO source assignment:
  `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
- KITTI source assignment: `save/attribute_kitti_qwen3_8B.csv`
- output directory: `bxz/qwen_human_audit/`
