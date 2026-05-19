# COCO Pose Estimation (Keypoint R-CNN)

This module provides a standard COCO person keypoint pipeline based on `torchvision`:
- training (`train`)
- evaluation with COCO metrics (`eval`)
- single-image inference visualization (`infer`)

## 1) Data preparation

Default expected COCO root:

```text
/data/oxx220000/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

Use:

```bash
./download_coco2017_pose.sh /data/oxx220000/coco
```

Or include test split:

```bash
./download_coco2017_pose.sh /data/oxx220000/coco true
```

## 2) Train

```bash
python pose_estimation/train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --epochs 12 \
  --batch-size 2 \
  --num-workers 4 \
  --output-dir logs/pose/pose_keypointrcnn

# From scratch baseline:
python pose_estimation/train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --pretrained false \
  --output-dir logs/pose/pose_keypointrcnn_scratch

# Add 1000 randomly sampled generated rainy images while keeping all capped originals:
python pose_estimation/train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --attribute rainy \
  --num-added-img 1000 \
  --sub-ratio 1.0 \
  --output-dir logs/pose/pose_keypointrcnn_rain
```

When `--attribute rainy` is enabled, the script loads generated images from:

```text
/data/oxx220000/coco-pose-weather-edits/generated_coco_train1500_rain/outputs
```

Legacy `--attribute rain` is still accepted as an alias and is normalized to the
canonical COCO weather value `weather=rainy`.

The generated PNG stems are matched against COCO `image_id`s and reuse the original
`person_keypoints_train2017.json` annotations during training.

## 3) Evaluate

```bash
python pose_estimation/train_pose_estimation.py eval \
  --data-root /data/oxx220000/coco \
  --checkpoint logs/pose/pose_keypointrcnn/best.pth \
  --output-json logs/pose/pose_eval_results.json
```

## 4) Inference visualization

```bash
python pose_estimation/train_pose_estimation.py infer \
  --image /path/to/image.jpg \
  --checkpoint logs/pose/pose_keypointrcnn/best.pth \
  --output logs/pose/pose_infer.jpg
```

## Dependencies

- `torch`
- `torchvision`
- `pycocotools`
- `Pillow`
