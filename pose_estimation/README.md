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
python train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --epochs 12 \
  --batch-size 2 \
  --num-workers 4 \
  --output-dir outputs/pose_keypointrcnn

# From scratch baseline:
python train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --pretrained false \
  --output-dir outputs/pose_keypointrcnn_scratch
```

## 3) Evaluate

```bash
python train_pose_estimation.py eval \
  --data-root /data/oxx220000/coco \
  --checkpoint outputs/pose_keypointrcnn/best.pth \
  --output-json outputs/pose_eval_results.json
```

## 4) Inference visualization

```bash
python train_pose_estimation.py infer \
  --image /path/to/image.jpg \
  --checkpoint outputs/pose_keypointrcnn/best.pth \
  --output outputs/pose_infer.jpg
```

## Dependencies

- `torch`
- `torchvision`
- `pycocotools`
- `Pillow`
