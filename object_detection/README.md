# KITTI Object Detection with YOLOv8 (First 5000 Images)

This workflow uses the first 5000 images (sorted by filename) from KITTI training split.

## 1) Download KITTI object dataset

```bash
./download_kitti_object.sh /data/oxx220000/kitti_object true
```

Expected extracted folders:

```text
/data/oxx220000/kitti_object/
└── training/
    ├── image_2/
    └── label_2/
```

## 2) Prepare YOLO-format subset (first 5000)

```bash
python3 object_detection/prepare_kitti_yolo.py \
  --kitti-root /data/oxx220000/kitti_object \
  --output-root /data/oxx220000/kitti_yolov8_first5000 \
  --subset-size 5000 \
  --val-size 500
```

This writes:
- `images/train`, `labels/train` from first 5000 images
- optional validation split from next images
- dataset yaml at `/data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml`

## 3) Train YOLOv8

Pretrained baseline:

```bash
python3 object_detection/train_yolov8_kitti.py \
  --data-yaml /data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml \
  --pretrained true \
  --weights yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

From scratch:

```bash
python3 object_detection/train_yolov8_kitti.py \
  --data-yaml /data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml \
  --pretrained false \
  --model-cfg yolov8n.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

## Dependencies

- `ultralytics`
- `Pillow`

