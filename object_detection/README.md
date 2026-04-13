# KITTI Object Detection with YOLOv8 (5000 Train / 1240 Val / 1241 Test)

This workflow uses the KITTI training split in deterministic filename order and keeps only `Car` and `Pedestrian` labels:
- first 5000 images for training
- next 1240 images for validation
- remaining 1241 images for test

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

## 2) Prepare YOLO-format split (5000 / 1240 / 1241)

```bash
python3 object_detection/prepare_kitti_yolo.py \
  --kitti-root /data/oxx220000/kitti_object \
  --output-root /data/oxx220000/kitti_yolov8_first5000 \
  --subset-size 5000 \
  --val-size 1240
```

This writes:
- `images/train`, `labels/train` from first 5000 images
- `images/val`, `labels/val` from next 1240 images
- `images/test`, `labels/test` from remaining 1241 images
- split ID files: `train_ids.txt`, `val_ids.txt`, `test_ids.txt`
- class names limited to `Car` and `Pedestrian` by default
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

After training, the script will also run `model.val(split="test")` automatically when the dataset yaml includes a `test:` entry.

## Dependencies

- `ultralytics`
- `Pillow`
