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

Use the local `fp` conda environment for KITTI YOLOv8 runs. The exported
environment snapshot is `utils/fp-kitti-env.yml`.

For the current local GPU workstation, prefer the launch helper when running on
physical GPU 3:

```bash
bash scripts/launch_kitti_yolov8_gpu3_e30.sh
```

The helper uses `conda activate fp`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and
Ultralytics `--device 3`. Do not change the Ultralytics device argument back to
`0` for this host; Ultralytics rewrites `CUDA_VISIBLE_DEVICES` from that value,
which sends the job to physical GPU 0.

For pretrained YOLOv8n fine-tuning at `imgsz=640` on a 24 GB RTX A5000,
`batch=64` is the current local default. It is less aggressive than targeting
maximum memory occupancy while still avoiding the very low throughput of
`batch=4`. Ultralytics documents `batch=16` as a normal fixed-size example and
`batch=-1` as AutoBatch for roughly 60% CUDA memory use when that is desired.

Pretrained baseline:

```bash
conda activate fp
python3 object_detection/train_yolov8_kitti.py \
  --data-yaml /data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml \
  --pretrained true \
  --weights save/weights/yolov8n.pt \
  --epochs 100 \
  --batch 64 \
  --imgsz 640 \
  --device 3
```

From scratch:

```bash
conda activate fp
python3 object_detection/train_yolov8_kitti.py \
  --data-yaml /data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml \
  --pretrained false \
  --model-cfg yolov8n.yaml \
  --epochs 100 \
  --batch 64 \
  --imgsz 640 \
  --device 3
```

After training, the script will also run `model.val(split="test")` automatically when the dataset yaml includes a `test:` entry.

Ultralytics writes resumable artifacts under
`save/runs/kitti_yolov8/<run-name>/save/weights/`. Resume from
`save/weights/last.pt`; use `save/weights/best.pt` for evaluation or model selection.
The previous completed local run
`save/runs/kitti_yolov8/first5000_gpu1_e30_20260409_194735/` has both files. A new
run only has `args.yaml` until it reaches the first checkpoint save point.
For the current YOLOv8n KITTI run, `best.pt` and `last.pt` are about `18.4 MB`
each, below GitHub's 100 MB hard file limit. Larger model families can exceed
that limit and should be checked before tracking.

## Dependencies

- `ultralytics`
- `Pillow`
