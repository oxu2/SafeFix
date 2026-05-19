# Dataset Download Reproduction (COCO + KITTI)

This file records the exact dataset download/prep workflow used in this repo, so others can reproduce the same paths and files.

## 0) Target paths used in this project

- COCO root: `/data/oxx220000/coco`
- KITTI object root: `/data/oxx220000/kitti_object`
- KITTI YOLO (first 5000) root: `/data/oxx220000/kitti_yolov8_first5000`

## 1) COCO 2017 download (for pose estimation)

### Option A (recommended): run repo script

```bash
cd /people/cs/o/oxx220000/safefix
bash download_coco2017_pose.sh /data/oxx220000/coco true
```

- `true` means also download `test2017` + `image_info_test2017`.
- Script file: `download_coco2017_pose.sh`

### Option B: equivalent manual commands

```bash
mkdir -p /data/oxx220000/coco
cd /data/oxx220000/coco

wget -c --no-check-certificate \
  https://images.cocodataset.org/zips/train2017.zip \
  https://images.cocodataset.org/zips/val2017.zip \
  https://images.cocodataset.org/zips/test2017.zip \
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  https://images.cocodataset.org/annotations/image_info_test2017.zip

unzip -q -n train2017.zip
unzip -q -n val2017.zip
unzip -q -n test2017.zip
unzip -q -n annotations_trainval2017.zip
unzip -q -n image_info_test2017.zip
```

### COCO quick check

```bash
ls /data/oxx220000/coco/annotations/person_keypoints_train2017.json
ls /data/oxx220000/coco/annotations/person_keypoints_val2017.json
```

Expected directories:

- `/data/oxx220000/coco/train2017`
- `/data/oxx220000/coco/val2017`
- `/data/oxx220000/coco/test2017` (if downloaded)
- `/data/oxx220000/coco/annotations`

Reference (official page source user provided):

- https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm

## 2) KITTI object detection download

### Option A (recommended): run repo script

```bash
cd /people/cs/o/oxx220000/safefix
bash download_kitti_object.sh /data/oxx220000/kitti_object true
```

- Script file: `download_kitti_object.sh`

### Option B: equivalent manual commands

```bash
mkdir -p /data/oxx220000/kitti_object
cd /data/oxx220000/kitti_object

wget -c \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

unzip -q -n data_object_image_2.zip
unzip -q -n data_object_label_2.zip
unzip -q -n data_object_calib.zip
```

### KITTI quick check

```bash
ls /data/oxx220000/kitti_object/training/image_2 | head
ls /data/oxx220000/kitti_object/training/label_2 | head
```

Expected directories:

- `/data/oxx220000/kitti_object/training/image_2`
- `/data/oxx220000/kitti_object/training/label_2`
- `/data/oxx220000/kitti_object/training/calib`

## 3) Prepare KITTI first-5000 in YOLO format

For the setting "YOLOv8 is trained on the first 5000 images of KITTI training set":

```bash
cd /people/cs/o/oxx220000/safefix
python object_detection/prepare_kitti_yolo.py \
  --kitti-root /data/oxx220000/kitti_object \
  --output-root /data/oxx220000/kitti_yolov8_first5000 \
  --subset-size 5000 \
  --val-size 500
```

Output key files:

- `/data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml`
- `/data/oxx220000/kitti_yolov8_first5000/train_ids.txt`
- `/data/oxx220000/kitti_yolov8_first5000/val_ids.txt` (if `--val-size > 0`)

## 4) Common troubleshooting

- Slow / interrupted download: keep `-c` (resume).
- Re-run safe unzip: keep `unzip -n` to avoid overriding existing files.
- Permission issue under `/data`: confirm account has write permission.
- If `wget` cert error for COCO, keep `--no-check-certificate` as in the script.
