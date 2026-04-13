#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/data/oxx220000/kitti_object}"
UNZIP_AFTER_DOWNLOAD="${2:-true}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "[KITTI] Target directory: ${TARGET_DIR}"
echo "[KITTI] Downloading object detection archives..."
wget -c \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

if [[ "${UNZIP_AFTER_DOWNLOAD}" == "true" ]]; then
  echo "[KITTI] Unzipping..."
  unzip -q -n data_object_image_2.zip
  unzip -q -n data_object_label_2.zip
  unzip -q -n data_object_calib.zip
fi

echo "[KITTI] Done."
echo "Training images: ${TARGET_DIR}/training/image_2"
echo "Training labels: ${TARGET_DIR}/training/label_2"

