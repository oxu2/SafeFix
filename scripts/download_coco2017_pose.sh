#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/data/oxx220000/coco}"
WITH_TEST="${2:-false}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "[COCO] Target directory: ${TARGET_DIR}"
echo "[COCO] Downloading train2017/val2017 + annotations..."
wget -c --no-check-certificate \
  https://images.cocodataset.org/zips/train2017.zip \
  https://images.cocodataset.org/zips/val2017.zip \
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip

if [[ "${WITH_TEST}" == "true" ]]; then
  echo "[COCO] Downloading test2017 + image_info_test2017..."
  wget -c --no-check-certificate \
    https://images.cocodataset.org/zips/test2017.zip \
    https://images.cocodataset.org/annotations/image_info_test2017.zip
fi

echo "[COCO] Unzipping..."
[[ -d train2017 ]] || unzip -q -n train2017.zip
[[ -d val2017 ]] || unzip -q -n val2017.zip
[[ -d annotations ]] || unzip -q -n annotations_trainval2017.zip

if [[ "${WITH_TEST}" == "true" ]]; then
  [[ -d test2017 ]] || unzip -q -n test2017.zip
  [[ -f annotations/image_info_test2017.json ]] || unzip -q -n image_info_test2017.zip
fi

echo "[COCO] Done."
echo "Train images: ${TARGET_DIR}/train2017"
echo "Val images:   ${TARGET_DIR}/val2017"
echo "Annotations:  ${TARGET_DIR}/annotations/person_keypoints_train2017.json"

