#!/usr/bin/env bash
set -euo pipefail

cd /people/cs/o/oxx220000/safefix

qwen_python="/people/cs/o/oxx220000/.conda/envs/qwen/bin/python"
nvjitlink_dir="/people/cs/o/oxx220000/.conda/envs/qwen/lib/python3.10/site-packages/nvidia/nvjitlink/lib"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

clean_ld_library_path=""
IFS=':' read -r -a ld_library_path_parts <<< "${LD_LIBRARY_PATH:-}"
for part in "${ld_library_path_parts[@]}"; do
  case "${part}" in
    ""|/usr/local/cuda-12.0/lib64|/usr/local/cuda-12.0/extras/CUPTI/lib64)
      continue
      ;;
  esac
  clean_ld_library_path="${clean_ld_library_path:+${clean_ld_library_path}:}${part}"
done
export LD_LIBRARY_PATH="${nvjitlink_dir}${clean_ld_library_path:+:${clean_ld_library_path}}"

"${qwen_python}" -u utils/assign_attribute.py \
  --dataset kitti \
  --kitti_root /data/oxx220000/kitti_yolov8_first5000 \
  --split val \
  --corpus save/corpus_kitti_gemini_clean.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output save/attribute_kitti_qwen3_8B_val.json \
  --csv-output save/attribute_kitti_qwen3_8B_val.csv \
  --partial save/attribute_kitti_qwen3_8B_val_partial.json \
  --save_every 250 \
  --max_memory_per_gpu 23

"${qwen_python}" -u utils/assign_attribute.py \
  --dataset kitti \
  --kitti_root /data/oxx220000/kitti_yolov8_first5000 \
  --split test \
  --corpus save/corpus_kitti_gemini_clean.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output save/attribute_kitti_qwen3_8B_test.json \
  --csv-output save/attribute_kitti_qwen3_8B_test.csv \
  --partial save/attribute_kitti_qwen3_8B_test_partial.json \
  --save_every 250 \
  --max_memory_per_gpu 23
