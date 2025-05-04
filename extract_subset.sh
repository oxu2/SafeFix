#!/usr/bin/env bash
set -euo pipefail

DST_DIR="/people/cs/o/oxx220000/HiBug/imagenet_subset"
mkdir -p "$DST_DIR"

SRC_DIR="/data/oxx220000/imagenet_train"

synsets=(
  n03891251  # park_bench
  n04099969  # rocking_chair
  n02791124  # barber_chair
  n03063599  # coffee_mug
  n02871525  # bookcase
  n03085013  # electric_guitar
  n03179701  # desk
  n02769748  # backpack
  n04026417  # purse
  n04557648  # water_bottle
)

for syn in "${synsets[@]}"; do
  tarfile="$SRC_DIR/${syn}.tar"
  if [[ -f "$tarfile" ]]; then
    echo "==> Extracting $syn"
    mkdir -p "$DST_DIR/$syn"
    tar -xf "$tarfile" -C "$DST_DIR/$syn"
  else
    echo "Warning: $tarfile not found, skipping" >&2
  fi
done

echo "✅ All classes extracted in: $DST_DIR"