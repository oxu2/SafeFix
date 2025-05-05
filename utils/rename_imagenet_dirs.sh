#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="./imagenet_subset"

# Declare an associative array mapping WNID → class name
declare -A MAPPING=(
  [n03891251]="park_bench"
  [n04099969]="rocking_chair"
  [n02791124]="barber_chair"
  [n03063599]="coffee_mug"
  [n02871525]="bookcase"
  [n03085013]="electric_guitar"
  [n03179701]="desk"
  [n02769748]="backpack"
  [n04026417]="purse"
  [n04557648]="water_bottle"
)

for wnid in "${!MAPPING[@]}"; do
  src="$BASE_DIR/$wnid"
  dst="$BASE_DIR/${MAPPING[$wnid]}"
  if [ -d "$src" ]; then
    echo "Renaming $src → $dst"
    mv "$src" "$dst"
  else
    echo "Warning: directory $src not found, skipping" >&2
  fi
done

echo "Done renaming folders."