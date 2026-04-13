#!/usr/bin/env python3
from pathlib import Path
import shutil

root = Path("/people/cs/o/oxx220000/safefix/data/pinkcolor_fabrictexture_instruct_pix2pix")
classes = [
    "backpack", "barber_chair", "coffee_mug", "desk", "electric_guitar",
    "park_bench", "pitcher", "purse", "rocking_chair", "water_bottle",
]
classes_sorted = sorted(classes, key=len, reverse=True)

for cls in classes:
    (root / cls).mkdir(exist_ok=True)

for jpg in root.glob("*.jpg"):
    target_dir = None
    for cls in classes_sorted:
        if jpg.name.startswith(f"{cls}_"):
            target_dir = cls
            break
    if target_dir is None:
        print(f"Skipped {jpg.name}: no matching class prefix")
        continue
    dest = root / target_dir / jpg.name
    shutil.move(str(jpg), dest)
    print(f"Moved {jpg.name} -> {dest.relative_to(root)}")