import os
import shutil

root_dir = 'exampleData/imagenet_simple'

class_mapping = {
    'backpack': 0,
    'barber_chair': 1,
    'coffee_mug': 2,
    'desk': 3,
    'electric_guitar': 4,
    'park_bench': 5,
    'pitcher': 6,
    'purse': 7,
    'rocking_chair': 8,
    'water_bottle': 9,
}

# sort keys by length so we try 'electric_guitar' before 'electric', etc.
class_keys = sorted(class_mapping.keys(), key=len, reverse=True)

for fname in os.listdir(root_dir):
    if not fname.lower().endswith('.png'):
        continue

    match = None
    for key in class_keys:
        if fname.startswith(key):
            match = key
            break

    if match is None:
        print(f"Warning: no matching class for {fname}, skipping")
        continue

    target_dir = os.path.join(root_dir, match)
    os.makedirs(target_dir, exist_ok=True)

    src = os.path.join(root_dir, fname)
    dst = os.path.join(target_dir, fname)
    shutil.move(src, dst)

print("Done.")