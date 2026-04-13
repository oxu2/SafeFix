import os
import shutil

src_dir = "data/imagenet_simple3"
dst_dir = "data/imagenet_simple"

# Ensure destination exists
os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if not filename.lower().endswith(".png"):
        continue

    base, ext = os.path.splitext(filename)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # If no clash, just copy
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
        continue

    # Otherwise append a numeric suffix until it is unique
    i = 2
    while True:
        new_name = f"{base}_{i}{ext}"
        dst_path = os.path.join(dst_dir, new_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            break
        i += 1