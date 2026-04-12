#!/usr/bin/env python3
import json
import sys
import os

def strip_one_zero(key: str) -> str:
    """Remove one leading '0' from key if it starts with '0'."""
    if key.startswith("0"):
        return key[1:]
    return key

def rename_images_in_dir(dir_path: str):
    """Rename each .jpg file in dir by removing a single leading '0' from filenames."""
    if not os.path.isdir(dir_path):
        print(f"Warning: {dir_path} is not a directory")
        return
    for fname in os.listdir(dir_path):
        if fname.lower().endswith('.jpg') and fname.startswith('0'):
            old_path = os.path.join(dir_path, fname)
            new_name = fname[1:]
            new_path = os.path.join(dir_path, new_name)
            os.replace(old_path, new_path)
            print(f"Renamed {fname} -> {new_name}")

def main(json_path: str):
    # If the path is a directory, rename image files instead of JSON
    if os.path.isdir(json_path):
        rename_images_in_dir(json_path)
        return

    # Read the JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Transform keys
    new_data = {}
    for old_key, value in data.items():
        new_key = strip_one_zero(old_key)
        new_data[new_key] = value

    # Write out the new JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(data)} JSON entries and updated in place.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-json>")
        sys.exit(1)
    main(sys.argv[1])
    # To rename image files, uncomment and set the correct directory:
    # rename_images_in_dir("/people/cs/o/oxx220000/safefix/data/redhair_brownskin")