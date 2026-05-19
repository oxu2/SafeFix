import os
import numpy as np
import shutil
import argparse
import re

def filter_images_by_train_idx(attribute_name, train_idx_path="data/celebA/train_idx.npy", train_idx_limit=80000):
    """
    Moves images from safefix_<attribute_name> to safefix_<attribute_name>_deleted
    if their base number does not correspond to an index in the first `train_idx_limit`
    entries of the specified train_idx file.

    Args:
        attribute_name (str): The attribute name used for directory naming (e.g., "redhair").
        train_idx_path (str): Path to the train_idx.npy file.
        train_idx_limit (int): The number of initial indices to consider from train_idx.npy.
    """
    source_dir = f"safefix_{attribute_name}"
    dest_dir = f"safefix_{attribute_name}_deleted"

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Destination directory for excluded images: '{dest_dir}'")

    # Load the relevant part of train_idx
    try:
        full_train_idx = np.load(train_idx_path)
        # Use only the first train_idx_limit indices, as used in predict.py
        train_indices_set = set(full_train_idx[:train_idx_limit])
        print(f"Loaded {len(train_indices_set)} indices from '{train_idx_path}' (up to limit {train_idx_limit}).")
    except FileNotFoundError:
        print(f"Error: Train index file '{train_idx_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading train index file: {e}")
        return

    moved_count = 0
    kept_count = 0
    skipped_count = 0

    print(f"Scanning files in '{source_dir}'...")
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.png'):
            # Extract base number (e.g., '000123' from 'attribute_000123.png')
            match = re.search(r'_(\d+)\.png$', filename, re.IGNORECASE)
            if match:
                base_number_str = match.group(1)
                try:
                    # Convert base number (1-based) to zero-based index
                    image_index = int(base_number_str) - 1

                    # Check if this index is in the allowed training indices
                    if image_index not in train_indices_set:
                        # Move the file
                        source_path = os.path.join(source_dir, filename)
                        dest_path = os.path.join(dest_dir, filename)
                        try:
                            shutil.move(source_path, dest_path)
                            # print(f"Moved '{filename}' (index {image_index}) to '{dest_dir}'") # Optional: verbose output
                            moved_count += 1
                        except Exception as e:
                            print(f"Error moving file '{filename}': {e}")
                            skipped_count += 1
                    else:
                        # Keep the file
                        kept_count += 1
                except ValueError:
                    print(f"Warning: Could not parse base number from '{filename}'. Skipping.")
                    skipped_count += 1
            else:
                print(f"Warning: Could not extract base number from filename format '{filename}'. Skipping.")
                skipped_count += 1
        else:
            # Optional: handle non-png files if necessary
            pass

    print("\nFiltering complete.")
    print(f" - Files kept in '{source_dir}': {kept_count}")
    print(f" - Files moved to '{dest_dir}': {moved_count}")
    print(f" - Files skipped (errors/format issues): {skipped_count}")

    # Verify remaining count
    remaining_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]
    print(f"Verification: Found {len(remaining_files)} PNG files remaining in '{source_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter safefix images based on CelebA train_idx.")
    parser.add_argument('--attributes', type=str, required=True,
                        help='Attribute name used in directory naming (e.g., "redhair").')
    parser.add_argument('--train_idx_path', type=str, default="data/celebA/train_idx.npy",
                        help='Path to the train_idx.npy file.')
    parser.add_argument('--train_idx_limit', type=int, default=80000,
                        help='Limit for indices loaded from train_idx.npy.')

    args = parser.parse_args()

    filter_images_by_train_idx(args.attributes, args.train_idx_path, args.train_idx_limit)
