import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import random
import math

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))

def plot_comparison(dataset_dir, num_images):
    # List all image files in the dataset directory
    image_files = [
        os.path.join(dataset_dir, fname)
        for fname in os.listdir(dataset_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    # Cap num_images to the number of available images
    if num_images > len(image_files):
        num_images = len(image_files)
    # Limit to max 3 images for 1x3 layout
    if num_images > 3:
        num_images = 3
    # Force 1x3 layout
    cols = 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes = axes.flatten()
    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)
    # sort the selected images by file name
    selected_images.sort()
    for idx, img_path in enumerate(selected_images):
        img = Image.open(img_path)
        img = center_crop_square(img)
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    plt.tight_layout()
    # Save the grid as a PDF named after the folder
    base_name = os.path.basename(dataset_dir.rstrip(os.sep))
    output_file = f"grid_{base_name}.pdf"
    plt.savefig(output_file, dpi=300, format='pdf')
    print(f"Saved 1x3 grid PDF to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='our',
        help='Path to the directory containing images'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=3,
        help='Number of images to select and plot (max 3)'
    )
    args = parser.parse_args()
    plot_comparison(args.dataset, args.num_images)
