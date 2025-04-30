"""
sd15.py
Stable Diffusion v1.5 inference script using Hugging Face diffusers.
All code comments are in English.
"""

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
# disable pipeline progress bars and lower verbosity
logging.disable_progress_bar()
logging.set_verbosity_error()
import torch
import itertools
import random
import datetime
import argparse
import os
import json
import numpy as np
parser = argparse.ArgumentParser(description="Stable Diffusion v1.5 image generation script.")
parser.add_argument('--num_added_img', type=int, default=10, help='Number of images to generate.')
parser.add_argument('--attributes', type=str, default="redhair_brownskin", 
                    help='attributes to detect, e.g. redhair_brownskin, redhair, etc.')
parser.add_argument('--attribute_file', type=str, default="exampleData/celebA/attribute.json",
                    help='Path to the attribute JSON file (default: exampleData/celebA/attribute.json)')
parser.add_argument('--gpu', type=int, default=0,
                    help='Index of the CUDA GPU to use (default: 0)')
parser.add_argument(
    '--num_inference_steps', type=int, default=30,
    help='Number of diffusion steps for generation (lower = faster; default: 30).'
)
args = parser.parse_args()

# Parse attribute overrides from args.attributes
attribute_parts = args.attributes.split('_')
hair_override = None
skin_override = None
emotion_override = None
for part in attribute_parts:
    if part.endswith("hair"):
        hair_override = part[:-4]
    elif part.endswith("skin"):
        skin_override = part[:-4]
    elif part.endswith("emotion"):
        emotion_override = part[:-7]

with open(args.attribute_file, 'r') as f:
    attr_json = json.load(f)

indices_file = f"exampleData/celebA/non_{args.attributes}_indices.txt"
with open(indices_file, 'r') as f:
    selected_indices = [int(line.strip()) for line in f if line.strip()][:args.num_added_img]

# Load mapping from original to filtered indices
with open("exampleData/celebA/idx_mapping.json", 'r') as f:
    original_to_filtered = json.load(f)["original_to_filtered"]

# Load lipstick labels (1 = wearing lipstick, 0 = no lipstick)
labels = np.load("exampleData/celebA/labels.npy")

prompts = []
for original_idx in selected_indices:
    # Map back to filtered index for attribute lookup
    filtered_idx = original_to_filtered[str(original_idx)]

    lipstick_flag = labels[original_idx]
    lipstick_clause = 'wearing lipstick' if lipstick_flag == 1 else 'no lipstick'

    # Determine hair color, skin, and emotion with overrides
    if hair_override:
        hair_color = hair_override
    else:
        hair_color = attr_json['hair color']['data'][filtered_idx]
    if skin_override:
        skin = skin_override
    else:
        skin = attr_json['skin']['data'][filtered_idx]
    if emotion_override:
        emotion = emotion_override
    else:
        emotion = attr_json['emotion']['data'][filtered_idx]

    # Build the structured prompt
    gender = attr_json['gender']['data'][filtered_idx]
    beard = attr_json['beard']['data'][filtered_idx]
    beard_clause = 'with beard' if beard == 'Yes' else 'without beard'
    makeup = attr_json['makeup']['data'][filtered_idx]
    makeup_clause = 'wearing makeup' if makeup == 'Yes' else 'no makeup'
    age = attr_json['young']['data'][filtered_idx]
    prompt = (
        f"A photo of a {gender} {beard_clause} {makeup_clause} {lipstick_clause} "
        f"with {hair_color} hair and {skin} skin, looking {emotion}, appearing {age}."
    )
    prompts.append(prompt)

def generate_images(prompt: str, num_images: int = 1, output_prefix: str = "output"):
    """
    Generate images using Stable Diffusion v1.5.

    Args:
        prompt (str): The text prompt for image generation.
        num_images (int): Number of images to generate.
        output_prefix (str): Prefix for output image filenames.
    """
    # Load the Stable Diffusion v1.5 pipeline with fp16 precision (revision removed)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    # Authentication is managed via `huggingface-cli login` or HF_TOKEN env var; no need for explicit use_auth_token
    pipeline = pipeline.to(f"cuda:{args.gpu}")
    pipeline.enable_attention_slicing()
    # disable generation-time progress bars
    pipeline.set_progress_bar_config(disable=True)

    # Perform image generation
    result = pipeline(
        prompt,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=num_images
    )

    # Save the single generated image
    img = result.images[0]
    filename = f"{output_prefix}.png"
    img.save(filename)
    print(f"Saved image: {filename}")

if __name__ == "__main__":
    time_now = datetime.datetime.now().strftime("%m%d%H%M")
    for i, user_prompt in enumerate(prompts):
        original_idx = selected_indices[i]
        output_prefix = f"hibug_{args.attributes}_{time_now}/{args.attributes}_{original_idx:06d}"
        # Ensure output directory exists
        output_dir = os.path.dirname(output_prefix)
        os.makedirs(output_dir, exist_ok=True)
        generate_images(user_prompt, num_images=1, output_prefix=output_prefix)
