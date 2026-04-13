import os
import sys
import contextlib
import io
import time


# Ensure this script runs in the 'control' conda environment
if os.environ.get('CONDA_DEFAULT_ENV') != 'control':
    sys.stderr.write("Info: Switching to the 'control' conda environment and re-running...\n")
    os.execvp(
        'conda',
        ['conda', 'run', '-n', 'control', 'python'] + sys.argv
    )
print("Info: Now running in the 'control' conda environment.")

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_controlnet'))

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

# ---------- Quiet mode tweaks ----------
import logging
# Suppress PyTorch Lightning info such as "Global seed set to ..."
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# Suppress MemoryEfficientCrossAttention setup messages
logging.getLogger("ldm").setLevel(logging.ERROR)
logging.getLogger("cldm").setLevel(logging.ERROR)
# Suppress global seed messages
logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.ERROR)

from pytorch_lightning import seed_everything

# Disable tqdm progress bars if output is non-interactive (such as nohup)
import functools, tqdm
tqdm.tqdm = functools.partial(tqdm.tqdm, disable=not sys.stdout.isatty())

# Ensure the local 'annotator' package in this repo is on the import path

from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
import re
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ControlNet batch inference")
parser.add_argument('--dataset', type=str, choices=['imagenet', 'celeba'], default='celeba',
                    help="Dataset to use for generation")
parser.add_argument('--attributes', type=str, default='redhair_brownskin_sademotion',
                    help="Attributes for generation, e.g., redhair_sademotion or redhair_brownskin")
parser.add_argument(
    '--stop_at',
    type=int,
    default=10234,
    help="Maximum total images to generate (including existing JPEGs in output_dir)"
)
args = parser.parse_args()

# Now import ControlNet share utilities and configuration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ControlNet')))
from utils.share import *
save_memory = False

# Build prompt text from attributesd
attr_tokens = args.attributes.split('_')
token_map = {
    'redhair': 'vibrant red hair',
    'brownskin': 'brown skin',
    'sademotion': 'sad emotion'
}
mapped = [token_map.get(tok, tok) for tok in attr_tokens]
prompt_text = 'a person with ' + ', '.join(mapped) + ', high detail, natural lighting'

apply_hed = HEDdetector()

# Determine model config and weights paths, with fallback
config_path = '/home/jovyan/ouser/ControlNet/models/cldm_v15.yaml'
weights_path = '/home/jovyan/ouser/ControlNet/models/control_sd15_hed.pth'
if not os.path.exists(config_path):
    config_path = 'model_controlnet/cldm_v15.yaml'
    weights_path = 'model_controlnet/control_sd15_hed.pth'

# Suppress verbose model loading output
f = io.StringIO()
with contextlib.redirect_stdout(f):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(weights_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)

        # Suppress verbose output from seed setting and sampling
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            seed_everything(seed, workers=True)
            if save_memory:
                model.low_vram_shift(is_diffusing=False)
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            if save_memory:
                model.low_vram_shift(is_diffusing=True)
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, num_samples,
                shape, cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results



def inference(image_path, prompt, 
              a_prompt='best quality, extremely detailed', 
              n_prompt='longbody, lowres, bad anatomy, bad hands', 
              num_samples=1, image_resolution=512, detect_resolution=512,
              ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0,
              seed=-1, eta=0.0, save_dir='./results'):
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    results = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

    hed_map = results[0] # detected_map
    generated_images = results[1:]

    # return results
      # Create output folder
    os.makedirs(save_dir, exist_ok=True)

    # Save HED map
    # hed_image = Image.fromarray(hed_map)
    # hed_path = os.path.join(save_dir, 'hed_map.png')
    # hed_image.save(hed_path)
    # print(f"HED map saved to {hed_path}")

    # Save generated images
    gen_paths = []
    for i, img in enumerate(generated_images):
        img_pil = Image.fromarray(img)
        img_path = os.path.join(save_dir, f'generated_{i+1}.jpg')
        img_pil.save(img_path)
        gen_paths.append(img_path)
        print(f"Generated image {i+1} saved to {img_path}")

    # return [hed_path] + gen_paths
    return gen_paths



# Determine input directory, try primary then fallback
if os.path.exists('/home/jovyan/ouser/dataset/img_align_celeba'):
    input_dir = '/home/jovyan/ouser/dataset/img_align_celeba'
else:
    input_dir = '/data/oxx220000/img_align_celeba'
# Output directory based on attributes
output_dir = f'./{args.attributes}'

os.makedirs(output_dir, exist_ok=True)
# Count existing JPEGs to know where to stop
existing_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
existing_count = len(existing_files)

# Determine indices file path with fallback
primary_indices = f'/home/jovyan/ouser/safefix/non_{args.attributes}_indices.txt'
fallback_indices = f'data/celebA/non_{args.attributes}_indices.txt'
if os.path.exists(primary_indices):
    indices_file = primary_indices
elif os.path.exists(fallback_indices):
    indices_file = fallback_indices
else:
    raise FileNotFoundError(f"Indices file not found: {primary_indices} or {fallback_indices}")
with open(indices_file, 'r') as f:
    indices = [int(line.strip()) for line in f if line.strip()]

image_files = [f"{idx + 1:06d}.jpg" for idx in indices]

# # List all jpg files in input folder
# image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])

# Start timing
start_time = time.time()
gpu_times = []  # Store GPU processing times for each image
last_1k_start_time = start_time
images_processed_in_batch = 0
new_images_generated = 0  # Track only newly generated images

print(f"Starting image generation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target: {args.stop_at} new images to generate")
print(f"Existing images in folder: {existing_count}")
print(f"Final folder will contain: {existing_count + args.stop_at} images")

# Loop through each file
for idx, filename in enumerate(image_files, start=1):
    # Stop if we've reached the limit of new images to generate
    if new_images_generated >= args.stop_at:
        print(f"Reached target of {args.stop_at} new images generated. Exiting.")
        break
    input_path = os.path.join(input_dir, filename)
    input_image = cv2.imread(input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Use a unique random seed for each image to vary outputs
    img_seed = random.randint(0, 65535)
    
    # Print seed every 100 images (starting from first image)
    if new_images_generated % 100 == 0:
        print(f"Processing image {new_images_generated + 1}, seed: {img_seed}")

    # Record GPU processing start time
    gpu_start_time = time.time()

    # Run your process (one image per input)
    results = process(
        input_image=input_image,
        prompt=prompt_text,
        a_prompt='best quality, extremely detailed',
        n_prompt='lowres, bad anatomy, bad hands',
        num_samples=1,
        image_resolution=512,
        detect_resolution=512,
        ddim_steps=30,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=img_seed,
        eta=0.0
    )

    # Record GPU processing end time
    gpu_end_time = time.time()
    gpu_time = gpu_end_time - gpu_start_time
    gpu_times.append(gpu_time)

    generated_image = results[1]  # Index 0 is the HED map

    # Save result, avoiding name collisions
    base = os.path.splitext(filename)[0]
    stem = f"{base}_{args.attributes}_{img_seed}"
    save_path = os.path.join(output_dir, f"{stem}.jpg")
    counter = 2
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, f"{stem}_{counter}.jpg")
        counter += 1
    Image.fromarray(generated_image).save(save_path)
    # Update count and print every 1000 images
    new_images_generated += 1
    images_processed_in_batch += 1
    total_images_in_folder = existing_count + new_images_generated
    
    # Print timing statistics every 1000 images
    if new_images_generated % 1000 == 0:
        current_time = time.time()
        batch_time = current_time - last_1k_start_time
        avg_gpu_time = sum(gpu_times[-images_processed_in_batch:]) / images_processed_in_batch
        avg_total_time = batch_time / images_processed_in_batch
        
        print(f"=== Batch of 1000 images completed ===")
        print(f"{new_images_generated} new images generated so far.")
        print(f"Total images in folder: {total_images_in_folder}")
        print(f"Last 1000 images took: {batch_time:.2f} seconds")
        print(f"Average GPU time per image: {avg_gpu_time:.2f} seconds")
        print(f"Average total time per image: {avg_total_time:.2f} seconds")
        print(f"GPU utilization: {(avg_gpu_time/avg_total_time)*100:.1f}%")
        
        # Reset for next batch
        last_1k_start_time = current_time
        images_processed_in_batch = 0

    # Print status every 500 images
    if new_images_generated % 500 == 0 and new_images_generated > 0:
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / new_images_generated if new_images_generated > 0 else 0
        remaining_images = args.stop_at - new_images_generated
        estimated_remaining_time = remaining_images * avg_time_per_image
        print(f"Generated {new_images_generated} new images, last saved: {save_path}")
        print(f"Elapsed time: {elapsed_time:.1f}s, Est. remaining: {estimated_remaining_time/60:.1f} min")

# Final timing summary
end_time = time.time()
total_time = end_time - start_time
total_images_processed = len(gpu_times)

print(f"\n=== Final Timing Summary ===")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
print(f"New images generated: {total_images_processed}")
print(f"Total images in folder: {existing_count + total_images_processed}")
if total_images_processed > 0:
    avg_gpu_time_overall = sum(gpu_times) / total_images_processed
    avg_total_time_overall = total_time / total_images_processed
    print(f"Average GPU time per image: {avg_gpu_time_overall:.2f} seconds")
    print(f"Average total time per image: {avg_total_time_overall:.2f} seconds")
    print(f"Overall GPU utilization: {(avg_gpu_time_overall/avg_total_time_overall)*100:.1f}%")
    print(f"Images per hour: {3600/avg_total_time_overall:.1f}")
print(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
