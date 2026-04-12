import os
import sys
import contextlib
import io
import time


# Ensure this script runs in the 'ada_work' conda environment
if os.environ.get('CONDA_DEFAULT_ENV') != 'ada_work':
    sys.stderr.write("Info: Switching to the 'ada_work' conda environment and re-running...\n")
    os.execvp(
        'conda',
        ['conda', 'run', '-n', 'ada_work', 'python'] + sys.argv
    )
print("Info: Now running in the 'ada_work' conda environment.")

import argparse
import logging
import warnings
import random

import cv2
import numpy as np
import torch
import transformers
import huggingface_hub
from PIL import Image
from pytorch_lightning import seed_everything

# Diffusers >=0.17 expects CLIPTextModelWithProjection; add a fallback alias when older transformers lack it.
if not hasattr(transformers, 'CLIPTextModelWithProjection'):
    from transformers import CLIPTextModel

    class CLIPTextModelWithProjection(CLIPTextModel):
        pass

    transformers.CLIPTextModelWithProjection = CLIPTextModelWithProjection

if not hasattr(transformers, 'AutoImageProcessor'):
    from transformers import AutoFeatureExtractor

    class AutoImageProcessor(AutoFeatureExtractor):
        pass

    transformers.AutoImageProcessor = AutoImageProcessor

if not hasattr(transformers, 'CLIPImageProcessor'):
    from transformers import CLIPFeatureExtractor

    class CLIPImageProcessor(CLIPFeatureExtractor):
        pass

    transformers.CLIPImageProcessor = CLIPImageProcessor

transformers_utils = transformers.utils
if not hasattr(transformers_utils, 'SAFE_WEIGHTS_NAME'):
    transformers_utils.SAFE_WEIGHTS_NAME = "model.safetensors"
if not hasattr(transformers_utils, 'SAFE_WEIGHTS_INDEX_NAME'):
    transformers_utils.SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub import hf_hub_download as _hf_hub_download

    def cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = cached_download

if not hasattr(huggingface_hub, 'hf_hub_download'):
    def _missing_hf_hub_download(*args, **kwargs):
        raise ImportError("hf_hub_download is required by diffusers but missing in huggingface_hub.")

    huggingface_hub.hf_hub_download = _missing_hf_hub_download

from diffusers import DiffusionPipeline


# Suppress verbose logging from libraries that are chatty by default
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="InstructPix2Pix batch inference")
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
parser.add_argument('--num_inference_steps', type=int, default=50,
                    help="Number of inference steps for the diffusion process")
parser.add_argument('--guidance_scale', type=float, default=9.0,
                    help="Classifier-free guidance scale")
parser.add_argument('--image_guidance_scale', type=float, default=1.5,
                    help="Image guidance scale balancing prompt vs. input image")
parser.add_argument('--max_memory', type=int, default=23,
                    help='Maximum memory allocation per GPU in GiB (default: 23)')
parser.add_argument('--image_resolution', type=int, default=128,
                    help='Resize both input and output images to this square resolution before diffusion (default: 128)')
args = parser.parse_args()

# Build prompt text from attributes
attr_tokens = args.attributes.split('_')
token_map = {
    'redhair': 'vibrant red hair',
    'brownskin': 'brown skin',
    'sademotion': 'sad emotion',
    'pinkcolor': 'pink color',
    'fabrictexture': 'fabric texture'
}
mapped = [token_map.get(tok, tok) for tok in attr_tokens]
attr_phrase = ', '.join(mapped)
negative_prompt = 'low detail, low resolution, distorted, bad anatomy'


def build_prompt(class_name: str = None) -> str:
    if class_name:
        class_phrase = class_name.replace('_', ' ')
        if attr_phrase:
            return f"a subject with {attr_phrase}"
        return f"a {class_phrase}"
    if attr_phrase:
        return f"a person with {attr_phrase}, high detail, natural lighting"
    return 'a person, high detail, natural lighting'


default_prompt = build_prompt()

# Determine input directory
if args.dataset == 'celeba':
    primary_input_dir = '/home/jovyan/ouser/dataset/img_align_celeba'
    fallback_input_dir = '/data/oxx220000/img_align_celeba'
elif args.dataset == 'imagenet':
    primary_input_dir = '/home/jovyan/ouser/dataset/imagenet_subset'
    fallback_input_dir = '/data/oxx220000/imagenet_subset'
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

if os.path.exists(primary_input_dir):
    input_dir = primary_input_dir
elif os.path.exists(fallback_input_dir):
    input_dir = fallback_input_dir
else:
    raise FileNotFoundError(f"Input directory not found: {primary_input_dir} or {fallback_input_dir}")

# GPU helper utilities (mirroring utils/filter.py logic)
def get_available_gpus():
    return list(range(torch.cuda.device_count()))


def configure_gpu_memory(available_gpus, max_memory_per_gpu_gb=23):
    max_memory = {}
    max_memory_str = f"{max_memory_per_gpu_gb}GiB"
    for idx, _ in enumerate(available_gpus):
        max_memory[idx] = max_memory_str
    if not max_memory:
        warnings.warn("No GPUs detected; falling back to CPU execution.")
        return "cpu"
    print(f"Using {len(max_memory)} GPU(s) with {max_memory_str} allocation each")
    return max_memory


available_gpus = get_available_gpus()
max_memory_config = configure_gpu_memory(available_gpus, args.max_memory)

# Output directory based on attributes
output_dir = f'./{args.attributes}_instruct_pix2pix'
os.makedirs(output_dir, exist_ok=True)

# Count existing JPEGs to know where to stop
existing_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
existing_count = len(existing_files)

# Build list of images to process
image_entries = []
if args.dataset == 'celeba':
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
    image_entries = [(None, os.path.join(input_dir, name), name) for name in image_files]
else:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    class_names = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d != 'backpack'
    ])
    if not class_names:
        raise FileNotFoundError(f"No class folders found under {input_dir}")
    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            _, ext = os.path.splitext(fname)
            if ext.lower() not in valid_exts:
                continue
            image_entries.append((class_name, os.path.join(class_dir, fname), fname))
    if not image_entries:
        raise FileNotFoundError(f"No images found under {input_dir}")
    random.shuffle(image_entries)

# Initialize pipeline with suppressed stdout
pipe_io = io.StringIO()
pipeline_kwargs = {
    "torch_dtype": torch.float16
}
if isinstance(max_memory_config, dict):
    pipeline_kwargs["device_map"] = "auto"
    pipeline_kwargs["max_memory"] = max_memory_config
elif max_memory_config == "cpu":
    pipeline_kwargs["device_map"] = "cpu"

with contextlib.redirect_stdout(pipe_io), contextlib.redirect_stderr(pipe_io):
    pipe = DiffusionPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        **pipeline_kwargs
    )
    if max_memory_config == "cpu":
        pipe.to("cpu")
    elif "device_map" not in pipeline_kwargs:
        pipe.to("cuda")
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload") and isinstance(max_memory_config, dict):
        pipe.enable_model_cpu_offload()
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)


def edit_image(input_image: np.ndarray, prompt: str, neg_prompt: str,
               num_inference_steps: int, guidance_scale: float,
               image_guidance_scale: float, seed: int, target_resolution: int) -> np.ndarray:
    input_pil = Image.fromarray(input_image)
    if target_resolution and target_resolution > 0:
        input_pil = input_pil.resize((target_resolution, target_resolution), Image.BILINEAR)
    if seed == -1:
        seed = random.randint(0, 65535)
    pipe_device = getattr(pipe, '_execution_device', getattr(pipe, 'device', torch.device('cpu')))
    if torch.cuda.is_available() and isinstance(pipe_device, torch.device) and pipe_device.type == 'cuda':
        generator = torch.Generator(device='cuda').manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)
    try:
        seed_everything(seed, workers=True)
    except TypeError:
        seed_everything(seed)
    autocast_context = torch.autocast('cuda') if torch.cuda.is_available() and isinstance(pipe_device, torch.device) and pipe_device.type == 'cuda' else contextlib.nullcontext()
    with autocast_context:
        result = pipe(
            prompt=prompt,
            image=input_pil,
            negative_prompt=neg_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator
        )
    output_image = result.images[0]
    if target_resolution and target_resolution > 0 and output_image.size != (target_resolution, target_resolution):
        output_image = output_image.resize((target_resolution, target_resolution), Image.LANCZOS)
    return np.asarray(output_image), seed


# Start timing
start_time = time.time()
gpu_times = []
last_1k_start_time = start_time
images_processed_in_batch = 0
new_images_generated = 0

print(f"Starting image generation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target: {args.stop_at} new images to generate")
print(f"Existing images in folder: {existing_count}")
print(f"Final folder will contain: {existing_count + args.stop_at} images")

for idx, (class_name, input_path, filename) in enumerate(image_entries, start=1):
    if new_images_generated >= args.stop_at:
        print(f"Reached target of {args.stop_at} new images generated. Exiting.")
        break
    input_image = cv2.imread(input_path)
    if input_image is None:
        print(f"Warning: Could not read {input_path}, skipping.")
        continue
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    img_seed = random.randint(0, 65535)
    if new_images_generated % 100 == 0:
        if class_name:
            print(f"Processing image {new_images_generated + 1} ({class_name}), seed: {img_seed}")
        else:
            print(f"Processing image {new_images_generated + 1}, seed: {img_seed}")

    prompt_current = build_prompt(class_name) if class_name else default_prompt

    gpu_start_time = time.time()
    edited_image, used_seed = edit_image(
        input_image=input_image,
        prompt=prompt_current,
        neg_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        seed=img_seed,
        target_resolution=args.image_resolution
    )
    gpu_end_time = time.time()
    gpu_times.append(gpu_end_time - gpu_start_time)

    base = os.path.splitext(filename)[0]
    if class_name:
        base = f"{class_name}_{base}"
    stem = f"{base}_{args.attributes}_{used_seed}"
    save_path = os.path.join(output_dir, f"{stem}.jpg")
    counter = 2
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, f"{stem}_{counter}.jpg")
        counter += 1
    Image.fromarray(edited_image).save(save_path)

    new_images_generated += 1
    images_processed_in_batch += 1
    total_images_in_folder = existing_count + new_images_generated

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
        last_1k_start_time = current_time
        images_processed_in_batch = 0

    if new_images_generated % 500 == 0 and new_images_generated > 0:
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / new_images_generated
        remaining_images = args.stop_at - new_images_generated
        estimated_remaining_time = remaining_images * avg_time_per_image
        print(f"Generated {new_images_generated} new images, last saved: {save_path}")
        print(f"Elapsed time: {elapsed_time:.1f}s, Est. remaining: {estimated_remaining_time/60:.1f} min")

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
