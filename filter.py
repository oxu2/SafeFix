 # Global flag for presence inclusion
INCLUDE_PRESENCE = False
import json
import random
import sys
import os
import datetime  # Add this import to support timestamps
import shutil
from typing import List
from utils.defines import *
# # Check if running in the correct conda environment
# def ensure_conda_env(env_name):
#     """Ensure script runs in the specified conda environment"""
#     current_env = os.environ.get('CONDA_DEFAULT_ENV')
#     if current_env != env_name:
#         # print(f"Switching to conda environment '{env_name}' (current: {current_env or 'None'})")
#         # Construct the conda run command
#         conda_path = os.environ.get('CONDA_EXE', 'conda')
#         script_path = os.path.abspath(sys.argv[0])
#         args = ' '.join(sys.argv[1:])
#         cmd = f"{conda_path} run -n {env_name} python {script_path} {args}"
#         # print(f"Executing: {cmd}")
#         os.system(cmd)
#         sys.exit(0)
#     else:
#         print(f"Running in conda environment: {env_name}")
#
# # Make the conda environment configurable
# conda_env = os.environ.get('CONDA_ENV', 'ad')  # Default to 'ad' but allow override via environment variable
# ensure_conda_env(conda_env)

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import base64
import accelerate
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-notebook environment
    def display(img):
        img.show()  # This will open the image in default image viewer


# @title inference function
def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False, verbose=False):
    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if verbose:
        print("text:", text)
    
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]
    
    

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# @title inference function with API
def inference_with_api(image_path, prompt, sys_prompt="You are a helpful assistant.", model_id="qwen2.5-vl-72b-instruct", min_pixels=512*28*28, max_pixels=2048*28*28):
    base64_image = encode_image(image_path)
    client = OpenAI(
        #If the environment variable is not configured, please replace the following line with the Dashscope API Key: api_key="sk-xxx". Access via https://bailian.console.alibabacloud.com/?apiKey=1 "
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


    messages=[
        {
            "role": "system",
            "content": [{"type":"text","text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
       
    )
    return completion.choices[0].message.content

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

# Get available CUDA devices from environment variable
def get_available_gpus():
    """Get available GPU IDs from CUDA_VISIBLE_DEVICES"""
    # With CUDA_VISIBLE_DEVICES set, PyTorch will see devices as 0,1,2,...
    # regardless of their actual physical IDs
    return list(range(torch.cuda.device_count()))

# Configure memory based on available GPUs
def configure_gpu_memory(available_gpus, max_memory_per_gpu_gb=23):
    """Configure memory allocation based on available GPUs."""
    max_memory = {}
    # Format the memory string
    max_memory_str = f"{max_memory_per_gpu_gb}GiB"

    for i, gpu_id in enumerate(available_gpus):
        # Use local device ID (0-indexed)
        max_memory[i] = max_memory_str  # Use the formatted string

    # If no GPUs specified, default to CPU
    if not max_memory:
        print("Warning: No GPUs available. Using CPU.")
        return "cpu"

    print(f"Using {len(max_memory)} GPUs with {max_memory_str} memory allocation each")
    return max_memory

def process_image(image_path, model, processor, tags: List[str], attributes: List[str], verbose=False):
    """Process a single image and return the binary results for both features."""
    # Special handling for ImageNet dataset
    if DATASET == 'imagenet':
        # Derive class name from parent directory
        class_name = os.path.basename(os.path.dirname(image_path))
        # Build questions: one per tag-attribute pair, then presence check
        questions = [f"Does the {class_name} have {t} {a}?" for t, a in zip(tags, attributes)]
        questions.append(f"Is there a {class_name} in the picture?")
        prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."
        response = inference(image_path, prompt, verbose=verbose)
        try:
            # Parse numeric answers
            nums = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
            # Map attribute results without class prefix: e.g., "yellowcolor"
            results = {
                f"{t}{a}": nums[i] if i < len(nums) else 0
                for i, (t, a) in enumerate(zip(tags, attributes))
            }
            # Optionally include presence if requested
            if INCLUDE_PRESENCE:
                presence_val = nums[len(tags)] if len(nums) > len(tags) else 0
                results["presence"] = presence_val
        except Exception as e:
            print(f"Warning: Error parsing response '{response}' for {image_path}: {e}")
            # On error, default all attributes to 0
            results = {f"{t}{a}": 0 for t, a in zip(tags, attributes)}
            if INCLUDE_PRESENCE:
                results["presence"] = 0
        return results
    else:
        # Existing CelebA logic for building questions follows below...
        questions = []
        for t, a in zip(tags, attributes):
            if a == "lipstick":
                questions.append("Is the person in this picture wearing lipstick?")
            elif a == "emotion":
                questions.append(
                    f"If you see any hint of {t} emotion in this picture, answer 1; "
                    "only answer 0 if you are completely sure there is no such emotion."
                )
            else:
                questions.append(f"Does the person in this picture have {t} {a}?")
        prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."
        response = inference(image_path, prompt, verbose=verbose)
        # Debug: print Qwen's raw response for emotion detection
        # if 'emotion' in attributes:
        # if True:
        #     print(f"Debug: {image_path} emotion raw response: {response}")
        
        # Extract the two numbers from the response
        try:
            numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
            results = {f"{t}_{a.replace(' ', '_')}": num for t, a, num in zip(tags, attributes, numbers)}
            # Fallback for emotion attributes: if inference missing or ambiguous, default to positive (1)
            for t, a in zip(tags, attributes):
                if a == "emotion":
                    key = f"{t}_{a.replace(' ', '_')}"
                    if key not in results:
                        results[key] = 1
            return results
        except Exception as e:
            print(f"Warning: Error parsing response '{response}' for {image_path}: {e}")
            # If parsing fails, at least return positive for emotion attributes
            fallback_results = {}
            for t, a in zip(tags, attributes):
                if a == "emotion":
                    fallback_results[f"{t}_{a.replace(' ', '_')}"] = 1
            return fallback_results

def process_directory(directory_path, output_json_path, tags, attributes, verbose=False, file_type="png", stop_at: int = 15000, stop_suffix=None):
    """
    Process images in directory and save results to JSON.
    Stops processing *before* encountering an image file whose base number
    is greater than or equal to stop_at.
    """
    # Ensure the save/json directory exists
    save_dir = "save/json"
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, os.path.basename(output_json_path))

    # Load existing results if present, so new runs only process newly added images
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        processed_set = set(results.keys())
        if verbose:
            print(f"Found {len(processed_set)} previously processed images, skipping them.")
    else:
        results = {}
        processed_set = set()
    stopped_early = False
    stop_filename = None # The file that triggered the stop (will not be processed)
    processed_count = 0
    if stop_at:
        print(f"Stop condition active: stop after processing {stop_at} images.")

    # Get image files recursively based on the specified file type
    if file_type.lower() == "all":
        exts = ['.png', '.jpg', '.jpeg']
    else:
        exts = ['.' + ext.lower() for ext in file_type.lower().split(',')]
    image_files = []
    for root, _, files in os.walk(directory_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                # Compute relative path to preserve subdirectory structure
                rel_dir = os.path.relpath(root, directory_path)
                rel_path = f if rel_dir == '.' else os.path.join(rel_dir, f)
                image_files.append(rel_path)
    # Sort files in ascending order
    image_files.sort()

    # Filter out files already processed
    image_files = [f for f in image_files if os.path.splitext(f)[0] not in processed_set]
    if verbose:
        print(f"{len(image_files)} new images to process after filtering out existing ones.")

    total_images = len(image_files)

    if verbose:
        print(f"Found {total_images} images with file type(s): {file_type}")
        print(f"Processing images in sorted order: {', '.join(image_files[:5])}{'...' if len(image_files) > 5 else ''}")
    else:
        print(f"Found {total_images} images with file type(s): {file_type}")


    processed_count = 0
    for idx, image_file in enumerate(image_files):
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(directory_path, image_file)
        filename = base_name

        if verbose:
            print(f"Processing {image_file}...")
        result = process_image(image_path, model, processor, tags, attributes, verbose=verbose)
        if result:
            results[filename] = result

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Check stop-after-images condition
        processed_count = len(results)
        if stop_at is not None and processed_count >= stop_at:
            stopped_early = True
            print(f"\nStopping after processing {processed_count} images (stop_at={stop_at})")
            break

        if (idx + 1) % 1000 == 0:
             print(f"[{idx + 1}/{total_images}] Processed...")

    if not stopped_early:
        processed_count = len(results)

    if verbose:
        print(f"Results saved to {output_json_path}")

    return results, processed_count, stopped_early, stop_filename

# Update function to extract features from results with configurable filtering
def create_feature_mapping(results_json, feasible_json="features.json", attributes: str = "redhair_brownskin"):
    """
    Create a JSON file mapping filenames to features for images with specific attributes
    Example: {"example_image": {"tag": "yellow", "attribute": "hair", "wearing_lipstick": 0}}
    Only include entries that have tag_attribute=1
    
    Parameters:
    - results_json: Path to results JSON or dictionary of results
    - feasible_json: Path to save the filtered results
    """
    # For ImageNet, simplify mapping to just the attribute value
    if DATASET == 'imagenet':
        # Load results dict if a filepath is given
        if isinstance(results_json, str):
            with open(results_json, 'r') as f:
                results = json.load(f)
        else:
            results = results_json
        # Build mapping: include only items where attribute==1
        feature_mapping = {}
        suffix = f"_{attributes}"
        for filename, result in results.items():
            # Find the value for the attribute key ending with suffix
            val = next((v for k, v in result.items() if k.endswith(suffix)), 0)
            if val == 1:
                feature_mapping[filename] = {attributes: val}
        # Save and return
        save_dir = os.path.dirname(feasible_json)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, os.path.basename(feasible_json))
        with open(out_path, 'w') as f:
            json.dump(feature_mapping, f, indent=4)
        return feature_mapping
    # --- Original logic for CelebA etc. is now disabled for ImageNet ---
    # Ensure the save/json directory exists
    # save_dir = "save/json"
    # os.makedirs(save_dir, exist_ok=True)
    # feasible_json = os.path.join(save_dir, os.path.basename(feasible_json))  # Fix duplicate path issue
    #
    # if isinstance(results_json, str):
    #     # If results_json is a file path
    #     with open(results_json, 'r') as f:
    #         results = json.load(f)
    # else:
    #     # If results_json is already a dictionary
    #     results = results_json
    # # Parse the attributes string into keys
    # attr_list = attributes.split('_')
    # # Build tags and attributes lists
    # tags = []
    # attrs = []
    # if DATASET == 'celeba':
    #     for attr in attr_list:
    #         if attr.endswith('hair'):
    #             tags.append(attr[:-4])
    #             attrs.append('hair')
    #         elif attr.endswith('skin'):
    #             tags.append(attr[:-4])
    #             attrs.append('skin')
    #         elif attr.endswith('emotion'):
    #             tags.append(attr[:-7])
    #             attrs.append('emotion')
    #         else:
    #             tags.append(attr)
    #             attrs.append(attr)
    #     # Always include lipstick for CelebA
    #     tags.append('wearing')
    #     attrs.append('lipstick')
    # else:  # imagenet
    #     for attr in attr_list:
    #         if attr.endswith('color'):
    #             tags.append(attr[:-5])
    #             attrs.append('color')
    #         elif attr.endswith('texture'):
    #             tags.append(attr[:-7])
    #             attrs.append('texture')
    # # Derive feature keys
    # keys = [f"{t}_{a}" for t, a in zip(tags, attrs)]
    # # Retain these attribute keys for filtering; ensure we always include lipstick feature in output
    # attr_keys = keys[:]
    # output_keys = attr_keys + ["wearing_lipstick"]
    #
    # feature_mapping = {}
    # for filename, result in results.items():
    #     # Require all specified attribute keys be positive
    #     if all(result.get(key, 0) == 1 for key in attr_keys):
    #         # Build mapping entry: include attribute keys and lipstick feature
    #         mapping_entry = {key: result.get(key, 0) for key in attr_keys}
    #         # Always include lipstick result (0 or 1)
    #         mapping_entry["wearing_lipstick"] = result.get("wearing_lipstick", 0)
    #         feature_mapping[filename] = mapping_entry
    #
    # # Save the feature mapping to a JSON file
    # with open(feasible_json, 'w') as f:
    #     json.dump(feature_mapping, f, indent=4)
    #
    # return feature_mapping

def rename_images_with_suffix(directory_path, suffix="_red", verbose=False):
    """
    Rename JPG files that have numeric filenames by adding a suffix before the extension.
    Example: 123.jpg becomes 123_red.jpg
    
    Parameters:
    - directory_path: Path to directory containing images to rename
    - suffix: The suffix to add before the extension (default: "_red")
    - verbose: Whether to print detailed renaming information
    
    Returns:
    - int: Number of files renamed
    """
    # Ensure directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return 0
        
    renamed_count = 0
    
    # Get all image files with .jpg or .jpeg extensions in the directory
    image_files = [
        f for f in os.listdir(directory_path)
        if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg'])
    ]

    for filename in image_files:
        # Get base name and extension
        base_name, ext = os.path.splitext(filename)
        
        # Skip files that already have the suffix
        if base_name.endswith(suffix):
            continue
            
        # Only rename files that are numeric (e.g., "123.jpg" or "123.jpeg")
        if base_name.isdigit():
            # Create new filename with suffix
            new_filename = f"{base_name}{suffix}{ext}"

            # Full paths
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            # Rename the file
            try:
                os.rename(old_path, new_path)
                renamed_count += 1
                if verbose:
                    print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
        elif verbose:
            print(f"Skipped non-numeric filename: {filename}")
    
    if verbose:
        print(f"Successfully renamed {renamed_count} numeric-named files in {directory_path}")
    return renamed_count

def main(directory_path="", 
         verbose: bool = False,
         file_type: str = "png",
         rename_files: bool = False,
         attributes: str = "brownskin",
         rename_suffix: str = None,
         max_memory_per_gpu: int = 23,
         stop_at: int = 15000,
         feasible_only: bool = False,
         results_file: str = None,
         copy_only: bool = False,
         feasible_json: str = None):
    """
    Main function to process images and create feature mapping.
    
    Parameters:
    - directory_path: Path to directory containing images to process
    - verbose: Whether to display detailed processing information
    - file_type: Type of image files to process ("png", "jpg", "png,jpg", "all")
    - rename_files: Whether to rename files by adding a suffix
    - rename_suffix: Suffix to add when renaming files (default: "_red")
    - max_memory_per_gpu: Max memory per GPU in GiB (e.g., 23)
    - stop_at: Base filename number to stop processing at (e.g., "077000")
    
    Returns:
    - Tuple of (results, feature_mapping)
    """
    # Copy-only mode: use existing feasible JSON to copy images, skip processing
    if copy_only:
        if feasible_json is None:
            print("Error: --feasible_json must be provided when using --copy_only")
            return {}, {}
        # Load feature mapping from provided JSON
        with open(feasible_json, 'r') as f:
            feature_mapping = json.load(f)
        # Copy feasible images
        feasible_count = len(feature_mapping)
        dest_dir = os.path.join(os.getcwd(), f"{attributes}_{feasible_count}")
        os.makedirs(dest_dir, exist_ok=True)
        for filename in feature_mapping:
            for ext in ['.png', '.jpg', '.jpeg']:
                src_path = os.path.join(directory_path, filename + ext)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(dest_dir, filename + ext))
                    break
        print(f"Copied {feasible_count} feasible images to {dest_dir}")
        return {}, feature_mapping

    # Configure GPU memory first using the passed argument
    available_gpus = get_available_gpus()
    device_memory = configure_gpu_memory(available_gpus, max_memory_per_gpu)

    # Load model and processor here, after configuring memory
    global model, processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=device_memory
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # If rename_files is True, rename the files in the directory
    if rename_files and rename_suffix:
        renamed_count = rename_images_with_suffix(directory_path, rename_suffix, verbose=verbose)
        if verbose:
            print(f"Renamed {renamed_count} files with suffix '{rename_suffix}'")
    
    # Parse attribute string into list and build tags/attrs lists depending on dataset
    attr_list = attributes.split('_')
    tags = []
    attrs = []
    if DATASET == 'celeba':
        for attr in attr_list:
            if attr.endswith('hair'):
                tags.append(attr[:-4])
                attrs.append('hair')
            elif attr.endswith('skin'):
                tags.append(attr[:-4])
                attrs.append('skin')
            elif attr.endswith('emotion'):
                tags.append(attr[:-7])
                attrs.append('emotion')
            else:
                tags.append(attr)
                attrs.append(attr)
        # Always include lipstick for CelebA
        tags.append('wearing')
        attrs.append('lipstick')
    else:  # imagenet
        for attr in attr_list:
            if attr.endswith('color'):
                tags.append(attr[:-5])
                attrs.append('color')
            elif attr.endswith('texture'):
                tags.append(attr[:-7])
                attrs.append('texture')

    # Build timestamped file paths
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    results_json_path = f"{attributes}_results_{timestamp}.json"
    feasible_json = f"{attributes}_feasible_{timestamp}.json"

    # Ensure the save/json directory exists
    save_dir = "save/json"
    os.makedirs(save_dir, exist_ok=True)
    results_json_path = os.path.join(save_dir, results_json_path)
    feasible_json = os.path.join(save_dir, feasible_json)
    
    print(f"Starting image processing for attributes: {attributes}...")

    # Feasible-only mode: skip inference, load existing results JSON
    if feasible_only:
        rf = results_file if results_file else results_json_path
        print(f"Feasible-only mode: loading results from {rf}")
        with open(rf, 'r') as f:
            results = json.load(f)
        feature_mapping = create_feature_mapping(results, feasible_json, attributes)
        print(f"Found {len(feature_mapping)} feasible images.")
        print(f"Feasible mapping saved to: {feasible_json}")
        return results, feature_mapping

    # (Removed file-based stop message)
    
    # Process all images in directory
    results, processed_count, stopped_early, stop_filename = process_directory(
           directory_path,
           results_json_path,
           tags,
           attrs,
           verbose=verbose,
           file_type=file_type,
           stop_at=stop_at,
           stop_suffix=rename_suffix
       )
    print("Image processing using Qwen VL model complete.")
    
    # Create feature mapping for images with specified attributes
    feature_mapping = create_feature_mapping(results, feasible_json, attributes)
    print("\n--- Processing Summary ---")
    if stopped_early:
        print(f"Processing stopped at file: {stop_filename}")
        print(f"Total images processed before stopping: {processed_count}")
    else:
        print(f"Processed all {processed_count} found images.")
    print(f"Found {len(feature_mapping)} feasible images matching the criteria.")
    # Print the ratio of feasible to processed images
    ratio = len(feature_mapping) / processed_count * 100 if processed_count > 0 else 0
    print(f"Feasible images ratio: {len(feature_mapping)}/{processed_count} ({ratio:.2f}%)")
    print(f"Results saved to: {results_json_path}")
    print(f"Feasible mapping saved to: {feasible_json}")
    print("--------------------------")
    # Filter and copy feasible images from the original dataset
    feasible_count = len(feature_mapping)
    dest_dir = os.path.join(os.getcwd(), f"{attributes}_{feasible_count}")
    os.makedirs(dest_dir, exist_ok=True)
    for filename in feature_mapping:
        # Attempt common image extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            src_path = os.path.join(directory_path, filename + ext)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(dest_dir, filename + ext))
                break
    print(f"Copied {feasible_count} feasible images to {dest_dir}")
    return results, feature_mapping

# Execute the main function with default parameters
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    import shlex
    
    parser = argparse.ArgumentParser(description='Process images for specific attributes')
    parser.add_argument(
        '--image_dir',
        type=str,
        default="",
        help='Directory containing images'
    )
    parser.add_argument(
        '--verbose',
        type=str2bool,
        default=False,
        help='Display detailed processing information (true/false)'
    )
    parser.add_argument('--file_type', type=str, default="jpg,jpeg",
                    help='Type of image files to process ("png", "jpg", "jpeg", "jpg,jpeg", "png,jpg,jpeg", "all")')
    parser.add_argument(
        '--rename_files',
        type=str2bool,
        default=False,
        help='Rename files by adding suffix before extension (true/false)'
    )
    parser.add_argument('--attributes', type=str, default="pinkcolor",
                        help='Attributes to detect; for celeba: e.g., redhair_brownskin, redhair; for imagenet: e.g., fabrictexture, pinkcolor')
    parser.add_argument('--rename_suffix', type=str, default=None,
                        help='Suffix to add when renaming files (default: "_red")')
    parser.add_argument('--max_memory', type=int, default=33, 
                        help='Maximum memory allocation per GPU in GiB (default: 33)')
    parser.add_argument('--stop_at', type=int, default=15000,
                        help='Number of images to process before stopping (default: 15000)')
    parser.add_argument('--feasible_only', action='store_true',
                        help='Only perform feasible mapping from existing results JSON.')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to existing results JSON file (used with --feasible_only).')
    parser.add_argument('--copy_only', type=str2bool,
                        help='Only copy feasible images using an existing feasible JSON (skip processing).')
    parser.add_argument('--feasible_json', type=str, default=None,
                        help='Path to existing feasible JSON file for copying images.')
    parser.add_argument(
        '--dataset',
        choices=['imagenet', 'celeba'],
        default='imagenet',
        help='Which dataset to use: imagenet (default) or celeba'
    )
    parser.add_argument(
        '--include_presence',
        type=str2bool,
        default=False,
        help='If true, include the <class>_presence field in ImageNet results (default: False)'
    )
    
args = parser.parse_args()

# Whether to include presence check in results
INCLUDE_PRESENCE = args.include_presence

if not args.image_dir:
    # Determine default directory based on dataset and attributes
    if args.dataset == 'celeba':
        default_dir = os.path.join("exampleData", args.attributes)
    else:  # imagenet
        default_dir = f"exampleData/imagenet_{args.attributes}"
    # Use default if it exists, otherwise try fallback paths
    if os.path.exists(default_dir):
        args.image_dir = default_dir
    else:
        possible_dirs = [
            f"/home/user/{args.dataset}_{args.attributes}",
            f"/data/user/{args.dataset}_{args.attributes}"
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                args.image_dir = d
                break
        else:
            args.image_dir = default_dir
    if args.verbose:
        print(f"No --image_dir provided or default not found; using: {args.image_dir}")

if args.rename_suffix is None:
    args.rename_suffix = "_" + args.attributes

# Set global dataset flag
DATASET = args.dataset

main(
   args.image_dir,
   args.verbose,
   args.file_type,
   args.rename_files,
   args.attributes,
   args.rename_suffix,
   args.max_memory,
   args.stop_at,
   args.feasible_only,
   args.results_file,
   args.copy_only,
   args.feasible_json
)