import json
import random
import sys
import os
import datetime  # Add this import to support timestamps
from typing import List
from utils.defines import *
# Check if running in the correct conda environment
def ensure_conda_env(env_name):
    """Ensure script runs in the specified conda environment"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != env_name:
        # print(f"Switching to conda environment '{env_name}' (current: {current_env or 'None'})")
        # Construct the conda run command
        conda_path = os.environ.get('CONDA_EXE', 'conda')
        script_path = os.path.abspath(sys.argv[0])
        args = ' '.join(sys.argv[1:])
        cmd = f"{conda_path} run -n {env_name} python {script_path} {args}"
        # print(f"Executing: {cmd}")
        os.system(cmd)
        sys.exit(0)
    else:
        print(f"Running in conda environment: {env_name}")

# Make the conda environment configurable
conda_env = os.environ.get('CONDA_ENV', 'ad')  # Default to 'ad' but allow override via environment variable
ensure_conda_env(conda_env)

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
def configure_gpu_memory(available_gpus, max_memory_per_gpu_gb=22):
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
    # Construct prompt based on tags and attributes
    questions = []
    for t, a in zip(tags, attributes):
        if a == "lipstick":
            questions.append("Is the person in this picture wearing lipstick?")
        else:
            questions.append(f"Does the person in this picture have {t} {a}?")
    prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."
    response = inference(image_path, prompt, verbose=verbose)
    
    # Extract the two numbers from the response
    try:
        numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
        results = {f"{t}_{a.replace(' ', '_')}": num for t, a, num in zip(tags, attributes, numbers)}
        return results
    except Exception as e:
        print(f"Warning: Error parsing response '{response}' for {image_path}: {e}")
        return {}

def process_directory(directory_path, output_json_path, tags, attributes, verbose=False, file_type="png", stop_at=None, stop_suffix=None):
    """
    Process images in directory and save results to JSON.
    Stops processing *before* encountering an image file whose base number
    is greater than or equal to stop_at.
    """
    # Ensure the save/json directory exists
    save_dir = "save/json"
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, os.path.basename(output_json_path))

    results = {}
    stopped_early = False
    stop_filename = None # The file that triggered the stop (will not be processed)
    processed_count = 0
    stop_at_num = None
    if stop_at:
        try:
            stop_at_num = int(stop_at)
        except ValueError:
            print(f"Warning: Invalid --stop_at value '{stop_at}'. It should be a number. Ignoring stop condition.")
            stop_at = None # Disable stopping if value is invalid

    # Get image files based on the specified file type
    if file_type.lower() == "all":
        image_files = [f for f in os.listdir(directory_path) if (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]
    else:
        extensions = ['.' + ext.lower() for ext in file_type.lower().split(',')]
        image_files = [f for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in extensions)]

    # Sort image files alphanumerically based on the numeric part
    # Handles names like '001.jpg', '002_suffix.jpg', '10.jpg' correctly
    def get_numeric_part(filename):
        base = os.path.splitext(filename)[0]
        numeric_str = base.split('_')[0]
        try:
            return int(numeric_str)
        except ValueError:
            return float('inf') # Place non-numeric filenames at the end

    image_files.sort(key=get_numeric_part)

    total_images = len(image_files)

    if verbose:
        print(f"Found {total_images} images with file type(s): {file_type}")
        print(f"Processing images in sorted order: {', '.join(image_files[:5])}{'...' if len(image_files) > 5 else ''}")
        if stop_at_num is not None:
            print(f"Will stop processing before any image with base number >= {stop_at}.")
    else:
        print(f"Found {total_images} images with file type(s): {file_type}")
        if stop_at_num is not None:
             print(f"Stop condition active: Stop before base number >= {stop_at}.")


    processed_count = 0
    for idx, image_file in enumerate(image_files):
        base_name = os.path.splitext(image_file)[0]
        numeric_part_str = base_name.split('_')[0]

        # Check for stopping condition based on numeric part
        if stop_at_num is not None:
            try:
                current_num = int(numeric_part_str)
                if current_num >= stop_at_num:
                    stopped_early = True
                    stop_filename = image_file
                    processed_count = idx
                    print(f"\nStopping before processing file: {stop_filename} (base number {current_num} >= stop_at {stop_at_num})")
                    break
            except ValueError:
                pass

        image_path = os.path.join(directory_path, image_file)
        filename = base_name

        if verbose:
            print(f"Processing {image_file}...")
        result = process_image(image_path, model, processor, tags, attributes, verbose=verbose)
        if result:
            results[filename] = result

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

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
    # Ensure the save/json directory exists
    save_dir = "save/json"
    os.makedirs(save_dir, exist_ok=True)
    feasible_json = os.path.join(save_dir, os.path.basename(feasible_json))  # Fix duplicate path issue

    if isinstance(results_json, str):
        # If results_json is a file path
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        # If results_json is already a dictionary
        results = results_json
    # Parse the attributes string into keys
    attr_list = attributes.split('_')
    # Build tags and attributes lists
    tags = []
    attrs = []
    for attr in attr_list:
        if attr.endswith('hair'):
            tags.append(attr[:-4])
            attrs.append('hair')
        elif attr.endswith('skin'):
            tags.append(attr[:-4])
            attrs.append('skin')
        else:
            tags.append(attr)
            attrs.append(attr)
    # Derive feature keys
    keys = [f"{t}_{a}" for t, a in zip(tags, attrs)]

    feature_mapping = {}
    for filename, result in results.items():
        # Include if all specified keys have value 1
        if all(result.get(key, 0) == 1 for key in keys):
            # Build mapping entry for these keys
            mapping_entry = { key: result.get(key, 0) for key in keys }
            feature_mapping[filename] = mapping_entry

    # Save the feature mapping to a JSON file
    with open(feasible_json, 'w') as f:
        json.dump(feature_mapping, f, indent=4)

    return feature_mapping

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
    
    # Get all jpg files in the directory
    jpg_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]
    
    for filename in jpg_files:
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Skip files that already have the suffix
        if base_name.endswith(suffix):
            continue
            
        # Only rename files that are numeric (e.g., "123.jpg")
        if base_name.isdigit():
            # Create new filename with suffix
            new_filename = f"{base_name}{suffix}.jpg"
            
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

def main(directory_path="./redhair", 
         verbose: bool = False,
         file_type: str = "png",
         rename_files: bool = False,
         attributes: str = "redhair_brownskin",
         rename_suffix: str = None,
         max_memory_per_gpu: int = 22,
         stop_at: str = None):
    """
    Main function to process images and create feature mapping.
    
    Parameters:
    - directory_path: Path to directory containing images to process
    - verbose: Whether to display detailed processing information
    - file_type: Type of image files to process ("png", "jpg", "png,jpg", "all")
    - rename_files: Whether to rename files by adding a suffix
    - rename_suffix: Suffix to add when renaming files (default: "_red")
    - max_memory_per_gpu: Max memory per GPU in GiB (e.g., 22)
    - stop_at: Base filename number to stop processing at (e.g., "077000")
    
    Returns:
    - Tuple of (results, feature_mapping)
    """
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
    if rename_files:
        renamed_count = rename_images_with_suffix(directory_path, rename_suffix, verbose=verbose)
        if verbose:
            print(f"Renamed {renamed_count} files with suffix '{rename_suffix}'")
    
    # Parse attribute string into list
    attr_list = attributes.split('_')
    tags = []
    attrs = []
    for attr in attr_list:
        if attr.endswith('hair'):
            tags.append(attr[:-4])
            attrs.append('hair')
        elif attr.endswith('skin'):
            tags.append(attr[:-4])
            attrs.append('skin')
        else:
            tags.append(attr)
            attrs.append(attr)
    # Always include lipstick
    tags.append('wearing')
    attrs.append('lipstick')

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
    if stop_at:
        print(f"Processing will stop if file '{stop_at}{rename_suffix}.jpg' (or similar) is found.")
    
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
    print(f"Results saved to: {results_json_path}")
    print(f"Feasible mapping saved to: {feasible_json}")
    print("--------------------------")
    
    return results, feature_mapping

# Execute the main function with default parameters
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    import shlex
    
    parser = argparse.ArgumentParser(description='Process images for specific attributes')
    parser.add_argument('--dir', type=str, default="exampleData/test", 
                        help='Directory containing images')
    parser.add_argument(
        '--verbose',
        type=str2bool,
        default=False,
        help='Display detailed processing information (true/false)'
    )
    parser.add_argument('--file_type', type=str, default="jpg", 
                        help='Type of image files to process ("png", "jpg", "png,jpg", "all")')
    parser.add_argument(
        '--rename-files',
        type=str2bool,
        default=True,
        help='Rename files by adding suffix before extension (true/false)'
    )
    parser.add_argument('--attributes', type=str, default="redhair_brownskin",
                        help='attributes to detect, e.g. redhair_brownskin, redhair, etc.')
    parser.add_argument('--rename_suffix', type=str, default=None,
                        help='Suffix to add when renaming files (default: "_red")')
    parser.add_argument('--max_memory', type=int, default=22, 
                        help='Maximum memory allocation per GPU in GiB (default: 22)')
    parser.add_argument('--stop_at', type=str, default="020000",
                        help='Base image number (e.g., "077000") to stop processing at. Processing stops *before* encountering any file whose base number is greater than or equal to this value.')
    
    args = parser.parse_args()

    if args.rename_suffix is None:
        args.rename_suffix = "_" + args.attributes

    main(
       args.dir,
       args.verbose,
       args.file_type,
       args.rename_files,
       args.attributes,
       args.rename_suffix,
       args.max_memory,
       args.stop_at
    )