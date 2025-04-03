import json
import random
import sys
import os

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
def configure_gpu_memory(available_gpus):
    """Configure memory allocation based on available GPUs"""
    max_memory = {}
    for i, gpu_id in enumerate(available_gpus):
        # Use local device ID (0-indexed)
        max_memory[i] = "22GiB"  # Increased from 20GiB to 22GiB
    
    # If no GPUs specified, default to CPU
    if not max_memory:
        print("Warning: No GPUs available. Using CPU.")
        return "cpu"
    
    print(f"Using {len(max_memory)} GPUs with 22GiB memory allocation each")
    return max_memory

# Get available GPUs and configure memory
available_gpus = get_available_gpus()
device_memory = configure_gpu_memory(available_gpus)

# Load model with dynamic device configuration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=device_memory
)

# Use fast processor to avoid warnings
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

def process_image(image_path, model, processor, tag="yellow", attribute="hair color", verbose=False):
    """Process a single image and return the binary results for both features."""
    # Customize prompt based on tag and attribute
    prompt = f"Does the person in this picture have {tag} {attribute}? Does the person in this picture wear lipstick? For each question, only answer with 1 (yes) or 0 (no). Format your answer as two numbers separated by a space: [tag_attribute_result] [lipstick_result]"
    response = inference(image_path, prompt, verbose=verbose)
    
    # Extract the two numbers from the response
    try:
        # Split the response and find the first two integers
        numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
        if len(numbers) >= 2:
            tag_attribute_result = numbers[0]
            lipstick_result = numbers[1]
            attribute_key = attribute.replace(" ", "_")  # Convert spaces to underscores for keys
            return {f"{tag}_{attribute_key}": tag_attribute_result, "wearing_lipstick": lipstick_result}
        else:
            print(f"Warning: Could not find two numbers in response '{response}' for {image_path}")
            attribute_key = attribute.replace(" ", "_")
            return {f"{tag}_{attribute_key}": None, "wearing_lipstick": None}
    except Exception as e:
        print(f"Warning: Error parsing response '{response}' for {image_path}: {e}")
        attribute_key = attribute.replace(" ", "_")
        return {f"{tag}_{attribute_key}": None, "wearing_lipstick": None}

def process_directory(directory_path, output_json_path="results.json", tag="yellow", attribute="hair color", verbose=False, file_type="png"):
    """Process all images in directory and save results to JSON."""
    results = {}
    
    # Get image files based on the specified file type
    if file_type.lower() == "all":
        # Process both PNG and JPG files
        image_files = [f for f in os.listdir(directory_path) if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]
    else:
        # Process only files of the specified type
        extensions = ['.' + ext for ext in file_type.lower().split(',')]
        image_files = [f for f in os.listdir(directory_path) if any(f.endswith(ext) for ext in extensions)]
    
    print(f"Found {len(image_files)} images with file type(s): {file_type}")
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        # Get filename without extension
        filename = os.path.splitext(image_file)[0]
        
        # print(f"Processing {image_file}...")
        result = process_image(image_path, model, processor, tag, attribute, verbose=verbose)
        results[filename] = result
        
        # Save after each image in case of interruption
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
    # print(f"Results saved to {output_json_path}")
    return results

# Update function to extract features from results with configurable filtering
def create_feature_mapping(results_json, output_feature_json="features.json", tag="yellow", attribute="hair color"):
    """
    Create a JSON file mapping filenames to features for images with specific attributes
    Example: {"example_image": {"tag": "yellow", "attribute": "hair color", "wearing_lipstick": 0}}
    Only include entries that have tag_attribute=1
    
    Parameters:
    - results_json: Path to results JSON or dictionary of results
    - output_feature_json: Path to save the filtered results
    - tag: The tag to filter for (e.g., "yellow")
    - attribute: The attribute to filter for (e.g., "hair color")
    """
    if isinstance(results_json, str):
        # If results_json is a file path
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        # If results_json is already a dictionary
        results = results_json
        
    feature_mapping = {}
    
    # Construct the key based on tag and attribute
    attribute_key = attribute.replace(" ", "_")
    tag_attribute_key = f"{tag}_{attribute_key}"
    
    for filename, result in results.items():
        # Check if the result contains the attribute key
        tag_attribute_result = result.get(tag_attribute_key)
        wearing_lipstick = result.get("wearing_lipstick")
        
        # Only include entries where tag_attribute_result is 1
        if tag_attribute_result == 1:
            # Store all features with new structure
            feature_mapping[filename] = {
                "tag": tag,
                "attribute": attribute,
                "wearing_lipstick": wearing_lipstick
            }
    
    # Save the feature mapping to a JSON file
    with open(output_feature_json, 'w') as f:
        json.dump(feature_mapping, f, indent=4)
    
    # print(f"Feature mapping saved to {output_feature_json} (filtered for {tag}_{attribute_key}=1)")
    return feature_mapping

def main(directory_path="/people/cs/o/oxx220000/HiBug/pictures", 
         tag="red", 
         attribute="hair color",
         verbose=False,
         file_type="png"):
    """
    Main function to process images and create feature mapping.
    
    Parameters:
    - directory_path: Path to directory containing images to process
    - tag: The visual feature tag to detect (e.g., "yellow")
    - attribute: The attribute to detect (e.g., "hair color")
    - verbose: Whether to display detailed processing information
    - file_type: Type of image files to process ("png", "jpg", "png,jpg", "all")
    
    Returns:
    - Tuple of (results, feature_mapping)
    """
    # Generate filenames based on tag and attribute
    attribute_key = attribute.replace(" ", "_")
    results_json_path = f"{tag}_{attribute_key}_results.json"
    output_feature_json = f"{tag}_{attribute_key}_feasible.json"
    
    print(f"Starting image processing for {tag} {attribute}...")
    # print(f"Processing directory: {directory_path}")
    # print(f"Processing file type(s): {file_type}")
    # print(f"Saving results to: {results_json_path}")
    # print(f"Saving filtered features to: {output_feature_json}")
    
    # Process all images in directory
    results = process_directory(directory_path, results_json_path, tag, attribute, verbose=verbose, file_type=file_type)
    print("Image processing using Qwen VL model complete.")
    
    # Create feature mapping for images with specified tag and attribute
    feature_mapping = create_feature_mapping(results, output_feature_json, tag, attribute)
    print(f"Feature mapping and filter complete. Found {len(feature_mapping)} images with {tag} {attribute}.")
    
    return results, feature_mapping

# Execute the main function with default parameters
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    import shlex
    
    parser = argparse.ArgumentParser(description='Process images for specific attributes')
    parser.add_argument('--dir', type=str, default="/people/cs/o/oxx220000/HiBug/pictures", 
                        help='Directory containing images')
    parser.add_argument('--tag', type=str, default="red", 
                        help='Tag to detect (e.g., "yellow", "red")')
    parser.add_argument('--attribute', type=str, default="hair color", 
                        help='Attribute to detect (e.g., "hair color", "eyes")')
    parser.add_argument('--verbose', action='store_true', 
                        help='Display detailed processing information')
    parser.add_argument('--file-type', type=str, default="png", 
                        help='Type of image files to process ("png", "jpg", "png,jpg", "all")')
    
    args = parser.parse_args()

    
    main(args.dir, args.tag, args.attribute, args.verbose, args.file_type)