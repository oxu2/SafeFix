import json
import random
import sys
import os

# Check if running in the correct conda environment
def ensure_conda_env(env_name):
    """Ensure script runs in the specified conda environment"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != env_name:
        print(f"Switching to conda environment '{env_name}' (current: {current_env or 'None'})")
        # Construct the conda run command
        conda_path = os.environ.get('CONDA_EXE', 'conda')
        script_path = os.path.abspath(sys.argv[0])
        args = ' '.join(sys.argv[1:])
        cmd = f"{conda_path} run -n {env_name} python {script_path} {args}"
        print(f"Executing: {cmd}")
        os.system(cmd)
        sys.exit(0)
    else:
        print(f"Running in correct conda environment: {env_name}")

# Ensure we're running in the 'ad' environment
ensure_conda_env('ad')

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
def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
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
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]
    
    



#  base 64 编码格式
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

def process_image(image_path, model, processor):
    """Process a single image and return the binary result."""
    prompt = "Is the person in this picture in yellow hair?, only 1 or 0 is accepted."
    response = inference(image_path, prompt)
    # Extract just the number from the response
    try:
        result = int(response.strip())
        return result
    except:
        print(f"Warning: Could not parse response '{response}' for {image_path}")
        return None

def process_directory(directory_path, output_json_path="results.json"):
    """Process all PNG images in directory and save results to JSON."""
    results = {}
    
    # Get all PNG files in directory that start with '0'
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.png') and f.startswith('0')]
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        # Get filename without extension
        filename = os.path.splitext(image_file)[0]
        
        print(f"Processing {image_file}...")
        result = process_image(image_path, model, processor)
        results[filename] = result
        
        # Save after each image in case of interruption
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_json_path}")
    return results

# Add function to extract colors from positive results
def create_color_mapping(results_json, output_color_json="colors.json"):
    """
    Create a JSON file mapping filenames to colors for images with result=1
    Example: "007_yellow": "yellow"
    Only include filenames that start with digit 0
    """
    if isinstance(results_json, str):
        # If results_json is a file path
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        # If results_json is already a dictionary
        results = results_json
        
    color_mapping = {}
    
    for filename, result in results.items():
        # Check if the filename starts with digit 0 and has result=1
        if filename.startswith('0') and result == 1:
            # Extract color from filename
            parts = filename.split('_')
            if len(parts) > 1:
                color = parts[-1]  # Get the last part after underscore
                color_mapping[filename] = color
    
    # Save the color mapping to a JSON file
    with open(output_color_json, 'w') as f:
        json.dump(color_mapping, f, indent=4)
    
    print(f"Color mapping for positive results (filenames starting with 0) saved to {output_color_json}")
    return color_mapping

# Replace the single image processing with directory processing
directory_path = "/people/cs/o/oxx220000/HiBug/pictures"
results = process_directory(directory_path)
print("Final results:", results)

# Create color mapping for positive results
color_mapping = create_color_mapping(results, "feasible.json")
print("Color mapping:", color_mapping)