 # Global flag for presence inclusion
INCLUDE_PRESENCE = False
import json
import random
import sys
import os
import datetime  # Add this import to support timestamps
import shutil
import time
import psutil
import GPUtil
from typing import List
from utils.defines import *
# # Check if running in the correct conda environment
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

# Set environment variables to prevent tokenizer warnings and improve memory management
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

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
    global current_lvlm
    
    image = Image.open(image_path)
    
    if current_lvlm == "qwen":
        # Qwen format
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
    
    elif current_lvlm.startswith("llava"):
        # LLaVA v1.5 format - needs specific prompt template
        prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to('cuda')
        if verbose:
            print("LLaVA prompt:", prompt_text)
    
    else:
        raise ValueError(f"Unsupported LVLM: {current_lvlm}")

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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

# Global variable to track current LVLM type
current_lvlm = "qwen"

# Resource monitoring class
class ResourceMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.inference_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.model_size_mb = 0
        
    def start_monitoring(self):
        """Start monitoring resources"""
        self.start_time = time.time()
        
    def stop_monitoring(self):
        """Stop monitoring and calculate final metrics"""
        self.end_time = time.time()
        
    def record_inference_time(self, inference_time):
        """Record individual inference time"""
        self.inference_times.append(inference_time)
        
    def record_memory_usage(self):
        """Record current memory usage"""
        # CPU memory
        cpu_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self.memory_usage.append(cpu_memory)
        
        # GPU memory
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = sum(gpu.memoryUsed for gpu in gpus) / 1024  # GB
                self.gpu_memory_usage.append(gpu_memory)
        except:
            pass
            
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        self.model_size_mb = (param_size + buffer_size) / (1024**2)
        return self.model_size_mb
        
    def get_statistics(self, num_images):
        """Get comprehensive resource statistics"""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        stats = {
            'total_processing_time_seconds': total_time,
            'total_processing_time_minutes': total_time / 60,
            'total_processing_time_hours': total_time / 3600,
            'num_images_processed': num_images,
            'avg_time_per_image_seconds': total_time / num_images if num_images > 0 else 0,
            'model_size_mb': self.model_size_mb,
            'model_size_gb': self.model_size_mb / 1024,
            'inference_statistics': {
                'total_inferences': len(self.inference_times),
                'avg_inference_time_seconds': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
                'min_inference_time_seconds': min(self.inference_times) if self.inference_times else 0,
                'max_inference_time_seconds': max(self.inference_times) if self.inference_times else 0,
                'total_inference_time_seconds': sum(self.inference_times)
            },
            'memory_statistics': {
                'avg_cpu_memory_gb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                'max_cpu_memory_gb': max(self.memory_usage) if self.memory_usage else 0,
                'avg_gpu_memory_gb': sum(self.gpu_memory_usage) / len(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
                'max_gpu_memory_gb': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
            },
            'estimated_cost_per_1k_images': {
                'time_seconds': (total_time / num_images * 1000) if num_images > 0 else 0,
                'time_minutes': (total_time / num_images * 1000 / 60) if num_images > 0 else 0,
                'time_hours': (total_time / num_images * 1000 / 3600) if num_images > 0 else 0
            }
        }
        
        return stats
        
    def print_statistics(self, num_images):
        """Print formatted resource statistics"""
        stats = self.get_statistics(num_images)
        
        print("\n" + "="*60)
        print("           RESOURCE USAGE STATISTICS")
        print("="*60)
        
        print(f"📊 Processing Overview:")
        print(f"   • Total images processed: {stats['num_images_processed']:,}")
        print(f"   • Total processing time: {stats['total_processing_time_minutes']:.2f} minutes ({stats['total_processing_time_hours']:.2f} hours)")
        print(f"   • Average time per image: {stats['avg_time_per_image_seconds']:.3f} seconds")
        
        print(f"\n🤖 Model Information:")
        print(f"   • Model size: {stats['model_size_mb']:.1f} MB ({stats['model_size_gb']:.2f} GB)")
        
        print(f"\n⚡ Inference Performance:")
        inf_stats = stats['inference_statistics']
        print(f"   • Total inferences: {inf_stats['total_inferences']:,}")
        print(f"   • Average inference time: {inf_stats['avg_inference_time_seconds']:.3f} seconds")
        print(f"   • Min inference time: {inf_stats['min_inference_time_seconds']:.3f} seconds")
        print(f"   • Max inference time: {inf_stats['max_inference_time_seconds']:.3f} seconds")
        print(f"   • Total inference time: {inf_stats['total_inference_time_seconds']:.2f} seconds")
        
        print(f"\n💾 Memory Usage:")
        mem_stats = stats['memory_statistics']
        print(f"   • Average CPU memory: {mem_stats['avg_cpu_memory_gb']:.2f} GB")
        print(f"   • Peak CPU memory: {mem_stats['max_cpu_memory_gb']:.2f} GB")
        if mem_stats['avg_gpu_memory_gb'] > 0:
            print(f"   • Average GPU memory: {mem_stats['avg_gpu_memory_gb']:.2f} GB")
            print(f"   • Peak GPU memory: {mem_stats['max_gpu_memory_gb']:.2f} GB")
        
        print(f"\n📈 Estimated Cost for 1,000 Images:")
        cost_1k = stats['estimated_cost_per_1k_images']
        print(f"   • Processing time: {cost_1k['time_minutes']:.2f} minutes ({cost_1k['time_hours']:.2f} hours)")
        print(f"   • Time per image: {cost_1k['time_seconds']/1000:.3f} seconds")
        
        # Calculate throughput
        if stats['total_processing_time_seconds'] > 0:
            throughput = num_images / stats['total_processing_time_seconds']
            print(f"\n🚀 Throughput:")
            print(f"   • Images per second: {throughput:.2f}")
            print(f"   • Images per minute: {throughput * 60:.1f}")
            print(f"   • Images per hour: {throughput * 3600:.0f}")
        
        print("="*60)

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
    # Format the memory string - reduce default memory allocation
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

def process_image(image_path, model, processor, tags: List[str], attributes: List[str], verbose=False, resource_monitor=None):
    """Process a single image and return the binary results for both features."""
    inference_start = time.time()
    
    try:
        # Special handling for ImageNet dataset
        if DATASET == 'imagenet':
            # Derive class name from parent directory
            class_name = os.path.basename(os.path.dirname(image_path))
            
            # Build questions with improved prompts for LLaVA
            questions = []
            for t, a in zip(tags, attributes):
                if a == 'texture' and current_lvlm.startswith("llava"):
                    # More specific texture prompts for LLaVA
                    if t == 'fabric':
                        questions.append(f"Does the {class_name} appear to be made of soft, flexible material like fabric, cloth, canvas, nylon, or textile (not hard materials like plastic or metal)?")
                    elif t == 'metal':
                        questions.append(f"Is the {class_name} made of metal or metallic material?")
                    elif t == 'wood':
                        questions.append(f"Is the {class_name} made of wood or wooden material?")
                    elif t == 'plastic':
                        questions.append(f"Is the {class_name} made of plastic or synthetic material?")
                    else:
                        questions.append(f"Does the {class_name} have a {t} texture or is made of {t} material?")
                elif a == 'color' and current_lvlm.startswith("llava"):
                    # More specific color prompts for LLaVA
                    questions.append(f"Is the {class_name} {t} in color or does it have {t} coloring?")
                else:
                    # Default format
                    questions.append(f"Does the {class_name} have {t} {a}?")
            
            # Add presence check
            questions.append(f"Is there a {class_name} in the picture?")
            
            # Different prompt format for LLaVA vs other models
            if current_lvlm.startswith("llava"):
                if len(questions) > 1:
                    prompt = "Look carefully at this image. Answer each question with YES or NO, separated by spaces:\n" + \
                            "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]) + \
                            "\n\nFor material questions: Examine the surface texture, flexibility, and visual properties carefully. If uncertain about fabric/textile material, lean towards YES."
                else:
                    prompt = f"Look carefully at this image. {questions[0]}\n\nFor material questions: Examine the surface texture and visual properties. If uncertain about fabric/textile material, lean towards YES.\n\nAnswer: YES or NO"
            else:
                prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."
            
            response = inference(image_path, prompt, verbose=verbose)
            try:
                # Parse numeric answers
                nums = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
                
                # If no numbers found, try to parse text responses (for LLaVA)
                if not nums and current_lvlm.startswith("llava"):
                    # Convert text responses to numbers
                    response_lower = response.lower()
                    nums = []
                    # Split by common separators and analyze each part
                    parts = response.replace(',', ' ').replace('.', ' ').split()
                    for part in parts:
                        part_lower = part.lower().strip('.,!?')
                        if part_lower in ['yes', 'true', '1']:
                            nums.append(1)
                        elif part_lower in ['no', 'false', '0']:
                            nums.append(0)
                    
                    # If still no clear answers, try pattern matching for sentences
                    if not nums:
                        # For each expected question, look for yes/no patterns
                        sentences = response.split('.')
                        for sentence in sentences[:len(tags)]:  # Only look at as many sentences as we have questions
                            sentence_lower = sentence.lower()
                            if 'yes' in sentence_lower or 'true' in sentence_lower or 'is ' in sentence_lower:
                                nums.append(1)
                            elif 'no' in sentence_lower or 'false' in sentence_lower or 'not' in sentence_lower:
                                nums.append(0)
                    
                    # If we still don't have enough answers, pad with defaults
                    while len(nums) < len(tags):
                        nums.append(0)
                
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
            
            # Extract the numbers from the response
            try:
                numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
                
                # If no numbers found, try to parse text responses (for LLaVA)
                if not numbers and current_lvlm.startswith("llava"):
                    # Convert text responses to numbers
                    response_lower = response.lower()
                    numbers = []
                    # Split by common separators and analyze each part
                    parts = response.replace(',', ' ').replace('.', ' ').split()
                    for part in parts:
                        part_lower = part.lower().strip('.,!?')
                        if part_lower in ['yes', 'true', '1']:
                            numbers.append(1)
                        elif part_lower in ['no', 'false', '0']:
                            numbers.append(0)
                    
                    # If still no clear answers, try pattern matching for sentences
                    if not numbers:
                        # For each expected question, look for yes/no patterns
                        sentences = response.split('.')
                        for sentence in sentences[:len(tags)]:  # Only look at as many sentences as we have questions
                            sentence_lower = sentence.lower()
                            if 'yes' in sentence_lower or 'true' in sentence_lower or 'is ' in sentence_lower:
                                numbers.append(1)
                            elif 'no' in sentence_lower or 'false' in sentence_lower or 'not' in sentence_lower:
                                numbers.append(0)
                    
                    # If we still don't have enough answers, pad with defaults
                    while len(numbers) < len(tags):
                        numbers.append(0)
                
                results = {}
                # Map each tag-attribute pair to its corresponding number
                for i, (t, a) in enumerate(zip(tags, attributes)):
                    key = f"{t}_{a.replace(' ', '_')}"
                    if i < len(numbers):
                        results[key] = numbers[i]
                    else:
                        # If no number available, default based on attribute type
                        if a == "emotion":
                            results[key] = 1  # Default to positive for emotion
                        else:
                            results[key] = 0  # Default to negative for other attributes
            except Exception as e:
                print(f"Warning: Error parsing response '{response}' for {image_path}: {e}")
                # If parsing fails, create default results for all attributes
                results = {}
                for t, a in zip(tags, attributes):
                    key = f"{t}_{a.replace(' ', '_')}"
                    if a == "emotion":
                        results[key] = 1  # Default to positive for emotion
                    else:
                        results[key] = 0  # Default to negative for other attributes
        
        # Record inference time
        inference_time = time.time() - inference_start
        if resource_monitor:
            resource_monitor.record_inference_time(inference_time)
        
        # Clear GPU cache periodically to prevent memory buildup
        if torch.cuda.is_available() and hasattr(process_image, '_call_count'):
            process_image._call_count += 1
            if process_image._call_count % 50 == 0:  # Clear every 50 images
                torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            process_image._call_count = 1
        
        return results
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return default results on any error
        if DATASET == 'imagenet':
            results = {f"{t}{a}": 0 for t, a in zip(tags, attributes)}
            if INCLUDE_PRESENCE:
                results["presence"] = 0
        else:
            results = {}
            for t, a in zip(tags, attributes):
                if a == "emotion":
                    results[f"{t}_{a.replace(' ', '_')}"] = 1
                else:
                    results[f"{t}_{a.replace(' ', '_')}"] = 0
        return results

def process_directory(directory_path, output_json_path, tags, attributes, verbose=False, file_type="png", stop_at: int = 15000, stop_suffix=None, resource_monitor=None):
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
        result = process_image(image_path, model, processor, tags, attributes, verbose=verbose, resource_monitor=resource_monitor)
        if result:
            results[filename] = result

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Record memory usage periodically
        if resource_monitor and (idx + 1) % 100 == 0:
            resource_monitor.record_memory_usage()

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
        
        # Parse attributes to get individual attribute names
        attr_list = attributes.split('_')
        individual_attrs = []
        for attr in attr_list:
            if attr.endswith('color'):
                individual_attrs.append(attr)
            elif attr.endswith('texture'):
                individual_attrs.append(attr)
            else:
                individual_attrs.append(attr)
        
        # Build mapping: include only items where ALL attributes == 1
        feature_mapping = {}
        for filename, result in results.items():
            # Check if all individual attributes are 1
            if all(result.get(attr, 0) == 1 for attr in individual_attrs):
                # Include all attribute values in the mapping
                mapping_entry = {attr: result.get(attr, 0) for attr in individual_attrs}
                feature_mapping[filename] = mapping_entry
        
        # Save and return
        save_dir = os.path.dirname(feasible_json)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, os.path.basename(feasible_json))
        with open(out_path, 'w') as f:
            json.dump(feature_mapping, f, indent=4)
        return feature_mapping
    
    # Original logic for CelebA
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
    # Derive feature keys
    keys = [f"{t}_{a}" for t, a in zip(tags, attrs)]
    # Retain these attribute keys for filtering; ensure we always include lipstick feature in output
    attr_keys = keys[:]
    output_keys = attr_keys + ["wearing_lipstick"]

    feature_mapping = {}
    for filename, result in results.items():
        # Require all specified attribute keys be positive
        if all(result.get(key, 0) == 1 for key in attr_keys):
            # Build mapping entry: include attribute keys and lipstick feature
            mapping_entry = {key: result.get(key, 0) for key in attr_keys}
            # Always include lipstick result (0 or 1)
            mapping_entry["wearing_lipstick"] = result.get("wearing_lipstick", 0)
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
         feasible_json: str = None,
         lvlm: str = "qwen"):
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
    - lvlm: Large vision-language model to use ("qwen" or "llava")
    
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

    # Initialize resource monitor
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring()

    # Load model and processor here, after configuring memory
    global model, processor, current_lvlm
    
    # Set global variable for current LVLM
    current_lvlm = lvlm
    
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Choose model path and classes based on lvlm parameter
    if lvlm == "qwen":
        model_path_to_use = "Qwen/Qwen2.5-VL-7B-Instruct"
        model_class = Qwen2_5_VLForConditionalGeneration
        processor_class = AutoProcessor
    elif lvlm == "llava-7b":
        model_path_to_use = "llava-hf/llava-1.5-7b-hf"
        model_class = LlavaForConditionalGeneration
        processor_class = LlavaProcessor
    elif lvlm == "llava-13b":
        model_path_to_use = "liuhaotian/llava-v1.5-13b"
        model_class = LlavaForConditionalGeneration
        processor_class = LlavaProcessor
    else:
        raise ValueError(f"Unsupported LVLM: {lvlm}")
    
    print(f"Loading {lvlm.upper()} model ({model_path_to_use}) with {max_memory_per_gpu}GB memory limit per GPU...")
    
    if lvlm.startswith("llava"):
        # Use more conservative loading for LLaVA
        print("Using conservative loading strategy for LLaVA...")
        model = model_class.from_pretrained(
            model_path_to_use,
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better compatibility
            device_map="auto",
            max_memory=device_memory,
            low_cpu_mem_usage=True,
            local_files_only=False,  # Allow downloading if needed
        )
    else:
        # Original Qwen loading
        model = model_class.from_pretrained(
            model_path_to_use,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=device_memory,
            low_cpu_mem_usage=True
        )
    
    processor = processor_class.from_pretrained(model_path_to_use, use_fast=True)
    
    # Calculate model size
    resource_monitor.get_model_size(model)
    print(f"Model loaded. Size: {resource_monitor.model_size_mb:.1f} MB ({resource_monitor.model_size_mb/1024:.2f} GB)")
    
    # Record initial memory usage
    resource_monitor.record_memory_usage()
    
    # Clear cache again after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
           stop_suffix=rename_suffix,
           resource_monitor=resource_monitor
       )
    
    # Stop monitoring and calculate statistics
    resource_monitor.stop_monitoring()

    print(f"Image processing using {args.lvlm} model complete.")

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
    
    # Print resource usage statistics
    resource_monitor.print_statistics(processed_count)
    
    # Save resource statistics to JSON file
    stats = resource_monitor.get_statistics(processed_count)
    stats_filename = f"{attributes}_resource_stats_{datetime.datetime.now().strftime('%m%d%H%M')}.json"
    stats_path = os.path.join("save/json", stats_filename)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n📊 Resource statistics saved to: {stats_path}")
    
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
    
    def str2bool(v):
        """Convert string to boolean value"""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    parser.add_argument('--max_memory', type=int, default=23, 
                        help='Maximum memory allocation per GPU in GiB (default: 23)')
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
        default='celeba',
        help='Which dataset to use: imagenet (default) or celeba'
    )
    parser.add_argument(
        '--include_presence',
        type=str2bool,
        default=False,
        help='If true, include the <class>_presence field in ImageNet results (default: False)'
    )
    parser.add_argument(
        '--lvlm',
        choices=['qwen', 'llava-7b', 'llava-13b'],
        default='qwen',
        help='Which large vision-language model to use: qwen (default), llava-7b, or llava-13b'
    )
    
args = parser.parse_args()

# Whether to include presence check in results
INCLUDE_PRESENCE = args.include_presence

if not args.image_dir:
    # Determine default directory based on dataset and attributes
    if args.dataset == 'celeba':
        default_dir = os.path.join("exampleData", args.attributes)
    else:  # imagenet
        default_dir = f"data/imagenet_{args.attributes}"
    # Use default if it exists, otherwise try fallback paths
    if os.path.exists(default_dir):
        args.image_dir = default_dir
    else:
        possible_dirs = [
            f"/home/jovyan/ouser/{args.dataset}_{args.attributes}",
            f"/data/oxx220000/{args.dataset}_{args.attributes}"
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
   args.feasible_json,
   args.lvlm
)