#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Quick test for LLaVA response format
import sys
sys.path.append('/people/cs/o/oxx220000/safefix')

from PIL import Image

# Test image path
test_image = "data/imagenet_pinkcolor_fabrictexture/backpack/n02769748_1032.JPEG"

# Load image and display info
image = Image.open(test_image)
print(f"Image: {test_image}")
print(f"Size: {image.size}")
print(f"Mode: {image.mode}")

# Test different prompt styles for LLaVA
test_prompts = [
    "Is this backpack pink? Answer only 1 for yes or 0 for no.",
    "Look at this image. Does the backpack have pink color? Reply with 1 (yes) or 0 (no).",
    "Answer with numbers only: Is the backpack pink? (1=yes, 0=no)",
    "Does the backpack have pink color? Does the backpack have fabric texture? For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."
]

# We'll manually test the LLaVA format here
print("Testing LLaVA prompt formats...")

# Import after setting CUDA device
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

print("Loading LLaVA...")
model_path = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_path)

for i, prompt in enumerate(test_prompts):
    print(f"\n--- Test {i+1} ---")
    print(f"Prompt: {prompt}")
    
    # Format for LLaVA
    llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    
    try:
        inputs = processor(images=image, text=llava_prompt, return_tensors="pt").to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        print(f"Raw response: '{response}'")
        
        # Try to extract numbers
        numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
        print(f"Extracted numbers: {numbers}")
        
    except Exception as e:
        print(f"Error: {e}")

print("\nTesting completed.")
