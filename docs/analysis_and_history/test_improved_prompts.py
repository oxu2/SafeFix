#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor

# Load LLaVA model for testing
print("Loading LLaVA 7B model for testing improved prompts...")
model_path = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = LlavaProcessor.from_pretrained(model_path, use_fast=True)

# Test image
test_image = "data/imagenet_pinkcolor_fabrictexture/backpack/n02769748_1032.JPEG"
image = Image.open(test_image)
print(f"Testing with image: {test_image}")

# Test the new improved prompts
class_name = 'backpack'

# Old prompt format
old_prompt = "Does the backpack have pink color? Does the backpack have fabric texture? For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."

# New improved prompt format
new_prompts = [
    "Is the backpack pink in color or does it have pink coloring?",
    "Is the backpack made of fabric, cloth, or textile material?"
]
new_prompt = " ".join(new_prompts) + " Answer each question with YES or NO, separated by spaces."

print(f"\n=== OLD PROMPT ===")
print(f"'{old_prompt}'")

print(f"\n=== NEW PROMPT ===")
print(f"'{new_prompt}'")

# Test both prompts
for prompt_name, prompt in [("OLD", old_prompt), ("NEW", new_prompt)]:
    print(f"\n--- Testing {prompt_name} Prompt ---")
    
    llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    
    try:
        inputs = processor(images=image, text=llava_prompt, return_tensors="pt").to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        print(f"Response: '{response}'")
        
        # Test parsing
        numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
        if not numbers:
            # Try text parsing
            response_lower = response.lower()
            parsed_nums = []
            parts = response.replace(',', ' ').replace('.', ' ').split()
            for part in parts:
                part_lower = part.lower().strip('.,!?')
                if part_lower in ['yes', 'true', '1']:
                    parsed_nums.append(1)
                elif part_lower in ['no', 'false', '0']:
                    parsed_nums.append(0)
            
            print(f"Parsed as: {parsed_nums}")
        else:
            print(f"Direct numbers: {numbers}")
            
    except Exception as e:
        print(f"Error: {e}")

print("\nTest completed.")
