#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor

# Load LLaVA model for testing
print("Loading LLaVA 7B model for debugging...")
model_path = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = LlavaProcessor.from_pretrained(model_path, use_fast=True)

# Test with a sample image
test_image = "data/imagenet_pinkcolor_fabrictexture/backpack/n02769748_1032.JPEG"
print(f"Testing with image: {test_image}")

if not os.path.exists(test_image):
    print(f"Image not found: {test_image}")
    exit(1)

image = Image.open(test_image)
print(f"Image loaded: {image.size}, {image.mode}")

# Test the ImageNet prompt construction
tags = ['pink', 'fabric']
attributes = ['color', 'texture']
class_name = 'backpack'

# Build the ImageNet-style prompt
questions = [f"Does the {class_name} have {t} {a}?" for t, a in zip(tags, attributes)]
prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."

print(f"\nPrompt being sent to LLaVA:")
print(f"'{prompt}'")

# Test the inference manually
llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
print(f"\nFull LLaVA prompt:")
print(f"'{llava_prompt}'")

try:
    inputs = processor(images=image, text=llava_prompt, return_tensors="pt").to('cuda')
    print("Input processed successfully")
    
    output_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    print(f"\nLLaVA Response: '{response}'")
    
    # Try to parse the response
    numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
    print(f"Parsed numbers: {numbers}")
    
    if not numbers:
        print("⚠️  No numbers found in response - this is the problem!")
        print("Let's try a simpler prompt...")
        
        simple_prompt = f"USER: <image>\nLook at this {class_name}. Is it pink? Answer 1 for yes, 0 for no.\nASSISTANT:"
        inputs2 = processor(images=image, text=simple_prompt, return_tensors="pt").to('cuda')
        output_ids2 = model.generate(**inputs2, max_new_tokens=50)
        generated_ids2 = [output_ids2[len(input_ids):] for input_ids, output_ids in zip(inputs2.input_ids, output_ids2)]
        simple_response = processor.batch_decode(generated_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"Simple response: '{simple_response}'")
        
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()

# Test with a sample image
test_image = "data/imagenet_pinkcolor_fabrictexture/backpack/n02769748_1032.JPEG"
print(f"Testing with image: {test_image}")

if not os.path.exists(test_image):
    print(f"Image not found: {test_image}")

# Test the ImageNet prompt construction
tags = ['pink', 'fabric']
attributes = ['color', 'texture']
class_name = 'backpack'

# Build the ImageNet-style prompt
questions = [f"Does the {class_name} have {t} {a}?" for t, a in zip(tags, attributes)]
prompt = " ".join(questions) + " For each question, only answer with 1 (yes) or 0 (no). Provide answers separated by spaces."

print(f"\nPrompt being sent to LLaVA:")
print(f"'{prompt}'")

# Test the inference
try:
    response = inference(test_image, prompt, verbose=True)
    print(f"\nLLaVA Response: '{response}'")
    
    # Try to parse the response
    numbers = [int(s) for s in response.replace(',', ' ').split() if s.isdigit()]
    print(f"Parsed numbers: {numbers}")
    
    if not numbers:
        print("⚠️  No numbers found in response - this is the problem!")
        print("Let's try a simpler prompt...")
        
        simple_prompt = f"Look at this {class_name}. Is it pink? Answer 1 for yes, 0 for no."
        simple_response = inference(test_image, simple_prompt, verbose=True)
        print(f"Simple response: '{simple_response}'")
        
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()
