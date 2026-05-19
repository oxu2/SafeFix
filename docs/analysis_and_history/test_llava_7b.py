#!/usr/bin/env python3
"""Test LLaVA 7B model availability"""

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

def test_llava_7b():
    """Test if LLaVA 7B model can be loaded"""
    print("Testing LLaVA 7B model availability...")
    
    model_paths_to_try = [
        "llava-hf/llava-1.5-7b-hf",
        "liuhaotian/llava-v1.5-7b", 
        "llava-hf/llava-v1.5-7b"
    ]
    
    for model_path in model_paths_to_try:
        try:
            print(f"\nTrying: {model_path}")
            
            # Try to load just the config first
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            print(f"✓ Config loaded for {model_path}")
            
            # Try to load processor
            processor = AutoProcessor.from_pretrained(model_path)
            print(f"✓ Processor loaded for {model_path}")
            
            print(f"✅ SUCCESS: {model_path} is available!")
            return model_path
            
        except Exception as e:
            print(f"❌ Failed {model_path}: {e}")
            continue
    
    print("\n❌ No working 7B model path found")
    return None

if __name__ == "__main__":
    working_path = test_llava_7b()
    if working_path:
        print(f"\n🎉 Use this path in your code: {working_path}")
    else:
        print("\n💡 Consider using the 13B model or downloading a 7B model manually")
