#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set up the environment
import sys
sys.path.append('/people/cs/o/oxx220000/HiBug')

from filter import main

# Test with one image to verify all attributes are detected
if __name__ == "__main__":
    print("Testing multi-attribute detection...")
    
    # Run with redhair_brownskin_sademotion - should detect all 3 attributes
    results, feature_mapping = main(
        directory_path="exampleData/redhair_brownskin_sademotion",
        verbose=True,
        attributes="redhair_brownskin_sademotion", 
        lvlm="llava-7b",
        stop_at=1
    )
    
    print("\n=== Results for first image ===")
    if results:
        first_result = next(iter(results.values()))
        print("Detected attributes:")
        for key, value in first_result.items():
            print(f"  {key}: {value}")
        
        expected_keys = ['red_hair', 'brown_skin', 'sad_emotion', 'wearing_lipstick']
        missing_keys = [k for k in expected_keys if k not in first_result]
        if missing_keys:
            print(f"\nMissing attributes: {missing_keys}")
        else:
            print("\n✅ All expected attributes detected!")
    else:
        print("No results found!")
