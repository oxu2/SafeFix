#!/usr/bin/env python3
"""
Test script for Active Learning integration in alpred.py
This script tests the integration of Core-set Active Learning strategy
"""

import subprocess
import sys
import os

def run_test(test_name, command):
    """Run a test command and report results"""
    print(f"\n{'='*60}")
    print(f"Running test: {test_name}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED (return code: {result.returncode})")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMED OUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return False

def main():
    print("Testing Active Learning Integration in alpred.py")
    print("=" * 60)
    
    # Test 1: ImageNet with Active Learning enabled
    test1_cmd = """python alpred.py \
        --dataset imagenet \
        --model resnet \
        --use_active_learning True \
        --sub_ratio 0.5 \
        --al_initial_labeled 500 \
        --al_embedding_dim 512 \
        --num_epochs 1 \
        --batch_size 16 \
        --num_added_img 100 \
        --verbose True \
        --attributes test_al"""
    
    success1 = run_test("ImageNet with Active Learning", test1_cmd)
    
    # Test 2: ImageNet with random sampling (baseline)
    test2_cmd = """python alpred.py \
        --dataset imagenet \
        --model resnet \
        --use_active_learning False \
        --sub_ratio 0.5 \
        --num_epochs 1 \
        --batch_size 16 \
        --num_added_img 100 \
        --verbose True \
        --attributes test_random"""
    
    success2 = run_test("ImageNet with Random Sampling", test2_cmd)
    
    # Test 3: CelebA with Active Learning enabled  
    test3_cmd = """python alpred.py \
        --dataset celeba \
        --model resnet \
        --use_active_learning True \
        --sub_ratio 0.3 \
        --al_initial_labeled 200 \
        --al_embedding_dim 256 \
        --num_epochs 1 \
        --batch_size 16 \
        --num_added_img 50 \
        --verbose True \
        --attributes test_al_celeba"""
    
    success3 = run_test("CelebA with Active Learning", test3_cmd)
    
    # Test 4: Test different models with Active Learning
    test4_cmd = """python alpred.py \
        --dataset imagenet \
        --model clip \
        --use_active_learning True \
        --sub_ratio 0.4 \
        --al_initial_labeled 300 \
        --num_epochs 1 \
        --batch_size 8 \
        --num_added_img 50 \
        --verbose True \
        --attributes test_al_clip"""
    
    success4 = run_test("ImageNet CLIP with Active Learning", test4_cmd)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("ImageNet with Active Learning", success1),
        ("ImageNet with Random Sampling", success2), 
        ("CelebA with Active Learning", success3),
        ("ImageNet CLIP with Active Learning", success4)
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Active Learning integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
