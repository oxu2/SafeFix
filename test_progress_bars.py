#!/usr/bin/env python3
"""
Test script to verify that progress bars are disabled
"""

import os
# Set environment variable before importing any other modules
os.environ['TQDM_DISABLE'] = '1'

# Try to disable tqdm if it's imported
try:
    from tqdm import tqdm
    tqdm.disable = True
except ImportError:
    pass

# Try to disable transformers progress bars
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

def test_progress_bars():
    """Test if progress bars are properly disabled"""
    
    print("Testing progress bar suppression...")
    
    # Test with range
    for i in range(10):
        pass
    
    print("Basic loop completed without progress bars.")
    
    # Test tqdm if available
    try:
        from tqdm import tqdm
        for i in tqdm(range(5), desc="Testing tqdm"):
            pass
        print("tqdm test completed.")
    except ImportError:
        print("tqdm not available, skipping tqdm test.")
    
    # Test transformers if available
    try:
        from transformers import AutoConfig
        print("Testing transformers loading...")
        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        print("Transformers test completed.")
    except ImportError:
        print("transformers not available, skipping transformers test.")
    except Exception as e:
        print(f"Transformers test failed: {e}")
    
    print("All progress bar tests completed!")

if __name__ == "__main__":
    test_progress_bars()
