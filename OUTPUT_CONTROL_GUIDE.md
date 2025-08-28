# Active Learning Output Control Guide

## Default Behavior (No Progress Bars)

By default, all progress bars are **disabled** to provide clean output. This includes:
- tqdm progress bars during training
- transformers library progress bars
- Any other progress indicators

## Running with Different Output Levels

### 1. Quiet Mode (Recommended)
```bash
# Using the quiet runner (minimal output, no progress bars)
python run_quiet_al.py

# Using main script with quiet flag
python active_learning_coreset.py --quiet

# Using bash script (quiet by default)
./run_active_learning.sh
```

### 2. Normal Mode (Some output, no progress bars)
```bash
# Default behavior
python active_learning_coreset.py

# With demo script
python demo_active_learning.py
```

### 3. Verbose Mode (Detailed output, no progress bars)
```bash
python active_learning_coreset.py --verbose
```

### 4. Enable Progress Bars (if needed for debugging)
```bash
python active_learning_coreset.py --enable-progress-bars --verbose
```

## Output Examples

### Quiet Mode Output:
```
Starting Quiet Active Learning...
Device: cuda
Model: resnet
AL Rounds: 5
--------------------------------------------------
Loaded 180000 images
==================================================
ACTIVE LEARNING COMPLETED
==================================================
Final Validation Accuracy: 85.67%
Best Validation Accuracy: 86.23%
Total AL Rounds: 5
```

### Normal Mode Output:
```
Loading CelebA dataset...
Loaded 180000 images
Label distribution: [90000 90000]

=== Active Learning Round 1/5 ===
Current labeled samples: 1000
Current unlabeled samples: 159000
Training model...
Round 1 - Train Acc: 78.45%, Val Acc: 79.12%
```

### What's Disabled:
- ❌ `Extracting embeddings: 76%|███████▌ | 3777/4969 [04:28<21:29, 1.08s/it]`
- ❌ `Epoch 1/10: 16%|█▌ | 5/32 [00:03<00:13, 1.96it/s]`
- ❌ Any other tqdm progress bars

## Configuration Files

You can also modify the default settings in the scripts:

### run_quiet_al.py
```python
config = {
    'verbose': False,  # Set to True for more output
    'disable_progress_bar': True,  # Set to False to enable progress bars
    # ... other settings
}
```

### Environment Variables
```bash
export TQDM_DISABLE=1  # Disable all tqdm progress bars
export TRANSFORMERS_VERBOSITY=error  # Reduce transformers output
```

## Recommendations

1. **For production runs**: Use `run_quiet_al.py` or bash script with `--quiet`
2. **For debugging**: Use `--verbose --enable-progress-bars`
3. **For demos**: Use `demo_active_learning.py` (clean output, no progress bars)

## Files for Different Use Cases

- `run_quiet_al.py` - Cleanest output, production ready
- `demo_active_learning.py` - Good for demonstrations
- `active_learning_coreset.py` - Full control with command line arguments
- `run_active_learning.sh` - Bash wrapper with logging
