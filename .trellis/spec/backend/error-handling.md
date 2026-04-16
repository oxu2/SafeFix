# Error Handling Guidelines

This document outlines the error handling protocols for the Safefix PyTorch backend. These practices ensure training stability and provide clear diagnostics for common deep learning failures.

## Tensor Shape and Dimension Mismatches

Dimension errors often occur during tensor operations or when passing data through model layers. When a shape mismatch occurs, catch the error and log the shapes of the involved tensors to facilitate debugging.

- **Protocol**: Wrap critical tensor operations or model forward passes in try-except blocks when debugging new architectures or data pipelines.
- **Logging**: Always include the tensor shapes in the format `[B, C, H, W]` (Batch, Channels, Height, Width) or the relevant dimensions for the specific operation.

Example:
```python
try:
    output = model(images)
except RuntimeError as e:
    if "size mismatch" in str(e) or "dimension" in str(e):
        print(f"Shape mismatch error: {e}")
        print(f"Input images shape: {images.shape}")
        # Log other relevant tensor shapes if applicable
    raise e
```

## CUDA Out of Memory (OOM)

CUDA OOM errors are common when training large models or using large batch sizes. Handling these errors gracefully allows for better resource management.

- **Protocol**: Catch `RuntimeError` and check if the error message contains "out of memory".
- **Mitigation**:
    - Suggest reducing the `batch_size`.
    - Clear the CUDA cache using `torch.cuda.empty_cache()`.
    - If the error persists, consider using gradient accumulation or a smaller model architecture.

Example:
```python
try:
    loss.backward()
    optimizer.step()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA Out of Memory detected.")
        torch.cuda.empty_cache()
        print("Suggestions: Reduce batch size or clear cache.")
    raise e
```

## NaNs in Loss and Gradients

Numerical instability can lead to `NaN` (Not a Number) values in the loss function, which pollutes gradients and ruins model weights.

- **Protocol**: Check for `NaN` values in the loss before calling `loss.backward()`.
- **Action**:
    - If `torch.isnan(loss)` is true, log the event and either skip the current batch or terminate training to prevent weight corruption.
    - Investigate learning rate, initialization, or data normalization if NaNs occur frequently.

Example:
```python
loss = criterion(outputs, labels)
if torch.isnan(loss):
    print("NaN loss detected. Skipping batch to protect gradients.")
    optimizer.zero_grad()
    continue  # Skip to the next iteration
```

## General Principles

1. **Fail Fast**: Terminate execution if an unrecoverable error occurs that would lead to invalid results.
2. **Informative Logging**: Ensure all error messages provide enough context (e.g., current epoch, batch index, and relevant hyperparameters) to reproduce the issue.
3. **Resource Cleanup**: Always ensure that resources like GPU memory or file handles are released when an error occurs.

