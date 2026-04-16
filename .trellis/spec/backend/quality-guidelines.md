# Quality Guidelines

> Code quality standards for backend development, with a focus on ML reproducibility and performance.

---

## ML Reproducibility

To ensure experiments are reproducible across different runs and environments, follow these standards.

### Seed Fixing
Always use seed-fixing functions for PyTorch, NumPy, and the built-in `random` module.

**Required Pattern:**
```python
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CUDNN if exact reproducibility is needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Usage at the start of the script
set_seed(42)
```

**Example from codebase:**
- `train.py` (lines 102-106)
- `alpred.py` (lines 247-253)
- `predict.py` (lines 217-223)

### Deterministic Behavior
When exact reproducibility is required (e.g., for debugging or final paper results), set `torch.backends.cudnn.deterministic = True`. Note that this may slightly reduce performance.

---

## Performance & Memory Management

### Evaluation Mode
Always use `model.eval()` and `torch.no_grad()` during inference, validation, or diagnostics. This prevents the computation graph from being stored, saving memory and preventing leaks.

**Required Pattern:**
```python
model.eval()
with torch.no_grad():
    for inputs, labels in data_loader:
        outputs = model(inputs)
        # ... processing ...
```

**Example from codebase:**
- `utils/training.py` (line 27, 33)
- `alpred.py` (line 1426, 1434)
- `predict.py` (line 469, 470)

### Vectorized Operations
Avoid raw `for` loops for tensor mathematics. Use PyTorch's vectorized operations to leverage GPU acceleration and improve readability.

**Anti-pattern:**
```python
# Slow and hard to read
entropy = torch.zeros(batch_size)
for i in range(batch_size):
    for j in range(num_classes):
        entropy[i] -= probs[i, j] * torch.log(probs[i, j] + 1e-8)
```

**Required Pattern:**
```python
# Fast and concise
entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
```

**Example from codebase:**
- `utils/active_learning.py` (line 1022)
- `model_controlnet/ldm/modules/diffusionmodules/model.py` (vectorized matmul/sum)

---

## Testing Requirements

- All new ML components (Active Learning strategies, model wrappers) should have corresponding tests in `tests/`.
- Use stable seeds in tests to ensure they are not flaky.

---

## Code Review Checklist

- [ ] Are random seeds fixed at the entry point?
- [ ] Is `model.eval()` called before inference?
- [ ] Are there any raw `for` loops that could be vectorized?
- [ ] Is `torch.no_grad()` used where gradients are not needed?
