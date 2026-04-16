# Directory Structure

> How the Safefix ML pipeline and supporting modules are organized.

---

## Overview

Safefix follows a modular ML project structure designed for experiment reproducibility and clear separation between core logic, data processing, and artifact storage.

- **Entrypoints**: Root-level scripts for running the end-to-end pipeline.
- **Core Logic**: Reusable modules for models, training, and data handling.
- **Diagnostics**: Specialized scripts for model evaluation and bug discovery.
- **Data & Artifacts**: Dedicated directories for datasets and generated outputs.

---

## Directory Layout

```text
safefix/
├── alpred.py                # Main Active Learning prediction pipeline
├── train.py                 # Core training entrypoint
├── predict.py               # Inference and prediction script
├── utils/                   # Reusable ML modules
│   ├── models.py            # Model architectures (ResNet, ViT, CLIP)
│   ├── data_utils.py        # Dataset classes and transforms
│   ├── training.py          # Training loops and evaluation logic
│   ├── active_learning.py   # Active Learning selection strategies
│   └── ...
├── processData/             # Diagnostic and evaluation scripts
│   ├── Diagnose.py          # Model failure and correlation analysis
│   ├── report.py            # Automated report generation
│   └── ...
├── data/                    # Raw datasets (e.g., CelebA, ImageNet)
├── exampleData/             # Generated or subset datasets for testing
└── save/                    # Experiment artifacts
    ├── weights/             # Saved model checkpoints
    ├── plot/                # Training metrics and visualizations
    ├── pred/                # Saved predictions (.npy files)
    └── idx/                 # Training/validation indices
```

---

## Module Organization

### Core Pipeline
`alpred.py` is the primary orchestrator. It integrates data selection, training, and diagnostics. 

**Rule**: New training features, data selection strategies, or model improvements must be implemented within the `utils/` directory and then imported into `alpred.py`. Do not add complex logic directly to the root entrypoints.

### Utilities (`utils/`)
- **`models.py`**: Define all neural network architectures here.
- **`data_utils.py`**: Handle all data loading, augmentation, and preprocessing.
- **`training.py`**: Contains the boilerplate for training loops, validation, and checkpointing.
- **`active_learning.py`**: Implementation of all intelligent data selection strategies.

### Diagnostics (`processData/`)
This directory is reserved for post-training analysis. Scripts here should focus on interpreting model behavior, identifying failure modes, and generating human-readable reports.

---

## Naming Conventions

- **Scripts**: Use `snake_case` for Python files (e.g., `data_utils.py`).
- **Classes**: Use `PascalCase` (e.g., `ActiveLearningStrategy`).
- **Artifacts**: Include dataset, model, and timestamp in saved filenames to ensure traceability.

---

## Examples

- **Adding a new AL strategy**: Implement the logic in `utils/active_learning.py` and add a corresponding argument to `alpred.py`.
- **Running a diagnostic**: Use `processData/Diagnose.py` to analyze predictions saved in `save/pred/`.
