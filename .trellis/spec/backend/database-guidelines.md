# Data Loading & Processing Guidelines

This project uses PyTorch for data loading and processing, specifically tailored for image datasets like CelebA and ImageNet. Instead of a traditional database, we manage data through structured file systems and metadata files.

## Dataset & DataLoader

*   Use PyTorch `Dataset` and `DataLoader` classes for efficient data handling.
*   Core dataset implementations are located in `utils/data_utils.py`, including `CelebADataset`, `GeneratedImageDataset`, and `SimpleImageDataset`.
*   Use the `make_loader` utility to create DataLoaders with consistent parameters like batch size, workers, and pin memory.

## Image Normalization

*   Always handle image normalization explicitly to match model requirements.
*   For standard models like ResNet and ViT, use the ImageNet-style normalization found in `default_img_transform`: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`.
*   For CLIP-based models, use the `CLIPImageProcessor` standard transforms via the `clip_processor_transform` utility.

## Data Storage & Indexing

*   Use `.npy` files for saving indices and labels to ensure fast loading and consistency across experiments.
*   Save training indices in `save/idx/` using the pattern `train_idx_<attr>_<n>imgs.npy`.
*   Save combined labels in `save/labels/` using the pattern `labels_<attr>_<n>imgs.npy`.

## Preprocessing & Augmentation

*   Separate original images from generated images during preprocessing.
*   Combine original and generated data using subsampling logic controlled by the `sub_ratio` parameter.
*   The `sub_ratio` (0 to 1) determines the fraction of original training data to include when combining with generated images.
*   Support different generated-image usage modes:
    *   `false`: Combine original data with generated images.
    *   `train_only`: Use generated images for training and validation only.
    *   `train_val_test`: Use generated images for all splits (8:1:1 ratio).

## Common Patterns

*   Use `ImageFolder` for directory-based datasets like the ImageNet subset.
*   Use `Subset` to manage specific splits (Train, Valid, Test, Unlabeled) based on saved indices.
*   Monitor dataset balance by logging class distributions using the `_print_class_distribution` utility.
