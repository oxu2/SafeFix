import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import os
import json
import argparse
from utils.defines import *
import torchvision
# print(torchvision.__version__) 
#0.14.1+cu117g
from torchvision.models import ResNet18_Weights
from transformers import ViTModel, ViTFeatureExtractor
from transformers import CLIPImageProcessor

import datetime
from timm import create_model
import matplotlib.pyplot as plt
from transformers import CLIPModel
from transformers import CLIPConfig
import random
import re  # Added for log parsing
import copy  # Added for saving best model state
from utils.stat import ModelDiagnose  # Import ModelDiagnose from local stat.py
from itertools import combinations
from processData.report import ReportGenerator
from torchvision.datasets import ImageFolder
from pathlib import Path

import torch.nn.functional as F

 
# Suppress the CLIPFeatureExtractor deprecation warning
import warnings

warnings.filterwarnings(
    "ignore",
    "The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers",
    FutureWarning
)
# Suppress the resume_download deprecation warning
warnings.filterwarnings(
    "ignore",
    "`resume_download` is deprecated and will be removed in version 1.0.0",
    FutureWarning
)

# Helper functions for transforms and loaders
def default_img_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

def clip_processor_transform(proc):
    return lambda img: proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True)

# Custom loss for smoothed BCE with logits
class SmoothBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy loss with logits and manual label smoothing.
    """
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Smooth targets: positive labels -> 1 - smoothing/2, negatives -> smoothing/2
        smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=self.pos_weight)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model on CelebA dataset')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='Enable verbose output (default: False)')
parser.add_argument('--feasible_file', type=str, default=None,
                    help='(Optional) Path to the feasible images JSON file. Defaults to save/feasible_<attributes>.json')
# parser.add_argument('--olabel', type=str2bool, default=False,
#                     help='Use original dataset labels (default: False); if False, uses labels from feasible_file')
parser.add_argument('--gen_image_dir', type=str, default=None,
                    help='(Optional) Directory containing generated images. Defaults to exampleData/<attributes>')
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Use pre-trained weights (default: False)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--attributes', type=str, default="redhair_brownskin", 
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, redhair, brownskin, yellowhair, sademotion, etc.')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, clip)')
parser.add_argument('--plot', type=str2bool, default=False, help='Enable plotting training metrics')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--num_added_img', type=int, default=1000,
                    help='Number of generated images to add (default: 4000)')
# Add argument for attribute file path
parser.add_argument('--attribute_file', type=str, default="exampleData/celebA/attribute.json",
                    help='Path to the attribute JSON file (default: exampleData/celebA/attribute.json)')
parser.add_argument('--gpu', type=int, default=0,
                    help='Index of the CUDA GPU to use (default: 0)')
parser.add_argument('--save_report', type=str2bool, default=False,
                    help='Generate and save diagnostic report (default: False)')
parser.add_argument('--gen_only', type=str2bool, default=False,
                    help='Use only generated images for training (default: False)')
parser.add_argument('--skip_train_unlabeled', type=str2bool, default=False,
                    help='Skip processing TRAIN and UNLABELED splits in diagnostics')
parser.add_argument('--no_controlnet', type=str2bool, default=False,
                    help='Ablation flag: disable ControlNet-based augmentation (default: False)')
parser.add_argument('--skip_feasible', type=str2bool, default=True,
                    help='Skip loading feasible JSON for any attributes (default: False)')
parser.add_argument(
    '--simple', type=str2bool, default=False,
    help='Simple mode: disable ControlNet and attribute editing (original HiBug behavior) (default: True).'
)
parser.add_argument(
    '--dataset',
    choices=['imagenet', 'celeba'],
    default='celeba',
    help='Which dataset to use: imagenet (default) or celeba')
# Add batch_size argument here
parser.add_argument('--batch_size', type=int, default=64,
                    help='Mini‑batch size for training and validation (default: 64)')
# Add workers argument here
parser.add_argument('--workers', type=int, default=4,
                    help='Number of DataLoader workers (default: 4)')
# Add random seed argument for reproducibility
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
# Separate learning rates for CLIP head and backbone (for scratch mode)
parser.add_argument('--lr_head', type=float, default=None,
                    help='Learning rate for CLIP classification head (scratch mode)')
parser.add_argument('--lr_backbone', type=float, default=None,
                    help='Learning rate for CLIP backbone (scratch mode)')
# Parse arguments
args = parser.parse_args()

# Set random seed for reproducibility
import random
import numpy as _np
import torch as _torch

random.seed(args.seed)
_np.random.seed(args.seed)
if _torch.cuda.is_available():
    _torch.manual_seed(args.seed)
    _torch.cuda.manual_seed_all(args.seed)
else:
    _torch.manual_seed(args.seed)
# Enforce deterministic algorithms
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False

if getattr(args, 'verbose', False):
    print(f"Random seed set to {args.seed}")
# Override attribute file for ImageNet dataset
if args.dataset == 'imagenet':
    args.attribute_file = 'attribute10.json'
# Automatically enable pretrained weights for CLIP and ViT on ImageNet
if args.dataset == 'imagenet' and args.model in ('clip', 'vit'):
    args.pretrained = True
# Dynamically lower batch size for memory‑intensive backbones to avoid CUDA OOM


# If using ViT, reduce epochs to 15 for faster training
if args.model == 'vit' and args.num_epochs == 20:
    args.num_epochs = 15
    if args.verbose:
        print(f"Adjusted number of epochs for ViT model to {args.num_epochs}")

if args.dataset == 'imagenet' and args.model == 'clip' and args.num_epochs == 20:
    args.num_epochs = 15

# Adjust number of epochs default for ImageNet subset
if args.dataset == 'imagenet' and args.num_epochs == 20 and args.model == 'resnet':
    args.num_epochs = 30
    if 'verbose' in locals() and args.verbose:
        print(f"Adjusted number of epochs for ImageNet dataset to {args.num_epochs}")
attributes = args.attributes

# Set default feasible_file based on attributes if not provided
if args.feasible_file is None:
    args.feasible_file = f"save/feasible_{attributes}.json"
# Set default gen_image_dir based on attributes if not provided
# Ablation: disable ControlNet-based augmentation if no_controlnet is True
if args.no_controlnet:
    args.gen_image_dir = f"no_controlnet_{attributes}"
    print(f"Ablation mode enabled (ControlNet disabled): using image directory {args.gen_image_dir}")
elif args.gen_image_dir is None:
    args.gen_image_dir = f"exampleData/{attributes}"

verbose = args.verbose

# Adjust learning rate default for ViT if not manually overridden
if args.model == 'vit' and args.lr == 1e-4:
    args.lr = 5e-5
    if verbose:
        print(f"Adjusted learning rate for ViT to {args.lr}")

# Specific adjustment for pretrained ViT on ImageNet to further reduce LR
if args.dataset == 'imagenet' and args.model == 'vit' and args.pretrained:
    # This can override the 5e-5 set above if conditions match,
    # or override a user-specified LR for this specific scenario.
    new_lr_val = 2e-5  # Target LR for this case
    if args.lr != new_lr_val: # Apply and print only if it's a change from current args.lr
        args.lr = new_lr_val
        if verbose:
            print(f"Specifically set learning rate for pretrained ViT on ImageNet to {args.lr}")

if args.dataset == 'imagenet' and args.lr == 1e-4 and args.model == 'resnet':
    args.lr = 1e-3

# Adjust LR for CLIP on ImageNet based on pretrained flag
elif args.dataset == 'imagenet' and args.model == 'clip' and args.lr == 1e-4:
    if args.pretrained:
        args.lr = 1e-5
        if verbose:
            print(f"Set lr for pretrained ImageNet CLIP to {args.lr}")
    else:
        # Scratch CLIP: define separate head and backbone LRs
        args.lr_head = 5e-3
        args.lr_backbone = 5e-4
        if verbose:
            print(f"Scratch CLIP: head lr={args.lr_head}, backbone lr={args.lr_backbone}")
# Leave non-CLIP/ImageNet default unchanged for other cases
# Adjust learning rate for CLIP and ViT on CelebA to match predict_0505.py
if args.dataset == 'celeba' and args.model in ('clip', 'vit'):
    args.lr = 1e-5
    if verbose:
        print(f"Adjusted learning rate for {args.model} on CelebA to {args.lr}")
time_now = datetime.datetime.now().strftime("%m%d%H%M%S")   
print("Current time:", time_now)

# Determine the comparison log file based on the model and dataset
compare_log_path = None
if args.model == 'resnet':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_resnet_compare.log"
elif args.model == 'vit':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_vit_compare.log"
elif args.model == 'clip':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_clip_compare.log"

if compare_log_path and not os.path.exists(compare_log_path):
    print(f"Warning: Default comparison log file for {args.model} not found at {compare_log_path}")
    compare_log_path = None # Set to None if file doesn't exist

class CLIPForBinary(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained=False, num_classes=1):
        super().__init__()
        # Initialize CLIP with or without pre-trained weights
        if pretrained:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.clip = CLIPModel(CLIPConfig())
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # Replace classifier to handle multiclass if needed
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, pixel_values):
        # Get image features and classify
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        feats = self.dropout(feats)
        feats = self.dropout2(feats)
        return self.classifier(feats)


# Load the feasible images from feasible.json unless skip_feasible is True
if not args.no_controlnet:
    if args.skip_feasible:
        feasible_images = {}
        if verbose:
            print(f"skip_feasible flag is True; skipping feasible JSON loading for {args.attributes}.")
    else:
        if os.path.exists(args.feasible_file):
            with open(args.feasible_file, 'r') as f:
                feasible_images = json.load(f)
            if verbose:
                print(f"Loaded {len(feasible_images)} feasible {args.attributes} images")
        else:
            feasible_images = {}
            if args.verbose:
                print(f"Feasible file '{args.feasible_file}' not found; skipping feasible JSON loading for '{args.attributes}'.")
else:
    feasible_images = {}  # Not used in ablation mode
    if verbose:
        print("Ablation (no_controlnet) mode: Skipping feasible image loading.")

if args.dataset == 'imagenet':
    # Load ImageNet subset from folders
    data_dir = Path("./imagenet_subset")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model == 'clip':
        train_transform = clip_processor_transform(feature_extractor)
        val_transform = train_transform
    else:
        train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)
    # Use no transform for index mapping, then create train/val datasets with correct transforms
    full_ds = ImageFolder(data_dir)
    train_idx = np.load('train_idx_imagenet.npy')
    val_idx = np.load('val_idx_imagenet.npy')
    valid_idx = val_idx  # alias for diagnostics compatibility
    train_ds = Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(ImageFolder(data_dir, transform=val_transform), val_idx)
    valid_loader = make_loader(val_ds, shuffle=False)
    valid_dataset = val_ds
    num_classes = len(full_ds.classes)

    # --- ImageNet augmentation modes ---
    if args.simple:
        # Simple mode: augment from exampleData/imagenet_simple
        simple_root = Path("exampleData/imagenet_simple")
        simple_paths, simple_labels = [], []
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = simple_root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    simple_paths.append(str(cls_dir / img_name))
                    simple_labels.append(cls_idx)
        # Sample up to args.num_added_img
        if len(simple_paths) > args.num_added_img:
            combined = list(zip(simple_paths, simple_labels))
            random.seed(args.seed)
            sampled = random.sample(combined, args.num_added_img)
            simple_paths, simple_labels = zip(*sampled)
            simple_paths, simple_labels = list(simple_paths), list(simple_labels)
        if simple_paths:
            class SimpleImageDataset(Dataset):
                def __init__(self, paths, labels, transform):
                    self.paths, self.labels, self.transform = paths, labels, transform
                def __len__(self): return len(self.paths)
                def __getitem__(self, i):
                    img = Image.open(self.paths[i]).convert("RGB")
                    return (self.transform(img), self.labels[i]) if self.transform else (img, self.labels[i])
            simple_ds = SimpleImageDataset(simple_paths, simple_labels, train_transform)
            train_dataset = torch.utils.data.ConcatDataset([train_ds, simple_ds])
            train_loader = make_loader(train_dataset, shuffle=True)
            print(f"Added {len(simple_ds)} ImageNet simple images: {len(train_ds)} -> {len(train_dataset)}")
        else:
            train_dataset, train_loader = train_ds, make_loader(train_ds, shuffle=True)
    else:
        # Normal mode: generated-images + feasible filter (mirrors CelebA logic)
        # Determine generated-image directory: use default if exists, otherwise try fallbacks
        default_gen_dir = f"exampleData/imagenet_{attributes}"
        fallback_dirs = [
            f"/home/user/imagenet_{attributes}",
            f"/data/user/imagenet_{attributes}"
        ]
        if os.path.exists(default_gen_dir):
            args.gen_image_dir = default_gen_dir
        else:
            for d in fallback_dirs:
                if os.path.exists(d):
                    args.gen_image_dir = d
                    break
            else:
                args.gen_image_dir = default_gen_dir
        if verbose:
            print(f"Using generated image directory: {args.gen_image_dir}")
        if verbose:
            print(f"[DEBUG] Normal mode: looking for generated images in directory: {args.gen_image_dir}")
        # --- Patch: Respect skip_feasible for feasible_images loading ---
        if args.skip_feasible:
            feasible_images = {}
            if verbose:
                print("[DEBUG] Skipping feasible filtering (skip_feasible=True)")
        else:
            if os.path.exists(args.feasible_file):
                with open(args.feasible_file) as f:
                    feasible_images = json.load(f)
            else:
                feasible_images = {}
            if verbose:
                print(f"[DEBUG] Feasible images loaded: {len(feasible_images)} entries")
                print(f"[DEBUG] Sample of feasible keys: {list(feasible_images.keys())[:10]}")
        if verbose:
            print(f"[DEBUG] Classes in full_ds: {list(full_ds.class_to_idx.keys())}")
        # Traverse each class subfolder for generated images
        from pathlib import Path as _Path
        gen_paths, gen_labels = [], []
        if verbose:
            print("[DEBUG] Starting traversal of class subfolders for generated images...")
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = _Path(args.gen_image_dir) / cls_name
            if verbose:
                print(f"[DEBUG] Checking class folder: {cls_dir} (exists: {cls_dir.is_dir()})")
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                if verbose:
                    print(f"[DEBUG] Found file in {cls_name}: {img_name}")
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                key = os.path.splitext(img_name)[0]
                full_key = f"{cls_name}/{key}"
                in_feasible = full_key in feasible_images
                if verbose:
                    print(f"[DEBUG] File key '{full_key}' in feasible_images: {in_feasible}")
                # --- Patch: Only filter if not args.skip_feasible ---
                if not args.skip_feasible and feasible_images and full_key not in feasible_images:
                    if verbose:
                        print(f"[DEBUG] Skipping '{full_key}' not in feasible_images")
                    continue
                gen_paths.append(str(cls_dir / img_name))
                gen_labels.append(cls_idx)
        # Debug before filtering by feasible_images
        total_before_filter = len(gen_paths)
        if verbose:
            print(f"[DEBUG] Candidate images before feasible filtering: {total_before_filter}")
            print(f"[DEBUG] Total candidate generated images before sampling: {len(gen_paths)}")
        if len(gen_paths) < args.num_added_img:
            print(f"Warning: requested {args.num_added_img}, but only {len(gen_paths)} available.")
        # Sample up to requested number
        random.seed(args.seed)
        if len(gen_paths) > args.num_added_img:
            idxs = random.sample(range(len(gen_paths)), args.num_added_img)
            gen_paths = [gen_paths[i] for i in idxs]
            gen_labels = [gen_labels[i] for i in idxs]
        modified_image_paths = gen_paths
        modified_image_labels = gen_labels
        if verbose:
            print(f"[DEBUG] Sampled generated images: {len(modified_image_paths)} items")
        print(f"Added {len(modified_image_labels)} generated images.")
        # Combine original train_ds and generated
        class GeneratedDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths, self.labels, self.transform = paths, labels, transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return (self.transform(img), self.labels[i]) if self.transform else (img, self.labels[i])
        gen_ds = GeneratedDataset(modified_image_paths, modified_image_labels, train_transform)
        train_dataset = torch.utils.data.ConcatDataset([train_ds, gen_ds])
        print(f"Added {len(modified_image_labels)} ImageNet {attributes} images: {len(train_ds)} -> {len(train_dataset)}")
        train_loader = make_loader(train_dataset, shuffle=True)

    # Skip CelebA loading
    all_train_imgs = None  # placeholder
    valid_imgs = None
    train_labels = None
    valid_labels = None
    modified_image_paths = []
    modified_image_labels = []
    unlabeled_predictions = np.array([])
    # Jump to model setup
else:
    with open("exampleData/celebA/list_attr_celeba.txt") as f:
        lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]
        

    celeba_dir1 = Path('/home/user/dataset/img_align_celeba')
    celeba_dir2 = Path('/data/user/img_align_celeba')
    input_dir = str(celeba_dir1) if celeba_dir1.exists() else str(celeba_dir2)
    all_datas = [input_dir + line for line in lines][1:]

    train_idx = np.load("exampleData/celebA/train_idx.npy")[:80000]
    valid_idx = np.arange(len(all_datas))[-100000:-80000]
    unlabel_idx = np.arange(len(all_datas))[-80000:]

    idxs = np.concatenate([train_idx, valid_idx, unlabel_idx], axis=0)
    labels = np.load("exampleData/celebA/labels.npy")[idxs]

    all_datas = [all_datas[i] for i in idxs]
    split = (
        [TRAIN for _ in train_idx] + 
        [VALID for _ in valid_idx] + 
        [UNLABELED for _ in unlabel_idx]
    )

    # Define Dataset
    class CelebADataset(Dataset):
        def __init__(self, img_paths, labels, transform=None):
            self.img_paths = img_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert("RGB")
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.transform:
                img = self.transform(img)

            return img, label

    # Initialize CLIP feature extractor
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model == 'clip':
        train_transform = clip_processor_transform(feature_extractor)
        val_transform = train_transform

    else:
        train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)

    # Create Datasets and Loaders
    train_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TRAIN]
    valid_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == VALID]
    unlabeled_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == UNLABELED]

    train_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TRAIN])
    valid_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == VALID])
    unlabeled_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == UNLABELED])

    # Add the modified hair images to the training set
    # These images' labels will be the same as their corresponding original images
    modified_image_paths = []
    modified_image_labels = []

    #
    # Prepare sample keys for ablation mode (was hibug mode)
    if args.no_controlnet:
        if args.simple:
            # Simple mode: read all PNGs from hibug_simple folder
            args.gen_image_dir = "hibug_simple"
            files = os.listdir(args.gen_image_dir)
            sample_keys = [os.path.splitext(f)[0] for f in files if f.lower().endswith('.png')]
            print(f"Simple ablation (no_controlnet) mode: Found {len(sample_keys)} images in {args.gen_image_dir}")
        else:
            # Existing ablation behavior: read from no_controlnet_<attributes> folder
            args.gen_image_dir = f"no_controlnet_{attributes}"
            if not os.path.isdir(args.gen_image_dir):
                print(f"Error: Ablation image directory {args.gen_image_dir} not found.")
                sample_keys = []
            else:
                all_files = os.listdir(args.gen_image_dir)
                sample_keys = [os.path.splitext(f)[0] for f in all_files if f.lower().endswith('.png')]
                print(f"Ablation (no_controlnet) mode: Found {len(sample_keys)} images in {args.gen_image_dir}")

    # Sample from all generated image files in the directory, ignoring JSON-based feasible_images
    if not args.no_controlnet:
        if not os.path.isdir(args.gen_image_dir):
            print(f"Error: Generated image directory {args.gen_image_dir} not found.")
            sample_keys = []
        else:
            import re  # ensure this import is at the top of the file
            all_files = os.listdir(args.gen_image_dir)
            # Match files named "<digits>_<attribute>" or "<digits>_<attribute>_<seed>"
            regex = re.compile(rf'^\d+_{re.escape(attributes)}(?:_\d+)?$')
            file_keys = [
                os.path.splitext(f)[0]
                for f in all_files
                if regex.match(os.path.splitext(f)[0])
            ]
            sample_keys = file_keys

    # Warn if fewer images are available than requested in the chosen sample_keys
    if len(sample_keys) < args.num_added_img:
        print(f"Warning: requested {args.num_added_img} generated {args.attributes} images, "
              f"but only {len(sample_keys)} available.")

    # Perform sampling
    if verbose:
        print("Perform sampling for generated images")
    selected_filenames = random.sample(sample_keys, min(args.num_added_img, len(sample_keys)))
    print(f"Selected {len(selected_filenames)} filenames for sampling")

    # Build a map from base image number to label for quick lookup (not needed for simple ablation mode)
    if not (args.no_controlnet and args.simple):
        # Include training, validation, and unlabeled images in the label map
        all_orig_paths = train_imgs + valid_imgs + unlabeled_imgs
        all_orig_labels = np.concatenate((train_labels, valid_labels, unlabeled_labels))
        base_to_label = {
            os.path.splitext(os.path.basename(p))[0]: lbl
            for p, lbl in zip(all_orig_paths, all_orig_labels)
        }
        
        if verbose:
            print(f"Populated base_to_label dictionary with {len(base_to_label)} entries.")
            example_keys = list(base_to_label.keys())[:5]
            print(f"Example keys in base_to_label: {example_keys}")

    # Sampling loop
    for filename in selected_filenames:
        extension = ".png" if args.no_controlnet else ".jpg"
        if args.no_controlnet and args.simple:
            # In simple ablation mode, infer label from filename prefix: 'yes'->1, 'no'->0
            prefix = filename.split('_')[0].lower()
            lbl = 1 if prefix == 'yes' else 0
            image_path = f"./{args.gen_image_dir}/{filename}{extension}"
            modified_image_paths.append(image_path)
            modified_image_labels.append(lbl)
            if verbose:
                print(f"Added {image_path} with label {lbl}")
            continue
        # Existing ablation or non-ablation logic follows...
        # Determine the correct extension based on ablation flag
        # Get the image file path...
        image_path = f"./{args.gen_image_dir}/{filename}{extension}"
        if not os.path.exists(image_path):
            print(f"Warning: Modified image {image_path} not found.")
            continue
        base = None
        if args.no_controlnet:
            parts = filename.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                base = parts[-1]
            else:
                print(f"Warning: Could not extract base number from ablation filename {filename}")
                continue
        else:
            base = filename.split('_')[0]
        if args.no_controlnet and base is None:
            print(f"Debug: Ablation filename {filename} resulted in base=None")
            continue
        lbl = base_to_label.get(base)
        if lbl is not None:
            modified_image_paths.append(image_path)
            modified_image_labels.append(lbl)
            if verbose:
                print(f"Added {image_path} with label {lbl}")
        else:
            if args.no_controlnet:
                print(f"Debug: Failed lookup for ablation file '{filename}'. Extracted base='{base}'. Key not in base_to_label.")
                print(f"Warning: Could not find original image label for base number {base} (from ablation file {filename})")
            else:
                print(f"Warning: Could not find original image for {filename}")

    if verbose:
        print("Finish sampling for generated images")

    # Print information about the added images
    if args.gen_only:
        # In gen_only mode, we’re only ever using the generated set
        print(f"Using only generated images: total training images = {len(modified_image_labels)}")
    else:
        added_count = len(modified_image_labels)
        print(f"Added {added_count} generated {attributes} images, "
              f"before training set has {len(train_labels)} images, "
              f"now {len(train_labels) + added_count}")
            
    # Load unlabeled predictions - moved outside conditional block
    unlabeled_predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
    unlabeled_predictions = np.array([unlabeled_predictions[i] for i in range(len(all_datas)) if split[i] == UNLABELED])
    if verbose:
        print(f"Unlabeled predictions length: {len(unlabeled_predictions)}")

    # Construct training data based on gen_only flag
    if args.gen_only:
        # Use only generated images: split into 80:20 train/validation sets
        indices = list(range(len(modified_image_paths)))
        random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        all_train_imgs = [modified_image_paths[i] for i in train_indices]
        all_train_labels = np.array(modified_image_labels)[train_indices]
        # Override valid set to use generated images
        valid_imgs = [modified_image_paths[i] for i in val_indices]
        valid_labels = np.array(modified_image_labels)[val_indices]
    elif modified_image_paths:
        # Combine original training data with modified images
        all_train_imgs = train_imgs + modified_image_paths
        all_train_labels = np.concatenate((train_labels, np.array(modified_image_labels)))
        combined_labels = np.concatenate((train_labels, valid_labels, np.array(modified_image_labels), unlabeled_labels))
    else:
        all_train_imgs = train_imgs
        all_train_labels = train_labels
        combined_labels = np.concatenate((train_labels, valid_labels, unlabeled_labels))

    # Save combined indices and labels when using generated images
    if modified_image_paths:
        modified_filenames = [os.path.basename(path).split('.')[0] for path in modified_image_paths]
        if args.gen_only:
            combined_train_idx = np.array(modified_filenames, dtype=object)
        else:
            combined_train_idx = np.array([*train_idx, *modified_filenames], dtype=object)
        # Determine naming for simple vs attribute-based cases
        name_key = "simple" if args.no_controlnet and args.simple else attributes
        if verbose:
            train_idx_path = f"save/idx/train_idx_{name_key}_{args.num_added_img}imgs.npy"
            labels_path    = f"save/labels/labels_{name_key}_{args.num_added_img}imgs.npy"
            if not os.path.exists(train_idx_path):
                np.save(train_idx_path, combined_train_idx)
                print(f"Saved combined train indices to {train_idx_path}")
            else:
                if verbose:
                    print(f"{train_idx_path} already exists, skipping save.")
            if not os.path.exists(labels_path):
                np.save(labels_path, combined_labels)
                print(f"Saved combined labels to {labels_path}")
            else:
                if verbose:
                    print(f"{labels_path} already exists, skipping save.")

    if verbose:
        print(f"Training images: {len(all_train_imgs)}, Labels: {len(all_train_labels)}")
        print(f"Validation images: {len(valid_imgs)}, Labels: {len(valid_labels)}")
        # 84050 20000
    # Create datasets using the combined data
    train_dataset = CelebADataset(all_train_imgs, all_train_labels, train_transform)
    valid_dataset = CelebADataset(valid_imgs, valid_labels, val_transform)

    train_loader = make_loader(train_dataset, shuffle=True)
    valid_loader = make_loader(valid_dataset, shuffle=False)
    num_classes = 2


# Define model, loss, and optimizer
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")
if verbose and device.type == "cuda":
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

if args.model == 'resnet':
    model = models.resnet18(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif args.model == 'vit':
    model = create_model('vit_base_patch16_224', pretrained=args.pretrained)
    # Replace head with a linear layer
    in_features = model.head.in_features if hasattr(model.head, 'in_features') else model.head.weight.shape[1]
    model.head = nn.Linear(in_features, num_classes)
elif args.model == 'clip':
    model = CLIPForBinary(pretrained=args.pretrained, num_classes=num_classes)


model = model.to(device)
print(f"Using {args.model} model with {'pretrained' if args.pretrained else 'random'} weights")

# For consistency, handle pos_weight and loss for both datasets
if args.dataset == 'imagenet':
    # Multi-class classification: use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
else:
    num_negative = (all_train_labels == 0).sum()
    num_positive = (all_train_labels == 1).sum()
    pos_weight = torch.tensor([num_negative / num_positive]).to(device)
    # Use smoothed BCE loss since label_smoothing param is unavailable
    criterion = SmoothBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=0.1)

# Optimize all model parameters for all architectures
if args.model == 'clip':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
elif args.model == 'resnet':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
elif args.model == 'vit':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

# Use StepLR for pretrained CLIP/ViT on ImageNet, otherwise CosineAnnealingLR for CLIP/ViT, StepLR for ResNet
if args.dataset == 'imagenet' and args.pretrained and args.model in ('clip', 'vit'):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
elif args.model == 'clip':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
elif args.model == 'vit':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
elif args.model == 'resnet':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

def evaluate(model, data_loader, save_preds=False):
    model.eval()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            else:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_correct += (preds == labels.int()).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    if save_preds:
        return acc, np.array(all_preds), np.array(all_labels)
    return acc, avg_loss

# After training, save predictions for each dataset
def save_predictions(model, train_loader, valid_loader):
    print("Generating and saving predictions...")

    # Get predictions for training set
    _, train_preds, _ = evaluate(model, train_loader, save_preds=True)

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)

    if args.dataset == 'imagenet':
        # Separate original ImageNet train preds (exclude simple added ones)
        original_count = len(train_idx)
        original_train_preds = train_preds[:original_count]
        # Combine only original train and validation preds
        combined_preds = np.concatenate((
            original_train_preds.flatten(),
            valid_preds.flatten()
        ))
    else:
        # Handle celeba or other cases
        if len(modified_image_paths) > 0:
            original_train_preds = train_preds[:-len(modified_image_labels)]
            modified_preds = train_preds[-len(modified_image_labels):]
            combined_preds = np.concatenate((
                original_train_preds.flatten(),
                valid_preds.flatten(),
                modified_preds.flatten(),
                unlabeled_predictions
            ))
        else:
            original_train_preds = train_preds
            combined_preds = np.concatenate((
                original_train_preds.flatten(),
                valid_preds.flatten(),
                unlabeled_predictions
            ))

    try:
        print(f"Combined predictions length: {len(combined_preds)} (should match combined_labels length: {len(combined_labels)})")
    except NameError:
        print(f"Combined predictions length: {len(combined_preds)}")
    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h" # Add 'h' for ablation/no_controlnet

    # Use simple naming if in simple ablation mode
    pred_key = "simple" if args.no_controlnet and args.simple else attributes
    np.save(
        f"save/pred/predictions_{args.dataset}_{args.model}_{pred_key}_{args.num_added_img}imgs{suffix}_{time_now}.npy",
        combined_preds
    )

# Modify the train_model function to save predictions after training
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=20):
    # Track best validation performance
    best_val_acc = 0.0
    best_model_state = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # For pretrained ViT and CLIP on ImageNet, freeze backbone for initial epochs
    freeze_epochs = 10
    if args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
        for name, param in model.named_parameters():
            # keep only classification head trainable
            if args.model == 'vit':
                if "head" not in name:
                    param.requires_grad = False
            else:  # clip
                if "classifier" not in name:
                    param.requires_grad = False
        if verbose:
            print(f"Froze {args.model} backbone parameters for initial {freeze_epochs} epochs")

    for epoch in range(num_epochs):
        # Unfreeze backbone after initial freeze_epochs for pretrained ImageNet CLIP/ViT
        if epoch == freeze_epochs and args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
            for param in model.parameters():
                param.requires_grad = True
            if verbose:
                print(f"Unfroze all {args.model} backbone parameters; now fine-tuning entire model")
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
            else:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                total_correct += ((torch.sigmoid(outputs) > 0.5).int() == labels.int()).sum().item()

        train_acc = total_correct / len(train_dataset)
        val_acc, val_loss = evaluate(model, valid_loader)
        # Update best model if this epoch's validation accuracy is higher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            if verbose:
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    if args.plot:
        epochs = range(1, num_epochs+1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0,0].plot(epochs, train_loss_history, marker='o')
        axes[0,0].set_title("Train Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")

        axes[0,1].plot(epochs, train_acc_history, marker='o')
        axes[0,1].set_title("Train Accuracy")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Accuracy")

        axes[1,0].plot(epochs, val_loss_history, marker='o')
        axes[1,0].set_title("Validation Loss")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")

        axes[1,1].plot(epochs, val_acc_history, marker='o')
        axes[1,1].set_title("Validation Accuracy")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig(f"save/plot/plot_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png")
        if args.dataset == 'imagenet':
            plt.savefig(f"./plot_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png")
        plt.close()

    # Save predictions after training
    save_predictions(model, train_loader, valid_loader)
    # Return training history for external use
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Function to parse log file (adapted from compare_acc.py)
def parse_log(log_path):
    """
    Parse the log file to extract (attribute, description, accuracy) tuples.
    """
    # 1：ACC: 0.893557  ... attribute "beard", description "No"
    pattern1 = re.compile(
        r'ACC:\s*(?P<acc>\d+\.\d+).*attribute\s+"(?P<attr>[^"]+)",\s*description\s+"(?P<desc>[^"]+)"'
    )
    # 2：The accuracy with "hair color" = "white" is 0.9303...
    pattern2 = re.compile(
        r'The accuracy with\s+"(?P<attr>[^"]+)"\s*=\s*"(?P<desc>[^"]+)"\s*is\s*(?P<acc>\d+\.\d+)'
    )
    pattern_overall = re.compile(r'Model Validation ACC:\s*(?P<acc>\d+\.\d+)')
    overall_acc = None

    results = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if overall_acc is None:
                    m0 = pattern_overall.search(line)
                    if m0:
                        overall_acc = float(m0.group('acc'))
                m1 = pattern1.search(line)
                if m1:
                    groups = m1.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
                    continue

                m2 = pattern2.search(line)
                if m2:
                    groups = m2.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
    except FileNotFoundError:
        print(f"Warning: Comparison log file not found at {log_path}")
        return [], None

    return results, overall_acc

# Start training with scheduler
 # Train the model and capture history for diagnostics
train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
    model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=args.num_epochs
)



# Run the model diagnostics after predictions are saved
# Add import for ImageFolder at the top if not present
from torchvision.datasets import ImageFolder

if args.dataset == 'imagenet':
    print("\nRunning model diagnostics on generated predictions...")
    # Define thresholds for diagnostics
    acc_diff_threshold = 0.01
    distribution_diff_threshold = 0.3
    dominant_label_in_failure_threshold = 0.8
    dominant_label_in_prediction_threshold = 0.8
    rare_label_threshold = 0.93
    dominant_error_threshold = 0.3
    error_cover_threshold = 0.9
    top_error = 3

    # Determine filename suffix for predictions
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h"
    # Load the predictions we just saved
    prediction_file = f"save/pred/predictions_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}.npy"
    predictions = np.load(prediction_file)

    # Prepare labels and split for diagnostics
    # Derive ground-truth labels from the ImageFolder ordering
    from pathlib import Path
    data_dir = Path("./imagenet_subset")
    full_ds = ImageFolder(data_dir)
    labels_all = np.array(full_ds.targets)
    # Build labels for train and validation splits
    labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx]])
    split = [TRAIN] * len(train_idx) + [VALID] * len(val_idx)

    # Instantiate ModelDiagnose and run diagnostics
    MD = ModelDiagnose(
        labels=labels_to_diagnose,
        predictions=predictions,
        split=split,
        attribute_dict_path=args.attribute_file,
        print=args.verbose,
        dataset=args.dataset,
        acc_diff_threshold=acc_diff_threshold,
        distribution_diff_threshold=distribution_diff_threshold,
        dominant_label_in_failure_threshold=dominant_label_in_failure_threshold,
        dominant_label_in_prediction_threshold=dominant_label_in_prediction_threshold,
        rare_label_threshold=rare_label_threshold,
        dominant_error_threshold=dominant_error_threshold,
        error_cover_threshold=error_cover_threshold,
        top_error=top_error
    )

    # print("Start detect_failure_by_label\n")
    # MD.detect_failure_by_label()
    print("Start detect_prediction_correlation\n")
    MD.detect_prediction_correlation()
    # print("Start detect_failure_prediction_correlation\n")
    # MD.detect_failure_prediction_correlation()

    # Optionally save a report
    if args.save_report:
        report_dir = "save/reports"
        os.makedirs(report_dir, exist_ok=True)
        MD.generate_report(f"{report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
        print(f"Diagnostic report saved to {report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")

    # --- Accuracy Comparison Logic ---
    if args.gen_only:
        print("Skipping accuracy comparison due to gen_only flag being True")
    else:
        if compare_log_path:
            print(f"\nComparing accuracy with log file: {compare_log_path}")
            orig_entries, orig_overall = parse_log(compare_log_path)
            if orig_entries is not None:
                orig_map = {(e['attribute'], e['description']): e['accuracy'] for e in orig_entries}
                current_overall = MD.valid_acc
                if orig_overall is not None and current_overall is not None:
                    if current_overall > orig_overall:
                        overall_status = 'improved'
                    elif current_overall < orig_overall:
                        overall_status = 'declined'
                    else:
                        overall_status = 'same'
                    print(f'Overall accuracy: {orig_overall:.6f} -> {current_overall:.6f}: {overall_status}')
                elif current_overall is not None:
                     print(f'Current overall accuracy: {current_overall:.6f} (Original overall accuracy not found in log)')
                else:
                     print(f'Original overall accuracy: {orig_overall:.6f} (Current overall accuracy not available)')

                print("\nAttribute-specific accuracy comparison:")
                current_compared_count = 0
                for (attribute, description), orig_acc in orig_map.items():
                    current_acc = None
                    if attribute in MD.attributes and description in MD.attributes[attribute]['valid acc']:
                        current_acc = MD.attributes[attribute]['valid acc'][description]
                    if current_acc is None:
                        continue
                    if current_acc > orig_acc:
                        status = 'improved'
                    elif current_acc < orig_acc:
                        status = 'declined'
                    else:
                        status = 'same'
                    print(f'{attribute}, {description}, {orig_acc:.6f} -> {current_acc:.6f}: {status}')
                    current_compared_count += 1

                print(f"\nCompared {current_compared_count} attribute-specific accuracies.")
            else:
                print(f"Could not parse or find the comparison log file: {compare_log_path}")
        else:
            print("\nSkipping accuracy comparison: No comparison log file specified for the selected model or file not found.")

elif args.dataset == 'celeba':
    print("\nRunning model diagnostics on generated predictions...")

    # Define the parameters for diagnostics
    acc_diff_threshold = 0.01
    distribution_diff_threshold = 0.3
    dominant_label_in_failure_threshold = 0.8
    dominant_label_in_prediction_threshold = 0.8
    rare_label_threshold = 0.93
    dominant_error_threshold = 0.3
    error_cover_threshold = 0.9
    top_error = 3

    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h"  # Add 'h' for ablation/no_controlnet

    # Load the predictions we just saved
    prediction_file = f"save/pred/predictions_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}.npy"
    predictions = np.load(prediction_file)

    # Adjust split to match combined_labels (including modified images in the training split)
    split = (
        [TRAIN] * len(train_labels) +
        [VALID] * len(valid_labels) +
        [TRAIN] * len(modified_image_labels) +
        [UNLABELED] * len(unlabeled_labels)
    )

    if not args.gen_only:
        MD = ModelDiagnose(labels=combined_labels,
                           predictions=predictions,
                           split=split,
                           attribute_dict_path=args.attribute_file,
                           print=args.verbose,
                           dataset=args.dataset,
                           acc_diff_threshold=acc_diff_threshold,
                           distribution_diff_threshold=distribution_diff_threshold,
                           dominant_label_in_failure_threshold=dominant_label_in_failure_threshold,
                           dominant_label_in_prediction_threshold=dominant_label_in_prediction_threshold,
                           rare_label_threshold=rare_label_threshold,
                           dominant_error_threshold=dominant_error_threshold,
                           error_cover_threshold=error_cover_threshold,
                           top_error=top_error)

        print("Start detect_failure_by_label\n")
        MD.detect_failure_by_label()
        print("Start detect_prediction_correlation\n")
        MD.detect_prediction_correlation()
        print("Start detect_failure_prediction_correlation\n")
        MD.detect_failure_prediction_correlation()
        print("Model diagnostics complete.")

        if args.save_report:
            report_dir = "save/reports"
            os.makedirs(report_dir, exist_ok=True)
            MD.generate_report(f"{report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
            print(f"Diagnostic report saved to {report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
    else:
        print("Skipping model diagnostics because gen_only=True")
        if len(train_acc_history) > 0:
            final_train_acc = train_acc_history[-1]
            print("\n" + "="*60)
            print(f"Final Epoch Train Accuracy (gen_only=True): {final_train_acc:.4f}")
            print("="*60 + "\n")

    # --- Accuracy Comparison Logic ---
    if args.gen_only:
        print("Skipping accuracy comparison due to gen_only flag being True")
    else:
        if compare_log_path:
            print(f"\nComparing accuracy with log file: {compare_log_path}")
            orig_entries, orig_overall = parse_log(compare_log_path)

            if orig_entries is not None:
                orig_map = {(e['attribute'], e['description']): e['accuracy'] for e in orig_entries}
                current_overall = MD.valid_acc
                if orig_overall is not None and current_overall is not None:
                    if current_overall > orig_overall:
                        overall_status = 'improved'
                    elif current_overall < orig_overall:
                        overall_status = 'declined'
                    else:
                        overall_status = 'same'
                    print(f'Overall accuracy: {orig_overall:.6f} -> {current_overall:.6f}: {overall_status}')
                elif current_overall is not None:
                     print(f'Current overall accuracy: {current_overall:.6f} (Original overall accuracy not found in log)')
                else:
                     print(f'Original overall accuracy: {orig_overall:.6f} (Current overall accuracy not available)')

                print("\nAttribute-specific accuracy comparison:")
                current_compared_count = 0
                for (attribute, description), orig_acc in orig_map.items():
                    current_acc = None
                    if attribute in MD.attributes and description in MD.attributes[attribute]['valid acc']:
                        current_acc = MD.attributes[attribute]['valid acc'][description]
                    if current_acc is None:
                        continue
                    if current_acc > orig_acc:
                        status = 'improved'
                    elif current_acc < orig_acc:
                        status = 'declined'
                    else:
                        status = 'same'
                    print(f'{attribute}, {description}, {orig_acc:.6f} -> {current_acc:.6f}: {status}')
                    current_compared_count += 1

                print(f"\nCompared {current_compared_count} attribute-specific accuracies.")
            else:
                print(f"Could not parse or find the comparison log file: {compare_log_path}")
        else:
            print("\nSkipping accuracy comparison: No comparison log file specified for the selected model or file not found.")

print("Current time:", time_now)