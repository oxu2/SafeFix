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
from transformers import BlipProcessor, BlipModel, BlipConfig
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

from torchvision.datasets import ImageFolder



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
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip', 'blip'],
                    help='Model architecture to use (resnet, vit, clip, or blip)')
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
parser.add_argument(
    '--save_pred_model',
    type=str,
    choices=['best', 'last'],
    default='best',
    help='Which model to use for saving predictions: best (default) or last'
)
parser.add_argument('--gen_only', type=str2bool, default=False,
                    help='Use only generated images for training (default: False)')
parser.add_argument('--skip_train_unlabeled', type=str2bool, default=False,
                    help='Skip processing TRAIN and UNLABELED splits in diagnostics')
parser.add_argument('--no_controlnet', type=str2bool, default=False,
                    help='Ablation flag: disable ControlNet-based augmentation (default: False)')
parser.add_argument('--skip_feasible', type=str2bool, default=False,
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
# Dynamically lower batch size for memory‑intensive backbones to avoid CUDA OOM
if args.model == 'blip' and args.batch_size > 32:
    args.batch_size = 32
    if args.verbose:
        print(f"Adjusted batch size for BLIP model to {args.batch_size} to fit GPU memory.")
# If the attribute list includes "emotion", automatically skip feasible loading
if "emotion" in args.attributes.lower():
    args.skip_feasible = True
    if args.verbose:
        print(f"'emotion' detected in attributes; setting skip_feasible=True")

# If using ViT, reduce epochs to 15 for faster training
if args.model == 'vit' and args.num_epochs == 20:
    args.num_epochs = 15
    if args.verbose:
        print(f"Adjusted number of epochs for ViT model to {args.num_epochs}")

# Adjust number of epochs default for ImageNet subset
if args.dataset == 'imagenet' and args.num_epochs == 20:
    args.num_epochs = 40
    if 'verbose' in locals() and verbose:
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

time_now = datetime.datetime.now().strftime("%m%d%H%M%S")   
print("Current time:", time_now)

# Determine the comparison log file based on the model
compare_log_path = None
if args.model == 'resnet':
    compare_log_path = 'logs/compare/stat_origin_resnet_compare.log'
elif args.model == 'vit':
    compare_log_path = 'logs/compare/stat_origin_vit_compare.log'
elif args.model == 'clip':
    compare_log_path = 'logs/compare/stat_origin_clip_compare.log'

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

# BLIP backbone + linear classification head
class BlipClassifier(nn.Module):
    """
    BLIP backbone + linear classification head.
    Works for both binary and multiclass tasks.
    """
    def __init__(self, num_classes=1, dropout_rate=0.5, pretrained=False):
        super().__init__()
        if pretrained:
            self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        else:
            self.blip = BlipModel(BlipConfig())
        self.dropout = nn.Dropout(dropout_rate)
        # Use the BLIP image-projection dimension for the classification head
        feat_dim = getattr(self.blip.config.vision_config, "projection_dim",
                           self.blip.config.vision_config.hidden_size)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, pixel_values):
        feats = self.blip.get_image_features(pixel_values=pixel_values)
        feats = self.dropout(feats)
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
        if args.pretrained:
            # Use CLIP’s standard preprocessing when fine-tuning
            train_transform = lambda img: feature_extractor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            val_transform   = train_transform
        else:
            # Strong augmentation for scratch CLIP
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
    elif args.model == 'blip':
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        train_transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        val_transform   = train_transform
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
    # Use no transform for index mapping, then create train/val datasets with correct transforms
    full_ds = ImageFolder(data_dir)
    train_idx = np.load('train_idx_imagenet.npy')
    val_idx = np.load('val_idx_imagenet.npy')
    valid_idx = val_idx  # alias for diagnostics compatibility
    print(f"Loaded pre-split ImageNet10 indices: {len(train_idx)} train, {len(val_idx)} validation.")
    train_ds = Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(ImageFolder(data_dir, transform=val_transform), val_idx)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    num_classes = len(full_ds.classes)
    # Expose datasets for training loop compatibility
    train_dataset = train_ds
    valid_dataset = val_ds
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
        

    if os.path.exists('/home/jovyan/ouyang/dataset/img_align_celeba/'):
        input_dir = '/home/jovyan/ouyang/dataset/img_align_celeba/'
    else:
        input_dir = '/data/oxx220000/img_align_celeba/'    
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
        if args.pretrained:
            # Use CLIP’s standard preprocessing when fine-tuning
            train_transform = lambda img: feature_extractor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            val_transform   = train_transform
        else:
            # Strong augmentation for scratch CLIP
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
    elif args.model == 'blip':
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        train_transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        val_transform   = train_transform
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
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
    # Use full ResNet-18 for multiclass ImageNet, simplified version for binary tasks (CelebA)
    if args.dataset == 'imagenet':
        # Load randomly initialized ResNet-18 and adjust final layer for multiclass
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Simplified ResNet for binary classification
        if args.pretrained:
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet18(weights=None)
        base_model.layer3 = nn.Identity()
        base_model.layer4 = nn.Identity()
        base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model = base_model
        model.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(128, 1)
        )
elif args.model == 'vit':
    # Initialize ViT base model
    model = create_model(
        'vit_base_patch16_224',
        pretrained=args.pretrained,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    )
    num_head_features = model.head.in_features
    if args.dataset == 'imagenet':
        # Multiclass head for ImageNet subset
        model.head = nn.Linear(num_head_features, num_classes)
    else:
        # Binary classification head for CelebA
        model.head = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(num_head_features, 1)
        )
elif args.model == 'clip':
    # Initialize CLIP-based classifier with appropriate output size
    model = CLIPForBinary(
        dropout_rate=0.5,
        pretrained=args.pretrained,
        num_classes=num_classes if args.dataset=='imagenet' else 1
    )
elif args.model == 'blip':
    # Initialize BLIP backbone + linear head
    model = BlipClassifier(
        num_classes=num_classes if args.dataset == 'imagenet' else 1,
        dropout_rate=0.5,
        pretrained=args.pretrained
    )

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
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

# Use CosineAnnealingLR for both CLIP and ViT, StepLR for ResNet
if args.model == 'clip':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
elif args.model == 'vit':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
# else for 'resnet'
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
    
    # Handle both cases: with or without modified images
    if len(modified_image_paths) > 0:
        original_train_preds = train_preds[:-len(modified_image_labels)]  # Separate original train predictions
        modified_preds = train_preds[-len(modified_image_labels):]  # Get modified image predictions
        # print(f"Original train predictions length: {len(original_train_preds)}")
        # print(f"Modified image predictions length: {len(modified_preds)}")
    else:
        original_train_preds = train_preds
        modified_preds = np.array([])  # Empty array if no modified images
        # print(f"Train predictions length: {len(original_train_preds)}")
    
    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)
    # print(f"Validation predictions length: {len(valid_preds)}")
    
    # Create combined predictions with correct handling for both cases
    if len(modified_image_paths) > 0:
        combined_preds = np.concatenate((
            original_train_preds.flatten(),
            valid_preds.flatten(),
            modified_preds.flatten(),
            unlabeled_predictions
        ))
    else:
        # Original + validation + unlabeled
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
    for epoch in range(num_epochs):
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

    # Load best or last model before saving predictions
    if args.save_pred_model == 'best' and best_model_state is not None:
        model.load_state_dict(best_model_state)
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