import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset

# GPUs are heavily utilized (~21GB used by other processes). Prevent 
# cuDNN from trying to allocate huge temporary workspaces for convolution profiling.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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
import time
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

def augmentation_pil_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
    ])

def augmentation_baseline_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return default_img_transform(train=False)

def clip_processor_transform(proc, pre_transform=None):
    def _apply(img):
        if pre_transform is not None:
            img = pre_transform(img)
        return proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    return _apply

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
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img, target, idx


class FocalLoss(nn.Module):
    """
    Focal Loss for multiclass classification (Lin et al., 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,) with class indices
        log_probs = F.log_softmax(logits, dim=1)         # (N, C)
        probs = torch.exp(log_probs)                      # (N, C)
        # Gather the log-prob and prob for the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)          # (N,)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)  # (N,)
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt                      # (N,)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                   # (N,)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Scalar weight for the positive class (class 1). The negative
               class gets weight (1 - alpha). Typically set to
               N_neg / (N_neg + N_pos) so the minority (positive) class
               receives higher weight.
        gamma: Focusing parameter. gamma=0 recovers (weighted) BCE.
        reduction: 'mean' | 'sum' | 'none'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, 1) or (N,), targets: same shape, values in {0, 1}
        probs = torch.sigmoid(logits)
        # p_t = p if y=1, else 1-p
        pt = targets * probs + (1 - targets) * (1 - probs)
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = (1 - pt) ** self.gamma
        # Standard BCE component
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).
    Enforces larger classification margins for minority classes.

    The per-class margin is: delta_j = C / n_j^(1/4)
    where n_j is the number of training samples in class j.

    Args:
        cls_num_list: List/array of per-class sample counts.
        max_m: Maximum margin value (default: 0.5).
        s: Scaling factor for logits (default: 30).
        weight: Optional per-class weight tensor for the CE component.
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        cls_num = np.array(cls_num_list, dtype=float)
        # Per-class margins: larger margin for rarer classes
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(cls_num, 1.0)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight

    def forward(self, logits, targets):
        # Subtract class-specific margins from the true-class logits
        m = self.m_list.to(logits.device)
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1).long(), True)
        logits_m = logits.clone()
        logits_m[index] -= m[targets]
        logits_m *= self.s
        return F.cross_entropy(logits_m, targets, weight=self.weight)


class BinaryLDAMLoss(nn.Module):
    """
    LDAM Loss adapted for binary classification with logits.
    Applies a margin shift to the logit for the true class.

    Args:
        cls_num_list: [n_negative, n_positive] counts.
        max_m: Maximum margin (default: 0.5).
        s: Logit scaling factor (default: 30).
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        cls_num = np.array(cls_num_list, dtype=float)
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(cls_num, 1.0)))
        m_list = m_list * (max_m / m_list.max())
        # m_list[0] = margin for class 0 (negative), m_list[1] = margin for class 1 (positive)
        self.m_neg = float(m_list[0])
        self.m_pos = float(m_list[1])
        self.s = s

    def forward(self, logits, targets):
        # logits: (N,1), targets: (N,1) in {0,1}
        # For positive targets: subtract m_pos from logit (make it harder)
        # For negative targets: add m_neg to logit (subtract -m_neg, make it harder)
        margin = targets * self.m_pos + (1 - targets) * (-self.m_neg)
        logits_m = (logits - margin) * self.s
        return F.binary_cross_entropy_with_logits(logits_m, targets)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model on CelebA dataset')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='Enable verbose output (default: False)')
parser.add_argument('--feasible_file', type=str, default=None,
                    help='(Optional) Path to the feasible images JSON file. Defaults to save/feasible_<attributes>.json')
parser.add_argument('--ltl_level', type=str, default='class', choices=['class', 'attribute'],
                    help='LTL computing level (class or attribute)')
parser.add_argument('--ltl_attributes', type=str, default='color,texture',
                    help='Comma-separated attributes to group by when ltl_level=attribute')
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
parser.add_argument('--lvlm', type=str, default=None,
                    help='LVLM model name (e.g., llava) to modify feasible file naming')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip', 'blip'],
                    help='Model architecture to use (resnet, vit, clip, or blip)')
parser.add_argument('--plot', type=str2bool, default=False, help='Enable plotting training metrics')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--num_added_img', type=int, default=1000,
                    help='Number of generated images to add (default: 4000)')
parser.add_argument('--augmentation_baseline', type=str2bool, default=False,
                    help='Use strong data augmentation baseline without adding generated images (default: False)')
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
parser.add_argument(
    '--cdm',
    choices=['controlnet', 'instruct_pix2pix'],
    default='controlnet',
    help='Conditional diffusion model used to generate images (default: controlnet)'
)
parser.add_argument(
    '--ltl_baseline',
    choices=['none', 'focal', 'class_weight', 'resample', 'ldam'],
    default='none',
    help='Long-tail learning baseline: none (default), focal (Focal Loss), '
         'class_weight (inverse-frequency weighted loss), '
         'resample (oversampling minority classes via WeightedRandomSampler), '
         'ldam (Label-Distribution-Aware Margin loss, Cao et al. NeurIPS 2019)'
)
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Gamma (focusing) parameter for Focal Loss (default: 2.0)')
parser.add_argument('--ldam_max_m', type=float, default=0.5,
                    help='Maximum margin for LDAM loss (default: 0.5)')
parser.add_argument('--ldam_s', type=float, default=30.0,
                    help='Logit scaling factor for LDAM loss (default: 30.0)')
# Add batch_size argument here
# Add batch_size argument here
parser.add_argument('--batch_size', type=int, default=64,
                    help='Mini‑batch size for training and validation (default: 64)')
# Add subsampling ratio argument for original training data when combining with modified images
# Add subsampling ratio argument for original training data when combining with modified images
parser.add_argument('--sub_ratio', type=float, 
                    # default=0.3,
                    default=1.0,
                    help='Subsampling ratio (0-1) for original training data when combining with modified images')
# Add pad_missing argument for subsampled original predictions
parser.add_argument('--pad_missing', action='store_true', default=True,
                    help='Pad missing original training image predictions with NaN when subsampling (enabled by default)')
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
    args.attribute_file = 'save/attribute10.json'
# Automatically enable pretrained weights for CLIP and ViT on ImageNet
if args.dataset == 'imagenet' and args.model in ('clip', 'vit'):
    args.pretrained = True
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

# Automatically skip feasible filtering for ImageNet
# if args.dataset == 'imagenet':
#     args.skip_feasible = True
#     if 'verbose' in locals():
#         if verbose:
#             print("Dataset is ImageNet; setting skip_feasible=True")
#     elif getattr(args, 'verbose', False):
#         print("Dataset is ImageNet; setting skip_feasible=True")

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
    if 'verbose' in locals() and verbose:
        print(f"Adjusted number of epochs for ImageNet dataset to {args.num_epochs}")
attributes = args.attributes

# Set default feasible_file based on attributes if not provided
if args.feasible_file is None:
    if args.lvlm == 'llava':
        if args.verbose:
            print(f"Using LLAVA model: setting feasible_file to save/feasible_{attributes}_{args.lvlm}.json")
        args.feasible_file = f"save/feasible_{attributes}_{args.lvlm}.json"
    else: #qwen
        args.feasible_file = f"save/feasible_{attributes}.json"
# Set default gen_image_dir based on attributes if not provided
# Ablation: disable ControlNet-based augmentation if no_controlnet is True
if args.no_controlnet:
    args.gen_image_dir = f"no_controlnet_{attributes}"
    print(f"Ablation mode enabled (ControlNet disabled): using image directory {args.gen_image_dir}")
elif args.gen_image_dir is None:
    args.gen_image_dir = f"exampleData/{attributes}"
    if args.cdm == 'instruct_pix2pix':
        print(f"Using InstructPix2Pix: setting gen_image_dir to {args.gen_image_dir}")
        args.gen_image_dir = f"exampleData/{attributes}_instruct_pix2pix"
verbose = args.verbose

if args.augmentation_baseline:
    if verbose and (args.gen_only or args.simple or args.num_added_img != 0):
        print("Augmentation baseline enabled: disabling generated-image options and setting num_added_img=0.")
    args.gen_only = False
    args.simple = False
    args.num_added_img = 0

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
    # Ensure generated images directory uses the correct prefix based on CDM
    if args.cdm == 'instruct_pix2pix':
        args.skip_feasible = True
        default_imagenet_gen_dir = f"exampleData/{attributes}_instruct_pix2pix"
    else:
        default_imagenet_gen_dir = f"exampleData/imagenet_{attributes}"
    if os.path.isdir(default_imagenet_gen_dir):
        args.gen_image_dir = default_imagenet_gen_dir
    elif not os.path.isdir(args.gen_image_dir):
        args.gen_image_dir = default_imagenet_gen_dir
    # Load ImageNet subset from folders
    data_dir = Path("exampleData/imagenet_subset")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model == 'clip':
        if args.augmentation_baseline:
            train_transform = clip_processor_transform(feature_extractor, pre_transform=augmentation_pil_transform())
        else:
            train_transform = clip_processor_transform(feature_extractor)
        val_transform = clip_processor_transform(feature_extractor)
    elif args.model == 'blip':
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        train_transform = clip_processor_transform(processor)
        val_transform = train_transform
    else:
        train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)
    # Use no transform for index mapping, then create train/val datasets with correct transforms
    full_ds = ImageFolder(data_dir)
    train_idx = np.load('save/train_idx_imagenet.npy')
    val_idx = np.load('save/val_idx_imagenet.npy')
    test_idx_path = Path('save/test_idx_imagenet.npy')
    if test_idx_path.exists():
        test_idx = np.load(test_idx_path)
    else:
        test_idx = np.array([], dtype=int)
    valid_idx = val_idx  # alias for diagnostics compatibility
    train_ds = Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(ImageFolder(data_dir, transform=val_transform), val_idx)
    test_ds = Subset(ImageFolder(data_dir, transform=val_transform), test_idx) if len(test_idx) else None
    valid_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False) if test_ds is not None else None
    valid_dataset = val_ds
    num_classes = len(full_ds.classes)

    # --- ImageNet augmentation modes ---
    if args.ltl_baseline != 'none':
        # LTL baselines: use original training data only, no generated images
        train_dataset = train_ds
        if args.ltl_level == 'attribute':
            train_dataset = IndexDataset(train_ds)
        train_loader = make_loader(train_dataset, shuffle=True)
        modified_image_paths = []
        modified_image_labels = []
        print(f"LTL baseline '{args.ltl_baseline}' active: training on original ImageNet subset only.")
    elif args.augmentation_baseline:
        train_dataset = train_ds
        train_loader = make_loader(train_ds, shuffle=True)
        if verbose:
            print("Augmentation baseline enabled: training on original ImageNet subset only.")
    elif args.gen_only:
        # Use only generated images for ImageNet: 80% train, 20% validation
        # Collect generated image paths and labels
        gen_paths, gen_labels = [], []
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = Path(args.gen_image_dir) / cls_name
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    gen_paths.append(str(cls_dir / img_name))
                    gen_labels.append(cls_idx)
        if len(gen_paths) < args.num_added_img:
            print(f"Warning: requested {args.num_added_img}, but only {len(gen_paths)} generated images available.")
        # Sample up to requested number
        random.seed(args.seed)
        idxs = list(range(len(gen_paths)))
        sampled = random.sample(idxs, min(len(gen_paths), args.num_added_img))
        gen_paths = [gen_paths[i] for i in sampled]
        gen_labels = [gen_labels[i] for i in sampled]
        # Split into train and validation sets (80/20)
        indices = list(range(len(gen_paths)))
        random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        train_idx_gen = indices[:split_idx]
        val_idx_gen = indices[split_idx:]
        class GeneratedImageDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths, self.labels, self.transform = paths, labels, transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return (self.transform(img), self.labels[i])
        train_ds = GeneratedImageDataset([gen_paths[i] for i in train_idx_gen],
                                         [gen_labels[i] for i in train_idx_gen],
                                         train_transform)
        val_ds = GeneratedImageDataset([gen_paths[i] for i in val_idx_gen],
                                       [gen_labels[i] for i in val_idx_gen],
                                       val_transform)
        train_loader = make_loader(train_ds, shuffle=True)
        valid_loader = make_loader(val_ds, shuffle=False)
        print(f"Using only generated images for ImageNet: train={len(train_ds)}, valid={len(val_ds)}")
    elif args.simple:
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
    elif not args.gen_only and not args.simple:
        # Normal mode: generated-images + feasible filter (mirrors CelebA logic)
        # Determine generated-image directory: use default if exists, otherwise try fallbacks
        if args.cdm == 'instruct_pix2pix':
            default_gen_dir = f"exampleData/{attributes}_instruct_pix2pix"
            fallback_dirs = [
                f"/home/jovyan/ouser/{attributes}_instruct_pix2pix",
                f"/data/oxx220000/{attributes}_instruct_pix2pix"
            ]
        else:
            default_gen_dir = f"exampleData/imagenet_{attributes}"
            fallback_dirs = [
                f"/home/jovyan/ouser/imagenet_{attributes}",
                f"/data/oxx220000/imagenet_{attributes}"
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
        print(f"[DEBUG] Using generated image directory: {args.gen_image_dir} (exists: {os.path.isdir(args.gen_image_dir)})")
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
        class_file_counts = {}
        if verbose:
            print("[DEBUG] Starting traversal of class subfolders for generated images...")
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = _Path(args.gen_image_dir) / cls_name
            if verbose:
                print(f"[DEBUG] Checking class folder: {cls_dir} (exists: {cls_dir.is_dir()})")
            if not cls_dir.is_dir():
                print(f"[DEBUG] Skipping class '{cls_name}' because directory {cls_dir} is missing")
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
                class_file_counts[cls_name] = class_file_counts.get(cls_name, 0) + 1
        if not gen_paths:
            print(f"[DEBUG] No generated files collected from {args.gen_image_dir}. class_file_counts={class_file_counts}")
        else:
            print(f"[DEBUG] Generated file counts per class: {class_file_counts}")
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
        # Subsample original train_ds based on sub_ratio
        orig_count = len(train_ds)
        sub_count = int(orig_count * args.sub_ratio)
        random.seed(args.seed)
        selected_indices = random.sample(range(orig_count), sub_count)
        # Map subset indices back to the original dataset indices
        sub_orig_indices = [train_ds.indices[i] for i in selected_indices]
        sub_train_ds = Subset(train_ds.dataset, sub_orig_indices)
        # Store subsampling info for padding in save_predictions
        args.selected_indices = selected_indices
        args.orig_train_count = orig_count
        # Combine subsampled original data with generated data
        train_dataset = torch.utils.data.ConcatDataset([sub_train_ds, gen_ds])
        print(f"Added {len(modified_image_labels)} ImageNet {attributes} images: {len(sub_train_ds)} -> {len(train_dataset)}")
        # preview_paths = modified_image_paths[:3]
        # if preview_paths:
        #     print("Sample generated image paths:")
        #     for p in preview_paths:
        #         print(f"  {p}")
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
elif args.dataset == 'celeba':
    with open("exampleData/celebA/list_attr_celeba.txt") as f:
        lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]
        

    if os.path.exists('/home/jovyan/ouser/dataset/img_align_celeba/'):
        input_dir = '/home/jovyan/ouser/dataset/img_align_celeba/'
    else:
        input_dir = '/data/oxx220000/img_align_celeba/'    
    all_datas = [input_dir + line for line in lines][1:]

    train_idx = np.load("exampleData/celebA/train_idx.npy")[:80000]

    split_cache_dir = "save"
    val_idx_path = os.path.join(split_cache_dir, "val_idx_celeba.npy")
    test_idx_path = os.path.join(split_cache_dir, "test_idx_celeba.npy")
    unlabel_idx_path = os.path.join(split_cache_dir, "unlabel_idx_celeba.npy")
    meta_path = os.path.join(split_cache_dir, "celeba_split_meta.json")

    cached_seed = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as meta_file:
                cached_seed = json.load(meta_file).get("seed")
        except Exception:
            cached_seed = None

    cache_ready = (
        os.path.exists(val_idx_path)
        and os.path.exists(test_idx_path)
        and os.path.exists(unlabel_idx_path)
        and cached_seed == args.seed
    )

    if cache_ready:
        valid_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
        unlabel_idx = np.load(unlabel_idx_path)
    else:
        val_pool = np.arange(len(all_datas))[-100000:]
        if val_pool.size < 20000:
            raise ValueError("Expected at least 20000 samples in CelebA validation/test pool.")
        rng = np.random.default_rng(args.seed)
        shuffled_pool = rng.permutation(val_pool)
        valid_idx = shuffled_pool[:10000]
        test_idx = shuffled_pool[10000:20000]
        unlabel_idx = shuffled_pool[20000:]
        os.makedirs(split_cache_dir, exist_ok=True)
        np.save(val_idx_path, valid_idx)
        np.save(test_idx_path, test_idx)
        np.save(unlabel_idx_path, unlabel_idx)
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump({
                "seed": int(args.seed),
                "timestamp": time.time()
            }, meta_file)

    idxs = np.concatenate([train_idx, valid_idx, test_idx, unlabel_idx], axis=0)
    labels = np.load("exampleData/celebA/labels.npy")[idxs]

    all_datas = [all_datas[i] for i in idxs]
    split = (
        [TRAIN for _ in train_idx] + 
        [VALID for _ in valid_idx] + 
        [TEST for _ in test_idx] +
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
        if args.augmentation_baseline:
            train_transform = clip_processor_transform(feature_extractor, pre_transform=augmentation_pil_transform())
        else:
            train_transform = clip_processor_transform(feature_extractor)
        val_transform = clip_processor_transform(feature_extractor)
    elif args.model == 'blip':
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        if args.augmentation_baseline:
            train_transform = clip_processor_transform(processor, pre_transform=augmentation_pil_transform())
        else:
            train_transform = clip_processor_transform(processor)
        val_transform = clip_processor_transform(processor)
    else:
        if args.augmentation_baseline:
            train_transform = augmentation_baseline_transform(train=True)
        else:
            train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)

    # Create Datasets and Loaders
    train_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TRAIN]
    valid_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == VALID]
    test_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TEST]
    unlabeled_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == UNLABELED]

    train_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TRAIN])
    valid_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == VALID])
    test_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TEST])
    unlabeled_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == UNLABELED])

    # When running LTL baselines, skip generated image loading entirely.
    # LTL baselines only reweight/resample the original training data.
    modified_image_paths = []
    modified_image_labels = []
    sample_keys = []
    selected_filenames = []

    if args.ltl_baseline != 'none':
        print(f"LTL baseline '{args.ltl_baseline}' active: skipping generated image loading.")
    elif args.augmentation_baseline:
        if verbose:
            print("Augmentation baseline enabled: skipping generated image loading.")
    else:
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
        # Include training, validation, test, and unlabeled images in the label map
        all_orig_paths = train_imgs + valid_imgs + test_imgs + unlabeled_imgs
        all_orig_labels = np.concatenate((train_labels, valid_labels, test_labels, unlabeled_labels))
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

    # Load unlabeled predictions
    unlabeled_predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
    unlabeled_predictions = np.array([unlabeled_predictions[i] for i in range(len(all_datas)) if split[i] == UNLABELED])
    if verbose:
        print(f"Unlabeled predictions length: {len(unlabeled_predictions)}")

    # Print information about the added images
    if not args.augmentation_baseline:
        if args.gen_only:
            # In gen_only mode, we’re only ever using the generated set
            print(f"Using only generated images: total training images = {len(modified_image_labels)}")
        else:
            added_count = len(modified_image_labels)
            print(f"Added {added_count} generated {attributes} images, "
                  f"before training set has {len(train_labels)} images, "
                  f"now {len(train_labels) + added_count}")

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
        preview_paths = modified_image_paths[:3]
        if preview_paths:
            print("Sample generated image paths:")
            for p in preview_paths:
                print(f"  {p}")
    elif modified_image_paths:
        # Subsample original training data based on sub_ratio and combine with modified images
        orig_count = len(train_imgs)
        sub_count = int(orig_count * args.sub_ratio)
        random.seed(args.seed)
        selected_indices = random.sample(range(orig_count), sub_count)
        sub_train_imgs = [train_imgs[i] for i in selected_indices]
        sub_train_labels = train_labels[selected_indices]
        all_train_imgs = sub_train_imgs + modified_image_paths
        all_train_labels = np.concatenate((sub_train_labels, np.array(modified_image_labels)))
        # Store subsampling info for later padding of predictions
        args.selected_indices = selected_indices
        args.orig_train_count = orig_count
        preview_paths = modified_image_paths[:3]
        if preview_paths:
            print("Sample generated image paths:")
            for p in preview_paths:
                print(f"  {p}")
    else:
        all_train_imgs = train_imgs
        all_train_labels = train_labels
    combined_labels = np.concatenate((all_train_labels, valid_labels, test_labels, unlabeled_labels))

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
    if len(test_imgs) > 0:
        test_dataset = CelebADataset(test_imgs, test_labels, val_transform)
        test_loader = make_loader(test_dataset, shuffle=False)
    else:
        test_dataset = None
        test_loader = None

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
        model.head = nn.Sequential(
            nn.Dropout(p=0.5),  # Added Dropout layer
            nn.Linear(num_head_features, num_classes)
        )
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
# --- Compute class counts (used by multiple LTL baselines) ---
if args.dataset == 'imagenet':
    # Use pre-loaded targets from ImageFolder (no image loading needed)
    _train_labels_arr = np.array(full_ds.targets)[train_idx]
    if args.ltl_level == 'attribute':
        # Load attribute json
        import json
        with open('save/attribute10.json', 'r') as f:
            attr_data = json.load(f)
        attrs_to_use = args.ltl_attributes.split(',')
        
        # Build unique subgroup list 
        subgroup_strings = []
        for i in train_idx:
            c_label = full_ds.targets[i]
            attr_str = f"class{c_label}"
            for attr in attrs_to_use:
                if attr in attr_data:
                    val = attr_data[attr]['data'][i]
                    attr_str += f"_{attr}:{val}"
            subgroup_strings.append(attr_str)
            
        # Convert to integer IDs
        unique_groups = list(set(subgroup_strings))
        group_to_id = {g: idx for idx, g in enumerate(unique_groups)}
        _subgroup_labels_arr = np.array([group_to_id[g] for g in subgroup_strings])
        _class_counts = np.bincount(_subgroup_labels_arr).astype(float)
        _num_subgroups = len(unique_groups)
        if verbose:
            print(f"Attribute-level LTL identified {_num_subgroups} unique subgroups (class + {args.ltl_attributes}).")
            # Print top 5 rarest groups
            rarest_idx = np.argsort(_class_counts)[:5]
            print("Rarest subgroups:")
            for ri in rarest_idx:
                print(f"  {unique_groups[ri]}: {int(_class_counts[ri])} samples")
    else:
        _subgroup_labels_arr = _train_labels_arr
        _class_counts = np.bincount(_train_labels_arr, minlength=num_classes).astype(float)
        _num_subgroups = num_classes
else:
    _num_negative = float((all_train_labels == 0).sum())
    _num_positive = float((all_train_labels == 1).sum())
    _class_counts = np.array([_num_negative, _num_positive])
    _subgroup_labels_arr = all_train_labels
    _num_subgroups = 2

# --- Select loss function based on LTL baseline ---
if args.ltl_baseline == 'focal':
    if args.dataset == 'imagenet':
        _inv_freq = 1.0 / np.maximum(_class_counts, 1.0)
        _alpha = _inv_freq / _inv_freq.sum() * num_classes
        alpha_tensor = torch.tensor(_alpha, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma)
        print(f"Using Focal Loss (multiclass) with gamma={args.focal_gamma}")
    else:
        alpha_val = _num_negative / (_num_negative + _num_positive)
        criterion = BinaryFocalLoss(alpha=alpha_val, gamma=args.focal_gamma)
        print(f"Using Binary Focal Loss with alpha={alpha_val:.4f}, gamma={args.focal_gamma}")

elif args.ltl_baseline == 'class_weight':
    # Class-Balanced Loss: inverse-frequency weights
    if args.dataset == 'imagenet':
        if args.ltl_level == 'attribute':
            _inv_freq = 1.0 / np.maximum(_class_counts, 1.0)
            _weights = _inv_freq / _inv_freq.sum() * _num_subgroups
            # Map weights to each training sample
            train_dataset_sample_weights = torch.tensor(_weights[_subgroup_labels_arr], dtype=torch.float32).to(device)
            # Use reduction='none' so we can weight per-sample in the training loop
            criterion = nn.CrossEntropyLoss(reduction='none')
            print(f"Using Attribute-Weighted CE Loss, {_num_subgroups} subgroup weights injected into Dataset.")
        else:
            _inv_freq = 1.0 / np.maximum(_class_counts, 1.0)
            _weights = _inv_freq / _inv_freq.sum() * num_classes
            weight_tensor = torch.tensor(_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"Using Class-Weighted CrossEntropyLoss, weights={_weights.round(4)}")
    else:
        pos_weight = torch.tensor([_num_negative / _num_positive]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using Class-Weighted BCEWithLogitsLoss, pos_weight={_num_negative/_num_positive:.4f}")

elif args.ltl_baseline == 'resample':
    # Oversampling: use standard (unweighted) loss; balancing is done via the sampler
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    # Build WeightedRandomSampler for the training set
    if args.dataset == 'imagenet':
        _sample_weights = 1.0 / np.maximum(_class_counts[_subgroup_labels_arr], 1.0)
    else:
        _sample_weights = np.where(all_train_labels == 1,
                                   1.0 / max(_num_positive, 1.0),
                                   1.0 / max(_num_negative, 1.0))
    _sample_weights_t = torch.DoubleTensor(_sample_weights)
    _sampler = torch.utils.data.WeightedRandomSampler(
        weights=_sample_weights_t,
        num_samples=len(_sample_weights_t),
        replacement=True
    )
    # Rebuild train_loader with the sampler (sampler is mutually exclusive with shuffle)
    if args.dataset == 'imagenet' and args.ltl_level == 'attribute':
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=_sampler,
            num_workers=args.workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=_sampler,
            num_workers=args.workers, pin_memory=True
        )
    print(f"Using Resample baseline with WeightedRandomSampler (class counts: {_class_counts.astype(int).tolist()})")

elif args.ltl_baseline == 'ldam':
    # LDAM: Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019)
    if args.dataset == 'imagenet':
        if args.ltl_level == 'attribute':
            # Subgroup LDAM: margin based on subgroup count, computed in custom loop
            _inv_freq = 1.0 / np.maximum(_class_counts, 1.0)
            _weights = _inv_freq / _inv_freq.sum() * _num_subgroups
            train_dataset_sample_weights = torch.tensor(_weights[_subgroup_labels_arr], dtype=torch.float32).to(device)
            # Margins per subgroup
            _m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(_class_counts, 1.0)))
            _m_list = _m_list * (args.ldam_max_m / _m_list.max())
            train_dataset_margins = torch.tensor(_m_list[_subgroup_labels_arr], dtype=torch.float32).to(device)

            # Standard reduction None CE, we will subtract margin manually in training loop
            criterion = nn.CrossEntropyLoss(reduction='none')
            print(f"Using Attribute-Level LDAM Loss, {_num_subgroups} subgroups. Margins embedded in Dataset.")
        else:
            _inv_freq = 1.0 / np.maximum(_class_counts, 1.0)
            _weights = _inv_freq / _inv_freq.sum() * num_classes
            weight_tensor = torch.tensor(_weights, dtype=torch.float32).to(device)
            criterion = LDAMLoss(
                cls_num_list=_class_counts.tolist(),
                max_m=args.ldam_max_m, s=args.ldam_s,
                weight=weight_tensor
            )
            print(f"Using LDAM Loss (multiclass) with max_m={args.ldam_max_m}, s={args.ldam_s}")
    else:
        criterion = BinaryLDAMLoss(
            cls_num_list=[_num_negative, _num_positive],
            max_m=args.ldam_max_m, s=args.ldam_s
        )
        print(f"Using Binary LDAM Loss with max_m={args.ldam_max_m}, s={args.ldam_s}, "
              f"class counts=[{int(_num_negative)}, {int(_num_positive)}]")

elif args.dataset == 'imagenet':
    # Default: unweighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
else:
    # Default: SmoothBCEWithLogitsLoss with pos_weight (original predict.py behavior)
    pos_weight = torch.tensor([_num_negative / _num_positive]).to(device)
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
    # start_time = time.time()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
                
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            elif args.dataset == 'celeba':
                # For CelebA, ensure labels are for binary classification
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_correct += (preds == labels.int()).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    # elapsed_time = time.time() - start_time
    # print(f"Evaluation time: {elapsed_time:.2f} seconds")
    if save_preds:
        return acc, np.array(all_preds), np.array(all_labels)
    return acc, avg_loss

# After training, save predictions for each dataset
def save_predictions(model, train_loader, valid_loader, test_loader=None):
    print("Generating and saving predictions...")

    # Get predictions for training set
    _, train_preds, _ = evaluate(model, train_loader, save_preds=True)

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)

    test_preds = None
    if test_loader is not None:
        test_acc, test_preds, _ = evaluate(model, test_loader, save_preds=True)

    if args.dataset == 'imagenet':
        selected_indices = getattr(args, "selected_indices", None)
        if selected_indices is None:
            # No subsampling (e.g., LTL baselines): train_preds covers all training samples
            combined_preds = np.concatenate((train_preds.flatten(), valid_preds.flatten()))
        else:
            original_count = len(train_idx)
            sub_count = len(selected_indices)
            sub_preds = train_preds[:sub_count]
            if getattr(args, 'pad_missing', False):
                full_orig_preds = np.full((original_count,), -1, dtype=int)
                for idx, pred in zip(selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                combined_preds = np.concatenate((full_orig_preds.flatten(), valid_preds.flatten()))
            else:
                combined_preds = np.concatenate((sub_preds.flatten(), valid_preds.flatten()))
        # Append test predictions if available
        if test_preds is not None:
            combined_preds = np.concatenate((combined_preds, test_preds.flatten()))
    else:
        # Handle celeba or other cases
        if len(modified_image_paths) > 0:
            gen_cnt = len(modified_image_labels)
            if getattr(args, "pad_missing", False):
                # Pad missing original training predictions with -1 (int) instead of NaN
                full_orig_preds = np.full((args.orig_train_count,), -1, dtype=int)
                sub_preds = train_preds[:len(args.selected_indices)]
                for idx, pred in zip(args.selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                modified_preds = train_preds[len(args.selected_indices):]
                combined_preds = np.concatenate((
                    full_orig_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
            else:
                original_train_preds = train_preds[:-gen_cnt]
                modified_preds = train_preds[-gen_cnt:]
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
                *([test_preds.flatten()] if test_preds is not None else []),
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
        
        for batch_data in train_loader:
            if len(batch_data) == 3:
                images, labels, batch_idx = batch_data
            else:
                images, labels = batch_data
                batch_idx = None
                
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                
                # Attribute-Level manual loss adjustments
                if args.ltl_level == 'attribute' and batch_idx is not None:
                    if args.ltl_baseline == 'ldam':
                        m = train_dataset_margins[batch_idx]
                        idx_mask = torch.zeros_like(outputs, dtype=torch.bool)
                        idx_mask.scatter_(1, labels.unsqueeze(1).long(), True)
                        outputs[idx_mask] -= m
                        outputs *= args.ldam_s
                        raw_loss = criterion(outputs, labels)
                        w = train_dataset_sample_weights[batch_idx]
                        loss = (raw_loss * w).mean()
                    elif args.ltl_baseline == 'class_weight':
                        raw_loss = criterion(outputs, labels)
                        w = train_dataset_sample_weights[batch_idx]
                        loss = (raw_loss * w).mean()
                    else:
                        loss = criterion(outputs, labels)
                else:
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

        train_acc = total_correct / len(train_loader.dataset)
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
    save_predictions(model, train_loader, valid_loader, test_loader=test_loader)
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
# Print training start message and record total training time
# print(f"Starting training for {args.num_epochs} epochs...")
total_start_time = time.time()
# Ensure test_loader is defined for datasets without a test split
if 'test_loader' not in dir():
    test_loader = None
train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
    model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=args.num_epochs
)
total_elapsed_time = time.time() - total_start_time
print(f"Total training time for {args.num_epochs} epochs: {total_elapsed_time:.2f} seconds")



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
    data_dir = Path("exampleData/imagenet_subset")
    full_ds = ImageFolder(data_dir)
    labels_all = np.array(full_ds.targets)
    # Dynamically check whether predictions include the test split
    expected_train = len(train_idx)
    expected_val = len(val_idx)
    test_idx_path = Path('save/test_idx_imagenet.npy')
    if test_idx_path.exists():
        _test_idx = np.load(test_idx_path)
    else:
        _test_idx = np.array([], dtype=int)
    expected_test = len(_test_idx)
    total_predictions = len(predictions)
    total_without_test = expected_train + expected_val
    total_with_test = total_without_test + expected_test

    if expected_test and total_predictions == total_with_test:
        labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx], labels_all[_test_idx]])
        split = [TRAIN] * expected_train + [VALID] * expected_val + [TEST] * expected_test
    elif total_predictions == total_without_test:
        labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx]])
        split = [TRAIN] * expected_train + [VALID] * expected_val
        if expected_test:
            print(
                "Warning: Predictions file omits test samples (expected total "
                f"{total_with_test}, got {total_predictions}). Diagnostics run on train+val only."
            )
    else:
        print(
            "Warning: Predictions length ({}) does not match expected counts ({}/{}).".format(
                total_predictions, total_without_test, total_with_test
            )
        )
        labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx]])
        split = [TRAIN] * expected_train + [VALID] * expected_val

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

    # Adjust split to match combined_labels (following alpred.py approach)
    split = (
        [TRAIN] * len(all_train_labels) +
        [VALID] * len(valid_labels) +
        [TEST] * len(test_labels) +
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