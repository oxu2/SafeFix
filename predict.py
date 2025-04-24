import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
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
import datetime
from timm import create_model
import matplotlib.pyplot as plt
from transformers import CLIPModel
from transformers import CLIPConfig
import random
import re  # Added for log parsing
from utils.stat import ModelDiagnose  # Import ModelDiagnose from local stat.py
from itertools import combinations
from processData.report import ReportGenerator

import torch.nn.functional as F

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
parser.add_argument('--olabel', type=str2bool, default=False,
                    help='Use original dataset labels (default: False); if False, uses labels from feasible_file')
parser.add_argument('--gen_image_dir', type=str, default=None,
                    help='(Optional) Directory containing generated images. Defaults to exampleData/<attributes>')
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Use pre-trained weights (default: False)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--attributes', type=str, default="redhair_brownskin", 
                    help='attributes to detect, e.g. redhair_brownskin, redhair, etc.')
parser.add_argument('--model', type=str, default='clip', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, or clip)')
parser.add_argument('--plot', type=str2bool, default=False, help='Enable plotting training metrics')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--num_added_img', type=int, default=4000,
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
args = parser.parse_args()
attributes = args.attributes

# Set default feasible_file based on attributes if not provided
if args.feasible_file is None:
    args.feasible_file = f"save/feasible_{attributes}.json"
# Set default gen_image_dir based on attributes if not provided
if args.gen_image_dir is None:
    args.gen_image_dir = f"exampleData/{attributes}"

verbose = args.verbose

# Adjust learning rate default for ViT if not manually overridden
if args.model == 'vit' and args.lr == 1e-4:
    args.lr = 5e-5
    if verbose:
        print(f"Adjusted learning rate for ViT to {args.lr}")

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
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # Initialize CLIP with random weights
        self.clip = CLIPModel(CLIPConfig())
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.clip.config.projection_dim, 1)

    def forward(self, pixel_values):
        # Get image features and classify
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        feats = self.dropout(feats)
        feats = self.dropout2(feats)
        return self.classifier(feats)

# Load the feasible images (color hair images) from feasible.json
if os.path.exists(args.feasible_file):
    with open(args.feasible_file, 'r') as f:
        feasible_images = json.load(f)
    if verbose:
        print(f"Loaded {len(feasible_images)} feasible {args.attributes} images")
else:
    feasible_images = {}
    attributes = "origin"  # fallback attributes when feasible_file is missing
    print(f"Warning: {args.feasible_file} not found. No additional {args.attributes} images will be added.")

with open("exampleData/celebA/list_attr_celeba.txt") as f:
    lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]
    
all_datas = ['/data/oxx220000/img_align_celeba/' + line for line in lines][1:]

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

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
# Prepare sample keys from JSON
json_keys = list(feasible_images.keys())

# If user requests more images than JSON entries, sample from all generated files instead
if args.num_added_img > len(json_keys):
    # List all .jpg files in the generated images directory and use their basenames
    all_files = os.listdir(args.gen_image_dir)
    file_keys = [os.path.splitext(f)[0] for f in all_files if f.lower().endswith('.jpg')]
    print(f"Warning: requested {args.num_added_img} generated {args.attributes} images, "
          f"but only {len(json_keys)} feasible. Sampling from all {len(file_keys)} files instead.")
    sample_keys = file_keys
else:
    sample_keys = json_keys

# Warn if fewer images are available than requested in the chosen sample_keys
if len(sample_keys) < args.num_added_img:
    print(f"Warning: requested {args.num_added_img} generated {args.attributes} images, "
          f"but only {len(sample_keys)} available.")

# Perform sampling
print("Perform sampling for generated images")
# 
selected_filenames = random.sample(sample_keys, min(args.num_added_img, len(sample_keys)))
print(f"Selected {len(selected_filenames)} filenames for sampling")
# Build a map from base image number to label for quick lookup
base_to_label = {}
for img_path, lbl in zip(train_imgs, train_labels):
    # Extract numeric basename without extension
    base = os.path.splitext(os.path.basename(img_path))[0]
    base_to_label[base] = lbl
for filename in selected_filenames:
    # Get the image file from the pictures directory
    # Use relative path since the script runs from the HiBug directory
    image_path = f"./{args.gen_image_dir}/{filename}.jpg"
    
    if not os.path.exists(image_path):
        print(f"Warning: Modified image {image_path} not found.")
        continue
    
    # Lookup label by base image number
    base = filename.split('_')[0]
    lbl = base_to_label.get(base)
    if lbl is not None:
        modified_image_paths.append(image_path)
        modified_image_labels.append(lbl)
        if verbose:
            print(f"Added {image_path} with label {lbl}")
    else:
        print(f"Warning: Could not find original image for {filename}")
print("Finish sampling for generated images")

# Print information about the added images
if args.gen_only:
    # In gen_only mode, we’re only ever using the generated set
    print(f"Using only generated images: total training images = {len(modified_image_labels)}")
else:
    print(f"Added {args.num_added_img} generated {attributes} images, "
          f"before training set has {len(train_labels)} images, "
          f"now {len(train_labels) + len(modified_image_labels)}")
          
# Load unlabeled predictions - moved outside conditional block
unlabeled_predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
unlabeled_predictions = np.array([unlabeled_predictions[i] for i in range(len(all_datas)) if split[i] == UNLABELED])
if verbose:
    print(f"Unlabeled predictions length: {len(unlabeled_predictions)}")

# Construct training data based on gen_only flag
if args.gen_only:
    # Use only generated images for training
    all_train_imgs = modified_image_paths
    all_train_labels = np.array(modified_image_labels)
    combined_labels = np.concatenate((all_train_labels, valid_labels, unlabeled_labels))
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
    train_idx_path = f"save/idx/train_idx_{attributes}_{args.num_added_img}imgs.npy"
    labels_path = f"save/labels/labels_{attributes}_{args.num_added_img}imgs.npy"
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
train_dataset = CelebADataset(all_train_imgs, all_train_labels, transform)
valid_dataset = CelebADataset(valid_imgs, valid_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

# Define model, loss, and optimizer
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")
if verbose and device.type == "cuda":
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

if args.model == 'resnet':
    # Create a simplified ResNet by removing deeper layers
    # Select resnet18 weights via new `weights` API
    if args.pretrained:
        base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        base_model = models.resnet18(weights=None)
    
    # Remove layers to reduce complexity
    base_model.layer3 = nn.Identity()
    base_model.layer4 = nn.Identity()
    
    # Replace the standard avgpool with an adaptive version for flexibility
    base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
    model = base_model
    
    # Adjust the fully-connected layer; since we removed layer3 and layer4, the output comes from layer2 (128 channels)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.7),  # Increased dropout probability for stronger regularization
        nn.Linear(128, 1)
    )
elif args.model == 'vit':
    # Initialize ViT with internal dropout and stochastic depth for regularization
    model = create_model(
        'vit_base_patch16_224',
        pretrained=args.pretrained,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    )
    # Modify the head for binary classification, adding dropout for regularization
    num_head_features = model.head.in_features
    model.head = nn.Sequential(
         nn.Dropout(p=0.7),  # Increased dropout probability for stronger regularization
         nn.Linear(num_head_features, 1)
    )
elif args.model == 'clip':
    # Use CLIP vision model for binary classification
    model = CLIPForBinary(dropout_rate=0.5)
    args.pretrained = False  # ensure downstream logs and prints reflect random weights for CLIP

model = model.to(device)
print(f"Using {args.model} model with {'pretrained' if args.pretrained else 'random'} weights")

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

# Use OneCycleLR scheduler for CLIP, CosineAnnealingLR for ViT, StepLR for resnet
if args.model == 'clip':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )
elif args.model == 'vit':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
else:  # resnet
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

def evaluate(model, data_loader, save_preds=False):
    model.eval()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
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
        print(f"Original train predictions length: {len(original_train_preds)}")
        print(f"Modified image predictions length: {len(modified_preds)}")
    else:
        original_train_preds = train_preds
        modified_preds = np.array([])  # Empty array if no modified images
        print(f"Train predictions length: {len(original_train_preds)}")
    
    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)
    print(f"Validation predictions length: {len(valid_preds)}")
    
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
    
    print(f"Combined predictions length: {len(combined_preds)} (should match combined_labels length: {len(combined_labels)})")
    # Determine filename suffix based on gen_only flag
    suffix = "_g" if args.gen_only else ""
    np.save(f"save/pred/predictions_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}.npy", combined_preds)
    

# Modify the train_model function to save predictions after training
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=20):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
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
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            if verbose:
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(total_loss)
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
print("Current time:", time_now)


# Run the model diagnostics after predictions are saved
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

# Determine filename suffix based on gen_only flag
suffix = "_g" if args.gen_only else ""
# Load the predictions we just saved
prediction_file = f"save/pred/predictions_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}.npy"
predictions = np.load(prediction_file)

# Adjust split to match combined_labels (including modified images in the training split)
split = (
    [TRAIN] * len(train_labels) +
    [VALID] * len(valid_labels) +
    [TRAIN] * len(modified_image_labels) +
    [UNLABELED] * len(unlabeled_labels)
)


if not args.gen_only:
    # Initialize diagnostics only if not gen_only
    MD = ModelDiagnose(labels=combined_labels,
                       predictions=predictions,
                       split=split,
                       attribute_dict_path=args.attribute_file,
                       print=args.verbose,
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

    # Generate and save report if requested
    if args.save_report:
        report_dir = "save/reports"
        os.makedirs(report_dir, exist_ok=True)
        MD.generate_report(f"{report_dir}/{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
        print(f"Diagnostic report saved to {report_dir}/{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
else:
    print("Skipping model diagnostics because gen_only=True")
    # Emphasize final epoch's Train Acc if diagnostics are skipped
    if len(train_acc_history) > 0:
        final_train_acc = train_acc_history[-1]
        print("\n" + "="*60)
        print(f"Final Epoch Train Accuracy (gen_only=True): {final_train_acc:.4f}")
        print("="*60 + "\n")

# --- Accuracy Comparison Logic ---
if args.gen_only:
    print("Skipping accuracy comparison due to gen_only flag being True")
else:
    # Use the determined compare_log_path instead of args.compare_log
    if compare_log_path:
        print(f"\nComparing accuracy with log file: {compare_log_path}")
        orig_entries, orig_overall = parse_log(compare_log_path)

        if orig_entries is not None:
            # Build lookup map for original accuracies
            orig_map = {(e['attribute'], e['description']): e['accuracy'] for e in orig_entries}

            # Get current overall accuracy from MD
            current_overall = MD.valid_acc

            # Compare overall accuracy
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


            # Compare attribute-specific accuracies
            print("\nAttribute-specific accuracy comparison:")
            current_compared_count = 0
            for (attribute, description), orig_acc in orig_map.items():
                # Retrieve current accuracy from MD.attributes
                current_acc = None
                if attribute in MD.attributes and description in MD.attributes[attribute]['valid acc']:
                    current_acc = MD.attributes[attribute]['valid acc'][description]

                if current_acc is None:
                    # print(f'{attribute}, {description}: not found in current run results') # Optional: uncomment to see missing ones
                    continue

                # Determine status
                if current_acc > orig_acc:
                    status = 'improved'
                elif current_acc < orig_acc:
                    status = 'declined'
                else:
                    status = 'same'
                print(f'{attribute}, {description}, {orig_acc:.6f} -> {current_acc:.6f}: {status}')
                current_compared_count += 1

            print(f"\nCompared {current_compared_count} attribute-specific accuracies.")

            # Check for entries present in current run but not in the original log (optional)
            # print("\nAccuracies present in current run but not in original log:")
            # for attribute, attr_data in MD.attributes.items():
            #     if 'valid acc' in attr_data:
            #         for description, current_acc in attr_data['valid acc'].items():
            #             if (attribute, description) not in orig_map:
            #                 print(f'{attribute}, {description}: {current_acc:.6f} (new)')

        else:
            # Use compare_log_path in the warning message
            print(f"Could not parse or find the comparison log file: {compare_log_path}")
    else:
        print("\nSkipping accuracy comparison: No comparison log file specified for the selected model or file not found.")
