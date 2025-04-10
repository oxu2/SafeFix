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
# print(torchvision.__version__) #0.11.1+cu113
from transformers import ViTModel, ViTFeatureExtractor
import datetime
from timm import create_model

time_now = datetime.datetime.now().strftime("%m%d%H%M")
print("Current time:", time_now)
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model on CelebA dataset')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Enable verbose output')
parser.add_argument('--feasible-file', type=str, default="red_hair_color_feasible_04080238.json",
                    help='Path to the feasible images JSON file')
parser.add_argument('--olabel', action='store_true', default=False,
                    help='Use original dataset labels. If False, uses labels from feasible_file')
parser.add_argument('--image_dir', type=str, default="redhair",
                    help='Directory containing images')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Use pre-trained weights (default: True)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--attribute', type=str, default="hair color", 
                    help='Attribute to detect (e.g., "hair color", "eyes")')
parser.add_argument('--model', type=str, default='vit', choices=['resnet18', 'vit'],
                    help='Model architecture to use (default: vit_b_16)')
args = parser.parse_args()
verbose = args.verbose

# Extract tag from the feasible_file name
feasible_file = args.feasible_file
tag = feasible_file.split('_')[0] if '_' in feasible_file else "unknown"
# Extract timestamp from feasible_file (assuming format like "red_hair_color_feasible_04080238.json")
timestamp = feasible_file.split('_')[-1].split('.')[0] if '_' in feasible_file else "unknown"

# Load the feasible images (color hair images) from feasible.json
if os.path.exists(feasible_file):
    with open(feasible_file, 'r') as f:
        feasible_images = json.load(f)
    if verbose:
        print(f"Loaded {len(feasible_images)} feasible {tag} {args.attribute} images")
    # print(f"Loaded {len(feasible_images)} feasible {tag} {args.attribute} images")
else:
    feasible_images = {}
    timestamp = "origin"  # fallback timestamp when feasible_file is missing
    print(f"Warning: {feasible_file} not found. No additional {tag} {args.attribute} images will be added.")

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
    transforms.Resize((224, 224)),  # Resize to 224x224 which is common for both ResNet and ViT
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

for filename in feasible_images.keys():
    # Get the image file from the pictures directory
    # Use relative path since the script runs from the HiBug directory
    image_path = f"./{args.image_dir}/{filename}.jpg"
    
    if not os.path.exists(image_path):
        print(f"Warning: Modified image {image_path} not found.")
        continue
    
    # Extract the base image number (e.g., "007" from "007_red")
    base_number = filename.split('_')[0]
    
    # Find the corresponding original image in train_imgs by number
    # We need to find the index of the original image to get its label
    original_idx = None
    for i, img_path in enumerate(train_imgs):
        if f"/{base_number}.jpg" in img_path or f"/{base_number}_" in img_path:
            original_idx = i
            break
    
    # If we found the original image, add the modified image with the same label
    if original_idx is not None:
        modified_image_paths.append(image_path)
        modified_image_labels.append(train_labels[original_idx])
        if verbose:
            print(f"Added {image_path} with label {train_labels[original_idx]}")
    else:
        print(f"Warning: Could not find original image for {filename}")

# Print information about the added images
print(f"Added {len(modified_image_paths)} generated {tag} {args.attribute} images, before training set has {len(train_labels)} images, now {len(train_labels) + len(modified_image_labels)}")

# Load unlabeled predictions - moved outside conditional block
unlabeled_predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
unlabeled_predictions = np.array([unlabeled_predictions[i] for i in range(len(all_datas)) if split[i] == UNLABELED])
if verbose:
    print(f"Unlabeled predictions length: {len(unlabeled_predictions)}")

# Add the modified images to the training data
if modified_image_paths:
    # Combine original training data with modified images
    all_train_imgs = train_imgs + modified_image_paths
    all_train_labels = np.concatenate((train_labels, np.array(modified_image_labels)))
    
    # Create combined_train_idx including the original train indices and modified image filenames
    modified_filenames = [os.path.basename(path).split('.')[0] for path in modified_image_paths]
    combined_train_idx = np.array([*train_idx, *modified_filenames], dtype=object)
    
    # Create combined_labels including train, valid, modified, and unlabeled labels
    combined_labels = np.concatenate((train_labels, valid_labels, np.array(modified_image_labels), unlabeled_labels))
    
    # Create combined predictions (train, valid, modified, unlabeled)
    # For the train and modified images, we'll use their actual labels as the predictions
    # (since we don't have actual predictions yet before training)
    initial_predictions = np.concatenate((train_labels, valid_labels, np.array(modified_image_labels), unlabeled_predictions))
    
    # Save the combined indices and labels with timestamp (skip if files already exist)
    train_idx_path = f"./combined_train_idx_{timestamp}.npy"
    labels_path = f"./combined_labels_{timestamp}.npy"

    if not os.path.exists(train_idx_path):
        np.save(train_idx_path, combined_train_idx)
        print(f"Saved combined train indices to {train_idx_path}")
    else:
        if verbose:
            print(f"{train_idx_path} already exists, skipping save.")

    if not os.path.exists(labels_path):
        np.save(labels_path)
        print(f"Saved combined labels to {labels_path}")
    else:
        if verbose:
            print(f"{labels_path} already exists, skipping save.")
else:
    all_train_imgs = train_imgs
    all_train_labels = train_labels
    # Define combined_labels here to be available for save_predictions function
    combined_labels = np.concatenate((train_labels, valid_labels, unlabeled_labels))

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model based on user choice
if args.model == 'resnet18':
    model = models.resnet18(pretrained=args.pretrained)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1)
    )  # Adjusting for binary classification
elif args.model == 'vit':
    # Use the smaller ViT-Base variant with proper import
    model = create_model('vit_base_patch16_224', pretrained=args.pretrained)
    # Modify the head for binary classification
    num_head_features = model.head.in_features
    model.head = nn.Linear(num_head_features, 1)

model = model.to(device)
print(f"Using {args.model} model with {'pretrained' if args.pretrained else 'random'} weights")

num_negative = (all_train_labels == 0).sum()
num_positive = (all_train_labels == 1).sum()
pos_weight = torch.tensor([num_negative / num_positive]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Use learning rate from command line args with weight decay for L2 regularization
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
# If args.model is 'vit', use cosine annealing scheduler to improve loss behavior; otherwise, use StepLR
if args.model == 'vit':
    # For ViT, use cosine annealing scheduler to improve loss behavior
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
else:
    # For ResNet and others, keep the StepLR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

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
            original_train_preds.flatten(),  # original train predictions
            valid_preds.flatten(),          # validation predictions
            modified_preds.flatten(),       # modified image predictions
            unlabeled_predictions           # existing unlabeled predictions
        ))
    else:
        combined_preds = np.concatenate((
            original_train_preds.flatten(),  # original train predictions
            valid_preds.flatten(),          # validation predictions
            unlabeled_predictions           # existing unlabeled predictions
        ))
    
    print(f"Combined predictions length: {len(combined_preds)} (should match combined_labels length: {len(combined_labels)})")
    np.save(f"./combined_predictions_{timestamp}_{args.model}_{time_now}.npy", combined_preds)
    
    if verbose:
        print(f"Saved predictions to ./combined_predictions_{timestamp}_{args.model}_{time_now}.npy")

# Modify the train_model function to save predictions after training
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save predictions after training
    save_predictions(model, train_loader, valid_loader)

# Start training with scheduler

train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=10)
print("Current time:", time_now)