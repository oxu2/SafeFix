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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train ResNet18 model on CelebA dataset')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Enable verbose output')
parser.add_argument('--feasible-file', type=str, default="red_hair_color_feasible_04080238.json",
                    help='Path to the feasible images JSON file')
parser.add_argument('--olabel', action='store_true', default=False,
                    help='Use original dataset labels. If False, uses labels from feasible_file')
parser.add_argument('--dir', type=str, default="redhair",
                    help='Directory containing images')
args = parser.parse_args()
verbose = args.verbose

# Extract tag from the feasible_file name
feasible_file = args.feasible_file
tag = feasible_file.split('_')[0] if '_' in feasible_file else "unknown"

# Load the feasible images (color hair images) from feasible.json
if os.path.exists(feasible_file):
    with open(feasible_file, 'r') as f:
        feasible_images = json.load(f)
    # if verbose:
    #     print(f"Loaded {len(feasible_images)} feasible {tag} hair images")
    print(f"Loaded {len(feasible_images)} feasible {tag} hair images")
else:
    feasible_images = {}
    print(f"Warning: {feasible_file} not found. No additional {tag} hair images will be added.")

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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create Datasets and Loaders
train_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TRAIN]
valid_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == VALID]

train_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TRAIN])
valid_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == VALID])

# Add the colored hair images to the training set
# These images' labels will be the same as their corresponding original images
colored_image_paths = []
colored_image_labels = []

for filename in feasible_images.keys():
    # Get the image file from the pictures directory
    # Use relative path since the script runs from the HiBug directory
    image_path = f"./{args.dir}/{filename}.jpg"
    
    # Extract the base image number (e.g., "007" from "007_red")
    base_number = filename.split('_')[0]
    
    # Find the corresponding original image in train_imgs by number
    # We need to find the index of the original image to get its label
    original_idx = None
    for i, img_path in enumerate(train_imgs):
        if f"/{base_number}.jpg" in img_path or f"/{base_number}_" in img_path:
            original_idx = i
            break
    
    # If we found the original image, add the colored hair image with the same label
    if original_idx is not None:
        colored_image_paths.append(image_path)
        colored_image_labels.append(train_labels[original_idx])
        if verbose:
            print(f"Added {image_path} with label {train_labels[original_idx]}")
    else:
        print(f"Warning: Could not find original image for {filename}")

# Print information about the added images
print(f"Added {len(colored_image_paths)} generated images, before training set has {len(train_labels)} images, now {len(train_labels) + len(colored_image_labels)}")

# Add the colored hair images to the training data
if colored_image_paths:
    # Combine original training data with colored hair images
    all_train_imgs = train_imgs + colored_image_paths
    all_train_labels = np.concatenate((train_labels, np.array(colored_image_labels)))
else:
    all_train_imgs = train_imgs
    all_train_labels = train_labels

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
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjusting for binary classification
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
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
        val_acc = evaluate(model, valid_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

def evaluate(model, data_loader):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            total_correct += (preds == labels.int()).sum().item()

    return total_correct / len(data_loader.dataset)

# Start training
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10)