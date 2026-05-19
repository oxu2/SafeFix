#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ResNet-18 on an ImageNet-style dataset'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./imagenet_subset',
        help='Root directory of dataset (must contain train/ and val/ subfolders; defaults to "./data")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for train and validation (default: 64)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of DataLoader workers (default: 4)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use (default: 0)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=False,
        help='If set, use ImageNet-pretrained ResNet-18 weights'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='resnet18_imagenet.pth',
        help='Path to save the trained model (default: resnet18_imagenet.pth)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    return parser.parse_args()

def default_img_transform(train: bool):
    """
    Returns a standard image transform for training or validation.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


args = parse_args()

# reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# Use standard ImageNet transforms for ResNet
train_transform = default_img_transform(train=True)
val_transform = default_img_transform(train=False)

# Load ImageNet subset
data_dir = Path(args.data_dir)
full_ds = ImageFolder(data_dir)
train_idx = np.load('save/train_idx_imagenet.npy')
val_idx = np.load('save/val_idx_imagenet.npy')
train_ds = Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
val_ds   = Subset(ImageFolder(data_dir, transform=val_transform), val_idx)
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)
num_classes = len(full_ds.classes)

# model setup
weights = models.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# training loop
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        train_loss += loss.item() * inputs.size(0)
        train_correct += torch.sum(preds == labels).item()

    train_loss /= len(train_ds)
    train_acc = train_correct / len(train_ds)

    # validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            val_loss += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels).item()

    val_loss /= len(val_ds)
    val_acc = val_correct / len(val_ds)

    print(
        f'Epoch {epoch}/{args.epochs} | '
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
    )

    scheduler.step()

# save final model
torch.save(model.state_dict(), args.save_path)
print(f'Model saved to {args.save_path}')
