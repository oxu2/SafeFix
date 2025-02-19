import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from utils.defines import *

# Load images, labels, predictions, and splits
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with open("exampleData/celebA/list_attr_celeba.txt") as f:
    lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]
    
all_datas = ['/data/oxx220000/img_align_celeba/' + line for line in lines][1:]

train_idx = np.load("exampleData/celebA/train_idx.npy")[:80000]
valid_idx = np.arange(len(all_datas))[-100000:-80000]
unlabel_idx = np.arange(len(all_datas))[-80000:]

idxs = np.concatenate([train_idx, valid_idx, unlabel_idx], axis=0)
labels = np.load("exampleData/celebA/labels.npy")[idxs]
predictions = np.load("exampleData/celebA/predictions.npy")[idxs]

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

train_dataset = CelebADataset(train_imgs, train_labels, transform)
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
        # print(f"Epoch {epoch+1}/{num_epochs}")
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