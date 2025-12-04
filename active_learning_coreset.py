"""
Active Learning for Classification using Core-set approach
Based on: Sener, Ozan, and Silvio Savarese. "Active learning for convolutional neural networks: A core-set approach." arXiv preprint arXiv:1708.00489 (2017).

This implementation provides a complete Active Learning pipeline with:
1. Core-set selection strategy
2. ResNet/CLIP/ViT backbone support
3. CelebA dataset integration
4. Iterative training with uncertainty sampling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import argparse
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Disable tqdm progress bars globally
import os
os.environ['TQDM_DISABLE'] = '1'

# Try to disable tqdm if it's imported
try:
    from tqdm import tqdm
    tqdm.disable = True
except ImportError:
    pass

# Import from your existing codebase
from utils.defines import TRAIN, VALID, TEST, UNLABELED

class CoreSetStrategy:
    """
    Core-set selection strategy for Active Learning
    Selects samples that are most representative of the unlabeled data distribution
    """
    
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        
    def greedy_k_center(self, labeled_embeddings, unlabeled_embeddings, n_select):
        """
        Greedy k-center algorithm for core-set selection
        
        Args:
            labeled_embeddings: embeddings of already labeled data [n_labeled, embedding_dim]
            unlabeled_embeddings: embeddings of unlabeled data [n_unlabeled, embedding_dim]
            n_select: number of samples to select
            
        Returns:
            selected_indices: indices of selected samples in unlabeled set
        """
        n_unlabeled = unlabeled_embeddings.shape[0]
        
        if n_select >= n_unlabeled:
            return list(range(n_unlabeled))
        
        # Convert to numpy if tensors
        if torch.is_tensor(labeled_embeddings):
            labeled_embeddings = labeled_embeddings.cpu().numpy()
        if torch.is_tensor(unlabeled_embeddings):
            unlabeled_embeddings = unlabeled_embeddings.cpu().numpy()
        
        # Initialize with all labeled embeddings
        selected_embeddings = labeled_embeddings.copy()
        selected_indices = []
        
        for _ in range(n_select):
            # Compute distances from all unlabeled points to all selected points
            distances = euclidean_distances(unlabeled_embeddings, selected_embeddings)
            
            # For each unlabeled point, find distance to closest selected point
            min_distances = np.min(distances, axis=1)
            
            # Select the point that is farthest from all selected points
            # (but exclude already selected points)
            for idx in selected_indices:
                min_distances[idx] = -1  # Mark as already selected
                
            best_idx = np.argmax(min_distances)
            selected_indices.append(best_idx)
            
            # Add the selected point's embedding to the selected set
            selected_embeddings = np.vstack([selected_embeddings, unlabeled_embeddings[best_idx:best_idx+1]])
        
        return selected_indices

class CelebADataset(Dataset):
    """CelebA Dataset for Active Learning"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default image or handle error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx

class ResNetClassifier(nn.Module):
    """ResNet-based classifier"""
    
    def __init__(self, num_classes=2, embedding_dim=512):
        super(ResNetClassifier, self).__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)  # Changed to False for training from scratch
        self.backbone.fc = nn.Identity()  # Remove final layer
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(2048, num_classes)
        self.feature_projection = nn.Linear(2048, embedding_dim)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)  # [batch_size, 2048]
        projected_features = self.feature_projection(features)  # [batch_size, 512]
        logits = self.classifier(features)
        
        if return_features:
            return logits, projected_features
        return logits

class CLIPClassifier(nn.Module):
    """CLIP-based classifier"""
    
    def __init__(self, num_classes=2, embedding_dim=512):
        super(CLIPClassifier, self).__init__()
        from transformers import CLIPModel, CLIPConfig
        
        # Create CLIP model from config without pretrained weights
        config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel(config)  # Initialize from config without pretrained weights
        
        self.embedding_dim = embedding_dim
        clip_dim = self.clip_model.config.vision_config.hidden_size  # 768 for CLIP ViT-B/32
        self.classifier = nn.Linear(clip_dim, num_classes)
        self.feature_projection = nn.Linear(clip_dim, embedding_dim)
        
    def forward(self, x, return_features=False):
        # CLIP expects pixel values in [0, 1] range
        if x.max() > 1.0:
            x = x / 255.0
            
        vision_outputs = self.clip_model.vision_model(pixel_values=x)
        features = vision_outputs.pooler_output  # [batch_size, 768]
        projected_features = self.feature_projection(features)  # [batch_size, 512]
        logits = self.classifier(features)
        
        if return_features:
            return logits, projected_features
        return logits

class ActiveLearningTrainer:
    """Main Active Learning trainer using Core-set strategy"""
    
    def __init__(self, 
                 model_type='resnet',
                 num_classes=2,
                 embedding_dim=512,
                 device='cuda',
                 batch_size=32,
                 learning_rate=1e-4,
                 num_epochs=20,
                 verbose=True,
                 disable_progress_bar=True):
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Device setup with validation
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("Warning: CUDA device requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.disable_progress_bar = disable_progress_bar
        
        # Disable progress bars if requested
        if disable_progress_bar:
            os.environ['TQDM_DISABLE'] = '1'
            # Try to disable transformers progress bars
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except ImportError:
                pass
        
        # Initialize model
        if model_type == 'resnet':
            self.model = ResNetClassifier(num_classes, embedding_dim)
        elif model_type == 'clip':
            self.model = CLIPClassifier(num_classes, embedding_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        
        # Print model info and GPU memory if using CUDA
        if self.device.startswith('cuda'):
            print(f"Model loaded on {self.device}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
                print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.3f} GB")
        else:
            print(f"Model loaded on CPU")
        
        # Initialize core-set strategy
        self.coreset_strategy = CoreSetStrategy(embedding_dim)
        
        # Training history
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'al_rounds': [],
            'selected_samples': []
        }
    
    def create_transforms(self):
        """Create data transforms"""
        if self.model_type == 'clip':
            # Manual CLIP-style transforms without pretrained dependency
            def clip_transform(image):
                # CLIP standard preprocessing: resize to 224x224, normalize to [0,1]
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),  # Converts to [0,1] range
                ])
                return transform(image)
            
            return clip_transform
        else:
            # Standard transforms for ResNet
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def extract_embeddings(self, dataloader):
        """Extract embeddings from the model"""
        self.model.eval()
        embeddings = []
        
        # if self.verbose:
        print(f"Extracting embeddings from {len(dataloader)} batches...")
        
        with torch.no_grad():
            for i, (batch_data, _, _) in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                _, features = self.model(batch_data, return_features=True)
                embeddings.append(features.cpu())
                
                # Print progress every 20% of batches
                if self.verbose and i > 0 and (i + 1) % max(1, len(dataloader) // 5) == 0:
                    progress = (i + 1) / len(dataloader) * 100
                    print(f"  Progress: {progress:.0f}% ({i+1}/{len(dataloader)} batches)")
        
        # if self.verbose:
        print("Embedding extraction completed.")
        
        return torch.cat(embeddings, dim=0)
    
    def train_model(self, train_loader, val_loader):
        """Train the model for one round"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i, (batch_data, batch_labels, _) in enumerate(train_loader):
                try:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "CUBLAS" in str(e):
                        print(f"CUDA error encountered: {e}")
                        print("Switching to CPU and retrying...")
                        self.device = 'cpu'
                        self.model = self.model.cpu()
                        batch_data = batch_data.cpu()
                        batch_labels = batch_labels.cpu()
                        
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Retry the forward pass on CPU
                        optimizer.zero_grad()
                        outputs = self.model(batch_data)
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += batch_labels.size(0)
                        train_correct += (predicted == batch_labels).sum().item()
                    else:
                        raise e
            
            train_acc = 100 * train_correct / train_total
            val_acc = self.evaluate_model(val_loader)
            
            # Always show epoch results regardless of verbose setting
            print(f"Epoch {epoch+1}/{self.num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model with unique name based on configuration
                model_save_path = f'best_model_{self.model_type}_{self.num_classes}classes.pth'
                torch.save(self.model.state_dict(), model_save_path)
                if self.verbose:
                    print(f"  Saved best model to {model_save_path}")
        
        # Load best model with the correct path
        model_save_path = f'best_model_{self.model_type}_{self.num_classes}classes.pth'
        if os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
            if self.verbose:
                print(f"  Loaded best model from {model_save_path}")
        else:
            print(f"Warning: Best model file {model_save_path} not found, using current model")
        return train_acc, best_val_acc
    
    def evaluate_model(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels, _ in val_loader:
                try:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "CUBLAS" in str(e):
                        # Switch to CPU if CUDA error
                        batch_data = batch_data.cpu()
                        batch_labels = batch_labels.cpu()
                        outputs = self.model.cpu()(batch_data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                    else:
                        raise e
        
        return 100 * correct / total
    
    def active_learning_loop(self, 
                           all_image_paths, 
                           all_labels, 
                           all_splits,
                           initial_labeled_size=1000,
                           al_rounds=5,
                           samples_per_round=500):
        """
        Main Active Learning loop
        
        Args:
            all_image_paths: list of all image paths
            all_labels: list of all labels  
            all_splits: list of splits (TRAIN, VALID, TEST, UNLABELED)
            initial_labeled_size: initial number of labeled samples
            al_rounds: number of active learning rounds
            samples_per_round: number of samples to select each round
        """
        
        # Separate data by split
        train_indices = [i for i, split in enumerate(all_splits) if split == TRAIN]
        val_indices = [i for i, split in enumerate(all_splits) if split == VALID]
        test_indices = [i for i, split in enumerate(all_splits) if split == TEST]
        unlabeled_indices = [i for i, split in enumerate(all_splits) if split == UNLABELED]

        print(f"Dataset statistics:")
        print(f"  Train: {len(train_indices)}")
        print(f"  Validation: {len(val_indices)}")
        print(f"  Test: {len(test_indices)}")
        print(f"  Unlabeled: {len(unlabeled_indices)}")
        
        # Start with initial labeled set (subset of training data)
        np.random.shuffle(train_indices)
        initial_labeled = train_indices[:initial_labeled_size]
        remaining_unlabeled = train_indices[initial_labeled_size:] + unlabeled_indices
        
        current_labeled = initial_labeled.copy()
        current_unlabeled = remaining_unlabeled.copy()
        
        # Create transforms
        transform = self.create_transforms()
        
        # Create validation dataset (remains constant)
        val_dataset = CelebADataset(
            [all_image_paths[i] for i in val_indices],
            [all_labels[i] for i in val_indices],
            transform=transform
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Starting Active Learning with {len(current_labeled)} labeled samples")
        
        for round_num in range(al_rounds):
            print(f"\n=== Active Learning Round {round_num + 1}/{al_rounds} ===")
            print(f"Current labeled samples: {len(current_labeled)}")
            print(f"Current unlabeled samples: {len(current_unlabeled)}")
            
            # Create current training dataset
            train_dataset = CelebADataset(
                [all_image_paths[i] for i in current_labeled],
                [all_labels[i] for i in current_labeled],
                transform=transform
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Train model on current labeled data
            print("Training model...")
            train_acc, val_acc = self.train_model(train_loader, val_loader)
            
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['al_rounds'].append(round_num + 1)
            
            print(f"Round {round_num + 1} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # If this is the last round, don't select more samples
            if round_num == al_rounds - 1:
                break
            
            # Core-set selection for next round
            if len(current_unlabeled) > 0:
                # if self.verbose:
                print("Performing core-set selection...")
                
                # Extract embeddings for labeled data
                labeled_embeddings = self.extract_embeddings(train_loader)
                
                # Extract embeddings for unlabeled data
                unlabeled_dataset = CelebADataset(
                    [all_image_paths[i] for i in current_unlabeled],
                    [all_labels[i] for i in current_unlabeled],  # Labels not used for selection
                    transform=transform
                )
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False)
                unlabeled_embeddings = self.extract_embeddings(unlabeled_loader)
                
                # Select samples using core-set strategy
                n_select = min(samples_per_round, len(current_unlabeled))
                # if self.verbose:
                print(f"Selecting {n_select} samples using core-set strategy...")
                selected_indices_in_unlabeled = self.coreset_strategy.greedy_k_center(
                    labeled_embeddings, unlabeled_embeddings, n_select
                )
                
                # Convert to original indices
                selected_indices = [current_unlabeled[i] for i in selected_indices_in_unlabeled]
                
                print(f"Selected {len(selected_indices)} new samples for labeling")
                
                # Update labeled and unlabeled sets
                current_labeled.extend(selected_indices)
                current_unlabeled = [idx for idx in current_unlabeled if idx not in selected_indices]
                
                self.history['selected_samples'].append(selected_indices)
        
        print("\nActive Learning completed!")
        return self.history
    
    def get_versioned_filename(self, base_filename):
        """
        Create versioned filename by checking if file exists and renaming old ones
        """
        if not os.path.exists(base_filename):
            return base_filename
        
        # Find the highest existing version number
        version = 1
        while True:
            old_filename = base_filename.replace('.', f'_old{version}.')
            if not os.path.exists(old_filename):
                break
            version += 1
        
        # Rename existing file to old version
        if version == 1:
            old_filename = base_filename.replace('.', '_old1.')
        else:
            # Shift all versions up by 1
            for v in range(version - 1, 0, -1):
                current_old = base_filename.replace('.', f'_old{v}.')
                next_old = base_filename.replace('.', f'_old{v+1}.')
                if os.path.exists(current_old):
                    os.rename(current_old, next_old)
            old_filename = base_filename.replace('.', '_old1.')
        
        os.rename(base_filename, old_filename)
        print(f"Renamed existing {base_filename} to {old_filename}")
        return base_filename
    
    def plot_results(self, save_path='al_results.png'):
        """Plot Active Learning results"""
        # Use versioned filename
        save_path = self.get_versioned_filename(save_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['al_rounds'], self.history['train_acc'], 'b-o', label='Train Accuracy')
        plt.plot(self.history['al_rounds'], self.history['val_acc'], 'r-o', label='Validation Accuracy')
        plt.xlabel('Active Learning Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Active Learning Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        
        print(f"Results plot saved to {save_path}")

def load_celeba_data(seed: int = 42):
    """Load CelebA data with reproducible train/validation/test/unlabeled splits."""

    # Load image paths
    with open("exampleData/celebA/list_attr_celeba.txt") as f:
        lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]

    # Determine base directory
    if os.path.exists('/home/jovyan/ouser/dataset/img_align_celeba/'):
        input_dir = '/home/jovyan/ouser/dataset/img_align_celeba/'
    else:
        input_dir = '/data/oxx220000/img_align_celeba/'

    all_datas = [input_dir + line for line in lines][1:]

    # Load indices
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
        and cached_seed == seed
    )

    if cache_ready:
        valid_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
        unlabel_idx = np.load(unlabel_idx_path)
    else:
        val_pool = np.arange(len(all_datas))[-100000:]
        if val_pool.size < 20000:
            raise ValueError("Expected at least 20000 samples in CelebA validation/test pool.")
        rng = np.random.default_rng(seed)
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
                "seed": int(seed),
                "timestamp": time.time()
            }, meta_file)

    # Concatenate and reorder
    idxs = np.concatenate([train_idx, valid_idx, test_idx, unlabel_idx], axis=0)
    labels = np.load("exampleData/celebA/labels.npy")[idxs]
    all_datas = [all_datas[i] for i in idxs]
    split = (
        [TRAIN for _ in train_idx] +
        [VALID for _ in valid_idx] +
        [TEST for _ in test_idx] +
        [UNLABELED for _ in unlabel_idx]
    )

    return all_datas, labels, split

def load_imagenet_data(data_dir):
    """Load ImageNet data following the predict.py structure"""
    from torchvision.datasets import ImageFolder
    from pathlib import Path
    
    # Load the full dataset to get image paths and labels
    full_ds = ImageFolder(data_dir)
    
    # Get all image paths
    all_image_paths = [path for path, _ in full_ds.samples]
    all_labels = np.array([label for _, label in full_ds.samples])
    
    # Load train/val indices (following predict.py pattern)
    try:
        train_idx = np.load('save/train_idx_imagenet.npy')
        val_idx = np.load('save/val_idx_imagenet.npy')
    except FileNotFoundError:
        print("File Not Found for save/train_idx_imagenet.npy")
        # Create indices if they don't exist
        total_samples = len(all_image_paths)
        train_size = int(0.8 * total_samples)
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:]
        
        # Save indices for consistency
        os.makedirs('save', exist_ok=True)
        np.save('save/train_idx_imagenet.npy', train_idx)
        np.save('save/val_idx_imagenet.npy', val_idx)
    
    # Create splits array
    splits = [VALID for _ in range(len(all_image_paths))]  # Initialize all as VALID
    
    # Set train indices
    for idx in train_idx:
        splits[idx] = TRAIN
    
    # Set validation indices  
    for idx in val_idx:
        splits[idx] = VALID
    
    # For active learning, we'll treat some training data as unlabeled
    # Keep first 1000 as labeled, rest as unlabeled
    unlabel_subset = train_idx[1000:]  # Take training samples beyond first 1000
    for idx in unlabel_subset:
        splits[idx] = UNLABELED
    
    return all_image_paths, all_labels, splits

def main():
    parser = argparse.ArgumentParser(description='Active Learning with Core-set Strategy')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['celeba', 'imagenet'],
                       help='Dataset to use (default: imagenet)')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'clip'],
                       help='Model type to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs per round')
    parser.add_argument('--initial-labeled', type=int, default=1000, help='Initial labeled samples')
    parser.add_argument('--al-rounds', type=int, default=5, help='Number of AL rounds')
    parser.add_argument('--samples-per-round', type=int, default=500, help='Samples to select per round')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', default=True, action='store_true', help='Disable progress output')
    parser.add_argument('--enable-progress-bars', action='store_true', help='Enable progress bars (disabled by default)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for CelebA validation/test split cache')
    
    args = parser.parse_args()
    
    # Set verbose based on arguments
    verbose = args.verbose and not args.quiet
    disable_progress_bar = not args.enable_progress_bars  # Default is to disable progress bars
    
    # Device setup with proper error handling
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = f'cuda:{torch.cuda.current_device()}'
            print(f"Using GPU: {device} ({torch.cuda.get_device_name()})")
            # Clear GPU cache
            torch.cuda.empty_cache()
        else:
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
    else:
        device = args.device
        print(f"Using device: {device}")
    
    # Load dataset based on choice
    if args.dataset == 'imagenet':
        data_dir = Path("exampleData/imagenet_subset")
        print("Loading ImageNet dataset...")
        all_image_paths, all_labels, all_splits = load_imagenet_data(data_dir)
    else:
        print("Loading CelebA dataset...")
        all_image_paths, all_labels, all_splits = load_celeba_data(seed=args.seed)
    
    print(f"Loaded {len(all_image_paths)} images")
    print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Determine number of classes from the data
    if args.dataset == 'imagenet':
        num_classes = len(np.unique(all_labels))
    else:
        num_classes = 2  # CelebA is binary classification
    
    print(f"Number of classes: {num_classes}")
    
    # Initialize trainer
    trainer = ActiveLearningTrainer(
        model_type=args.model,
        num_classes=num_classes,
        embedding_dim=512,
        device=device,  # Use the detected device
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        verbose=verbose,
        disable_progress_bar=disable_progress_bar
    )
    
    # Run Active Learning
    history = trainer.active_learning_loop(
        all_image_paths=all_image_paths,
        all_labels=all_labels,
        all_splits=all_splits,
        initial_labeled_size=args.initial_labeled,
        al_rounds=args.al_rounds,
        samples_per_round=args.samples_per_round
    )
    
    # Create filename with dataset and pretrained info
    pretrained_suffix = "nopretrain"  # Since we changed to no pretrained weights
    base_filename = f'al_results_{args.dataset}_{args.model}_{pretrained_suffix}'
    
    # Plot and save results
    trainer.plot_results(f'{base_filename}.png')
    
    # Save history with versioning
    history_filename = f'al_history_{args.dataset}_{args.model}_{pretrained_suffix}.json'
    
    # Use versioning for history file too
    if os.path.exists(history_filename):
        history_filename = trainer.get_versioned_filename(history_filename)
    
    with open(history_filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'], 
            'al_rounds': history['al_rounds'],
            'selected_samples': [list(map(int, samples)) for samples in history['selected_samples']]
        }
        json.dump(history_json, f, indent=2)
    
    print(f"History saved to {history_filename}")

if __name__ == "__main__":
    main()
