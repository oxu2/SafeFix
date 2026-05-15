"""
Data processing utilities and dataset classes.
"""
from utils.imports import *
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE

def tsne(embedding):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(embedding)
    embedding = tsne.embedding_
    return embedding

def kmeans_selection(embedding,n):
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embedding)
    
    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idxs


# Transform functions
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

def make_loader(dataset, shuffle, batch_size=None, workers=None):
    """Create a DataLoader with flexible arguments for backward compatibility."""
    # If called with only 2 args (old style), get batch_size and workers from args
    if batch_size is None or workers is None:
        # This assumes args is accessible in the global scope - not ideal but keeps compatibility
        import sys
        frame = sys._getframe(1)
        args = frame.f_globals.get('args')
        if args:
            batch_size = batch_size or args.batch_size
            workers = workers or args.workers
        else:
            # Fallback defaults
            batch_size = batch_size or 32
            workers = workers or 4
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True)


# Dataset classes
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


class GeneratedImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)


class SimpleImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    
    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)


class GeneratedDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)

