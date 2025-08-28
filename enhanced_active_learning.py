"""
Enhanced Active Learning Framework
Supporting multiple sampling strategies: Core-set, Uncertainty, Random, Diversity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from active_learning_coreset import ActiveLearningTrainer, CoreSetStrategy
import random
import os

# Disable progress bars by default
os.environ['TQDM_DISABLE'] = '1'

# Try to disable tqdm if it's imported
try:
    from tqdm import tqdm
    tqdm.disable = True
except ImportError:
    pass

# Try to disable transformers progress bars
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

class UncertaintyStrategy:
    """Uncertainty-based sampling strategy"""
    
    def __init__(self, strategy_type='entropy'):
        """
        Args:
            strategy_type: 'entropy', 'least_confidence', 'margin'
        """
        self.strategy_type = strategy_type
    
    def calculate_uncertainty(self, probabilities):
        """Calculate uncertainty scores"""
        if self.strategy_type == 'entropy':
            # Entropy uncertainty
            return -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        elif self.strategy_type == 'least_confidence':
            # Least confidence
            return 1 - np.max(probabilities, axis=1)
        elif self.strategy_type == 'margin':
            # Margin sampling (difference between max and second max probability)
            sorted_probs = np.sort(probabilities, axis=1)
            return 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def select_samples(self, probabilities, n_select):
        """Select samples based on uncertainty"""
        uncertainties = self.calculate_uncertainty(probabilities)
        selected_indices = np.argsort(uncertainties)[-n_select:]
        return selected_indices.tolist()

class DiversityStrategy:
    """Diversity-based sampling strategy"""
    
    def __init__(self, strategy_type='kmeans'):
        """
        Args:
            strategy_type: 'kmeans', 'cosine_diversity'
        """
        self.strategy_type = strategy_type
    
    def kmeans_sampling(self, embeddings, n_select):
        """K-means clustering sampling"""
        if n_select >= len(embeddings):
            return list(range(len(embeddings)))
        
        # Use K-means clustering
        kmeans = KMeans(n_clusters=n_select, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select the sample closest to cluster center from each cluster
        selected_indices = []
        for i in range(n_select):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                # Find sample closest to cluster center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return selected_indices
    
    def cosine_diversity_sampling(self, embeddings, n_select):
        """Cosine distance-based diversity sampling"""
        if n_select >= len(embeddings):
            return list(range(len(embeddings)))
        
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))
        
        # Randomly select first sample
        first_idx = random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Greedily select remaining samples
        for _ in range(n_select - 1):
            if not remaining_indices:
                break
            
            max_min_distance = -1
            best_idx = None
            
            for candidate_idx in remaining_indices:
                # Calculate minimum distance from candidate to selected samples
                candidate_embedding = embeddings[candidate_idx:candidate_idx+1]
                selected_embeddings = embeddings[selected_indices]
                
                distances = cosine_distances(candidate_embedding, selected_embeddings)[0]
                min_distance = np.min(distances)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def select_samples(self, embeddings, n_select):
        """Select diverse samples"""
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        if self.strategy_type == 'kmeans':
            return self.kmeans_sampling(embeddings, n_select)
        elif self.strategy_type == 'cosine_diversity':
            return self.cosine_diversity_sampling(embeddings, n_select)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

class RandomStrategy:
    """Random sampling strategy (baseline method)"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def select_samples(self, n_total, n_select):
        """Randomly select samples"""
        return np.random.choice(n_total, size=min(n_select, n_total), replace=False).tolist()

class EnhancedActiveTrainer(ActiveLearningTrainer):
    """Enhanced Active Learning trainer supporting multiple strategies"""
    
    def __init__(self, strategy='coreset', **kwargs):
        """
        Args:
            strategy: 'coreset', 'uncertainty', 'diversity', 'random'
        """
        super().__init__(**kwargs)
        
        self.strategy_name = strategy
        
        # Initialize strategy
        if strategy == 'coreset':
            self.strategy = CoreSetStrategy(self.embedding_dim)
        elif strategy == 'uncertainty':
            self.strategy = UncertaintyStrategy('entropy')
        elif strategy == 'diversity':
            self.strategy = DiversityStrategy('kmeans')
        elif strategy == 'random':
            self.strategy = RandomStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_predictions_and_features(self, dataloader):
        """Get prediction probabilities and features"""
        self.model.eval()
        all_probabilities = []
        all_features = []
        
        with torch.no_grad():
            for batch_data, _, _ in dataloader:
                batch_data = batch_data.to(self.device)
                logits, features = self.model(batch_data, return_features=True)
                probabilities = F.softmax(logits, dim=1)
                
                all_probabilities.append(probabilities.cpu())
                all_features.append(features.cpu())
        
        probabilities = torch.cat(all_probabilities, dim=0)
        features = torch.cat(all_features, dim=0)
        
        return probabilities, features
    
    def select_samples_with_strategy(self, train_loader, unlabeled_loader, n_select):
        """Select samples using specified strategy"""
        
        if self.strategy_name == 'random':
            # Random strategy doesn't need model predictions
            n_unlabeled = len(unlabeled_loader.dataset)
            return self.strategy.select_samples(n_unlabeled, n_select)
        
        # Get predictions and features for unlabeled data
        unlabeled_probs, unlabeled_features = self.get_predictions_and_features(unlabeled_loader)
        
        if self.strategy_name == 'coreset':
            # Core-set strategy needs labeled data features
            labeled_probs, labeled_features = self.get_predictions_and_features(train_loader)
            return self.strategy.greedy_k_center(labeled_features, unlabeled_features, n_select)
        
        elif self.strategy_name == 'uncertainty':
            # Uncertainty strategy
            return self.strategy.select_samples(unlabeled_probs.numpy(), n_select)
        
        elif self.strategy_name == 'diversity':
            # Diversity strategy
            return self.strategy.select_samples(unlabeled_features, n_select)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
    
    def active_learning_loop(self, *args, **kwargs):
        """Override parent method to use new sampling strategy"""
        
        # Call parent method, but use new sampling logic
        all_image_paths = args[0]
        all_labels = args[1] 
        all_splits = args[2]
        initial_labeled_size = kwargs.get('initial_labeled_size', 1000)
        al_rounds = kwargs.get('al_rounds', 5)
        samples_per_round = kwargs.get('samples_per_round', 500)
        
        from utils.defines import TRAIN, VALID, UNLABELED
        
        # Separate data
        train_indices = [i for i, split in enumerate(all_splits) if split == TRAIN]
        val_indices = [i for i, split in enumerate(all_splits) if split == VALID]
        unlabeled_indices = [i for i, split in enumerate(all_splits) if split == UNLABELED]
        
        print(f"Using strategy: {self.strategy_name}")
        print(f"Data statistics: Train={len(train_indices)}, Val={len(val_indices)}, Unlabeled={len(unlabeled_indices)}")
        
        # Initialize
        np.random.shuffle(train_indices)
        initial_labeled = train_indices[:initial_labeled_size]
        remaining_unlabeled = train_indices[initial_labeled_size:] + unlabeled_indices
        
        current_labeled = initial_labeled.copy()
        current_unlabeled = remaining_unlabeled.copy()
        
        # Create transforms
        transform = self.create_transforms()
        
        # Create validation dataset
        from active_learning_coreset import CelebADataset
        from torch.utils.data import DataLoader
        
        val_dataset = CelebADataset(
            [all_image_paths[i] for i in val_indices],
            [all_labels[i] for i in val_indices],
            transform=transform
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Active Learning loop
        for round_num in range(al_rounds):
            print(f"\n=== Active Learning Round {round_num + 1}/{al_rounds} ===")
            print(f"Current labeled samples: {len(current_labeled)}")
            print(f"Current unlabeled samples: {len(current_unlabeled)}")
            
            # Create training dataset
            train_dataset = CelebADataset(
                [all_image_paths[i] for i in current_labeled],
                [all_labels[i] for i in current_labeled],
                transform=transform
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Train model
            print("Training model...")
            train_acc, val_acc = self.train_model(train_loader, val_loader)
            
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc) 
            self.history['al_rounds'].append(round_num + 1)
            
            print(f"Round {round_num + 1} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Last round doesn't need to select new samples
            if round_num == al_rounds - 1:
                break
            
            # Sample selection
            if len(current_unlabeled) > 0:
                print(f"Using {self.strategy_name} strategy to select samples...")
                
                # Create unlabeled dataset
                unlabeled_dataset = CelebADataset(
                    [all_image_paths[i] for i in current_unlabeled],
                    [all_labels[i] for i in current_unlabeled],
                    transform=transform
                )
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False)
                
                # Select samples
                n_select = min(samples_per_round, len(current_unlabeled))
                selected_indices_in_unlabeled = self.select_samples_with_strategy(
                    train_loader, unlabeled_loader, n_select
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

def compare_strategies():
    """Compare different Active Learning strategies"""
    
    print("=== Active Learning Strategy Comparison ===")
    
    strategies = ['random', 'uncertainty', 'diversity', 'coreset']
    results = {}
    
    # Load data
    from active_learning_coreset import load_celeba_data
    all_image_paths, all_labels, all_splits = load_celeba_data()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for strategy in strategies:
        print(f"\n--- Testing strategy: {strategy} ---")
        
        trainer = EnhancedActiveTrainer(
            strategy=strategy,
            model_type='resnet',
            num_classes=2,
            embedding_dim=512,
            device=device,
            batch_size=16,
            learning_rate=1e-4,
            num_epochs=3,  # Quick test
            verbose=False,  # Disable verbose output for comparison
            disable_progress_bar=True  # Disable progress bars
        )
        
        try:
            history = trainer.active_learning_loop(
                all_image_paths, all_labels, all_splits,
                initial_labeled_size=300,
                al_rounds=3,
                samples_per_round=100
            )
            
            results[strategy] = history
            
            # Plot results
            trainer.plot_results(f'results_{strategy}.png')
            
        except Exception as e:
            print(f"Strategy {strategy} test failed: {e}")
            results[strategy] = None
    
    # Compare results
    print("\n=== Strategy Comparison Results ===")
    for strategy, history in results.items():
        if history:
            final_val_acc = history['val_acc'][-1]
            print(f"{strategy}: Final validation accuracy = {final_val_acc:.2f}%")
        else:
            print(f"{strategy}: Test failed")
    
    return results

if __name__ == "__main__":
    # Run strategy comparison
    results = compare_strategies()
    
    print("\nStrategy comparison completed!")
    print("Check generated plot files for detailed results")
