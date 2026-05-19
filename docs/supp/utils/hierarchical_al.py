"""
Hierarchical Two-Stage Active Learning Implementation

This module implements the two-level hierarchy active learning method:
- Stage A: Slice-level priority allocation using attribute groups
- Stage B: Instance-level selection within slices using uncertainty and diversity
- Two-round hierarchical scheduling (single attributes -> attribute pairs)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import itertools
from typing import Dict, List, Tuple, Set, Optional
import json
import os

class HierarchicalActiveStrategy:
    """
    Hierarchical Two-Stage Active Learning Strategy
    
    Performs data selection in two stages:
    1. Stage A: Slice-level priority allocation
    2. Stage B: Instance-level selection within slices
    """
    
    def __init__(self, 
                 w_err: float = 0.6,
                 w_unc: float = 0.4,
                 alpha: float = 0.4,
                 beta: float = 0.4, 
                 gamma: float = 0.2,
                 lambda_div: float = 0.5,
                 round1_ratio: float = 0.7,
                 min_slice_quota: int = 5,
                 max_slice_quota: int = 100,
                 embedding_dim: int = 512,
                 dataset: str = 'imagenet'):
        """
        Initialize hierarchical active learning strategy
        
        Args:
            w_err: Weight for error rate in difficulty calculation
            w_unc: Weight for uncertainty in difficulty calculation
            alpha: Weight for difficulty in priority score
            beta: Weight for coverage gap in priority score
            gamma: Weight for correlation penalty in priority score
            lambda_div: Weight for diversity in instance selection
            round1_ratio: Fraction of budget allocated to Round 1 (single attributes)
            min_slice_quota: Minimum samples per slice
            max_slice_quota: Maximum samples per slice
            embedding_dim: Dimension for feature embeddings
            dataset: Dataset type ('imagenet' or 'celeba')
        """
        self.w_err = w_err
        self.w_unc = w_unc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_div = lambda_div
        self.round1_ratio = round1_ratio
        self.min_slice_quota = min_slice_quota
        self.max_slice_quota = max_slice_quota
        self.embedding_dim = embedding_dim
        self.dataset = dataset
        self.preferred_attribute_keys = []
        self.exact_slice_greedy_max_budget = 128
        self.exact_slice_greedy_max_candidates = 4096
        self.approx_slice_candidate_multiplier = 4
        self.approx_slice_candidate_min = 2048
        self._slice_membership_cache = {}
        
        # Define attribute groups based on dataset
        if dataset == 'imagenet':
            # Use actual attribute values from attribute10.json
            self.attribute_groups = {
                'color': ['black', 'blue', 'brown', 'gray', 'green', 'maroon', 'multicolored', 
                         'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow'],
                'texture': ['ceramic', 'fabric', 'glass', 'leather', 'metallic', 'plastic', 'wooden'],
                'background': ['bridge', 'buildings', 'grass', 'house', 'inside a house', 'rocks', 
                              'sky', 'snow', 'street', 'trees', 'wall', 'water', 'wild']
            }
            self.preferred_attribute_keys = ['color', 'texture', 'background']
            # Load attribute data from JSON file
            self.attribute_data_path = 'save/attribute10.json'
        else:  # celeba
            self.attribute_groups = {
                'hair color': ['brown', 'black', 'blonde', 'white', 'yellow', 'red', 'gray'],
                'emotion': ['excited', 'surprised', 'serious', 'neutral', 'happy', 'calm', 'pensive', 'sad', 'angry'],
                'skin': ['brown', 'white', 'black'],
                'beard': ['No', 'Yes']
            }
            self.preferred_attribute_keys = ['hair color', 'emotion', 'skin', 'beard']
            # Load attribute data from JSON file
            self.attribute_data_path = 'data/celebA/attribute.json'
        
        # Load attribute data at initialization
        self.attribute_data = None
        if os.path.exists(self.attribute_data_path):
            try:
                with open(self.attribute_data_path, 'r') as f:
                    raw_data = json.load(f)

                # Common JSON layout in this repo is attribute -> {"data": [...], ...}.
                # Convert that into sample-index -> {attribute: value, ...} so slice
                # membership works for both ImageNet and CelebA.
                if (
                    isinstance(raw_data, dict)
                    and raw_data
                    and all(isinstance(v, dict) and isinstance(v.get('data'), list) for v in raw_data.values())
                ):
                    lengths = [len(v['data']) for v in raw_data.values() if isinstance(v.get('data'), list)]
                    min_length = min(lengths) if lengths else 0
                    samplewise_data = {}
                    for i in range(min_length):
                        samplewise_data[str(i)] = {
                            attr_name: attr_info['data'][i]
                            for attr_name, attr_info in raw_data.items()
                            if i < len(attr_info['data'])
                        }
                    self.attribute_data = samplewise_data

                    # Override default value lists with the exact values present in the
                    # metadata file so CelebA slices match the available annotations.
                    dynamic_groups = {}
                    for key in self.preferred_attribute_keys:
                        attr_info = raw_data.get(key)
                        if not isinstance(attr_info, dict):
                            continue
                        values = attr_info.get('data', [])
                        if not isinstance(values, list):
                            continue
                        unique_values = list(dict.fromkeys(values))
                        if unique_values:
                            dynamic_groups[key] = unique_values
                    if dynamic_groups:
                        self.attribute_groups = dynamic_groups
                else:
                    # Fallback: assume data is already in the expected samplewise format.
                    self.attribute_data = raw_data

            except Exception as e:
                print(f"Warning: Could not load attribute data from {self.attribute_data_path}: {e}")
                self.attribute_data = None
    
    def get_attribute_slices(self, round_num: int = 1) -> List[Dict]:
        """
        Generate attribute slices for the given round
        
        Args:
            round_num: 1 for single attributes, 2 for attribute pairs
            
        Returns:
            List of slice dictionaries with attribute combinations
        """
        slices = []
        
        if round_num == 1:
            # Single attribute slices
            for group_name, attributes in self.attribute_groups.items():
                for attr in attributes:
                    slices.append({
                        'attributes': {group_name: attr},
                        'name': f"{group_name}={attr}",
                        'type': 'single'
                    })
        else:
            # Attribute pair slices
            groups = list(self.attribute_groups.keys())
            for group1, group2 in itertools.combinations(groups, 2):
                for attr1 in self.attribute_groups[group1]:
                    for attr2 in self.attribute_groups[group2]:
                        slices.append({
                            'attributes': {group1: attr1, group2: attr2},
                            'name': f"{group1}={attr1}&{group2}={attr2}",
                            'type': 'pair'
                        })
        
        return slices
    
    def compute_slice_membership(self, data_indices: np.ndarray, slice_def: Dict) -> np.ndarray:
        """
        Compute which data samples belong to a given slice
        
        Args:
            data_indices: Indices of data samples
            slice_def: Slice definition with attributes
            
        Returns:
            Boolean mask indicating slice membership
        """
        cache_key = None
        if (
            len(data_indices) > 0
            and np.array_equal(data_indices, np.arange(len(data_indices)))
            and 'name' in slice_def
        ):
            cache_key = (len(data_indices), slice_def['name'])
            cached_mask = self._slice_membership_cache.get(cache_key)
            if cached_mask is not None:
                return cached_mask.copy()

        mask = np.zeros(len(data_indices), dtype=bool)
        
        if self.attribute_data is None:
            # Fallback to simple simulation if no attribute data
            slice_hash = hash(slice_def['name'])
            np.random.seed(slice_hash % 1000)
            membership_prob = 0.1 + (slice_hash % 100) / 1000  # 0.1 to 0.199
            mask = np.random.random(len(data_indices)) < membership_prob
            if cache_key is not None:
                self._slice_membership_cache[cache_key] = mask.copy()
            return mask
        
        # Use real attribute data for slice membership
        slice_type = slice_def.get('type', 'single')
        attributes = slice_def.get('attributes', {})
        
        for i, idx in enumerate(data_indices):
            idx_str = str(idx)
            
            if idx_str not in self.attribute_data:
                continue
                
            attrs = self.attribute_data[idx_str]
            is_member = True
            
            # Check if all required attributes match
            for attr_name, attr_value in attributes.items():
                if attr_name not in attrs or attrs[attr_name] != attr_value:
                    is_member = False
                    break
            
            mask[i] = is_member
        
        if cache_key is not None:
            self._slice_membership_cache[cache_key] = mask.copy()
        return mask

    def _greedy_select_within_slice(self,
                                  slice_indices: np.ndarray,
                                  slice_uncertainties: np.ndarray,
                                  slice_embeddings: np.ndarray,
                                  budget: int) -> np.ndarray:
        selected_local = []
        remaining_local = list(range(len(slice_indices)))

        first_idx = int(np.argmax(slice_uncertainties))
        selected_local.append(first_idx)
        remaining_local.remove(first_idx)

        for _ in range(budget - 1):
            if not remaining_local:
                break

            best_score = -float('inf')
            best_idx = None
            selected_embeddings = slice_embeddings[selected_local]

            for idx in remaining_local:
                uncertainty_score = slice_uncertainties[idx]
                current_embedding = slice_embeddings[idx:idx + 1]
                distances = euclidean_distances(current_embedding, selected_embeddings)[0]
                diversity_score = np.min(distances) if len(distances) > 0 else 0.0
                total_score = uncertainty_score + self.lambda_div * diversity_score

                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx

            if best_idx is not None:
                selected_local.append(best_idx)
                remaining_local.remove(best_idx)

        return slice_indices[selected_local]
    
    def compute_difficulty(self, slice_mask: np.ndarray, 
                          validation_errors: np.ndarray,
                          uncertainties: np.ndarray,
                          val_data_indices: np.ndarray = None) -> float:
        """
        Compute difficulty score for a slice
        
        Args:
            slice_mask: Boolean mask for slice membership (for training data)
            validation_errors: Error indicators for validation samples
            uncertainties: Uncertainty scores for training samples
            val_data_indices: Indices of validation data (optional)
            
        Returns:
            Difficulty score D(τ)
        """
        if not np.any(slice_mask):
            return 0.0
        
        # For training uncertainties, use the slice mask directly
        slice_uncertainties = uncertainties[slice_mask]
        mean_uncertainty = np.mean(slice_uncertainties) if len(slice_uncertainties) > 0 else 0.0
        
        # For validation errors, we need to handle the size mismatch
        # Since validation set is separate, we'll estimate error rate based on uncertainty
        if val_data_indices is not None and len(val_data_indices) == len(validation_errors):
            # If we have validation indices, try to map them to slice membership
            # For now, use a simplified approach: sample validation errors randomly
            # In a real implementation, you'd match validation samples to slices properly
            if len(validation_errors) > 0:
                error_rate = np.mean(validation_errors)  # Use overall validation error rate
            else:
                error_rate = 0.0
        else:
            # Fallback: estimate error rate from uncertainty distribution
            # Higher uncertainty suggests higher error rate
            uncertainty_threshold = np.percentile(uncertainties, 75)  # Top 25% uncertainty
            high_uncertainty_mask = slice_uncertainties > uncertainty_threshold
            error_rate = np.mean(high_uncertainty_mask) if len(slice_uncertainties) > 0 else 0.0
        
        difficulty = self.w_err * error_rate + self.w_unc * mean_uncertainty
        return difficulty
    
    def compute_coverage_gap(self, slice_mask: np.ndarray,
                           selected_mask: np.ndarray,
                           generated_mask: np.ndarray,
                           target_ratio: float = 0.1) -> float:
        """
        Compute coverage gap for a slice
        
        Args:
            slice_mask: Boolean mask for slice membership
            selected_mask: Boolean mask for already selected samples
            generated_mask: Boolean mask for generated samples
            target_ratio: Target coverage ratio for the slice
            
        Returns:
            Coverage gap C(τ)
        """
        slice_size = np.sum(slice_mask)
        if slice_size == 0:
            return 0.0
        
        selected_in_slice = np.sum(slice_mask & selected_mask)
        generated_in_slice = np.sum(slice_mask & generated_mask)
        current_coverage = selected_in_slice + generated_in_slice
        
        target_count = max(1, int(slice_size * target_ratio))
        coverage_ratio = min(1.0, current_coverage / target_count)
        
        coverage_gap = 1.0 - coverage_ratio
        return coverage_gap
    
    def compute_correlation_penalty(self, slice_def: Dict, 
                                  slice_budgets: Dict[str, int]) -> float:
        """
        Compute correlation penalty for a slice
        
        Args:
            slice_def: Definition of current slice
            slice_budgets: Current budget allocations for all slices
            
        Returns:
            Correlation penalty R(τ)
        """
        penalty = 0.0
        current_attrs = set(slice_def['attributes'].items())
        
        for other_slice_name, other_budget in slice_budgets.items():
            if other_budget == 0:
                continue
                
            # Parse other slice attributes (simplified)
            if '&' in other_slice_name:
                # Attribute pair
                parts = other_slice_name.split('&')
                other_attrs = set()
                for part in parts:
                    group, attr = part.split('=')
                    other_attrs.add((group, attr))
            else:
                # Single attribute
                group, attr = other_slice_name.split('=')
                other_attrs = {(group, attr)}
            
            # Compute correlation coefficient (simplified as overlap ratio)
            overlap = len(current_attrs.intersection(other_attrs))
            total = len(current_attrs.union(other_attrs))
            correlation = overlap / total if total > 0 else 0.0
            
            penalty += correlation * other_budget
        
        return penalty
    
    def allocate_slice_budgets(self, data_indices: np.ndarray,
                             validation_errors: np.ndarray,
                             uncertainties: np.ndarray,
                             selected_mask: np.ndarray,
                             generated_mask: np.ndarray,
                             total_budget: int,
                             round_num: int = 1,
                             val_data_indices: np.ndarray = None) -> Dict[str, int]:
        """
        Stage A: Allocate budgets to slices based on priority scores
        
        Args:
            data_indices: Indices of available data
            validation_errors: Error indicators for validation
            uncertainties: Uncertainty scores
            selected_mask: Already selected samples
            generated_mask: Generated samples
            total_budget: Total selection budget
            round_num: Round number (1 or 2)
            val_data_indices: Indices of validation data
            
        Returns:
            Dictionary mapping slice names to budgets
        """
        slices = self.get_attribute_slices(round_num)
        slice_priorities = {}
        slice_budgets = {}
        
        # Compute priority scores for each slice
        for slice_def in slices:
            slice_mask = self.compute_slice_membership(data_indices, slice_def)
            
            if not np.any(slice_mask):
                slice_priorities[slice_def['name']] = 0.0
                continue
            
            # Compute components
            difficulty = self.compute_difficulty(slice_mask, validation_errors, uncertainties, val_data_indices)
            coverage_gap = self.compute_coverage_gap(slice_mask, selected_mask, generated_mask)
            correlation_penalty = self.compute_correlation_penalty(slice_def, slice_budgets)
            
            # Compute priority score
            priority = (self.alpha * difficulty + 
                       self.beta * coverage_gap - 
                       self.gamma * correlation_penalty)
            
            slice_priorities[slice_def['name']] = max(0.0, priority)
        
        # Normalize priorities and allocate budgets
        total_priority = sum(slice_priorities.values())
        if total_priority == 0:
            # Fallback: equal distribution
            valid_slices = [name for name, priority in slice_priorities.items() if priority >= 0]
            if valid_slices:
                budget_per_slice = max(1, total_budget // len(valid_slices))
                for slice_name in valid_slices:
                    slice_budgets[slice_name] = min(budget_per_slice, self.max_slice_quota)
        else:
            for slice_name, priority in slice_priorities.items():
                if priority > 0:
                    raw_budget = int((priority / total_priority) * total_budget)
                    slice_budgets[slice_name] = max(self.min_slice_quota, 
                                                  min(raw_budget, self.max_slice_quota))
        
        return slice_budgets
    
    def compute_uncertainty_scores(self, model, data_loader, device):
        """
        Compute uncertainty scores for samples using model predictions
        
        Args:
            model: Trained model
            data_loader: Data loader for samples
            device: Computing device
            
        Returns:
            Array of uncertainty scores
        """
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for batch_images, _ in data_loader:
                batch_images = batch_images.to(device)
                outputs = model(batch_images)
                
                if outputs.shape[1] == 1:  # Binary classification
                    probs = torch.sigmoid(outputs).squeeze()
                    # Use margin-based uncertainty for binary case
                    uncertainty = 1.0 - torch.abs(probs - 0.5) * 2.0
                else:  # Multi-class classification
                    probs = F.softmax(outputs, dim=1)
                    # Use entropy-based uncertainty
                    uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                uncertainties.append(uncertainty.cpu().numpy())
        
        return np.concatenate(uncertainties)
    
    def extract_embeddings(self, model, data_loader, device):
        """
        Extract feature embeddings for diversity computation
        
        Args:
            model: Trained model
            data_loader: Data loader for samples
            device: Computing device
            
        Returns:
            Array of feature embeddings
        """
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_images, _ in data_loader:
                batch_images = batch_images.to(device)
                
                # Extract features before final classification layer
                if hasattr(model, 'features'):
                    features = model.features(batch_images)
                    features = features.view(features.size(0), -1)
                elif hasattr(model, 'backbone'):
                    features = model.backbone(batch_images)
                else:
                    # Fallback: use penultimate layer
                    features = model(batch_images)
                    if len(features.shape) > 2:
                        features = features.view(features.size(0), -1)
                
                # Reduce dimensionality if needed
                if features.shape[1] > self.embedding_dim:
                    # Simple dimensionality reduction via averaging
                    stride = features.shape[1] // self.embedding_dim
                    features = features[:, ::stride][:, :self.embedding_dim]
                
                embeddings.append(features.cpu().numpy())
        
        return np.concatenate(embeddings)
    
    def select_within_slice(self, slice_indices: np.ndarray,
                          uncertainties: np.ndarray,
                          embeddings: np.ndarray,
                          budget: int) -> np.ndarray:
        """
        Stage B: Select instances within a slice using uncertainty and diversity
        
        Args:
            slice_indices: Indices of samples in the slice
            uncertainties: Uncertainty scores for all samples
            embeddings: Feature embeddings for all samples
            budget: Number of samples to select from this slice
            
        Returns:
            Selected indices from the slice
        """
        if len(slice_indices) == 0 or budget <= 0:
            return np.array([], dtype=int)
        
        budget = min(budget, len(slice_indices))
        
        if budget == len(slice_indices):
            return slice_indices
        
        # Extract slice-specific data
        slice_uncertainties = uncertainties[slice_indices]
        slice_embeddings = embeddings[slice_indices]

        use_exact_greedy = (
            budget <= self.exact_slice_greedy_max_budget
            and len(slice_indices) <= self.exact_slice_greedy_max_candidates
        )

        if use_exact_greedy:
            return self._greedy_select_within_slice(
                slice_indices,
                slice_uncertainties,
                slice_embeddings,
                budget,
            )

        candidate_pool_size = min(
            len(slice_indices),
            self.exact_slice_greedy_max_candidates,
            max(self.approx_slice_candidate_min, budget * self.approx_slice_candidate_multiplier),
        )
        candidate_order = np.argsort(slice_uncertainties)[::-1][:candidate_pool_size]

        reduced_slice_indices = slice_indices[candidate_order]
        reduced_uncertainties = slice_uncertainties[candidate_order]
        reduced_embeddings = slice_embeddings[candidate_order]

        return self._greedy_select_within_slice(
            reduced_slice_indices,
            reduced_uncertainties,
            reduced_embeddings,
            budget,
        )
    
    def select(self, dataset, val_dataset, model, device, n_select: int, verbose: bool = False):
        """
        Main selection method implementing hierarchical two-stage active learning
        
        Args:
            dataset: Unlabeled dataset
            val_dataset: Validation dataset for computing errors
            model: Current model
            device: Computing device
            n_select: Number of samples to select
            verbose: Whether to print detailed information
            
        Returns:
            Selected sample indices
        """
        if verbose:
            print(f"Starting hierarchical active learning selection for {n_select} samples")

        self._slice_membership_cache.clear()
        
        # Create data loaders
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Get all available indices
        data_indices = np.arange(len(dataset))
        val_indices = np.arange(len(val_dataset))
        
        # Compute uncertainty scores and embeddings
        if verbose:
            print("Computing uncertainty scores...")
        uncertainties = self.compute_uncertainty_scores(model, data_loader, device)
        
        if verbose:
            print("Extracting feature embeddings...")
        embeddings = self.extract_embeddings(model, data_loader, device)
        
        # Compute validation errors (simplified)
        val_uncertainties = self.compute_uncertainty_scores(model, val_loader, device)
        validation_errors = (val_uncertainties > np.median(val_uncertainties)).astype(float)
        
        # Initialize masks
        selected_mask = np.zeros(len(dataset), dtype=bool)
        generated_mask = np.zeros(len(dataset), dtype=bool)  # Would be set based on actual data
        
        # Two-round hierarchical scheduling
        round1_budget = int(n_select * self.round1_ratio)
        round2_budget = n_select - round1_budget
        
        selected_indices = []
        
        # Round 1: Single-attribute slices
        if verbose:
            print(f"Round 1: Allocating {round1_budget} samples to single-attribute slices")
        
        slice_budgets_r1 = self.allocate_slice_budgets(
            data_indices, validation_errors, uncertainties, 
            selected_mask, generated_mask, round1_budget, round_num=1,
            val_data_indices=val_indices
        )
        
        for slice_name, budget in slice_budgets_r1.items():
            if budget > 0:
                # Find slice definition
                slice_def = {'name': slice_name, 'attributes': {}}
                if '=' in slice_name:
                    group, attr = slice_name.split('=')
                    slice_def['attributes'][group] = attr
                
                slice_mask = self.compute_slice_membership(data_indices, slice_def)
                slice_indices = data_indices[slice_mask & ~selected_mask]
                
                if len(slice_indices) > 0:
                    selected_slice = self.select_within_slice(
                        slice_indices, uncertainties, embeddings, budget
                    )
                    selected_indices.extend(selected_slice)
                    selected_mask[selected_slice] = True
                    
                    if verbose:
                        print(f"  {slice_name}: selected {len(selected_slice)}/{budget} samples")
        
        # Round 2: Attribute-pair slices
        if round2_budget > 0:
            if verbose:
                print(f"Round 2: Allocating {round2_budget} samples to attribute-pair slices")
            
            slice_budgets_r2 = self.allocate_slice_budgets(
                data_indices, validation_errors, uncertainties,
                selected_mask, generated_mask, round2_budget, round_num=2,
                val_data_indices=val_indices
            )
            
            for slice_name, budget in slice_budgets_r2.items():
                if budget > 0:
                    # Parse attribute pairs
                    slice_def = {'name': slice_name, 'attributes': {}}
                    if '&' in slice_name:
                        parts = slice_name.split('&')
                        for part in parts:
                            if '=' in part:
                                group, attr = part.split('=')
                                slice_def['attributes'][group] = attr
                    
                    slice_mask = self.compute_slice_membership(data_indices, slice_def)
                    slice_indices = data_indices[slice_mask & ~selected_mask]
                    
                    if len(slice_indices) > 0:
                        selected_slice = self.select_within_slice(
                            slice_indices, uncertainties, embeddings, budget
                        )
                        selected_indices.extend(selected_slice)
                        selected_mask[selected_slice] = True
                        
                        if verbose:
                            print(f"  {slice_name}: selected {len(selected_slice)}/{budget} samples")
        
        # Fill remaining budget with uncertainty-based selection if needed
        remaining_budget = n_select - len(selected_indices)
        if remaining_budget > 0:
            if verbose:
                print(f"Filling remaining {remaining_budget} samples with uncertainty-based selection")
            
            remaining_indices = data_indices[~selected_mask]
            if len(remaining_indices) > 0:
                remaining_uncertainties = uncertainties[remaining_indices]
                sorted_indices = remaining_indices[np.argsort(remaining_uncertainties)[::-1]]
                additional_selected = sorted_indices[:remaining_budget]
                selected_indices.extend(additional_selected)
        
        final_selected = np.array(selected_indices[:n_select])
        
        if verbose:
            print(f"Hierarchical AL selected {len(final_selected)} samples total")
            print(f"  Round 1 (single attr): {round1_budget} samples")
            print(f"  Round 2 (attr pairs): {round2_budget} samples")
            print(f"  Additional: {len(final_selected) - round1_budget - round2_budget} samples")
        
        return final_selected
