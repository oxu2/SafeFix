"""
Active Learning utilities for the HiBug project.
This module contains multiple active learning strategies and related functions.
"""

from utils.imports import *
from utils.defines import *
from utils.hierarchical_al import HierarchicalActiveStrategy
from collections import Counter, defaultdict


def parse_attribute_priority_log(log_path):
    """Parse rare or hard attribute cases from a diagnostic log file."""
    if not log_path or not os.path.exists(log_path):
        return {}

    pattern = re.compile(
        r"(?:Rare Case|Hard case):\s*attribute\s*\"([^\"]+)\",\s*description\s*\"([^\"]+)\"",
        re.IGNORECASE,
    )

    priorities = defaultdict(lambda: {"rare": set(), "hard": set()})

    with open(log_path, "r") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if not match:
                continue

            attribute, description = match.groups()
            category = "rare" if "rare" in line.lower() else "hard"
            priorities[attribute][category].add(description)

    # Convert defaultdict to regular dict for downstream usage
    return {attr: {k: set(v) for k, v in buckets.items()} for attr, buckets in priorities.items() if any(buckets.values())}


class CoreSetStrategy:
    """
    Core-set selection strategy for Active Learning
    Selects samples that are most representative of the unlabeled data distribution
    """
    
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        
    def greedy_k_center(self, labeled_embeddings, unlabeled_embeddings, n_select):
        """
        Greedy k-center algorithm for core-set selection with progress tracking
        
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
        
        print(f"Core-set selection: choosing {n_select} from {n_unlabeled} samples...")
        print("Progress: ", end="", flush=True)
        
        # Initialize with all labeled embeddings
        selected_embeddings = labeled_embeddings.copy()
        selected_indices = []
        
        # Pre-compute distances to labeled data (one-time cost)
        distances_to_labeled = euclidean_distances(unlabeled_embeddings, labeled_embeddings)
        min_distances = np.min(distances_to_labeled, axis=1)
        
        progress_step = max(1, n_select // 20)  # Show progress every 5%
        
        for i in range(n_select):
            # Show progress
            if i % progress_step == 0:
                progress = int(20 * i / n_select)
                print(f"{'█' * progress}{'░' * (20 - progress)} {i}/{n_select}", end="\r", flush=True)
            
            # Select the point that is farthest from all selected points
            selected_idx = np.argmax(min_distances)
            selected_indices.append(selected_idx)
            
            # Update minimum distances efficiently
            new_point = unlabeled_embeddings[selected_idx:selected_idx+1]
            new_distances = euclidean_distances(unlabeled_embeddings, new_point).flatten()
            min_distances = np.minimum(min_distances, new_distances)
        
        print(f"{'█' * 20} {n_select}/{n_select} - Complete!")
        return selected_indices


class StratifiedStrategy:
    """
    Stratified sampling strategy that maintains class balance
    """
    
    def __init__(self):
        pass
        
    def stratified_sample(self, targets, n_select, random_seed=42):
        """
        Stratified sampling to maintain class balance
        
        Args:
            targets: array of class labels
            n_select: number of samples to select
            random_seed: random seed for reproducibility
            
        Returns:
            selected_indices: indices of selected samples
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        n_classes = len(unique_classes)
        
        print(f"Stratified sampling: choosing {n_select} from {len(targets)} samples across {n_classes} classes...")
        
        # Calculate samples per class
        samples_per_class = n_select // n_classes
        remaining_samples = n_select % n_classes
        
        selected_indices = []
        
        for i, cls in enumerate(unique_classes):
            cls_indices = np.where(targets == cls)[0]
            n_cls_samples = samples_per_class + (1 if i < remaining_samples else 0)
            n_cls_samples = min(n_cls_samples, len(cls_indices))
            
            # Random sample from this class
            selected_cls_indices = np.random.choice(cls_indices, n_cls_samples, replace=False)
            selected_indices.extend(selected_cls_indices)
            
            print(f"  Class {cls}: selected {n_cls_samples}/{len(cls_indices)} samples")
        
        return selected_indices


class UncertaintyStrategy:
    """
    Uncertainty-based sampling strategy
    Selects samples where the model is most uncertain
    """
    
    def __init__(self):
        pass
        
    def uncertainty_sample(self, model, dataloader, device, n_select, model_type='resnet'):
        """
        Select samples based on prediction uncertainty
        
        Args:
            model: trained model for uncertainty estimation
            dataloader: DataLoader for the dataset
            device: computation device
            n_select: number of samples to select
            model_type: type of model
            
        Returns:
            selected_indices: indices of selected samples
        """
        print(f"Uncertainty sampling: choosing {n_select} most uncertain samples...")
        
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                outputs = model(data)
                
                if model_type in ['resnet', 'vit', 'clip'] and outputs.dim() > 1:
                    # Multi-class: use entropy
                    probs = torch.softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties.extend(entropy.cpu().numpy())
                else:
                    # Binary: use prediction confidence
                    probs = torch.sigmoid(outputs.squeeze())
                    uncertainty = 1 - torch.abs(probs - 0.5) * 2  # Distance from 0.5
                    uncertainties.extend(uncertainty.cpu().numpy())
        
        uncertainties = np.array(uncertainties)
        # Select indices with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_select:].tolist()
        
        print(f"Selected samples with uncertainty range: {uncertainties[selected_indices].min():.4f} - {uncertainties[selected_indices].max():.4f}")
        return selected_indices


class HybridStrategy:
    """
    Hybrid strategy combining multiple sampling methods
    """
    
    def __init__(self, coreset_ratio=0.5, stratified_ratio=0.3, random_ratio=0.2):
        self.coreset_ratio = coreset_ratio
        self.stratified_ratio = stratified_ratio  
        self.random_ratio = random_ratio
        
        assert abs(coreset_ratio + stratified_ratio + random_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
    def hybrid_sample(self, labeled_embeddings, unlabeled_embeddings, targets, n_select, random_seed=42):
        """
        Combine core-set, stratified, and random sampling
        
        Args:
            labeled_embeddings: embeddings of labeled data
            unlabeled_embeddings: embeddings of unlabeled data  
            targets: class labels for unlabeled data
            n_select: total number of samples to select
            random_seed: random seed
            
        Returns:
            selected_indices: indices of selected samples
        """
        print(f"Hybrid sampling: {self.coreset_ratio:.1%} core-set + {self.stratified_ratio:.1%} stratified + {self.random_ratio:.1%} random")
        
        # Calculate sample counts for each method
        n_coreset = int(n_select * self.coreset_ratio)
        n_stratified = int(n_select * self.stratified_ratio)
        n_random = n_select - n_coreset - n_stratified
        
        selected_indices = set()
        
        # 1. Core-set selection
        if n_coreset > 0:
            coreset_strategy = CoreSetStrategy()
            coreset_indices = coreset_strategy.greedy_k_center(
                labeled_embeddings, unlabeled_embeddings, n_coreset
            )
            selected_indices.update(coreset_indices)
            print(f"Core-set selected: {len(coreset_indices)} samples")
        
        # 2. Stratified selection (from remaining samples)
        if n_stratified > 0:
            remaining_indices = [i for i in range(len(targets)) if i not in selected_indices]
            remaining_targets = targets[remaining_indices]
            
            stratified_strategy = StratifiedStrategy()
            stratified_local_indices = stratified_strategy.stratified_sample(
                remaining_targets, n_stratified, random_seed
            )
            stratified_indices = [remaining_indices[i] for i in stratified_local_indices]
            selected_indices.update(stratified_indices)
            print(f"Stratified selected: {len(stratified_indices)} samples")
        
        # 3. Random selection (from remaining samples)
        if n_random > 0:
            remaining_indices = [i for i in range(len(targets)) if i not in selected_indices]
            random.seed(random_seed)
            random_indices = random.sample(remaining_indices, min(n_random, len(remaining_indices)))
            selected_indices.update(random_indices)
            print(f"Random selected: {len(random_indices)} samples")
        
        return list(selected_indices)


class DiversityStrategy:
    """
    Diversity-based sampling using feature clustering
    """
    
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        
    def diversity_sample(self, embeddings, n_select, random_seed=42):
        """
        Select diverse samples using K-means clustering
        
        Args:
            embeddings: feature embeddings
            n_select: number of samples to select
            random_seed: random seed
            
        Returns:
            selected_indices: indices of selected samples
        """
        from sklearn.cluster import KMeans
        
        print(f"Diversity sampling: clustering into {self.n_clusters or n_select} groups...")
        
        # Convert to numpy
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Use n_select clusters by default, or specified number
        n_clusters = self.n_clusters or min(n_select, len(embeddings_np))
        
        # Perform K-means clustering
        random.seed(random_seed)
        np.random.seed(random_seed)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        selected_indices = []
        samples_per_cluster = n_select // n_clusters
        remaining_samples = n_select % n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Number of samples to select from this cluster
            n_cluster_samples = samples_per_cluster + (1 if cluster_id < remaining_samples else 0)
            n_cluster_samples = min(n_cluster_samples, len(cluster_indices))
            
            # Select samples closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(embeddings_np[cluster_indices] - cluster_center, axis=1)
            closest_indices = cluster_indices[np.argsort(distances)[:n_cluster_samples]]
            
            selected_indices.extend(closest_indices)
            print(f"  Cluster {cluster_id}: selected {n_cluster_samples}/{len(cluster_indices)} samples")
        
        return selected_indices


class AttributeFocusedStrategy:
    """
    Attribute-driven sampling strategy that prioritizes underrepresented
    or uncertain attribute slices defined in an attribute metadata JSON.
    """

    def __init__(
        self,
        attribute_metadata,
        focus_attributes=None,
        rarity_power=1.0,
        uncertain_boost=1.0,
        epsilon=1e-8,
        value_boosts=None,
    ):
        """
        Args:
            attribute_metadata: Path to attribute JSON or pre-loaded dict.
            focus_attributes: Optional iterable of attribute keys to use.
            rarity_power: Exponent applied to rarity weighting (>= 0).
            uncertain_boost: Score bonus added when a sample belongs to an
                uncertain attribute slice.
            epsilon: Numerical stability term for rarity weighting.
            value_boosts: Optional dict {attribute: {value: bonus}} to
                incorporate external priority signals (e.g., from logs).
        """
        if isinstance(attribute_metadata, str):
            if not os.path.exists(attribute_metadata):
                raise FileNotFoundError(
                    f"Attribute metadata file '{attribute_metadata}' not found."
                )
            with open(attribute_metadata, "r") as f:
                attribute_metadata = json.load(f)

        if not isinstance(attribute_metadata, dict):
            raise ValueError("Attribute metadata must be a dict or path to a JSON file")

        if focus_attributes:
            focus_set = {attr.strip() for attr in focus_attributes if attr.strip()}
            attribute_metadata = {
                key: value for key, value in attribute_metadata.items() if key in focus_set
            }

        if len(attribute_metadata) == 0:
            raise ValueError("No attribute metadata available for AttributeFocusedStrategy")

        # Pre-process attribute metadata for efficient lookup
        self.attributes = {}
        for attr_name, attr_info in attribute_metadata.items():
            data = attr_info.get("data", [])
            uncertain_idx = attr_info.get("uncertain_idx", [])
            self.attributes[attr_name] = {
                "data": data,
                "uncertain": set(uncertain_idx) if uncertain_idx else set(),
            }

        self.rarity_power = max(rarity_power, 0.0)
        self.uncertain_boost = max(uncertain_boost, 0.0)
        self.epsilon = epsilon
        self.value_boosts = value_boosts or {}

    def _resolve_metadata_indices(self, dataset):
        """Map dataset-local indices to attribute metadata indices."""
        from torch.utils.data import Subset, ConcatDataset

        if isinstance(dataset, Subset):
            parent_mapping = self._resolve_metadata_indices(dataset.dataset)
            if parent_mapping is None:
                return list(dataset.indices)
            return [parent_mapping[idx] for idx in dataset.indices]

        if isinstance(dataset, ConcatDataset):
            indices = []
            for sub_dataset in dataset.datasets:
                sub_map = self._resolve_metadata_indices(sub_dataset)
                if sub_map is None:
                    indices.extend([None] * len(sub_dataset))
                else:
                    indices.extend(sub_map)
            return indices

        # Base dataset: assume attribute metadata aligns with dataset order
        dataset_len = len(dataset) if hasattr(dataset, "__len__") else 0

        # Heuristic: generated datasets typically do not have attribute metadata
        class_name = dataset.__class__.__name__.lower()
        if "generated" in class_name or dataset_len == 0:
            return [None] * dataset_len

        return list(range(dataset_len))

    def _compute_attribute_counts(self, metadata_indices):
        """Compute frequency of each attribute value within the dataset subset."""
        counts = {attr: Counter() for attr in self.attributes}
        for meta_idx in metadata_indices:
            if meta_idx is None:
                continue
            for attr_name, attr_info in self.attributes.items():
                data = attr_info.get("data", [])
                if meta_idx < len(data):
                    value = data[meta_idx]
                    counts[attr_name][value] += 1
        return counts

    def select(self, dataset, n_select, random_seed=42):
        """
        Score each sample by attribute rarity and uncertainty, then pick the top ones.
        """
        total_samples = len(dataset)
        if n_select >= total_samples:
            return list(range(total_samples))

        metadata_indices = self._resolve_metadata_indices(dataset)
        if len(metadata_indices) != total_samples:
            raise RuntimeError("Metadata index mapping length mismatch with dataset")

        attribute_counts = self._compute_attribute_counts(metadata_indices)

        scores = np.zeros(total_samples, dtype=np.float32)
        rng = np.random.default_rng(random_seed)

        for local_idx, meta_idx in enumerate(metadata_indices):
            if meta_idx is None:
                continue

            sample_score = 0.0
            for attr_name, attr_info in self.attributes.items():
                data = attr_info["data"]
                if meta_idx >= len(data):
                    continue
                value = data[meta_idx]
                frequency = attribute_counts[attr_name].get(value, 0)
                rarity = 1.0 / ((frequency if frequency > 0 else 1) + self.epsilon)
                if self.rarity_power != 1.0:
                    rarity = rarity ** self.rarity_power
                sample_score += rarity

                if meta_idx in attr_info["uncertain"]:
                    sample_score += self.uncertain_boost

                if self.value_boosts:
                    attr_boosts = self.value_boosts.get(attr_name, {})
                    if attr_boosts:
                        sample_score += attr_boosts.get(value, 0.0)

            scores[local_idx] = sample_score

        if np.all(scores == 0):
            # Fall back to random selection if no attribute information applied
            rng = np.random.default_rng(random_seed)
            return rng.choice(total_samples, size=n_select, replace=False).tolist()

        # Tie-break by adding a small amount of random noise (stable selection)
        noise = rng.uniform(low=0.0, high=1e-6, size=total_samples)
        ranked_indices = np.argsort(-(scores + noise))
        return ranked_indices[:n_select].tolist()


def extract_embeddings_for_al(model, dataloader, device, model_type='resnet', embedding_dim=512):
    """
    Extract embeddings from the model for Active Learning
    
    Args:
        model: trained model
        dataloader: DataLoader for the dataset
        device: computation device
        model_type: type of model ('resnet', 'clip', 'vit', 'blip')
        embedding_dim: target embedding dimension
        
    Returns:
        embeddings: tensor of shape [n_samples, embedding_dim]
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            if model_type == 'resnet':
                # For ResNet, use the features before the final classifier
                if hasattr(model, 'backbone'):
                    features = model.backbone(data)  # [batch_size, 2048]
                else:
                    # For standard ResNet18
                    x = model.conv1(data)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    features = torch.flatten(x, 1)  # [batch_size, 512] for ResNet18
                
                # Project to target embedding dimension if needed
                if features.shape[1] != embedding_dim:
                    if not hasattr(model, 'al_projection'):
                        model.al_projection = nn.Linear(features.shape[1], embedding_dim).to(device)
                    features = model.al_projection(features)
                    
            elif model_type == 'clip':
                # For CLIP, use the image features
                features = model.get_image_features(pixel_values=data)
                
            elif model_type == 'vit':
                # For ViT, use the [CLS] token features
                outputs = model(data)
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state[:, 0]  # [CLS] token
                elif hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    # Fallback for different ViT implementations
                    features = outputs[0][:, 0] if isinstance(outputs, tuple) else outputs[:, 0]
                
                # Project to target embedding dimension if needed
                if features.shape[1] != embedding_dim:
                    if not hasattr(model, 'al_projection'):
                        model.al_projection = nn.Linear(features.shape[1], embedding_dim).to(device)
                    features = model.al_projection(features)
                
            elif model_type == 'blip':
                # For BLIP, use the image features
                features = model.get_image_features(pixel_values=data)
                
            embeddings.append(features.cpu())
    
    return torch.cat(embeddings, dim=0)


def active_learning_selection(train_dataset, val_dataset, args, device, verbose=False):
    """
    Perform Active Learning selection using specified strategy
    
    Args:
        train_dataset: full training dataset
        val_dataset: validation dataset  
        args: parsed arguments
        device: computation device
        verbose: whether to print verbose output
        
    Returns:
        selected_indices: indices of selected samples for training
    """
    # Import here to avoid circular imports
    from transformers import CLIPModel, ViTModel
    
    if not args.use_active_learning:
        # Fallback to random selection if AL is disabled
        total_samples = len(train_dataset)
        target_samples = int(total_samples * args.sub_ratio)
        random.seed(args.seed)
        return random.sample(range(total_samples), target_samples)
    
    # Get AL strategy from args (default to 'coreset')
    al_strategy = getattr(args, 'al_strategy', 'coreset')
    
    if verbose:
        print(f"Starting Active Learning data selection using '{al_strategy}' strategy...")
        print(f"Dataset: {args.dataset}, Model: {args.model}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Target selection ratio: {args.sub_ratio}")
        print(f"Embedding dimension: {args.al_embedding_dim}")
        print(f"Initial labeled samples: {args.al_initial_labeled}")
    
    # Calculate target number of samples to select
    total_train_samples = len(train_dataset)
    target_samples = int(total_train_samples * args.sub_ratio)
    
    # Handle pure stratified sampling (doesn't need embeddings)
    if al_strategy == 'stratified':
        try:
            # Get targets from dataset
            if hasattr(train_dataset, 'targets'):
                targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
                targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            else:
                if verbose:
                    print("Warning: Could not extract targets for stratified sampling, falling back to random")
                random.seed(args.seed)
                return random.sample(range(total_train_samples), target_samples)
            
            strategy = StratifiedStrategy()
            selected_indices = strategy.stratified_sample(targets, target_samples, args.seed)
            
            if verbose:
                print(f"Stratified Active Learning selection completed. Selected {len(selected_indices)} samples.")
            return selected_indices
            
        except Exception as e:
            if verbose:
                print(f"Stratified sampling failed: {e}, falling back to random")
            random.seed(args.seed)
            return random.sample(range(total_train_samples), target_samples)

    elif al_strategy == 'attribute':
        attribute_path = getattr(args, 'attribute_file', None)
        focus_raw = getattr(args, 'al_attribute_focus', None)
        rarity_power = getattr(args, 'al_attribute_rarity_power', 1.0)
        uncertain_boost = getattr(args, 'al_attribute_uncertainty_boost', 1.0)
        log_path = getattr(args, 'al_attribute_log_path', None)
        include_hard = getattr(args, 'al_attribute_include_hard', True)
        rare_boost = getattr(args, 'al_attribute_log_rare_boost', 1.0)
        hard_boost = getattr(args, 'al_attribute_log_hard_boost', 0.0)

        focus_attributes = None
        if isinstance(focus_raw, str) and focus_raw.strip():
            focus_attributes = [part.strip() for part in focus_raw.split(',') if part.strip()]
        elif isinstance(focus_raw, (list, tuple)):
            focus_attributes = list(focus_raw)

        if not log_path and getattr(args, 'dataset', None) and getattr(args, 'model', None):
            default_log = Path('logs/compare') / f"stat_{args.dataset}_{args.model}_compare.log"
            if default_log.exists():
                log_path = str(default_log)
            elif verbose:
                print(f"Attribute strategy: default log {default_log} not found.")

        value_boosts = None
        if log_path:
            priority_map = parse_attribute_priority_log(log_path)
            if priority_map:
                value_boosts = {}
                for attr_name, buckets in priority_map.items():
                    boost_map = {}
                    for value in buckets.get('rare', set()):
                        boost_map[value] = boost_map.get(value, 0.0) + rare_boost
                    if include_hard:
                        for value in buckets.get('hard', set()):
                            boost_map[value] = boost_map.get(value, 0.0) + hard_boost
                    if boost_map:
                        value_boosts[attr_name] = boost_map
                if verbose:
                    print(f"Loaded attribute priorities from {log_path}: {len(value_boosts or {})} attributes with boosts")
            elif verbose:
                print(f"Attribute strategy: No priority entries found in {log_path}")
        elif verbose:
            print("Attribute strategy: No log file provided for priority boosts")

        try:
            strategy = AttributeFocusedStrategy(
                attribute_metadata=attribute_path,
                focus_attributes=focus_attributes,
                rarity_power=rarity_power,
                uncertain_boost=uncertain_boost,
                value_boosts=value_boosts,
            )
            selected_indices = strategy.select(train_dataset, target_samples, random_seed=args.seed)

            if verbose:
                print(
                    "Attribute-focused Active Learning completed. "
                    f"Selected {len(selected_indices)} samples with focus={focus_attributes or 'all'}"
                )
                # Resolve indices for logging (subset order and base dataset order if available)
                resolved_indices = selected_indices
                if hasattr(train_dataset, 'indices'):
                    try:
                        resolved_indices = [train_dataset.indices[i] for i in selected_indices]
                    except Exception:
                        resolved_indices = selected_indices

                print(f"Attribute strategy selected subset indices: {selected_indices}")
                if resolved_indices is not selected_indices:
                    print(f"Attribute strategy corresponding base indices: {resolved_indices}")
            return selected_indices
        except Exception as exc:
            if verbose:
                print(f"Attribute strategy failed ({exc}). Falling back to random selection.")
            random.seed(args.seed)
            return random.sample(range(total_train_samples), target_samples)
    
    # For other strategies, we need feature extraction
    # Create a simple model for feature extraction
    if args.model == 'resnet':
        feature_model = models.resnet18(pretrained=True)
        feature_model.fc = nn.Identity()
    elif args.model == 'clip':
        feature_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif args.model == 'vit':
        feature_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    else:
        # Default to ResNet for feature extraction
        feature_model = models.resnet18(pretrained=True)
        feature_model.fc = nn.Identity()
    
    feature_model = feature_model.to(device)
    feature_model.eval()
    
    # Adjust batch size for feature extraction to avoid memory issues
    al_batch_size = min(args.batch_size, 32)  # Use smaller batch size for AL
    if verbose:
        print(f"Using batch size {al_batch_size} for Active Learning feature extraction")
    
    # Create data loaders for feature extraction
    train_loader = DataLoader(train_dataset, batch_size=al_batch_size, shuffle=False, 
                             num_workers=args.workers, pin_memory=True)
    
    # Extract embeddings
    if verbose:
        print(f"Extracting embeddings from {len(train_dataset)} training samples...")
    train_embeddings = extract_embeddings_for_al(feature_model, train_loader, device, 
                                                  args.model, args.al_embedding_dim)
    
    if verbose:
        print(f"Train embeddings shape: {train_embeddings.shape}")
    
    # Apply the selected strategy
    if al_strategy == 'coreset':
        # Traditional core-set approach
        val_loader = DataLoader(val_dataset, batch_size=al_batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        
        if verbose:
            print(f"Extracting embeddings from {len(val_dataset)} validation samples...")
        val_embeddings = extract_embeddings_for_al(feature_model, val_loader, device,
                                                   args.model, args.al_embedding_dim)
        
        # Start with initial labeled set (using validation as proxy for labeled data)
        max_initial_size = min(len(val_embeddings), len(train_embeddings) // 10)
        initial_size = min(args.al_initial_labeled, max_initial_size)
        labeled_embeddings = val_embeddings[:initial_size]
        
        strategy = CoreSetStrategy(embedding_dim=args.al_embedding_dim)
        selected_indices = strategy.greedy_k_center(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=train_embeddings,
            n_select=target_samples
        )
        
    elif al_strategy == 'diversity':
        # Diversity-based clustering
        n_clusters = getattr(args, 'al_clusters', target_samples // 10)
        strategy = DiversityStrategy(n_clusters=n_clusters)
        selected_indices = strategy.diversity_sample(
            embeddings=train_embeddings,
            n_select=target_samples,
            random_seed=args.seed
        )
        
    elif al_strategy == 'hybrid':
        # Hybrid approach combining multiple methods
        # Get targets for stratified component
        try:
            if hasattr(train_dataset, 'targets'):
                targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
                targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            else:
                # Fallback to diversity if no targets available
                if verbose:
                    print("No targets available for hybrid, falling back to diversity")
                strategy = DiversityStrategy()
                selected_indices = strategy.diversity_sample(train_embeddings, target_samples, args.seed)
            
            if 'targets' in locals():
                # Use validation embeddings for labeled set
                val_loader = DataLoader(val_dataset, batch_size=al_batch_size, shuffle=False,
                                       num_workers=args.workers, pin_memory=True)
                val_embeddings = extract_embeddings_for_al(feature_model, val_loader, device,
                                                           args.model, args.al_embedding_dim)
                
                max_initial_size = min(len(val_embeddings), len(train_embeddings) // 10)
                initial_size = min(args.al_initial_labeled, max_initial_size)
                labeled_embeddings = val_embeddings[:initial_size]
                
                # Configure hybrid ratios
                coreset_ratio = getattr(args, 'al_coreset_ratio', 0.4)
                stratified_ratio = getattr(args, 'al_stratified_ratio', 0.4)
                random_ratio = 1.0 - coreset_ratio - stratified_ratio
                
                strategy = HybridStrategy(coreset_ratio, stratified_ratio, random_ratio)
                selected_indices = strategy.hybrid_sample(
                    labeled_embeddings=labeled_embeddings,
                    unlabeled_embeddings=train_embeddings,
                    targets=targets,
                    n_select=target_samples,
                    random_seed=args.seed
                )
        except Exception as e:
            if verbose:
                print(f"Hybrid sampling failed: {e}, falling back to coreset")
            # Fallback to coreset
            val_loader = DataLoader(val_dataset, batch_size=al_batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
            val_embeddings = extract_embeddings_for_al(feature_model, val_loader, device,
                                                       args.model, args.al_embedding_dim)
            max_initial_size = min(len(val_embeddings), len(train_embeddings) // 10)
            initial_size = min(args.al_initial_labeled, max_initial_size)
            labeled_embeddings = val_embeddings[:initial_size]
            
            strategy = CoreSetStrategy(embedding_dim=args.al_embedding_dim)
            selected_indices = strategy.greedy_k_center(labeled_embeddings, train_embeddings, target_samples)
            
    elif al_strategy == 'hierarchical':
        # Hierarchical two-stage active learning
        # This strategy requires a trained model for uncertainty computation
        if verbose:
            print("Using hierarchical two-stage active learning")
        
        # Initialize hierarchical strategy with parameters from args
        strategy = HierarchicalActiveStrategy(
            w_err=getattr(args, 'al_w_err', 0.6),
            w_unc=getattr(args, 'al_w_unc', 0.4),
            alpha=getattr(args, 'al_alpha', 0.4),
            beta=getattr(args, 'al_beta', 0.4),
            gamma=getattr(args, 'al_gamma', 0.2),
            lambda_div=getattr(args, 'al_lambda_div', 0.5),
            round1_ratio=getattr(args, 'al_round1_ratio', 0.7),
            min_slice_quota=getattr(args, 'al_min_slice_quota', 5),
            max_slice_quota=getattr(args, 'al_max_slice_quota', 100),
            embedding_dim=args.al_embedding_dim,
            dataset=args.dataset
        )
        
        # For hierarchical AL, we need a pre-trained model to compute uncertainties
        # Use the feature extraction model as a proxy
        selected_indices = strategy.select(
            dataset=train_dataset,
            val_dataset=val_dataset,
            model=feature_model,
            device=device,
            n_select=target_samples,
            verbose=verbose
        )
            
    else:
        # Unknown strategy, fall back to random
        if verbose:
            print(f"Unknown AL strategy '{al_strategy}', falling back to random selection")
        random.seed(args.seed)
        selected_indices = random.sample(range(total_train_samples), target_samples)
    
    # Diagnostic: Check class distribution in selected samples
    if verbose:
        try:
            # For ImageNet subset, check class balance
            if hasattr(train_dataset, 'targets') or (hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets')):
                if hasattr(train_dataset, 'targets'):
                    all_targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
                else:
                    all_targets = np.array([train_dataset.dataset.targets[i] for i in train_dataset.indices])
                    
                selected_targets = all_targets[selected_indices]
                
                print(f"\nClass distribution in selected samples:")
                unique_classes, class_counts = np.unique(selected_targets, return_counts=True)
                for cls, count in zip(unique_classes, class_counts):
                    percentage = count / len(selected_targets) * 100
                    print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
                
                # Compare with random baseline
                random.seed(args.seed + 1)  # Different seed for comparison
                random_indices = random.sample(range(len(train_dataset)), target_samples)
                random_targets = all_targets[random_indices]
                random_unique, random_counts = np.unique(random_targets, return_counts=True)
                
                print(f"\nClass distribution in random baseline:")
                for cls, count in zip(random_unique, random_counts):
                    percentage = count / len(random_targets) * 100
                    print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
                    
        except Exception as e:
            if verbose:
                print(f"Could not analyze class distribution: {e}")
    
    if verbose:
        print(f"Active Learning selection completed. Selected {len(selected_indices)} samples using '{al_strategy}' strategy.")
    
    return selected_indices
