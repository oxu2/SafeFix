"""
Active Learning utilities for the safefix project.
This module contains multiple active learning strategies and related functions.
"""

from utils.imports import *
from utils.defines import *
from utils.hierarchical_al import HierarchicalActiveStrategy
from collections import Counter, defaultdict
import math


def parse_attribute_priority_log(log_path):
    """Parse Rare Case entries and associated accuracies from a diagnostic log file."""
    if not log_path or not os.path.exists(log_path):
        return {}

    pattern = re.compile(
        r"ACC:\s*([0-9]*\.?[0-9]+).*?Rare Case:\s*attribute\s*\"([^\"]+)\",\s*description\s*\"([^\"]+)\"",
        re.IGNORECASE,
    )

    priorities = defaultdict(dict)

    with open(log_path, "r") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if not match:
                continue

            acc_value, attribute, description = match.groups()
            try:
                accuracy = float(acc_value)
            except ValueError:
                continue

            if accuracy > 1.0:
                accuracy /= 100.0

            accuracy = max(0.0, min(accuracy, 1.0))
            failure_weight = 1.0 - accuracy

            priorities[attribute][description] = failure_weight

    return {attr: dict(values) for attr, values in priorities.items() if values}


def load_attribute_metadata(metadata_source):
    """Load attribute metadata from a path or return an existing dict."""
    if metadata_source is None:
        raise ValueError("Attribute metadata path must be provided for attribute strategy")

    if isinstance(metadata_source, dict):
        return metadata_source

    if not isinstance(metadata_source, str):
        raise ValueError("Attribute metadata must be a dict or path to a JSON file")

    if not os.path.exists(metadata_source):
        raise FileNotFoundError(f"Attribute metadata file '{metadata_source}' not found.")

    with open(metadata_source, "r") as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, dict):
        raise ValueError("Attribute metadata JSON must contain a dictionary")
    return metadata


def parse_attribute_value_counts(source, default_count=None):
    """Parse extra attribute counts either from an inline string or a JSON file."""
    if not source:
        return {}

    if os.path.exists(source):
        with open(source, "r") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Attribute count JSON must be a dict of attribute -> value -> count")
        parsed = {}
        for attr_name, value_map in data.items():
            if not isinstance(value_map, dict):
                raise ValueError(
                    f"Attribute '{attr_name}' in extra counts JSON must map to a dict of values"
                )
            parsed[attr_name] = {
                str(val): float(count) for val, count in value_map.items()
            }
        return parsed

    parsed = defaultdict(dict)
    entries = re.split(r"[;,]", source)
    for raw_entry in entries:
        entry = raw_entry.strip()
        if not entry:
            continue
        count = default_count
        attr_value = entry
        if "=" in entry:
            attr_value, count_str = entry.split("=", 1)
            try:
                count = float(count_str)
            except ValueError as exc:
                raise ValueError(f"Invalid count '{count_str}' for entry '{entry}'") from exc
        elif default_count is None:
            raise ValueError(
                "Extra attribute counts must include '=count' when no default_count is provided"
            )

        if ":" not in attr_value:
            raise ValueError(
                "Extra attribute count entries must follow attr:value or attr:value=count"
            )
        attr, value = attr_value.split(":", 1)
        if count is None:
            raise ValueError(
                f"No count specified for entry '{entry}' and default_count is None"
            )
        parsed[attr.strip()][value.strip()] = float(count)

    return {attr: dict(values) for attr, values in parsed.items() if values}


def parse_float_with_infinity(value, default=0.0):
    """Parse floats that may include textual infinity tokens."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = str(value).strip().lower()
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if text in {"-inf", "-infinity"}:
            return 0.0
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"Could not parse '{value}' as a float or infinity") from exc

    return parsed if parsed > 0 else 0.0


def infer_generated_attribute_counts(metadata, attribute_string, default_count, focus_attributes=None):
    """Infer extra counts from attribute string like 'pinkcolor_fabrictexture'."""
    if not attribute_string or not default_count or default_count <= 0:
        return {}

    metadata_dict = load_attribute_metadata(metadata)
    attributes_scope = (
        {k: v for k, v in metadata_dict.items() if not focus_attributes or k in focus_attributes}
        if isinstance(metadata_dict, dict)
        else {}
    )
    if not attributes_scope:
        return {}

    tokens = [tok for tok in re.split(r"[_\s]+", attribute_string) if tok]
    if not tokens:
        return {}

    inferred = defaultdict(dict)

    token_pairs = [(tok, tok.lower()) for tok in tokens]

    for attr_name, attr_info in attributes_scope.items():
        attr_suffix = attr_name.lower()
        if not attr_suffix:
            continue
        values = attr_info.get("data", [])
        value_lookup = {}
        for val in values:
            if val is None:
                continue
            lowered = str(val).lower()
            if lowered not in value_lookup:
                value_lookup[lowered] = val
        if not value_lookup:
            continue

        suffix_len = len(attr_suffix)
        for token_original, token_lower in token_pairs:
            if len(token_lower) <= suffix_len:
                continue
            if not token_lower.endswith(attr_suffix):
                continue
            candidate = token_lower[:-suffix_len]
            canonical_value = value_lookup.get(candidate)
            if canonical_value is None:
                continue
            inferred[attr_name][canonical_value] = float(default_count)

    return {attr: values for attr, values in inferred.items() if values}


def _coerce_targets_array(candidates):
    """Convert common target containers into a 1D integer numpy array."""
    if candidates is None:
        return None

    if torch.is_tensor(candidates):
        candidates = candidates.detach().cpu().numpy()
    elif not isinstance(candidates, np.ndarray):
        candidates = np.asarray(list(candidates))

    if candidates.ndim == 0:
        candidates = candidates.reshape(1)

    if candidates.ndim != 1:
        try:
            candidates = candidates.reshape(-1)
        except ValueError:
            return None

    try:
        return candidates.astype(int)
    except (TypeError, ValueError):
        cleaned = []
        for item in candidates.tolist():
            if torch.is_tensor(item):
                if item.numel() != 1:
                    return None
                item = item.item()
            elif isinstance(item, np.ndarray):
                if item.size != 1:
                    return None
                item = item.item()
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                return None
        return np.array(cleaned, dtype=int)


def extract_targets_from_dataset(dataset):
    """Extract per-sample class labels from common dataset wrappers."""
    from torch.utils.data import ConcatDataset, Subset

    if hasattr(dataset, "al_targets"):
        targets = _coerce_targets_array(getattr(dataset, "al_targets"))
        if targets is not None:
            return targets

    if isinstance(dataset, Subset):
        parent_targets = extract_targets_from_dataset(dataset.dataset)
        if parent_targets is None:
            return None
        return np.array([parent_targets[int(idx)] for idx in dataset.indices], dtype=int)

    if isinstance(dataset, ConcatDataset):
        all_targets = []
        for sub_dataset in dataset.datasets:
            sub_targets = extract_targets_from_dataset(sub_dataset)
            if sub_targets is None:
                return None
            all_targets.append(np.asarray(sub_targets, dtype=int))
        if not all_targets:
            return np.array([], dtype=int)
        return np.concatenate(all_targets, axis=0)

    for attr_name in ("targets", "labels"):
        if hasattr(dataset, attr_name):
            targets = _coerce_targets_array(getattr(dataset, attr_name))
            if targets is not None:
                return targets

    if hasattr(dataset, "samples"):
        targets = _coerce_targets_array([sample[1] for sample in dataset.samples])
        if targets is not None:
            return targets

    if hasattr(dataset, "imgs"):
        targets = _coerce_targets_array([sample[1] for sample in dataset.imgs])
        if targets is not None:
            return targets

    return None


class CoreSetStrategy:
    """
    Core-set selection strategy for Active Learning
    Selects samples that are most representative of the unlabeled data distribution
    """
    
    def __init__(self, embedding_dim=512, exact_kcenter_max_select=4096, exact_kcenter_max_samples=20000):
        self.embedding_dim = embedding_dim
        self.exact_kcenter_max_select = max(int(exact_kcenter_max_select), 1)
        self.exact_kcenter_max_samples = max(int(exact_kcenter_max_samples), 1)

    def _weighted_sample_without_replacement(self, weights, n_select, random_seed=42):
        n_samples = len(weights)
        if n_select >= n_samples:
            return list(range(n_samples))

        rng = np.random.default_rng(random_seed)
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        total_weight = float(np.sum(weights))

        if not np.isfinite(total_weight) or total_weight <= 0.0:
            return rng.choice(n_samples, size=n_select, replace=False).tolist()

        probabilities = weights / total_weight
        return rng.choice(n_samples, size=n_select, replace=False, p=probabilities).tolist()
        
    def greedy_k_center(self, labeled_embeddings, unlabeled_embeddings, n_select, random_seed=42):
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

        # Pre-compute distances to labeled data (one-time cost)
        distances_to_labeled = euclidean_distances(unlabeled_embeddings, labeled_embeddings)
        min_distances = np.min(distances_to_labeled, axis=1)

        use_exact_kcenter = (
            n_select <= self.exact_kcenter_max_select
            and n_unlabeled <= self.exact_kcenter_max_samples
        )

        if not use_exact_kcenter:
            print(
                "Using approximate core-set selection for large job: "
                f"{n_select} selections from {n_unlabeled} samples."
            )
            return self._weighted_sample_without_replacement(
                min_distances,
                n_select,
                random_seed=random_seed,
            )

        print("Progress: ", end="", flush=True)
        selected_indices = []
        
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


class BadgeStrategy:
    """
    BADGE selection via gradient embeddings and k-means++ seeding.

    This implementation uses a lightweight centroid classifier over the same
    pretrained embeddings already extracted for other AL strategies. That keeps
    the method usable before any task-specific fine-tuning has happened while
    still coupling uncertainty and diversity in the batch selection stage.
    """

    def __init__(self, temperature=1.0, exact_kmeanspp_max_select=512, exact_kmeanspp_max_samples=20000):
        self.temperature = max(float(temperature), 1e-6)
        self.exact_kmeanspp_max_select = max(int(exact_kmeanspp_max_select), 1)
        self.exact_kmeanspp_max_samples = max(int(exact_kmeanspp_max_samples), 1)
        self.last_proxy_stats = {}

    def _normalize_rows(self, array):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return array / norms

    def _softmax(self, logits):
        stabilized = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(stabilized)
        normalizer = np.sum(exp_logits, axis=1, keepdims=True)
        return exp_logits / np.maximum(normalizer, 1e-12)

    def _build_centroids(self, labeled_embeddings, labeled_targets):
        unique_targets = np.unique(labeled_targets)
        if unique_targets.size < 2:
            raise ValueError("BADGE requires at least two classes in the labeled proxy set")

        centroids = []
        centroid_labels = []
        for target in unique_targets:
            mask = labeled_targets == target
            if not np.any(mask):
                continue
            centroids.append(np.mean(labeled_embeddings[mask], axis=0))
            centroid_labels.append(int(target))

        if len(centroids) < 2:
            raise ValueError("BADGE could not form at least two class centroids")

        return np.stack(centroids, axis=0), np.array(centroid_labels, dtype=int)

    def _compute_gradient_embeddings(self, labeled_embeddings, unlabeled_embeddings, labeled_targets):
        labeled_embeddings = self._normalize_rows(np.asarray(labeled_embeddings, dtype=np.float32))
        unlabeled_embeddings = self._normalize_rows(np.asarray(unlabeled_embeddings, dtype=np.float32))
        labeled_targets = np.asarray(labeled_targets, dtype=int)

        centroids, centroid_labels = self._build_centroids(labeled_embeddings, labeled_targets)
        distances = euclidean_distances(unlabeled_embeddings, centroids, squared=True)
        probs = self._softmax(-distances / self.temperature)
        predicted_classes = np.argmax(probs, axis=1)

        n_samples, embedding_dim = unlabeled_embeddings.shape
        n_classes = probs.shape[1]
        gradient_embeddings = np.zeros((n_samples, embedding_dim * n_classes), dtype=np.float32)

        for class_idx in range(n_classes):
            coeff = probs[:, class_idx].copy()
            coeff[predicted_classes == class_idx] -= 1.0
            start = class_idx * embedding_dim
            end = start + embedding_dim
            gradient_embeddings[:, start:end] = unlabeled_embeddings * coeff[:, None]

        uncertainty = 1.0 - np.max(probs, axis=1)
        predicted_labels = centroid_labels[predicted_classes]
        self.last_proxy_stats = {
            "n_proxy_classes": int(len(centroid_labels)),
            "uncertainty_min": float(np.min(uncertainty)),
            "uncertainty_mean": float(np.mean(uncertainty)),
            "uncertainty_max": float(np.max(uncertainty)),
            "predicted_class_hist": {
                int(label): int(count)
                for label, count in zip(*np.unique(predicted_labels, return_counts=True))
            },
        }

        return gradient_embeddings

    def _kmeanspp_seeding(self, embeddings, n_select, random_seed=42):
        n_samples = embeddings.shape[0]
        if n_select >= n_samples:
            return list(range(n_samples))

        rng = np.random.default_rng(random_seed)
        selected = [int(np.argmax(np.sum(embeddings * embeddings, axis=1)))]
        min_distances = euclidean_distances(
            embeddings, embeddings[selected[0]:selected[0] + 1], squared=True
        ).reshape(-1)
        min_distances[selected[0]] = 0.0

        while len(selected) < n_select:
            total_distance = float(np.sum(min_distances))
            if not np.isfinite(total_distance) or total_distance <= 1e-12:
                remaining = np.setdiff1d(np.arange(n_samples), np.array(selected), assume_unique=False)
                if remaining.size == 0:
                    break
                next_idx = int(rng.choice(remaining))
            else:
                probs = min_distances / total_distance
                next_idx = int(rng.choice(n_samples, p=probs))
                if next_idx in selected:
                    remaining = np.setdiff1d(np.arange(n_samples), np.array(selected), assume_unique=False)
                    if remaining.size == 0:
                        break
                    next_idx = int(remaining[np.argmax(min_distances[remaining])])

            selected.append(next_idx)
            new_distances = euclidean_distances(
                embeddings, embeddings[next_idx:next_idx + 1], squared=True
            ).reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[selected] = 0.0

        return selected

    def _weighted_sample_without_replacement(self, weights, n_select, random_seed=42):
        n_samples = len(weights)
        if n_select >= n_samples:
            return list(range(n_samples))

        rng = np.random.default_rng(random_seed)
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, 0.0)

        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            return rng.choice(n_samples, size=n_select, replace=False).tolist()

        probabilities = weights / total_weight
        return rng.choice(n_samples, size=n_select, replace=False, p=probabilities).tolist()

    def select(self, labeled_embeddings, unlabeled_embeddings, labeled_targets, n_select, random_seed=42):
        gradient_embeddings = self._compute_gradient_embeddings(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=unlabeled_embeddings,
            labeled_targets=labeled_targets,
        )

        n_samples = gradient_embeddings.shape[0]
        if n_select >= n_samples:
            self.last_proxy_stats["selection_mode"] = "all_samples"
            return list(range(n_samples))

        # Exact BADGE kmeans++ becomes impractical on CelebA-sized runs because it updates
        # distances against the full candidate set once per selected point. Switch to a
        # gradient-norm weighted approximation for larger jobs so the selection stage
        # finishes in a reasonable amount of time.
        use_exact_kmeanspp = (
            n_select <= self.exact_kmeanspp_max_select
            and n_samples <= self.exact_kmeanspp_max_samples
        )

        if use_exact_kmeanspp:
            self.last_proxy_stats["selection_mode"] = "exact_badge_kmeanspp"
            return self._kmeanspp_seeding(gradient_embeddings, n_select, random_seed=random_seed)

        gradient_norms = np.linalg.norm(gradient_embeddings, axis=1)
        self.last_proxy_stats["selection_mode"] = "approx_badge_weighted_gradient_norm"
        self.last_proxy_stats["gradient_norm_min"] = float(np.min(gradient_norms))
        self.last_proxy_stats["gradient_norm_mean"] = float(np.mean(gradient_norms))
        self.last_proxy_stats["gradient_norm_max"] = float(np.max(gradient_norms))
        return self._weighted_sample_without_replacement(
            gradient_norms,
            n_select,
            random_seed=random_seed,
        )


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
                labeled_embeddings, unlabeled_embeddings, n_coreset, random_seed=random_seed
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


class UncertaintyStratifiedStrategy:
    """
    Uncertainty-Weighted Stratified Sampling Strategy.
    Maintains strict class quotas but selects the most uncertain samples within each class.
    """

    def __init__(self):
        pass

    def select(self, model, dataloader, device, n_select, model_type='resnet'):
        """
        Select samples using stratified uncertainty (entropy/confidence).

        Args:
            model: trained model or feature extractor
            dataloader: DataLoader (must not be shuffled to preserve indices)
            device: computation device
            n_select: number of samples to select
            model_type: type of model

        Returns:
            selected_indices: indices of selected samples
        """
        print(f"Uncertainty-Stratified: choosing {n_select} samples...")

        # 1. Collect predictions and targets
        model.eval()
        all_uncertainties = []
        all_targets = []
        
        # We need to extract targets from the dataloader while collecting uncertainties
        # This assumes the dataloader returns (data, target) tuples
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.to(device)
                outputs = model(data)
                
                # Compute uncertainty
                if model_type in ['resnet', 'vit', 'clip'] and outputs.dim() > 1:
                    # Multi-class: use entropy
                    probs = torch.softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties = entropy.cpu().numpy()
                elif outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                    # Binary: use confidence (distance from 0.5)
                    probs = torch.sigmoid(outputs.squeeze())
                    confidence = torch.abs(probs - 0.5) * 2
                    uncertainties = 1.0 - confidence.cpu().numpy() # Higher is more uncertain
                else:
                    # Fallback
                    uncertainties = np.random.rand(len(targets))
                
                all_uncertainties.extend(uncertainties)
                all_targets.extend(targets.numpy())

        all_uncertainties = np.array(all_uncertainties)
        all_targets = np.array(all_targets)

        # 2. Determine stratified quotas
        unique_classes, class_counts = np.unique(all_targets, return_counts=True)
        n_classes = len(unique_classes)
        samples_per_class = n_select // n_classes
        remaining_samples = n_select % n_classes

        print(f"  Class breakdown: {samples_per_class} per class (+{remaining_samples} remainder)")

        selected_indices = []

        # 3. Select top uncertainty per class
        for i, cls in enumerate(unique_classes):
            # Indices matching this class
            cls_indices_global = np.where(all_targets == cls)[0]
            
            # Quota for this class
            n_cls_target = samples_per_class + (1 if i < remaining_samples else 0)
            n_cls_target = min(n_cls_target, len(cls_indices_global))
            
            # Get uncertainties for this class
            cls_uncertainties = all_uncertainties[cls_indices_global]
            
            # Sort by uncertainty (descending)
            # argsort returns indices into cls_uncertainties
            sorted_local_indices = np.argsort(cls_uncertainties)[::-1] # High to low
            top_local_indices = sorted_local_indices[:n_cls_target]
            
            # Map back to global indices
            top_global_indices = cls_indices_global[top_local_indices]
            selected_indices.extend(top_global_indices)
            
            # print(f"    Class {cls}: selected {len(top_global_indices)} (mean unc: {cls_uncertainties[top_local_indices].mean():.4f})")

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
        value_weights=None,
        log_acc_scale=1.0,
        extra_counts=None,
        frequency_penalty=0.0,
        class_balance_weight=0.0,
        conditional_rarity=False,
        use_entropy=False,
    ):

        """
        Args:
            attribute_metadata: Path to attribute JSON or pre-loaded dict.
            focus_attributes: Optional iterable of attribute keys to use.
            rarity_power: Exponent applied to rarity weighting (>= 0).
            uncertain_boost: Score bonus added when a sample belongs to an
                uncertain attribute slice.
            epsilon: Numerical stability term for rarity weighting.
            value_weights: Optional dict {attribute: {value: weight}} derived
                from diagnostic logs.
            log_acc_scale: Scaling factor applied to each log-based weight.
            extra_counts: Optional dict of additional counts per attribute value
                used to augment observed frequencies (e.g., generated data).
            frequency_penalty: Penalty multiplier for frequent attribute values.
            class_balance_weight: Non-negative weight controlling class-level
                stratification (use float('inf') for strict round-robin quotas).
            conditional_rarity: If True, compute rarity relative to the class label
                (P(val|class)) instead of global P(val).
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
        self.value_weights = value_weights or {}
        self.log_acc_scale = log_acc_scale
        self.extra_counts = {
            attr: {val: float(count) for val, count in value_map.items()}
            for attr, value_map in (extra_counts or {}).items()
        }
        self.frequency_penalty = max(frequency_penalty, 0.0)
        try:
            parsed_balance = float(class_balance_weight)
        except (TypeError, ValueError):
            parsed_balance = 0.0
        self.class_balance_weight = parsed_balance if math.isinf(parsed_balance) else max(parsed_balance, 0.0)
        self.conditional_rarity = conditional_rarity
        self.use_entropy = use_entropy

    def _log_attribute_distribution(self, metadata_indices, local_indices, label):
        if not local_indices:
            return

        print(f"\n{label}:")
        for attr_name, attr_info in self.attributes.items():
            value_counts = Counter()
            valid_total = 0
            data = attr_info.get("data", [])
            for local_idx in local_indices:
                if local_idx >= len(metadata_indices):
                    continue
                meta_idx = metadata_indices[local_idx]
                if meta_idx is None or meta_idx >= len(data):
                    continue
                value = data[meta_idx]
                value_counts[value] += 1
                valid_total += 1

            if valid_total == 0:
                print(f"  Attribute '{attr_name}': no coverage")
                continue

            for value, count in value_counts.most_common():
                percentage = (count / valid_total) * 100.0 if valid_total else 0.0
                print(f"  [{attr_name}] {value}: {count} samples ({percentage:.1f}%)")

    def _resolve_metadata_indices(self, dataset):
        """Map dataset-local indices to attribute metadata indices."""
        from torch.utils.data import Subset, ConcatDataset

        dataset_len = len(dataset) if hasattr(dataset, "__len__") else 0

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

        if hasattr(dataset, "attribute_metadata_indices"):
            custom_indices = list(getattr(dataset, "attribute_metadata_indices"))
            if len(custom_indices) != dataset_len:
                raise ValueError(
                    "attribute_metadata_indices length must match dataset length"
                )
            return custom_indices

        # Base dataset: assume attribute metadata aligns with dataset order
        # Heuristic: generated datasets typically do not have attribute metadata
        class_name = dataset.__class__.__name__.lower()
        if "generated" in class_name or dataset_len == 0:
            return [None] * dataset_len

        return list(range(dataset_len))

    def _resolve_class_labels(self, dataset):
        """Map dataset-local indices to class labels when available."""
        from torch.utils.data import Subset, ConcatDataset

        if isinstance(dataset, Subset):
            parent_labels = self._resolve_class_labels(dataset.dataset)
            if parent_labels is None:
                return [None] * len(dataset)
            resolved = []
            for idx in dataset.indices:
                if idx is None or idx >= len(parent_labels):
                    resolved.append(None)
                else:
                    resolved.append(parent_labels[idx])
            return resolved

        if isinstance(dataset, ConcatDataset):
            labels = []
            for sub_dataset in dataset.datasets:
                sub_labels = self._resolve_class_labels(sub_dataset)
                if sub_labels is None:
                    sub_len = len(sub_dataset) if hasattr(sub_dataset, "__len__") else 0
                    labels.extend([None] * sub_len)
                else:
                    labels.extend(sub_labels)
            return labels

        if hasattr(dataset, "al_targets"):
            candidates = list(getattr(dataset, "al_targets"))
        else:
            candidates = None

        if candidates is None:
            for attr_name in ("targets", "labels"):
                if hasattr(dataset, attr_name):
                    candidates = getattr(dataset, attr_name)
                    break

        if candidates is None and hasattr(dataset, "samples"):
            candidates = [sample[1] for sample in dataset.samples]
        if candidates is None and hasattr(dataset, "imgs"):
            candidates = [sample[1] for sample in dataset.imgs]

        if candidates is not None:
            if isinstance(candidates, np.ndarray):
                candidates = candidates.tolist()
            else:
                candidates = list(candidates)
            resolved = []
            for label in candidates:
                try:
                    resolved.append(int(label) if label is not None else None)
                except (TypeError, ValueError):
                    resolved.append(None)
            return resolved

        dataset_len = len(dataset) if hasattr(dataset, "__len__") else 0
        return [None] * dataset_len

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
        if self.extra_counts:
            for attr_name, value_map in self.extra_counts.items():
                if attr_name not in counts:
                    counts[attr_name] = Counter()
                for value, bonus in value_map.items():
                    counts[attr_name][value] += bonus
        return counts

    def _compute_conditional_attribute_counts(self, metadata_indices, class_labels):
        """Compute frequency of each attribute value per class."""
        # Structure: {attr: {class_label: Counter()}}
        counts = {attr: defaultdict(Counter) for attr in self.attributes}
        
        if not class_labels or len(class_labels) != len(metadata_indices):
            return counts

        for i, meta_idx in enumerate(metadata_indices):
            if meta_idx is None:
                continue
            
            label = class_labels[i]
            if label is None:
                continue
                
            label = int(label)
            
            for attr_name, attr_info in self.attributes.items():
                data = attr_info.get("data", [])
                if meta_idx < len(data):
                    value = data[meta_idx]
                    counts[attr_name][label][value] += 1
                    
        # Apply extra counts globally (since they might not be class-specific in current format)
        # OR just ignore extra counts for conditional rarity for now to keep it simple
        # The prompt implies we want to find "Red Banana", so we rely on observed data.
        return counts

    def _select_with_class_balance(self, base_scores, class_labels, n_select):
        if class_labels is None or n_select <= 0:
            return None

        total_samples = len(base_scores)
        if total_samples == 0 or len(class_labels) != total_samples:
            return None

        label_to_indices = defaultdict(list)
        unlabeled_indices = []
        for idx, label in enumerate(class_labels):
            if label is None:
                unlabeled_indices.append(idx)
            else:
                try:
                    label_to_indices[int(label)].append(idx)
                except (TypeError, ValueError):
                    unlabeled_indices.append(idx)

        active_labels = [label for label, queue in label_to_indices.items() if queue]
        if len(active_labels) <= 1:
            return None

        for queue in label_to_indices.values():
            queue.sort(key=lambda value_idx: base_scores[value_idx], reverse=True)
        unlabeled_indices.sort(key=lambda value_idx: base_scores[value_idx], reverse=True)

        if math.isinf(self.class_balance_weight):
            preliminary = self._round_robin_class_selection(label_to_indices, n_select)
        else:
            preliminary = self._soft_class_selection(
                label_to_indices,
                unlabeled_indices,
                base_scores,
                n_select,
            )

        return self._finalize_balanced_selection(base_scores, preliminary, n_select)

    def _round_robin_class_selection(self, label_to_indices, n_select):
        ordered_labels = [label for label, queue in label_to_indices.items() if queue]
        if not ordered_labels:
            return []

        selected = []
        active_labels = ordered_labels.copy()
        pointer = 0
        while len(selected) < n_select and active_labels:
            idx = pointer % len(active_labels)
            label = active_labels[idx]
            queue = label_to_indices[label]
            if queue:
                selected.append(queue.pop(0))
                pointer += 1
            else:
                active_labels.pop(idx)
                if not active_labels:
                    break
        return selected

    def _soft_class_selection(self, label_to_indices, unlabeled_indices, base_scores, n_select):
        if not label_to_indices:
            return []

        available_labels = [label for label, queue in label_to_indices.items() if queue]
        if not available_labels:
            return []

        n_classes = len(available_labels)
        target = n_select / n_classes if n_classes else 0.0
        denominator = max(target, 1e-6)
        class_counts = {label: 0 for label in label_to_indices}
        unlabeled_ptr = 0
        selected = []
        max_iterations = max(4 * n_select, len(base_scores))

        for _ in range(max_iterations):
            if len(selected) >= n_select:
                break

            best_label = None
            best_idx = None
            best_score = -np.inf

            for label, queue in label_to_indices.items():
                if not queue:
                    continue
                candidate_idx = queue[0]
                deficit = target - class_counts.get(label, 0)
                bonus = self.class_balance_weight * (deficit / denominator)
                effective = base_scores[candidate_idx] + bonus
                if effective > best_score:
                    best_score = effective
                    best_label = label
                    best_idx = candidate_idx

            unlabeled_candidate = None
            unlabeled_score = -np.inf
            if unlabeled_ptr < len(unlabeled_indices):
                unlabeled_candidate = unlabeled_indices[unlabeled_ptr]
                unlabeled_score = base_scores[unlabeled_candidate]
            if unlabeled_candidate is not None and unlabeled_score > best_score:
                best_label = None
                best_idx = unlabeled_candidate
                best_score = unlabeled_score

            if best_idx is None:
                break

            selected.append(best_idx)
            if best_label is None:
                unlabeled_ptr += 1
            else:
                class_counts[best_label] = class_counts.get(best_label, 0) + 1
                label_to_indices[best_label].pop(0)

        return selected

    def _finalize_balanced_selection(self, base_scores, preliminary_indices, n_select):
        if preliminary_indices is None:
            preliminary_indices = []

        ordered = []
        seen = set()
        for idx in preliminary_indices:
            idx_int = int(idx)
            if idx_int in seen:
                continue
            ordered.append(idx_int)
            seen.add(idx_int)
            if len(ordered) >= n_select:
                return ordered[:n_select]

        ranked = np.argsort(-base_scores)
        for idx in ranked:
            idx_int = int(idx)
            if idx_int in seen:
                continue
            ordered.append(idx_int)
            seen.add(idx_int)
            if len(ordered) >= n_select:
                break

        return ordered[:n_select]

    def select(self, dataset, n_select, random_seed=42, debug_attribute=None, model=None, unlabeled_dataloader=None, device=None):
        """
        Score each sample by attribute rarity and uncertainty, then pick the top ones.

        Args:
            dataset: Dataset or subset aligned with attribute metadata indices.
            n_select: Number of samples to select.
            random_seed: Seed used when generating tie-break noise.
            debug_attribute: Optional attribute key to emit component breakdown (R, U, L).
        """
        total_samples = len(dataset)
        if n_select >= total_samples:
            return list(range(total_samples))

        metadata_indices = self._resolve_metadata_indices(dataset)
        if len(metadata_indices) != total_samples:
            raise RuntimeError("Metadata index mapping length mismatch with dataset")

        class_labels = None
        if self.conditional_rarity or self.class_balance_weight > 0:
            labels_candidate = self._resolve_class_labels(dataset)
            if labels_candidate and len(labels_candidate) == total_samples:
                class_labels = labels_candidate
        entropy_scores = None
        if self.use_entropy:
            if model is None or unlabeled_dataloader is None or device is None:
                raise ValueError("model, unlabeled_dataloader, and device must be provided when use_entropy is True")
            
            model.eval()
            entropy_list = []
            
            with torch.no_grad():
                for batch in unlabeled_dataloader:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        probs = torch.softmax(outputs, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    else:
                        probs = torch.sigmoid(outputs.squeeze())
                        confidence = torch.abs(probs - 0.5) * 2
                        entropy = 1.0 - confidence
                        
                    entropy_list.extend(entropy.cpu().numpy())
            
            entropy_scores = np.array(entropy_list)
            if len(entropy_scores) != total_samples:
                print(f"Warning: Calculated {len(entropy_scores)} entropy scores but dataset has {total_samples} samples.")

        attribute_counts = self._compute_attribute_counts(metadata_indices)
        conditional_counts = None
        if self.conditional_rarity and class_labels is not None:
             conditional_counts = self._compute_conditional_attribute_counts(metadata_indices, class_labels)

        attribute_totals = {
            attr: float(sum(counter.values()))
            for attr, counter in attribute_counts.items()
        }

        scores = np.zeros(total_samples, dtype=np.float32)
        rng = np.random.default_rng(random_seed)

        per_attribute_breakdown = (
            defaultdict(
                lambda: defaultdict(
                    lambda: {
                        "frequency": 0,
                        "rarity": 0.0,
                        "log": 0.0,
                        "balance": 0.0,
                        "indices": [],
                        "uncertain_count": 0,
                    }
                )
            )
            if debug_attribute
            else None
        )

        for local_idx, meta_idx in enumerate(metadata_indices):
            if meta_idx is None:
                continue

            sample_score = 0.0
            for attr_name, attr_info in self.attributes.items():
                data = attr_info["data"]
                if meta_idx >= len(data):
                    continue
                value = data[meta_idx]
                
                if self.conditional_rarity and conditional_counts and class_labels[local_idx] is not None:
                    # Conditional Rarity: P(val | class)
                    cls_label = int(class_labels[local_idx])
                    frequency = conditional_counts[attr_name][cls_label].get(value, 0)
                else:
                    # Global Rarity: P(val)
                    frequency = attribute_counts[attr_name].get(value, 0)
                
                rarity = 1.0 / ((frequency if frequency > 0 else 1) + self.epsilon)
                if self.rarity_power != 1.0:
                    rarity = rarity ** self.rarity_power
                sample_score += rarity

                if self.use_entropy:
                    if entropy_scores is not None and local_idx < len(entropy_scores):
                        sample_score += self.uncertain_boost * entropy_scores[local_idx]
                elif meta_idx in attr_info["uncertain"]:
                    sample_score += self.uncertain_boost

                log_increment = 0.0
                if self.value_weights:
                    attr_weights = self.value_weights.get(attr_name, {})
                    if attr_weights:
                        log_increment = self.log_acc_scale * attr_weights.get(value, 0.0)
                        sample_score += log_increment

                penalty_component = 0.0
                if self.frequency_penalty > 0:
                    total_count = attribute_totals.get(attr_name, 0.0)
                    if total_count > 0.0:
                        freq_ratio = frequency / total_count
                        penalty_component = -self.frequency_penalty * freq_ratio
                        sample_score += penalty_component

                if debug_attribute and attr_name == debug_attribute:
                    stats = per_attribute_breakdown[attr_name][value]
                    stats["frequency"] = frequency
                    stats["rarity"] = float(rarity)
                    stats["log"] = float(log_increment)
                    stats["penalty"] = float(penalty_component)
                    stats["indices"].append(meta_idx)
                    if meta_idx in attr_info["uncertain"]:
                        stats["uncertain_count"] += 1

            scores[local_idx] = sample_score

        if np.all(scores == 0):
            noise = rng.uniform(low=0.0, high=1.0, size=total_samples)
            if self.class_balance_weight > 0 and class_labels:
                balanced_pick = self._select_with_class_balance(noise, class_labels, n_select)
                if balanced_pick is not None:
                    return balanced_pick
            return rng.choice(total_samples, size=n_select, replace=False).tolist()

        # Tie-break by adding a small amount of random noise (stable selection)
        noise = rng.uniform(low=0.0, high=1e-6, size=total_samples)
        base_scores = scores + noise

        selected_indices = None
        if self.class_balance_weight > 0 and class_labels:
            balanced_selection = self._select_with_class_balance(base_scores, class_labels, n_select)
            if balanced_selection is not None:
                selected_indices = balanced_selection

        if selected_indices is None:
            ranked_indices = np.argsort(-base_scores)
            selected_indices = ranked_indices[:n_select].tolist()

        if selected_indices:
            self._log_attribute_distribution(
                metadata_indices=metadata_indices,
                local_indices=selected_indices,
                label="Attribute distribution in selected samples",
            )

        if debug_attribute and per_attribute_breakdown:
            value_stats = per_attribute_breakdown.get(debug_attribute, {})
            if value_stats:
                print(f"Debug attribute breakdown for '{debug_attribute}':")
                sorted_items = sorted(value_stats.items(), key=lambda kv: (kv[1]["frequency"], kv[0]))
                for value, stats in sorted_items:
                    rarity = stats["rarity"]
                    log_component = stats["log"]
                    uncertainty_component = self.uncertain_boost if stats["uncertain_count"] > 0 else 0.0
                    balance_component = stats.get("balance", 0.0)
                    sorted_indices = sorted(stats["indices"])
                    if len(sorted_indices) > 10:
                        indices_preview = ",".join(str(idx) for idx in sorted_indices[:10]) + ",..."
                    else:
                        indices_preview = ",".join(str(idx) for idx in sorted_indices)
                    print(
                        f"  value={value}, freq={stats['frequency']}, indices=[{indices_preview}], "
                        f"R={rarity:.4f}, U={uncertainty_component:.4f}, L={log_component:.4f}, "
                        f"B={balance_component:.4f}"
                    )

        return selected_indices


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
    al_strategy = str(getattr(args, 'al_strategy', 'coreset') or 'random').lower().strip()

    # Normalize falsy / alias values -> 'random'
    _RANDOM_ALIASES = {'none', '0', 'false', 'no', 'random', ''}
    if al_strategy in _RANDOM_ALIASES:
        al_strategy = 'random'

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

    # Explicit random shortcut (also covers 'none'/'0'/'false' after normalization above)
    if al_strategy == 'random':
        if verbose:
            print(f"Random selection: choosing {target_samples} samples from {total_train_samples}")
        random.seed(args.seed)
        return random.sample(range(total_train_samples), target_samples)

    # Handle pure stratified sampling (doesn't need embeddings)
    if al_strategy == 'stratified':
        try:
            targets = extract_targets_from_dataset(train_dataset)
            if targets is None:
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
        log_acc_scale = getattr(args, 'al_attribute_log_acc_scale', 1.0)
        extra_counts_raw = getattr(args, 'al_attribute_extra_counts', None)
        extra_count_scale = getattr(args, 'al_attribute_extra_count_scale', 1.0)
        frequency_penalty = getattr(args, 'al_attribute_frequency_penalty', 0.0)
        class_balance_raw = getattr(args, 'al_attribute_class_balance', 0.0)
        class_balance_weight = parse_float_with_infinity(class_balance_raw)
        use_entropy = getattr(args, 'al_attribute_use_entropy', False)

        try:
            extra_count_scale = float(extra_count_scale)
        except (TypeError, ValueError):
            extra_count_scale = 1.0
        extra_count_scale = max(extra_count_scale, 0.0)

        focus_attributes = None
        if isinstance(focus_raw, str) and focus_raw.strip():
            focus_attributes = [part.strip() for part in focus_raw.split(',') if part.strip()]
        elif isinstance(focus_raw, (list, tuple)):
            focus_attributes = list(focus_raw)

        extra_counts = None
        if isinstance(extra_counts_raw, str) and extra_counts_raw.strip():
            try:
                extra_counts = parse_attribute_value_counts(
                    extra_counts_raw.strip(),
                    default_count=getattr(args, 'num_added_img', None)
                )
            except ValueError as exc:
                if verbose:
                    print(f"Attribute strategy: failed to parse extra counts ({exc})")
                raise

        attribute_metadata_obj = load_attribute_metadata(attribute_path)

        if extra_counts is None and getattr(args, 'num_added_img', 0):
            inferred_counts = infer_generated_attribute_counts(
                attribute_metadata_obj,
                getattr(args, 'attributes', ''),
                getattr(args, 'num_added_img', 0),
                focus_attributes=focus_attributes,
            )
            if inferred_counts:
                extra_counts = inferred_counts
                if verbose:
                    print(
                        "Attribute strategy: auto-injected extra counts for generated attributes "
                        f"{inferred_counts}"
                    )

        if extra_counts and extra_count_scale != 1.0:
            extra_counts = {
                attr_name: {
                    value: float(count) * extra_count_scale
                    for value, count in value_map.items()
                }
                for attr_name, value_map in extra_counts.items()
            }
            if verbose:
                print(
                    "Attribute strategy: scaled extra counts by "
                    f"{extra_count_scale:g} -> {extra_counts}"
                )

        if not log_path and getattr(args, 'dataset', None) and getattr(args, 'model', None):
            default_log = Path('logs/compare') / f"stat_{args.dataset}_{args.model}_compare.log"
            if default_log.exists():
                log_path = str(default_log)
            elif verbose:
                print(f"Attribute strategy: default log {default_log} not found.")

        value_weights = None
        if log_path:
            priority_map = parse_attribute_priority_log(log_path)
            if priority_map:
                value_weights = priority_map
                if verbose:
                    print(f"Loaded attribute accuracies from {log_path}: {len(value_weights)} attributes with weights")
            elif verbose:
                print(f"Attribute strategy: No Rare Case entries found in {log_path}")
        elif verbose:
            print("Attribute strategy: No log file provided for attribute weights")

        try:
            strategy = AttributeFocusedStrategy(
                attribute_metadata=attribute_metadata_obj,
                focus_attributes=focus_attributes,
                rarity_power=rarity_power,
                uncertain_boost=uncertain_boost,
                value_weights=value_weights,
                log_acc_scale=log_acc_scale,
                extra_counts=extra_counts,
                frequency_penalty=frequency_penalty,
                class_balance_weight=class_balance_weight,
                conditional_rarity=getattr(args, 'al_conditional_rarity', False),
                use_entropy=use_entropy,
            )
            debug_attr = None
            if verbose:
                preferred_attr = "color"
                if preferred_attr in strategy.attributes:
                    debug_attr = preferred_attr
                elif focus_attributes:
                    for attr in focus_attributes:
                        if attr in strategy.attributes:
                            debug_attr = attr
                            break

            model = None
            train_loader = None
            if use_entropy:
                if verbose:
                    print("Attribute strategy: Loading model and dataloader for dynamic entropy calculation...")
                if args.model == 'resnet':
                    model = models.resnet18(pretrained=True).to(device)
                else:
                    model = models.resnet18(pretrained=True).to(device)
                
                al_batch_size = min(getattr(args, 'batch_size', 32), 32)
                train_loader = DataLoader(train_dataset, batch_size=al_batch_size, shuffle=False,
                                          num_workers=getattr(args, 'workers', 2), pin_memory=True)

            selected_indices = strategy.select(
                train_dataset,
                target_samples,
                random_seed=args.seed,
                debug_attribute=debug_attr,
                model=model,
                unlabeled_dataloader=train_loader,
                device=device
            )

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

    elif al_strategy == 'gen_probe':
        if verbose:
            print(f"Using Generation-Guided Active Learning (gen_probe) Strategy")
        
        probe_path = getattr(args, 'al_probe_model', None)
        if not probe_path or not os.path.exists(probe_path):
            if verbose:
                print(f"Error: gen_probe requires --al_probe_model to point to a valid checkpoint. Found: {probe_path}")
                print("Falling back to random selection.")
            random.seed(args.seed)
            return random.sample(range(total_train_samples), target_samples)

        # 1. Load Probe Model
        # Dynamically determine number of classes from target data
        if hasattr(train_dataset, 'targets'):
            num_classes = len(np.unique(train_dataset.targets))
        elif hasattr(train_dataset, 'labels'):
            num_classes = len(np.unique(train_dataset.labels))
        elif args.dataset == 'imagenet':
            num_classes = 10
        else:
            num_classes = 2 # fallback binary


        if args.model == 'resnet':
            probe_model = models.resnet18(pretrained=False) # Wait to load weights
            # Resnet18 defaults to 1000 classes, but our checkpoint has num_classes
            probe_model.fc = nn.Linear(probe_model.fc.in_features, num_classes)
        else:
            if verbose:
                print(f"Warning: Model {args.model} gen_probe not fully supported yet, using ResNet18 proxy")
            probe_model = models.resnet18(pretrained=False)
            probe_model.fc = nn.Linear(probe_model.fc.in_features, num_classes)

        if verbose:
            print(f"Loading probe model from {probe_path}")
        ckpt = torch.load(probe_path, map_location=device)
        try:
            probe_model.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            if verbose:
                print(f"Error loading probe model dict: {e}. Falling back to random.")
            random.seed(args.seed)
            return random.sample(range(total_train_samples), target_samples)

        probe_model = probe_model.to(device)
        probe_model.eval()

        # 2. Iterate and score all training samples
        al_batch_size = min(args.batch_size, 32)
        train_loader = DataLoader(train_dataset, batch_size=al_batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        
        all_entropies = []
        if verbose:
            print(f"Scoring {total_train_samples} source samples using the probe model...")
        
        with torch.no_grad():
            for batch_data in train_loader:
                if isinstance(batch_data, list):
                    inputs = batch_data[0].to(device, non_blocking=True)
                else:
                    inputs = batch_data.to(device, non_blocking=True)
                
                outputs = probe_model(inputs)
                probs = F.softmax(outputs, dim=1)
                # Calculate Entropy: -sum(p * log(p))
                log_probs = F.log_softmax(outputs, dim=1)
                entropy = -(probs * log_probs).sum(dim=1)
                all_entropies.extend(entropy.cpu().numpy())
        
        all_entropies = np.array(all_entropies)

        # 3. Select top K most uncertain (highest entropy)
        # argsort is ascending, so [::-1] makes it descending
        sorted_indices = np.argsort(all_entropies)[::-1]
        selected_indices = sorted_indices[:target_samples].tolist()
        
        if verbose:
            print(f"Generation-Guided Active Learning completed.")
            print(f"Selected {len(selected_indices)} hardest samples (highest probe entropy).")
            print(f"Max entropy: {all_entropies[selected_indices[0]]:.4f}, Min entropy in subset: {all_entropies[selected_indices[-1]]:.4f}")
            
        return selected_indices

    elif al_strategy == 'uncertainty_stratified':
        # Combined uncertainty + stratified sampling
        # Requires a model with a classification head
        if verbose:
            print("Using Uncertainty-Weighted Stratified Strategy")

        # 1. Load Model (with head)
        if args.model == 'resnet':
            model = models.resnet18(pretrained=True)
            # Default ResNet18 has 1000 classes, matching ImageNet
        elif args.model == 'clip':
             # CLIP not straightforward for classification uncertainty without prompts
             # Fallback to feature extraction -> linear probe?
             # For simplicity, warn and use ResNet
             if verbose:
                print("Warning: CLIP uncertainty not fully supported, using ResNet18 proxy for uncertainty")
             model = models.resnet18(pretrained=True)
        else:
            if verbose:
                print(f"Warning: Model {args.model} uncertainty not fully supported, using ResNet18 proxy")
            model = models.resnet18(pretrained=True)
        
        model = model.to(device)
        model.eval()

        # 2. Create DataLoader (no shuffle!)
        # Check if train_dataset is a Subset, if so, we need proper collation
        al_batch_size = min(args.batch_size, 32)
        train_loader = DataLoader(train_dataset, batch_size=al_batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

        # 3. Select
        strategy = UncertaintyStratifiedStrategy()
        selected_indices = strategy.select(
            model=model,
            dataloader=train_loader,
            device=device,
            n_select=target_samples,
            model_type=args.model
        )
        return selected_indices
    
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
            n_select=target_samples,
            random_seed=args.seed
        )
        
    elif al_strategy == 'badge':
        val_loader = DataLoader(val_dataset, batch_size=al_batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        if verbose:
            print(f"Extracting embeddings from {len(val_dataset)} validation samples for BADGE...")
        val_embeddings = extract_embeddings_for_al(feature_model, val_loader, device,
                                                   args.model, args.al_embedding_dim)

        val_targets = extract_targets_from_dataset(val_dataset)
        if val_targets is None:
            if verbose:
                print("BADGE failed: could not extract validation targets for the proxy classifier. Falling back to diversity.")
            strategy = DiversityStrategy(n_clusters=getattr(args, 'al_clusters', target_samples // 10))
            selected_indices = strategy.diversity_sample(
                embeddings=train_embeddings,
                n_select=target_samples,
                random_seed=args.seed
            )
        else:
            try:
                strategy = BadgeStrategy()
                selected_indices = strategy.select(
                    labeled_embeddings=val_embeddings,
                    unlabeled_embeddings=train_embeddings,
                    labeled_targets=val_targets,
                    n_select=target_samples,
                    random_seed=args.seed
                )
                if verbose and strategy.last_proxy_stats:
                    stats = strategy.last_proxy_stats
                    print(
                        "BADGE proxy classifier: "
                        f"{stats['n_proxy_classes']} classes, "
                        f"uncertainty min/mean/max="
                        f"{stats['uncertainty_min']:.4f}/"
                        f"{stats['uncertainty_mean']:.4f}/"
                        f"{stats['uncertainty_max']:.4f}"
                    )
                    print(f"BADGE predicted class histogram: {stats['predicted_class_hist']}")
            except Exception as exc:
                if verbose:
                    print(f"BADGE selection failed ({exc}). Falling back to diversity.")
                strategy = DiversityStrategy(n_clusters=getattr(args, 'al_clusters', target_samples // 10))
                selected_indices = strategy.diversity_sample(
                    embeddings=train_embeddings,
                    n_select=target_samples,
                    random_seed=args.seed
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
            targets = extract_targets_from_dataset(train_dataset)
            if targets is None:
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
            selected_indices = strategy.greedy_k_center(
                labeled_embeddings,
                train_embeddings,
                target_samples,
                random_seed=args.seed,
            )
            
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
    try:
        base_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        subset_indices = getattr(train_dataset, 'indices', list(range(len(train_dataset))))
        base_targets = None
        if hasattr(train_dataset, 'targets') and isinstance(train_dataset.targets, (list, np.ndarray)):
            base_targets = np.array(train_dataset.targets)
        elif hasattr(base_dataset, 'targets'):
            base_targets = np.array([base_dataset.targets[i] for i in subset_indices])

        if base_targets is not None and len(base_targets) == len(train_dataset):
            n_classes = None
            class_names = None
            if hasattr(base_dataset, 'classes'):
                class_names = list(base_dataset.classes)
                n_classes = len(class_names)
            if n_classes is None:
                n_classes = int(base_targets.max()) + 1 if base_targets.size else 0

            selected_targets = base_targets[selected_indices]
            print("\nClass distribution in selected samples:")
            for cls in range(n_classes):
                count = int(np.sum(selected_targets == cls))
                percentage = (count / len(selected_targets) * 100.0) if len(selected_targets) > 0 else 0.0
                label = class_names[cls] if class_names and cls < len(class_names) else cls
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")

            random.seed(args.seed + 1)
            random_indices = random.sample(range(len(train_dataset)), target_samples)
            random_targets = base_targets[random_indices]
            print("\nClass distribution in random baseline:")
            for cls in range(n_classes):
                count = int(np.sum(random_targets == cls))
                percentage = (count / len(random_targets) * 100.0) if len(random_targets) > 0 else 0.0
                label = class_names[cls] if class_names and cls < len(class_names) else cls
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")

    except Exception as e:
        if verbose:
            print(f"Could not analyze class distribution: {e}")
    
    if verbose:
        print(f"Active Learning selection completed. Selected {len(selected_indices)} samples using '{al_strategy}' strategy.")
    
    return selected_indices
