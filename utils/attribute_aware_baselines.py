"""Simple attribute-aware source-selection baselines for rebuttal experiments.

The baselines in this module use the same source attribute records, generated
attribute support, class/proxy labels, and source budget as AttriSelect. They
select source examples only; the generated repair set remains fixed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TARGET_SLICE_STRATIFIED = "target_slice_stratified"
ATTRIBUTE_FILTERED_RANDOM = "attribute_filtered_random"
ATTRIBUTE_FILTERED_ENTROPY = "attribute_filtered_entropy"
GENERATED_COMPLEMENT = "generated_complement"

ATTRIBUTE_AWARE_BASELINES = (
    TARGET_SLICE_STRATIFIED,
    ATTRIBUTE_FILTERED_RANDOM,
    ATTRIBUTE_FILTERED_ENTROPY,
    GENERATED_COMPLEMENT,
)


def normalize_attribute_aware_strategy(strategy: str) -> str:
    """Return the canonical name for a rebuttal attribute-aware baseline."""
    normalized = str(strategy or "").strip().lower().replace("-", "_")
    aliases = {
        "exact_slice_stratified": TARGET_SLICE_STRATIFIED,
        "target_slice_plus_stratified": TARGET_SLICE_STRATIFIED,
        "attribute_filtered_random": ATTRIBUTE_FILTERED_RANDOM,
        "filtered_random": ATTRIBUTE_FILTERED_RANDOM,
        "attribute_filtered_entropy": ATTRIBUTE_FILTERED_ENTROPY,
        "filtered_entropy": ATTRIBUTE_FILTERED_ENTROPY,
        "generated_complement": GENERATED_COMPLEMENT,
        "generated_complement_class_balance": GENERATED_COMPLEMENT,
    }
    canonical = aliases.get(normalized, normalized)
    if canonical not in ATTRIBUTE_AWARE_BASELINES:
        raise ValueError(
            f"Unsupported attribute-aware baseline '{strategy}'. "
            f"Expected one of {ATTRIBUTE_AWARE_BASELINES}."
        )
    return canonical


def target_values_from_generated_counts(
    generated_attribute_counts: Mapping[str, Mapping[Any, float]],
) -> Dict[str, Tuple[Any, ...]]:
    """Extract generated-supported target values from positive attribute counts."""
    target_values: Dict[str, Tuple[Any, ...]] = {}
    for attribute, value_counts in generated_attribute_counts.items():
        positive_values = [
            value
            for value, count in value_counts.items()
            if float(count) > 0.0
        ]
        if positive_values:
            target_values[str(attribute)] = tuple(positive_values)
    if not target_values:
        raise ValueError(
            "Generated attribute counts do not contain any positive target values."
        )
    return target_values


def _normalized_value(value: Any) -> str:
    return str(value).strip().casefold()


def compute_generated_complement_scores(
    *,
    attribute_metadata: Mapping[str, Mapping[str, Any]],
    metadata_indices: Sequence[Optional[int]],
    generated_attribute_counts: Mapping[str, Mapping[Any, float]],
) -> List[float]:
    """Score source samples by attributes underrepresented in generated data.

    Each valid repair-relevant field contributes ``1 / (1 + f_gen(value))``.
    Contributions are averaged across valid fields so missing metadata does not
    change the scale merely by reducing the number of available fields.
    """
    arrays = _metadata_arrays(attribute_metadata, generated_attribute_counts)
    normalized_counts = {
        str(attribute): {
            _normalized_value(value): max(float(count), 0.0)
            for value, count in value_counts.items()
        }
        for attribute, value_counts in generated_attribute_counts.items()
    }

    scores: List[float] = []
    for metadata_index in metadata_indices:
        contributions: List[float] = []
        for attribute, value_counts in normalized_counts.items():
            data = arrays[attribute]
            if (
                metadata_index is None
                or metadata_index < 0
                or metadata_index >= len(data)
            ):
                continue
            value = _normalized_value(data[metadata_index])
            contributions.append(1.0 / (1.0 + value_counts.get(value, 0.0)))
        scores.append(
            sum(contributions) / len(contributions) if contributions else 0.0
        )
    return scores


def _metadata_arrays(
    attribute_metadata: Mapping[str, Mapping[str, Any]],
    target_values: Mapping[str, Sequence[Any]],
) -> Dict[str, Sequence[Any]]:
    arrays: Dict[str, Sequence[Any]] = {}
    missing_attributes = []
    for attribute in target_values:
        info = attribute_metadata.get(attribute)
        if not isinstance(info, Mapping) or not isinstance(info.get("data"), Sequence):
            missing_attributes.append(attribute)
            continue
        arrays[attribute] = info["data"]
    if missing_attributes:
        raise ValueError(
            "Attribute metadata is missing target data arrays for: "
            + ", ".join(sorted(missing_attributes))
        )
    return arrays


def build_attribute_match_pools(
    *,
    attribute_metadata: Mapping[str, Mapping[str, Any]],
    metadata_indices: Sequence[Optional[int]],
    target_values: Mapping[str, Sequence[Any]],
) -> Tuple[List[int], List[int], List[int]]:
    """Build exact-conjunction, partial-match, and background candidate pools.

    The exact pool matches at least one generated-supported value for every
    target field. The partial pool matches at least one target field. Missing
    metadata never counts as a match.
    """
    arrays = _metadata_arrays(attribute_metadata, target_values)
    normalized_targets = {
        attribute: {_normalized_value(value) for value in values}
        for attribute, values in target_values.items()
    }

    exact_pool: List[int] = []
    partial_pool: List[int] = []
    background_pool: List[int] = []

    for local_index, metadata_index in enumerate(metadata_indices):
        matches: List[bool] = []
        for attribute, values in normalized_targets.items():
            data = arrays[attribute]
            if (
                metadata_index is None
                or metadata_index < 0
                or metadata_index >= len(data)
            ):
                matches.append(False)
                continue
            matches.append(_normalized_value(data[metadata_index]) in values)

        if matches and all(matches):
            exact_pool.append(local_index)
            partial_pool.append(local_index)
        elif any(matches):
            partial_pool.append(local_index)
        else:
            background_pool.append(local_index)

    return exact_pool, partial_pool, background_pool


def _json_label(label: Any) -> str:
    return "__missing__" if label is None else str(label)


def _balanced_select(
    candidate_indices: Iterable[int],
    *,
    n_select: int,
    class_labels: Sequence[Any],
    rng: random.Random,
    initial_class_counts: Optional[Counter] = None,
    scores: Optional[Sequence[float]] = None,
) -> List[int]:
    """Select candidates with deterministic hard class/proxy-label balancing."""
    candidate_list = list(dict.fromkeys(int(index) for index in candidate_indices))
    if n_select <= 0 or not candidate_list:
        return []
    n_select = min(int(n_select), len(candidate_list))

    grouped: Dict[Any, List[int]] = defaultdict(list)
    for index in candidate_list:
        label = class_labels[index] if index < len(class_labels) else None
        grouped[label].append(index)

    for queue in grouped.values():
        if scores is None:
            rng.shuffle(queue)
        else:
            tie_break = {index: rng.random() for index in queue}
            queue.sort(
                key=lambda index: (
                    -float(scores[index]),
                    tie_break[index],
                    index,
                )
            )

    class_counts = Counter(initial_class_counts or {})
    selected: List[int] = []
    while len(selected) < n_select:
        active_labels = [label for label, queue in grouped.items() if queue]
        if not active_labels:
            break
        selected_label = min(
            active_labels,
            key=lambda label: (class_counts[label], _json_label(label)),
        )
        selected.append(grouped[selected_label].pop(0))
        class_counts[selected_label] += 1
    return selected


def _balanced_priority_select(
    *,
    priority_indices: Iterable[int],
    complement_indices: Iterable[int],
    n_select: int,
    class_labels: Sequence[Any],
    rng: random.Random,
    priority_scores: Optional[Sequence[float]] = None,
    complement_scores: Optional[Sequence[float]] = None,
) -> List[int]:
    """Balance classes while preferring the attribute-filtered queue per class."""

    def build_grouped_queues(indices, scores):
        grouped: Dict[Any, List[int]] = defaultdict(list)
        for index in dict.fromkeys(int(value) for value in indices):
            label = class_labels[index] if index < len(class_labels) else None
            grouped[label].append(index)
        for queue in grouped.values():
            if scores is None:
                rng.shuffle(queue)
            else:
                tie_break = {index: rng.random() for index in queue}
                queue.sort(
                    key=lambda index: (
                        -float(scores[index]),
                        tie_break[index],
                        index,
                    )
                )
        return grouped

    priority_queues = build_grouped_queues(priority_indices, priority_scores)
    complement_queues = build_grouped_queues(complement_indices, complement_scores)
    class_counts = Counter()
    selected: List[int] = []

    while len(selected) < n_select:
        active_labels = {
            label
            for label, queue in priority_queues.items()
            if queue
        } | {
            label
            for label, queue in complement_queues.items()
            if queue
        }
        if not active_labels:
            break
        selected_label = min(
            active_labels,
            key=lambda label: (class_counts[label], _json_label(label)),
        )
        if priority_queues.get(selected_label):
            selected_index = priority_queues[selected_label].pop(0)
        else:
            selected_index = complement_queues[selected_label].pop(0)
        selected.append(selected_index)
        class_counts[selected_label] += 1

    return selected


def _selected_class_counts(
    selected_indices: Sequence[int],
    class_labels: Sequence[Any],
) -> Counter:
    return Counter(
        class_labels[index] if index < len(class_labels) else None
        for index in selected_indices
    )


def select_attribute_aware_baseline(
    *,
    strategy: str,
    attribute_metadata: Mapping[str, Mapping[str, Any]],
    metadata_indices: Sequence[Optional[int]],
    class_labels: Optional[Sequence[Any]],
    generated_attribute_counts: Mapping[str, Mapping[Any, float]],
    n_select: int,
    seed: int,
    entropy_scores: Optional[Sequence[float]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select source-local indices for one same-information baseline."""
    canonical_strategy = normalize_attribute_aware_strategy(strategy)
    total_candidates = len(metadata_indices)
    if n_select < 0:
        raise ValueError(f"n_select must be non-negative, got {n_select}")
    n_select = min(int(n_select), total_candidates)

    if class_labels is None:
        resolved_labels: Sequence[Any] = [None] * total_candidates
    else:
        if len(class_labels) != total_candidates:
            raise ValueError(
                f"class_labels has length {len(class_labels)}, expected {total_candidates}."
            )
        resolved_labels = class_labels

    if canonical_strategy == ATTRIBUTE_FILTERED_ENTROPY:
        if entropy_scores is None:
            raise ValueError("Attribute-Filtered Entropy requires candidate entropy scores.")
        if len(entropy_scores) != total_candidates:
            raise ValueError(
                f"entropy_scores has length {len(entropy_scores)}, expected {total_candidates}."
            )

    complement_scores = None
    if canonical_strategy == GENERATED_COMPLEMENT:
        complement_scores = compute_generated_complement_scores(
            attribute_metadata=attribute_metadata,
            metadata_indices=metadata_indices,
            generated_attribute_counts=generated_attribute_counts,
        )

    target_values = target_values_from_generated_counts(generated_attribute_counts)
    exact_pool, partial_pool, background_pool = build_attribute_match_pools(
        attribute_metadata=attribute_metadata,
        metadata_indices=metadata_indices,
        target_values=target_values,
    )

    rng = random.Random(int(seed))
    if canonical_strategy == TARGET_SLICE_STRATIFIED:
        priority_pool = exact_pool
        priority_scores = None
    elif canonical_strategy == ATTRIBUTE_FILTERED_RANDOM:
        priority_pool = partial_pool
        priority_scores = None
    elif canonical_strategy == GENERATED_COMPLEMENT:
        priority_pool = range(total_candidates)
        priority_scores = complement_scores
    else:
        priority_pool = partial_pool
        priority_scores = entropy_scores

    priority_set = set(priority_pool)
    complement = [
        index for index in range(total_candidates) if index not in priority_set
    ]
    fill_scores = entropy_scores if canonical_strategy == ATTRIBUTE_FILTERED_ENTROPY else None
    selected = _balanced_priority_select(
        priority_indices=priority_pool,
        complement_indices=complement,
        n_select=n_select,
        class_labels=resolved_labels,
        rng=rng,
        priority_scores=priority_scores,
        complement_scores=fill_scores,
    )

    if len(selected) != n_select or len(set(selected)) != len(selected):
        raise RuntimeError(
            f"{canonical_strategy} selected {len(selected)} unique={len(set(selected))}; "
            f"expected {n_select}."
        )

    exact_set = set(exact_pool)
    partial_set = set(partial_pool)
    selected_exact = sum(index in exact_set for index in selected)
    selected_partial = sum(index in partial_set for index in selected)
    selected_counts = _selected_class_counts(selected, resolved_labels)
    selected_complement_scores = (
        [float(complement_scores[index]) for index in selected]
        if complement_scores is not None
        else []
    )
    class_count_values = list(selected_counts.values())
    diagnostics: Dict[str, Any] = {
        "strategy": canonical_strategy,
        "seed": int(seed),
        "candidate_count": total_candidates,
        "requested_count": n_select,
        "selected_count": len(selected),
        "target_values": {
            attribute: [str(value) for value in values]
            for attribute, values in target_values.items()
        },
        "generated_attribute_counts": {
            str(attribute): {
                str(value): float(count) for value, count in value_counts.items()
            }
            for attribute, value_counts in generated_attribute_counts.items()
        },
        "exact_pool_count": len(exact_pool),
        "partial_pool_count": len(partial_pool),
        "background_pool_count": len(background_pool),
        "selected_exact_count": selected_exact,
        "selected_partial_count": selected_partial,
        "selected_exact_fraction": selected_exact / len(selected) if selected else 0.0,
        "selected_partial_fraction": selected_partial / len(selected) if selected else 0.0,
        "selected_class_counts": {
            _json_label(label): int(count)
            for label, count in sorted(
                selected_counts.items(), key=lambda item: _json_label(item[0])
            )
        },
        "class_count_imbalance": (
            max(class_count_values) - min(class_count_values)
            if class_count_values
            else 0
        ),
        "selected_mean_complement_score": (
            sum(selected_complement_scores) / len(selected_complement_scores)
            if selected_complement_scores
            else None
        ),
        "selected_min_complement_score": (
            min(selected_complement_scores) if selected_complement_scores else None
        ),
        "selected_max_complement_score": (
            max(selected_complement_scores) if selected_complement_scores else None
        ),
    }
    return selected, diagnostics
