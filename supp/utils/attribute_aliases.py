from typing import Dict, Optional


COCO_ATTRIBUTE_VALUE_ALIASES = {
    "weather": {
        "rain": "rainy",
    },
}

COCO_POSE_GENERATED_ATTRIBUTE_ALIASES = {
    "rain": "rainy",
    "brown_rain": "brown_rainy",
    "brown-rain": "brown_rainy",
    "brown-rainy": "brown_rainy",
}

COCO_POSE_GENERATED_ATTRIBUTE_CANONICAL = {
    "brown_rainy",
    "rainy",
}


def canonicalize_attribute_value(
    attribute_name: str,
    value: str,
    *,
    dataset_name: Optional[str] = None,
) -> str:
    """Map dataset-specific attribute aliases onto canonical metadata values."""
    if value is None:
        return value

    normalized_dataset = (dataset_name or "").strip().lower()
    normalized_attr = str(attribute_name).strip()
    normalized_value = str(value).strip()

    if normalized_dataset not in {"coco", "coco_pose"}:
        return normalized_value

    alias_map = COCO_ATTRIBUTE_VALUE_ALIASES.get(normalized_attr, {})
    return alias_map.get(normalized_value.lower(), normalized_value)


def canonicalize_attribute_counts(
    counts: Dict[str, Dict[str, float]],
    *,
    dataset_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Normalize nested attribute-count maps onto canonical attribute values."""
    canonicalized: Dict[str, Dict[str, float]] = {}
    for attr_name, value_map in counts.items():
        attr_counts = canonicalized.setdefault(attr_name, {})
        for value, count in value_map.items():
            canonical_value = canonicalize_attribute_value(
                attr_name,
                value,
                dataset_name=dataset_name,
            )
            attr_counts[canonical_value] = attr_counts.get(canonical_value, 0.0) + float(count)
    return canonicalized


def canonicalize_coco_pose_generated_attribute(value: Optional[str]) -> Optional[str]:
    """Normalize COCO pose generated-attribute aliases used by the CLI."""
    if value is None:
        return value
    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return "none"
    return COCO_POSE_GENERATED_ATTRIBUTE_ALIASES.get(normalized, normalized)


def coco_pose_generated_attribute_choices():
    return sorted(COCO_POSE_GENERATED_ATTRIBUTE_CANONICAL | set(COCO_POSE_GENERATED_ATTRIBUTE_ALIASES))
