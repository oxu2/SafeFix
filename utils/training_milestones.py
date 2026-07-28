"""Helpers for validating cumulative training-evaluation milestones."""


def parse_evaluation_milestones(value, *, num_epochs):
    """Parse a comma-separated milestone list into sorted unique epoch numbers."""
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = [item.strip() for item in str(value).split(",")]

    milestones = []
    for raw_value in raw_values:
        if raw_value == "" or raw_value is None:
            continue
        try:
            milestone = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Evaluation milestones must be integers, got: {raw_value!r}."
            ) from exc
        if milestone <= 0:
            raise ValueError(
                f"Evaluation milestones must be positive, got: {milestone}."
            )
        if milestone > int(num_epochs):
            raise ValueError(
                f"Evaluation milestone {milestone} exceeds --num_epochs={num_epochs}."
            )
        milestones.append(milestone)

    return sorted(set(milestones))
