from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SAVE_ROOT = REPO_ROOT / "save"


def _strip_save_prefix(path_like):
    path = Path(path_like)
    parts = path.parts
    if not parts:
        return None
    if parts[0] == "save":
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    if len(parts) >= 2 and parts[0] == "outputs" and parts[1] == "save":
        return Path(*parts[2:]) if len(parts) > 2 else Path()
    return None


def canonical_save_path(relative_path="", create_parent=False):
    target = PRIMARY_SAVE_ROOT / Path(relative_path)
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def artifact_candidates(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return [path]

    save_relative = _strip_save_prefix(path_like)
    if save_relative is not None:
        return [PRIMARY_SAVE_ROOT / save_relative]

    return [REPO_ROOT / path]


def resolve_artifact_path(path_like, create_parent=False):
    candidates = artifact_candidates(path_like)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    target = candidates[0]
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def resolve_existing_artifact(*path_likes):
    checked = []
    for path_like in path_likes:
        resolved = resolve_artifact_path(path_like)
        checked.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError("Could not find any of: " + ", ".join(checked))


def dataset_default_attribute_path(dataset_name):
    if dataset_name == "imagenet":
        return resolve_artifact_path("save/attribute10.json")
    if dataset_name == "celeba":
        return resolve_artifact_path("save/attribute.json")
    return resolve_artifact_path("save/attribute.json")
