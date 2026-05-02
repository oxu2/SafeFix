# Import all common dependencies from utils
from utils.imports import *
from utils.defines import *

# Modular imports
from utils.active_learning import CoreSetStrategy, extract_embeddings_for_al, active_learning_selection
from utils.models import CLIPForBinary, SmoothBCEWithLogitsLoss
from utils.path_utils import (
    LEGACY_SAVE_ROOT,
    canonical_save_path,
    dataset_default_attribute_path,
    resolve_artifact_path,
    resolve_existing_artifact,
)
from utils.data_utils import (default_img_transform, clip_processor_transform, make_loader, 
                             CelebADataset, GeneratedImageDataset, SimpleImageDataset, GeneratedDataset)
from utils.training import evaluate, train_model, save_predictions, parse_log


_llm_data_selection = None
_LLM_SELECTION_IMPORT_ERROR = None


def _require_llm_data_selection():
    """Import LLM selection lazily so non-LLM runs do not depend on LVLM packages."""
    global _llm_data_selection, _LLM_SELECTION_IMPORT_ERROR

    if _llm_data_selection is None:
        try:
            from utils.llm_selection import llm_data_selection as imported_llm_data_selection
        except Exception as exc:
            _LLM_SELECTION_IMPORT_ERROR = exc
        else:
            _llm_data_selection = imported_llm_data_selection
            _LLM_SELECTION_IMPORT_ERROR = None

    if _llm_data_selection is None:
        raise ImportError(
            "LLM-based selection requires optional LVLM dependencies. "
            "Install a compatible transformers/LVLM stack or run with --use_llm_selection False."
        ) from _LLM_SELECTION_IMPORT_ERROR

    return _llm_data_selection


def _resolve_existing_path(*candidates):
    """Return the first existing path from a small ordered candidate list."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of the required files: "
        + ", ".join(str(Path(candidate)) for candidate in candidates)
    )


def _first_existing_dir(*candidates):
    """Return the first existing directory from an ordered candidate list."""
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    return None


def _juno_scratch_data_dir(*parts):
    return Path("/home/oxx220000/scratch/safefix_cache/data").joinpath(*parts)


def _default_generated_dir(dataset, attributes):
    """Resolve a generated-image directory, preferring recent Juno scratch copies."""
    if dataset == "imagenet":
        candidates = [
            Path(f"data/{attributes}_instruct_pix2pix_seed42_1k"),
            _juno_scratch_data_dir(f"imagenet_{attributes}_seed42_100perclass_1k"),
            _juno_scratch_data_dir(f"{attributes}_instruct_pix2pix_seed42_1k"),
            _juno_scratch_data_dir(f"{attributes}_instruct_pix2pix_seed42_100perclass_1k"),
            _juno_scratch_data_dir(f"imagenet_{attributes}"),
            Path(f"data/imagenet_{attributes}"),
            Path(f"/data/oxx220000/imagenet_{attributes}"),
        ]
    else:
        candidates = [
            Path(f"data/hibug_{attributes}_seed42_1k"),
            Path(f"data/{attributes}_instruct_pix2pix_seed42_1k"),
            _juno_scratch_data_dir(f"hibug_{attributes}_seed42_1k"),
            _juno_scratch_data_dir(f"{attributes}_instruct_pix2pix_seed42_1k"),
            Path(f"exampleData/{attributes}"),
            Path(f"/data/oxx220000/{attributes}"),
        ]
    return _first_existing_dir(*candidates)


def _extract_celeba_base_id(generated_stem, attributes):
    """Extract the original CelebA image id from Safefix or HiBUG generated names."""
    if re.match(r"^\d+", generated_stem):
        return generated_stem.split("_", 1)[0]

    attr_prefix = f"{attributes}_"
    if generated_stem.startswith(attr_prefix):
        suffix = generated_stem[len(attr_prefix):].split("_")[-1]
        if suffix.isdigit():
            return suffix

    trailing_digits = re.search(r"(\d+)$", generated_stem)
    if trailing_digits:
        return trailing_digits.group(1)

    return None


def _files_match(path_a, path_b, chunk_size=1024 * 1024):
    """Return True when two files have identical contents."""
    path_a = Path(path_a)
    path_b = Path(path_b)

    if not path_a.exists() or not path_b.exists():
        return False
    if path_a.stat().st_size != path_b.stat().st_size:
        return False

    with path_a.open("rb") as fa, path_b.open("rb") as fb:
        while True:
            chunk_a = fa.read(chunk_size)
            chunk_b = fb.read(chunk_size)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def _find_resume_log_for_checkpoint(checkpoint_path):
    """Find the training log that originally saved the checkpoint, if available."""
    checkpoint_path = Path(checkpoint_path).resolve()
    candidate_model_paths = [checkpoint_path]

    if checkpoint_path.name == "model_celeba_resnet_base.pth":
        for candidate in sorted(Path("save/models").glob("model_celeba_resnet_*_0imgs_*.pth")):
            if _files_match(checkpoint_path, candidate):
                candidate_model_paths.append(candidate.resolve())

    candidate_log_paths = list(Path("logs").rglob("*.log"))
    for candidate_model_path in candidate_model_paths:
        model_rel = candidate_model_path.relative_to(Path.cwd()) if candidate_model_path.is_relative_to(Path.cwd()) else None
        model_markers = {str(candidate_model_path)}
        if model_rel is not None:
            model_markers.add(str(model_rel))

        for log_path in candidate_log_paths:
            try:
                with log_path.open("r") as handle:
                    for line in handle:
                        if "Model saved to" not in line:
                            continue
                        if any(marker in line for marker in model_markers):
                            return log_path
            except OSError:
                continue

    return None


def _lookup_resume_test_acc(checkpoint_path):
    """Extract the saved run's overall test accuracy from its training log, if available."""
    resume_log = _find_resume_log_for_checkpoint(checkpoint_path)
    if resume_log is None:
        return None, None

    pattern_overall = re.compile(
        r'(?:Model Test ACC|Held-out test accuracy|Model Validation ACC):\s*(?P<acc>\d+\.\d+)'
    )
    overall_acc = None
    try:
        with resume_log.open("r") as handle:
            for line in handle:
                match = pattern_overall.search(line)
                if match:
                    overall_acc = float(match.group("acc"))
                    break
    except OSError:
        return None, None

    return overall_acc, resume_log


def _resolve_cuda_device_index(requested_gpu):
    """Map a requested GPU id to the process-local CUDA index."""
    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None

    try:
        requested_idx = int(requested_gpu)
    except (TypeError, ValueError):
        requested_idx = 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if visible_tokens:
        requested_str = str(requested_idx)
        if requested_str in visible_tokens:
            mapped_idx = visible_tokens.index(requested_str)
            if mapped_idx < device_count:
                return mapped_idx
        if device_count == 1:
            return 0

    if 0 <= requested_idx < device_count:
        return requested_idx

    return 0


def _resolve_runtime_device(requested_gpu):
    """Return the correct torch.device for the current process visibility mask."""
    resolved_idx = _resolve_cuda_device_index(requested_gpu)
    if resolved_idx is None:
        return torch.device("cpu")
    return torch.device(f"cuda:{resolved_idx}")


def _print_class_distribution(subset, description, verbose=False):
    """Log per-class counts for a torch.utils.data.Subset built on ImageFolder."""
    from torch.utils.data import Subset as TorchSubset

    try:
        if not isinstance(subset, TorchSubset):
            return False

        base_dataset = subset.dataset
        indices = subset.indices
        if base_dataset is None or indices is None:
            return False

        base_targets = None
        if hasattr(base_dataset, "targets"):
            base_targets = base_dataset.targets
        elif hasattr(base_dataset, "samples"):
            base_targets = [sample[1] for sample in base_dataset.samples]

        if base_targets is None:
            return False

        labels = np.array([base_targets[i] for i in indices])
        if labels.size == 0:
            return False

        class_names = getattr(base_dataset, "classes", None)
        n_classes = len(class_names) if class_names else int(labels.max()) + 1

        print(f"\n[{description}] class distribution:")
        for cls in range(n_classes):
            count = int(np.sum(labels == cls))
            percentage = (count / len(labels) * 100.0) if len(labels) > 0 else 0.0
            label = class_names[cls] if class_names and cls < len(class_names) else cls
            print(f"  Class {label}: {count} samples ({percentage:.1f}%)")

        return True
    except Exception as exc:
        if verbose:
            print(f"Warning: could not compute class distribution for {description}: {exc}")
        return False

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model on CelebA dataset')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='Enable verbose output (default: False)')
parser.add_argument('--feasible_file', type=str, default=None,
                    help='(Optional) Path to the feasible images JSON file. Defaults to save/feasible_<attributes>.json')
# parser.add_argument('--olabel', type=str2bool, default=False,
#                     help='Use original dataset labels (default: False); if False, uses labels from feasible_file')
parser.add_argument('--gen_image_dir', type=str, default=None,
                    help='(Optional) Directory containing generated images. Defaults to exampleData/<attributes>')
parser.add_argument('--data_dir', type=str, default=None,
                    help='(Optional) Source dataset directory. Defaults to Juno scratch copies when available, then local paths.')
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Use pre-trained weights (default: False)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--attributes', type=str, default="redhair_brownskin", 
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, redhair, brownskin, yellowhair, sademotion, etc.')
parser.add_argument('--lvlm', type=str, default='qwen',
                    help='LVLM model name (e.g., llava) to modify feasible file naming')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, or clip)')
parser.add_argument('--plot', type=str2bool, default=False, help='Enable plotting training metrics')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--num_added_img', type=int, default=1000,
                    help='Number of generated images to add (default: 4000)')
# Add argument for attribute file path
parser.add_argument(
    '--attribute_file',
    type=str,
    default=None,
    help='Path to the attribute JSON file (default: auto; ImageNet -> save/attribute10.json, CelebA -> save/attribute.json)',
)
parser.add_argument('--gpu', type=int, default=0,
                    help='Index of the CUDA GPU to use (default: 0)')
parser.add_argument('--save_report', type=str2bool, default=False,
                    help='Generate and save diagnostic report (default: False)')
parser.add_argument('--save_preds', type=str2bool, default=False,
                    help='Save model predictions after training (default: False)')
parser.add_argument('--save_model', type=str2bool, default=False,
                    help='Save model checkpoint (.pth) to save/models/ after training (default: False)')
parser.add_argument('--resume_model', type=str, default=None,
                    help='Path to a saved .pth checkpoint to resume/fine-tune from (default: None, train from scratch)')
parser.add_argument('--freeze_backbone', type=str2bool, default=False,
                    help='Freeze early backbone layers when fine-tuning from checkpoint (default: False)')
parser.add_argument(
    '--save_pred_model',
    type=str,
    choices=['best', 'last'],
    default='best',
    help='Which model to use for saving predictions: best (default) or last'
)
parser.add_argument(
    '--gen_only',
    type=str,
    default='false',
    choices=['false', 'train_only', 'train_val_test'],
    help='Generated-image usage: false (off), train_only (use generated for train/val), train_val_test (8:1:1 split for train/val/test)'
)
parser.add_argument('--skip_train_unlabeled', type=str2bool, default=False,
                    help='Skip processing TRAIN and UNLABELED splits in diagnostics')
parser.add_argument('--no_controlnet', type=str2bool, default=False,
                    help='Ablation flag: disable ControlNet-based augmentation (default: False)')
parser.add_argument('--skip_feasible', type=str2bool, default=False,
                    help='Skip loading feasible JSON for any attributes (default: False)')
parser.add_argument(
    '--simple', type=str2bool, default=False,
    help='Simple mode: disable ControlNet and attribute editing (original safefix behavior) (default: True).'
)
parser.add_argument(
    '--dataset',
    choices=['imagenet', 'celeba'],
    default='celeba',
    help='Which dataset to use: imagenet (default) or celeba')
# Add batch_size argument here
# Add batch_size argument here
parser.add_argument('--batch_size', type=int, default=64,
                    help='Mini‑batch size for training and validation (default: 64)')
# Add subsampling ratio argument for original training data when combining with modified images
# Add subsampling ratio argument for original training data when combining with modified images
parser.add_argument('--sub_ratio', type=float, 
                    # default=0.3,
                    default=1.0,
                    help='Subsampling ratio (0-1) for original training data when combining with modified images')
# Add pad_missing argument for subsampled original predictions
parser.add_argument('--pad_missing', action='store_true', default=True,
                    help='Pad missing original training image predictions with NaN when subsampling (enabled by default)')
# Add workers argument here
parser.add_argument('--workers', type=int, default=4,
                    help='Number of DataLoader workers (default: 4)')
# Add random seed argument for reproducibility
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
# Separate learning rates for CLIP head and backbone (for scratch mode)
parser.add_argument('--lr_head', type=float, default=None,
                    help='Learning rate for CLIP classification head (scratch mode)')
parser.add_argument('--lr_backbone', type=float, default=None,
                    help='Learning rate for CLIP backbone (scratch mode)')
# Active Learning arguments
parser.add_argument('--use_active_learning', type=str2bool, default=True,
                    help='Use Active Learning strategy for data selection (default: True)')
parser.add_argument('--al_strategy', type=str, default='attribute',
                    choices=['coreset', 'badge', 'stratified', 'diversity', 'hybrid', 'hierarchical',
                             'attribute', 'uncertainty_stratified', 'gen_probe',
                             'random', 'none', '0', 'false', 'no'],
                    help='Active Learning strategy: coreset, badge, stratified, diversity, hybrid, hierarchical, '
                         'attribute, uncertainty_stratified, gen_probe, or random/none/0/false (all map to plain random selection)')
parser.add_argument('--al_probe_model', type=str, default=None,
                    help='Path to the model trained ONLY on generated images, used for the gen_probe AL strategy.')
parser.add_argument('--al_embedding_dim', type=int, default=512,
                    help='Embedding dimension for Active Learning (default: 512)')
parser.add_argument('--al_initial_labeled', type=int, default=400,
                    help='Initial number of labeled samples for Active Learning (default: 400)')
# Hybrid strategy parameters
parser.add_argument('--al_coreset_ratio', type=float, default=0.4,
                    help='Ratio of samples selected by core-set in hybrid strategy (default: 0.4)')
parser.add_argument('--al_stratified_ratio', type=float, default=0.4,
                    help='Ratio of samples selected by stratified sampling in hybrid strategy (default: 0.4)')
# Diversity strategy parameters  
parser.add_argument('--al_clusters', type=int, default=None,
                    help='Number of clusters for diversity strategy (default: n_select//10)')
# Hierarchical strategy parameters
parser.add_argument('--al_w_err', type=float, default=0.6,
                    help='Weight for error rate in hierarchical difficulty calculation (default: 0.6)')
parser.add_argument('--al_w_unc', type=float, default=0.4,
                    help='Weight for uncertainty in hierarchical difficulty calculation (default: 0.4)')
# Attribute-driven strategy parameters
parser.add_argument('--al_attribute_focus', type=str, default=None,
                    help='Comma-separated attribute keys to prioritize (defaults to all attributes in metadata)')
parser.add_argument('--al_attribute_rarity_power', type=float, default=0.01,
                    help='Exponent controlling emphasis on rare attribute values (default: 0.01)')
parser.add_argument('--al_attribute_use_rarity', type=str2bool, default=True,
                    help='Attribute strategy: include the rarity score term. Set False to remove R from RPC ablations.')
parser.add_argument('--al_attribute_uncertainty_boost', type=float, default=1.0,
                    help='Score bonus for samples flagged as uncertain in attribute metadata (default: 1.0)')
parser.add_argument('--al_attribute_use_entropy', type=str2bool, default=False,
                    help='Attribute strategy: compute and use dynamic Information Entropy for the U term')
parser.add_argument('--al_attribute_log_path', type=str, default=None,
                    help='Optional path to attribute diagnostics log for Rare Case accuracies')
parser.add_argument('--al_attribute_log_acc_scale', type=float, default=0.0,
                    help='Scaling factor applied to log-derived failure weights (default: 0.0)')
parser.add_argument('--al_attribute_extra_counts', type=str, default=None,
                    help='Extra counts per attribute value to reflect generated data (attr:value=count entries or JSON path)')
parser.add_argument('--al_attribute_extra_count_scale', type=float, default=1.0,
                    help='Multiplier applied to parsed/inferred extra counts before scoring (default: 1.0)')
parser.add_argument('--al_attribute_frequency_penalty', type=float, default=5.0,
                    help='Penalty strength applied to over-represented attribute values (default: 5.0)')
parser.add_argument('--al_attribute_class_balance', type=str, default="inf",
                    help='Class-level stratified weight; pass "inf" to enforce hard per-class quotas (default: 0)')
parser.add_argument('--al_alpha', type=float, default=0.4,
                    help='Weight for difficulty in hierarchical priority score (default: 0.4)')
parser.add_argument('--al_beta', type=float, default=0.4,
                    help='Weight for coverage gap in hierarchical priority score (default: 0.4)')
parser.add_argument('--al_gamma', type=float, default=0.2,
                    help='Weight for correlation penalty in hierarchical priority score (default: 0.2)')
parser.add_argument('--al_lambda_div', type=float, default=0.5,
                    help='Weight for diversity in hierarchical instance selection (default: 0.5)')
parser.add_argument('--al_round1_ratio', type=float, default=0.7,
                    help='Fraction of budget allocated to Round 1 in hierarchical AL (default: 0.7)')
parser.add_argument('--al_min_slice_quota', type=int, default=5,
                    help='Minimum samples per slice in hierarchical AL (default: 5)')
parser.add_argument('--al_max_slice_quota', type=int, default=100,
                    help='Maximum samples per slice in hierarchical AL (default: 100)')
parser.add_argument('--al_conditional_rarity', type=str2bool, default=True,
                    help='Attribute strategy: use class-conditional rarity (P(val|class)) instead of global rarity')
# Combined Active Learning arguments
parser.add_argument('--al_combined', type=str2bool, default=False,
                    help='Apply Active Learning on combined dataset (generated + original) instead of just original (default: False)')
parser.add_argument('--combined_ratio', type=float, default=0.7,
                    help='Ratio of samples to select from combined dataset when using --al_combined (default: 0.7)')
# LLM-based selection arguments (alternative to traditional Active Learning)
parser.add_argument('--use_llm_selection', type=str2bool, default=False,
                    help='Use LLM-based data selection instead of traditional Active Learning (default: False)')
parser.add_argument('--llm_model', type=str, default='qwen', 
                    choices=['qwen', 'llava-7b', 'llava-13b'],
                    help='LLM model for data selection: qwen (default), llava-7b, or llava-13b')
parser.add_argument('--llm_strategy', type=str, default='diversity_difficulty',
                    choices=['diversity', 'difficulty', 'diversity_difficulty', 'llm_random'],
                    help='LLM selection strategy: diversity, difficulty, diversity_difficulty (default), or llm_random')
parser.add_argument('--llm_max_memory', type=int, default=23,
                    help='Maximum memory per GPU in GB for LLM model (default: 23)')
# Parse arguments
args = parser.parse_args()

# Normalize al_strategy: treat none/0/false/no/random as 'random' (plain random selection)
_strategy_str = str(args.al_strategy).lower().strip() if args.al_strategy is not None else 'random'
if _strategy_str in ('none', '0', 'false', 'no', 'random', ''):
    args.al_strategy = 'random'
    args.use_active_learning = False  # Disable AL so random branch is taken

# Normalize generated-image mode
gen_mode = str(args.gen_only).lower()
valid_gen_modes = ('false', 'train_only', 'train_val_test')
if gen_mode not in valid_gen_modes:
    raise ValueError(f"Invalid --gen_only value '{args.gen_only}'. Choose from {valid_gen_modes}.")
args.gen_only = gen_mode
gen_use_generated = gen_mode != 'false'
gen_three_split = gen_mode == 'train_val_test'

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
else:
    torch.manual_seed(args.seed)
# Enforce deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if getattr(args, 'verbose', False):
    print(f"Random seed set to {args.seed}")

# Record overall start time (end-to-end, includes data prep, AL selection, training)
overall_start_time = time.time()
# Accumulator for total selection time (Active Learning or LLM-based selection)
al_total_time = 0.0
diag_labels_imagenet = None
diag_split_imagenet = None
predictions_for_diagnostics = None

if args.attribute_file is None:
    args.attribute_file = str(dataset_default_attribute_path(args.dataset))
else:
    args.attribute_file = str(resolve_artifact_path(args.attribute_file))

# Automatically enable pretrained weights for CLIP and ViT on ImageNet
if args.dataset == 'imagenet' and args.model in ('clip', 'vit'):
    args.pretrained = True
# If the attribute list includes "emotion", automatically skip feasible loading
if "emotion" in args.attributes.lower():
    args.skip_feasible = True
    if args.verbose:
        print(f"'emotion' detected in attributes; setting skip_feasible=True")

if args.model == 'vit' and args.num_epochs == 20:
    args.num_epochs = 15
    if args.verbose:
        print(f"Adjusted number of epochs for ViT model to {args.num_epochs}")

if args.dataset == 'imagenet' and args.model == 'clip' and args.num_epochs == 20:
    args.num_epochs = 15

# Adjust number of epochs default for ImageNet subset
# if args.dataset == 'imagenet' and args.num_epochs == 20 and args.model == 'resnet':
#     args.num_epochs = 30
#     if 'verbose' in locals() and verbose:
#         print(f"Adjusted number of epochs for ImageNet dataset to {args.num_epochs}")

# Adjust Active Learning parameters for ImageNet dataset (13000 images, 10 classes)
if args.dataset == 'imagenet':
    # For ImageNet with 13000 images total (1300 per class, 10 classes)
    # Adjust initial labeled size to be proportional to dataset size
    if args.al_initial_labeled == 2000:  # Default value
        args.al_initial_labeled = 1300  # About 10% of total dataset
        if args.verbose:
            print(f"Adjusted al_initial_labeled for ImageNet to {args.al_initial_labeled}")
    
    # Ensure embedding dimension is appropriate for the model
    if args.model == 'resnet' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 512  # ResNet18 feature dimension before FC
    elif args.model == 'clip' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 512  # CLIP projection dimension
    elif args.model == 'vit' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 768  # ViT hidden dimension
        if args.verbose:
            print(f"Adjusted al_embedding_dim for ViT to {args.al_embedding_dim}")

attributes = args.attributes

# Set default feasible_file based on attributes if not provided
if args.feasible_file is None:
    if args.lvlm == 'llava':
        if args.verbose:
            print(f"Using LLAVA model: setting feasible_file to save/feasible_{attributes}_{args.lvlm}.json")
        args.feasible_file = str(resolve_artifact_path(f"save/feasible_{attributes}_{args.lvlm}.json"))
    else: #qwen
        args.feasible_file = str(resolve_artifact_path(f"save/feasible_{attributes}.json"))
else:
    args.feasible_file = str(resolve_artifact_path(args.feasible_file))
# Set default gen_image_dir based on attributes if not provided
# Ablation: disable ControlNet-based augmentation if no_controlnet is True
if args.no_controlnet:
    args.gen_image_dir = f"no_controlnet_{attributes}"
    print(f"Ablation mode enabled (ControlNet disabled): using image directory {args.gen_image_dir}")
elif args.gen_image_dir is None:
    default_generated_dir = _default_generated_dir(args.dataset, attributes)
    args.gen_image_dir = str(default_generated_dir) if default_generated_dir else f"exampleData/{attributes}"

verbose = args.verbose
test_loader = None

# Adjust Active Learning parameters based on dataset size
if args.dataset == 'imagenet':
    # ImageNet subset has ~13000 images (1300 per class × 10 classes)
    # Adjust al_initial_labeled to be reasonable for this scale
    if args.al_initial_labeled == 500:  # Default value
        # Use about 3-5% of total dataset as initial labeled set
        args.al_initial_labeled = min(500, max(200, int(13000 * 0.04)))  # ~520, capped at 500
        if verbose:
            print(f"Adjusted al_initial_labeled for ImageNet scale to {args.al_initial_labeled}")
    
    # For ImageNet with 10 classes, ensure we have representation from all classes
    # Validation set typically uses 20% of data, so ~2600 images
    # This is sufficient for initial labeled set
    
    # Adjust embedding dimension based on model type for ImageNet
    if args.model == 'resnet' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 256  # Lighter for ResNet features
        if verbose:
            print(f"Adjusted al_embedding_dim for ResNet on ImageNet to {args.al_embedding_dim}")
    elif args.model == 'clip' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 512  # Keep 512 for CLIP
    elif args.model == 'vit' and args.al_embedding_dim == 512:
        args.al_embedding_dim = 768  # ViT typically uses 768
        if verbose:
            print(f"Adjusted al_embedding_dim for ViT on ImageNet to {args.al_embedding_dim}")

elif args.dataset == 'celeba':
    # CelebA has much larger scale (~80k training images)
    # Adjust parameters accordingly
    if args.al_initial_labeled == 500:  # Default value
        args.al_initial_labeled = 1000  # More samples for larger dataset
        if verbose:
            print(f"Adjusted al_initial_labeled for CelebA scale to {args.al_initial_labeled}")

# Validation: Ensure only one selection method is enabled
if args.use_active_learning and args.use_llm_selection:
    print("Warning: Both --use_active_learning and --use_llm_selection are enabled.")
    print("Disabling Active Learning in favor of LLM selection.")
    args.use_active_learning = False

if args.use_llm_selection:
    _require_llm_data_selection()

# Adjust learning rate default for ViT if not manually overridden
if args.model == 'vit' and args.lr == 1e-4:
    args.lr = 5e-5
    if verbose:
        print(f"Adjusted learning rate for ViT to {args.lr}")

# Specific adjustment for pretrained ViT on ImageNet to further reduce LR
if args.dataset == 'imagenet' and args.model == 'vit' and args.pretrained:
    # This can override the 5e-5 set above if conditions match,
    # or override a user-specified LR for this specific scenario.
    new_lr_val = 2e-5  # Target LR for this case
    if args.lr != new_lr_val: # Apply and print only if it's a change from current args.lr
        args.lr = new_lr_val
        if verbose:
            print(f"Specifically set learning rate for pretrained ViT on ImageNet to {args.lr}")

if args.dataset == 'imagenet' and args.lr == 1e-4 and args.model == 'resnet':
    # If resuming from checkpoint, use much lower LR
    if args.resume_model:
        args.lr = 1e-5
        if verbose:
            if args.freeze_backbone:
                print(f"Using fine-tuning learning rate {args.lr} for resumed model with frozen backbone")
            else:
                print(f"Using fine-tuning learning rate {args.lr} for resumed model")
    else:
        args.lr = 1e-3

elif args.dataset == 'celeba' and args.lr == 1e-4 and args.model == 'resnet':
    if args.resume_model:
        args.lr = 1e-5
        if verbose:
            if args.freeze_backbone:
                print(f"Using fine-tuning learning rate {args.lr} for resumed CelebA ResNet with frozen backbone")
            else:
                print(f"Using fine-tuning learning rate {args.lr} for resumed CelebA ResNet")

# Adjust LR for CLIP on ImageNet based on pretrained flag
elif args.dataset == 'imagenet' and args.model == 'clip' and args.lr == 1e-4:
    if args.pretrained:
        args.lr = 1e-5
        if verbose:
            print(f"Set lr for pretrained ImageNet CLIP to {args.lr}")
    else:
        # Scratch CLIP: define separate head and backbone LRs
        args.lr_head = 5e-3
        args.lr_backbone = 5e-4
        if verbose:
            print(f"Scratch CLIP: head lr={args.lr_head}, backbone lr={args.lr_backbone}")
# Leave non-CLIP/ImageNet default unchanged for other cases
# Adjust learning rate for CLIP and ViT on CelebA to match predict_0505.py
if args.dataset == 'celeba' and args.model in ('clip', 'vit'):
    args.lr = 1e-5
    if verbose:
        print(f"Adjusted learning rate for {args.model} on CelebA to {args.lr}")
time_now = datetime.datetime.now().strftime("%m%d%H%M%S")   
print("Current time:", time_now)

# Determine the comparison log file based on the model and dataset
compare_log_path = None
if args.model == 'resnet':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_resnet_compare.log"
elif args.model == 'vit':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_vit_compare.log"
elif args.model == 'clip':
    compare_log_path = f"logs/compare/stat_{'imagenet' if args.dataset=='imagenet' else 'origin'}_clip_compare.log"

if compare_log_path and not os.path.exists(compare_log_path):
    print(f"Warning: Default comparison log file for {args.model} not found at {compare_log_path}")
    compare_log_path = None # Set to None if file doesn't exist

# Load the feasible images from feasible.json unless skip_feasible is True
if not args.no_controlnet:
    if args.skip_feasible:
        feasible_images = {}
        if verbose:
            print(f"skip_feasible flag is True; skipping feasible JSON loading for {args.attributes}.")
    else:
        if os.path.exists(args.feasible_file):
            with open(args.feasible_file, 'r') as f:
                feasible_images = json.load(f)
            if verbose:
                print(f"Loaded {len(feasible_images)} feasible {args.attributes} images")
        else:
            feasible_images = {}
            if args.verbose:
                print(f"Feasible file '{args.feasible_file}' not found; skipping feasible JSON loading for '{args.attributes}'.")
else:
    feasible_images = {}  # Not used in ablation mode
    if verbose:
        print("Ablation (no_controlnet) mode: Skipping feasible image loading.")

if args.dataset == 'imagenet':
    # Ensure generated images directory uses the 'imagenet_' prefix
    if args.gen_image_dir is None or not os.path.isdir(args.gen_image_dir):
        default_imagenet_gen_dir = _default_generated_dir("imagenet", attributes)
        if default_imagenet_gen_dir is not None:
            args.gen_image_dir = str(default_imagenet_gen_dir)
    # Load ImageNet subset from folders
    data_dir = _first_existing_dir(
        args.data_dir,
        _juno_scratch_data_dir("imagenet_subset"),
        Path("data/imagenet_subset"),
    )
    if data_dir is None:
        raise FileNotFoundError(
            "ImageNet source data directory not found. Pass --data_dir or copy it to "
            f"{_juno_scratch_data_dir('imagenet_subset')}"
        )
    if verbose:
        print(f"Using ImageNet source data directory: {data_dir}")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model == 'clip':
        train_transform = clip_processor_transform(feature_extractor)
        val_transform = train_transform
    else:
        train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)
    # Use no transform for index mapping, then create train/val datasets with correct transforms
    full_ds = ImageFolder(data_dir)
    train_idx_path = resolve_existing_artifact('save/train_idx_imagenet.npy')
    val_idx_path = resolve_existing_artifact('save/val_idx_imagenet.npy')
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    try:
        test_idx_path = resolve_existing_artifact('save/test_idx_imagenet.npy')
    except FileNotFoundError:
        test_idx_path = None
    if test_idx_path is not None:
        test_idx = np.load(test_idx_path)
    else:
        test_idx = np.array([], dtype=int)
    if verbose and (train_idx_path.parent == LEGACY_SAVE_ROOT or val_idx_path.parent == LEGACY_SAVE_ROOT):
        print(
            "Using ImageNet split files from "
            f"{train_idx_path.parent} (train) and {val_idx_path.parent} (val)"
        )
    valid_idx = val_idx  # alias for diagnostics compatibility
    train_ds = Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(ImageFolder(data_dir, transform=val_transform), val_idx)
    test_ds = Subset(ImageFolder(data_dir, transform=val_transform), test_idx) if len(test_idx) else None
    valid_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False) if test_ds is not None else None
    valid_dataset = val_ds
    num_classes = len(full_ds.classes)

    labels_all = np.array(full_ds.targets)

    # --- ImageNet augmentation modes ---
    if gen_use_generated:
        # Use only generated images for ImageNet
        # Collect generated image paths and labels
        gen_paths, gen_labels = [], []
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = Path(args.gen_image_dir) / cls_name
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    gen_paths.append(str(cls_dir / img_name))
                    gen_labels.append(cls_idx)
        if len(gen_paths) < args.num_added_img:
            print(f"Warning: requested {args.num_added_img}, but only {len(gen_paths)} generated images available.")
        # Sample up to requested number
        random.seed(args.seed)
        idxs = list(range(len(gen_paths)))
        sampled = random.sample(idxs, min(len(gen_paths), args.num_added_img))
        gen_paths = [gen_paths[i] for i in sampled]
        gen_labels = [gen_labels[i] for i in sampled]
        # Split generated set
        indices = list(range(len(gen_paths)))
        random.shuffle(indices)
        if gen_three_split:
            train_end = int(0.8 * len(indices))
            val_end = int(0.9 * len(indices))
            train_idx_gen = indices[:train_end]
            val_idx_gen = indices[train_end:val_end]
            test_idx_gen = indices[val_end:]
        else:
            # 8:1 split for train_only
            train_end = max(1, (len(indices) * 8) // 9)
            if train_end >= len(indices):
                train_end = max(len(indices) - 1, 0)
            train_idx_gen = indices[:train_end]
            val_idx_gen = indices[train_end:]
            test_idx_gen = []
        class GeneratedImageDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths, self.labels, self.transform = paths, labels, transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return (self.transform(img), self.labels[i])
        train_ds = GeneratedImageDataset([gen_paths[i] for i in train_idx_gen],
                                         [gen_labels[i] for i in train_idx_gen],
                                         train_transform)
        val_ds = GeneratedImageDataset([gen_paths[i] for i in val_idx_gen],
                                       [gen_labels[i] for i in val_idx_gen],
                                       val_transform)
        train_loader = make_loader(train_ds, shuffle=True)
        valid_loader = make_loader(val_ds, shuffle=False)
        if gen_three_split:
            test_ds_gen = GeneratedImageDataset([gen_paths[i] for i in test_idx_gen],
                                                [gen_labels[i] for i in test_idx_gen],
                                                val_transform)
            test_loader = make_loader(test_ds_gen, shuffle=False)
        print(
            f"Using generated images for ImageNet ({gen_mode}): "
            f"train={len(train_ds)}, valid={len(val_ds)}"
            + (f", test={len(test_idx_gen)}" if gen_three_split else "")
        )

        # Prepare diagnostics labels/split for generated modes
        gen_labels_np = np.array(gen_labels)
        train_labels_gen = gen_labels_np[train_idx_gen]
        val_labels_gen = gen_labels_np[val_idx_gen]
        if gen_three_split:
            test_labels_gen = gen_labels_np[test_idx_gen]
        else:
            test_labels_gen = labels_all[test_idx] if len(test_idx) else np.array([])
        diag_labels_imagenet = np.concatenate([train_labels_gen, val_labels_gen, test_labels_gen])
        diag_split_imagenet = (
            [TRAIN] * len(train_labels_gen)
            + [VALID] * len(val_labels_gen)
            + [TEST] * len(test_labels_gen)
        )
    elif args.simple:
        # Simple mode: augment from data/imagenet_simple
        simple_root = Path("data/imagenet_simple")
        simple_paths, simple_labels = [], []
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = simple_root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    simple_paths.append(str(cls_dir / img_name))
                    simple_labels.append(cls_idx)
        # Sample up to args.num_added_img
        if len(simple_paths) > args.num_added_img:
            combined = list(zip(simple_paths, simple_labels))
            random.seed(args.seed)
            sampled = random.sample(combined, args.num_added_img)
            simple_paths, simple_labels = zip(*sampled)
            simple_paths, simple_labels = list(simple_paths), list(simple_labels)
        if simple_paths:
            class SimpleImageDataset(Dataset):
                def __init__(self, paths, labels, transform):
                    self.paths, self.labels, self.transform = paths, labels, transform
                def __len__(self): return len(self.paths)
                def __getitem__(self, i):
                    img = Image.open(self.paths[i]).convert("RGB")
                    return (self.transform(img), self.labels[i]) if self.transform else (img, self.labels[i])
            simple_ds = SimpleImageDataset(simple_paths, simple_labels, train_transform)
            train_dataset = torch.utils.data.ConcatDataset([train_ds, simple_ds])
            train_loader = make_loader(train_dataset, shuffle=True)
            print(f"Added {len(simple_ds)} ImageNet simple images: {len(train_ds)} -> {len(train_dataset)}")
        else:
            train_dataset, train_loader = train_ds, make_loader(train_ds, shuffle=True)
    elif not gen_use_generated and not args.simple:
        # Normal mode: generated-images + feasible filter (mirrors CelebA logic)
        # Determine generated-image directory: use default if exists, otherwise try fallbacks
        if args.gen_image_dir is None or not os.path.exists(args.gen_image_dir):
            default_gen_dir = _default_generated_dir("imagenet", attributes)
            args.gen_image_dir = str(default_gen_dir) if default_gen_dir else f"data/imagenet_{attributes}"
        if verbose:
            print(f"Using generated image directory: {args.gen_image_dir}")
        if verbose:
            print(f"[DEBUG] Normal mode: looking for generated images in directory: {args.gen_image_dir}")
        # --- Patch: Respect skip_feasible for feasible_images loading ---
        if args.skip_feasible:
            feasible_images = {}
            # if verbose:
            #     print("[DEBUG] Skipping feasible filtering (skip_feasible=True)")
        else:
            if os.path.exists(args.feasible_file):
                with open(args.feasible_file) as f:
                    feasible_images = json.load(f)
            else:
                feasible_images = {}
            # if verbose:
            #     print(f"[DEBUG] Feasible images loaded: {len(feasible_images)} entries")
            #     print(f"[DEBUG] Sample of feasible keys: {list(feasible_images.keys())[:10]}")
        # if verbose:
        #     print(f"[DEBUG] Classes in full_ds: {list(full_ds.class_to_idx.keys())}")
        # Traverse each class subfolder for generated images
        gen_paths, gen_labels = [], []
        if verbose:
            print("[DEBUG] Starting traversal of class subfolders for generated images...")
        for cls_name, cls_idx in full_ds.class_to_idx.items():
            cls_dir = Path(args.gen_image_dir) / cls_name
            # if verbose:
            #     print(f"[DEBUG] Checking class folder: {cls_dir} (exists: {cls_dir.is_dir()})")
            if not cls_dir.is_dir():
                continue
            for img_name in os.listdir(cls_dir):
                # if verbose:
                #     print(f"[DEBUG] Found file in {cls_name}: {img_name}")
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                key = os.path.splitext(img_name)[0]
                full_key = f"{cls_name}/{key}"
                in_feasible = full_key in feasible_images
                # if verbose:
                #     print(f"[DEBUG] File key '{full_key}' in feasible_images: {in_feasible}")
                # --- Patch: Only filter if not args.skip_feasible ---
                if not args.skip_feasible and feasible_images and full_key not in feasible_images:
                    # if verbose:
                    #     print(f"[DEBUG] Skipping '{full_key}' not in feasible_images")
                    continue
                gen_paths.append(str(cls_dir / img_name))
                gen_labels.append(cls_idx)
        # Debug before filtering by feasible_images
        total_before_filter = len(gen_paths)
        if verbose:
            print(f"[DEBUG] Candidate images before feasible filtering: {total_before_filter}")
            print(f"[DEBUG] Total candidate generated images before sampling: {len(gen_paths)}")
        if len(gen_paths) < args.num_added_img:
            print(f"Warning: requested {args.num_added_img}, but only {len(gen_paths)} available.")
        # Sample up to requested number
        random.seed(args.seed)
        if len(gen_paths) > args.num_added_img:
            idxs = random.sample(range(len(gen_paths)), args.num_added_img)
            gen_paths = [gen_paths[i] for i in idxs]
            gen_labels = [gen_labels[i] for i in idxs]
        modified_image_paths = gen_paths
        modified_image_labels = gen_labels
        if verbose:
            print(f"[DEBUG] Sampled generated images: {len(modified_image_paths)} items")
        print(f"Added {len(modified_image_labels)} generated images.")
        # Combine original train_ds and generated
        class GeneratedDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths, self.labels, self.transform = paths, labels, transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return (self.transform(img), self.labels[i]) if self.transform else (img, self.labels[i])
        gen_ds = GeneratedDataset(modified_image_paths, modified_image_labels, train_transform)
        
        # Handle combined Active Learning or original approach
        if args.al_combined and args.use_active_learning:
            # Combined approach: Apply AL on combined dataset (original + generated)
            orig_count = len(train_ds)
            gen_count = len(gen_ds)
            combined_dataset = torch.utils.data.ConcatDataset([train_ds, gen_ds])
            combined_count = len(combined_dataset)
            combined_target_count = int(combined_count * args.combined_ratio)
            
            if verbose:
                print(f"Combined AL approach: selecting {combined_target_count} samples from {combined_count} (original: {orig_count}, generated: {gen_count})")
            
            # Define device for Active Learning
            al_device = _resolve_runtime_device(args.gpu)
            
            # Use Active Learning for data selection on combined dataset (timed)
            al_select_start = time.time()
            selected_indices = active_learning_selection(combined_dataset, val_ds, args, al_device, verbose)
            al_duration = time.time() - al_select_start
            al_total_time += al_duration
            if verbose:
                print(f"[TIMING] Active Learning selection (Combined ImageNet) took {al_duration:.2f} seconds")
            
            # Ensure we don't exceed the target count
            if len(selected_indices) > combined_target_count:
                random.seed(args.seed)
                selected_indices = random.sample(selected_indices, combined_target_count)
            
            # Analyze selection: count original vs generated
            orig_selected = sum(1 for idx in selected_indices if idx < orig_count)
            gen_selected = len(selected_indices) - orig_selected
            print(f"AL selected from combined dataset: {orig_selected} original + {gen_selected} generated = {len(selected_indices)} total")
            
            # Create final training dataset from selected indices
            train_dataset = Subset(combined_dataset, selected_indices)
            selection_method = f"Combined Active Learning ({args.al_strategy})"
            
        else:
            # Original approach: Apply AL/random on original data, then add all generated
            orig_count = len(train_ds)
            sub_count = int(orig_count * args.sub_ratio)
            
            if args.use_active_learning and args.sub_ratio < 1.0:
                if verbose:
                    print(f"Using Active Learning strategy to select {sub_count} samples from {orig_count}")
                # Define device for Active Learning if not already defined
                al_device = _resolve_runtime_device(args.gpu)
                # Use Active Learning for data selection (timed)
                al_select_start = time.time()
                selected_indices = active_learning_selection(train_ds, val_ds, args, al_device, verbose)
                al_duration = time.time() - al_select_start
                al_total_time += al_duration
                if verbose:
                    print(f"[TIMING] Active Learning selection (ImageNet) took {al_duration:.2f} seconds")
                # Ensure we don't exceed the target count
                if len(selected_indices) > sub_count:
                    random.seed(args.seed)
                    selected_indices = random.sample(selected_indices, sub_count)
            elif args.use_llm_selection and args.sub_ratio < 1.0:
                if verbose:
                    print(f"Using LLM-based selection ({args.llm_strategy}) to select {sub_count} samples from {orig_count}")
                
                # Extract image paths and labels from ImageNet Subset dataset
                # train_ds is a Subset, so we need to access the underlying dataset
                underlying_dataset = train_ds.dataset
                subset_indices = train_ds.indices
                image_paths = [underlying_dataset.samples[idx][0] for idx in subset_indices]
                dataset_labels = np.array([underlying_dataset.samples[idx][1] for idx in subset_indices])
                
                # Use LLM-based selection (timed)
                llm_select_start = time.time()
                llm_data_selection_fn = _require_llm_data_selection()
                selected_indices = llm_data_selection_fn(
                    image_paths=image_paths,
                    labels=dataset_labels,
                    sub_ratio=args.sub_ratio,
                    dataset_type='imagenet',
                    lvlm_model=args.llm_model,
                    strategy=args.llm_strategy,
                    device=_resolve_runtime_device(args.gpu),
                    random_seed=args.seed,
                    verbose=verbose
                )
                llm_duration = time.time() - llm_select_start
                al_total_time += llm_duration  # Add to AL time for comparison
                if verbose:
                    print(f"[TIMING] LLM selection (ImageNet) took {llm_duration:.2f} seconds")
            else:
                if verbose:
                    print(f"Using random sampling to select {sub_count} samples from {orig_count}")
                # Use random sampling (original behavior)
                random.seed(args.seed)
                selected_indices = random.sample(range(orig_count), sub_count)
            
            # Map subset indices back to the original dataset indices
            sub_orig_indices = [train_ds.indices[i] for i in selected_indices]
            sub_train_ds = Subset(train_ds.dataset, sub_orig_indices)

            if args.dataset == 'imagenet':
                _print_class_distribution(
                    sub_train_ds,
                    description="Active Learning subset (before adding generated images)",
                    verbose=verbose,
                )
            # Combine subsampled original data with generated data
            train_dataset = torch.utils.data.ConcatDataset([sub_train_ds, gen_ds])
            if args.use_active_learning and args.sub_ratio < 1.0:
                selection_method = f"Active Learning ({args.al_strategy})"
            elif args.use_llm_selection and args.sub_ratio < 1.0:
                selection_method = f"LLM-based selection ({args.llm_strategy})"
            else:
                selection_method = "Random sampling"
            print(f"Added {len(modified_image_labels)} ImageNet {attributes} images using {selection_method}: {len(sub_train_ds)} -> {len(train_dataset)}")
        
        # Store subsampling info for padding in save_predictions
        if not args.al_combined:
            args.selected_indices = selected_indices
            args.orig_train_count = orig_count
        else:
            args.selected_indices = None  # Combined mode doesn't need this
            args.orig_train_count = None
        
        train_loader = make_loader(train_dataset, shuffle=True)

    # Skip CelebA loading
    all_train_imgs = None  # placeholder
    valid_imgs = None
    train_labels = None
    valid_labels = None
    modified_image_paths = []
    modified_image_labels = []
    unlabeled_predictions = np.array([])
    # Jump to model setup
elif args.dataset == 'celeba':
    with open("data/celebA/list_attr_celeba.txt") as f:
        lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]
        

    input_dir = _first_existing_dir(
        args.data_dir,
        _juno_scratch_data_dir("img_align_celeba"),
        Path("/home/jovyan/ouser/dataset/img_align_celeba"),
        Path("/data/oxx220000/img_align_celeba"),
    )
    if input_dir is None:
        raise FileNotFoundError(
            "CelebA source data directory not found. Pass --data_dir or copy it to "
            f"{_juno_scratch_data_dir('img_align_celeba')}"
        )
    if verbose:
        print(f"Using CelebA source data directory: {input_dir}")
    all_datas = [str(input_dir / line) for line in lines][1:]

    train_idx = np.load("data/celebA/train_idx.npy")[:80000]
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
        and cached_seed == args.seed
    )

    if cache_ready:
        valid_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
        unlabel_idx = np.load(unlabel_idx_path)
    else:
        val_pool = np.arange(len(all_datas))[-100000:]
        if val_pool.size < 20000:
            raise ValueError("Expected at least 20000 samples in CelebA validation/test pool.")
        rng = np.random.default_rng(args.seed)
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
                "seed": int(args.seed),
                "timestamp": time.time()
            }, meta_file)

    idxs = np.concatenate([train_idx, valid_idx, test_idx, unlabel_idx], axis=0)
    labels = np.load("data/celebA/labels.npy")[idxs]

    all_datas = [all_datas[i] for i in idxs]
    split = (
        [TRAIN] * len(train_idx)
        + [VALID] * len(valid_idx)
        + [TEST] * len(test_idx)
        + [UNLABELED] * len(unlabel_idx)
    )

    # Define Dataset
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

    # Initialize CLIP feature extractor
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model == 'clip':
        train_transform = clip_processor_transform(feature_extractor)
        val_transform = train_transform
    else:
        train_transform = default_img_transform(train=True)
        val_transform = default_img_transform(train=False)

    # Create Datasets and Loaders
    train_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TRAIN]
    valid_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == VALID]
    test_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == TEST]
    unlabeled_imgs = [all_datas[i] for i in range(len(all_datas)) if split[i] == UNLABELED]

    train_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TRAIN])
    valid_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == VALID])
    test_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == TEST])
    unlabeled_labels = np.array([labels[i] for i in range(len(all_datas)) if split[i] == UNLABELED])

    # Add the modified hair images to the training set
    # These images' labels will be the same as their corresponding original images
    modified_image_paths = []
    modified_image_labels = []

    #
    # Prepare sample keys for ablation mode (was safefix mode)
    if args.no_controlnet:
        if args.simple:
            # Simple mode: read all PNGs from safefix_simple folder
            args.gen_image_dir = "safefix_simple"
            files = os.listdir(args.gen_image_dir)
            sample_keys = [os.path.splitext(f)[0] for f in files if f.lower().endswith('.png')]
            print(f"Simple ablation (no_controlnet) mode: Found {len(sample_keys)} images in {args.gen_image_dir}")
        else:
            # Existing ablation behavior: read from no_controlnet_<attributes> folder
            args.gen_image_dir = f"no_controlnet_{attributes}"
            if not os.path.isdir(args.gen_image_dir):
                print(f"Error: Ablation image directory {args.gen_image_dir} not found.")
                sample_keys = []
            else:
                all_files = os.listdir(args.gen_image_dir)
                sample_keys = [os.path.splitext(f)[0] for f in all_files if f.lower().endswith('.png')]
                print(f"Ablation (no_controlnet) mode: Found {len(sample_keys)} images in {args.gen_image_dir}")

    # Sample from all generated image files in the directory, ignoring JSON-based feasible_images
    if not args.no_controlnet:
        if not os.path.isdir(args.gen_image_dir):
            print(f"Error: Generated image directory {args.gen_image_dir} not found.")
            sample_keys = []
        else:
            import re  # ensure this import is at the top of the file
            all_files = os.listdir(args.gen_image_dir)
            # Match Safefix names "<digits>_<attribute>[_seed]" and HiBUG names "<attribute>_<digits>".
            regex = re.compile(rf'^\d+_{re.escape(attributes)}(?:_\d+)?$')
            hibug_regex = re.compile(rf'^{re.escape(attributes)}_\d+$')
            file_keys = [
                os.path.splitext(f)[0]
                for f in all_files
                if regex.match(os.path.splitext(f)[0]) or hibug_regex.match(os.path.splitext(f)[0])
            ]
            sample_keys = file_keys

    # Warn if fewer images are available than requested in the chosen sample_keys
    if len(sample_keys) < args.num_added_img:
        print(f"Warning: requested {args.num_added_img} generated {args.attributes} images, "
              f"but only {len(sample_keys)} available.")

    # Perform sampling
    if verbose:
        print("Perform sampling for generated images")
    selected_filenames = random.sample(sample_keys, min(args.num_added_img, len(sample_keys)))
    print(f"Selected {len(selected_filenames)} filenames for sampling")

    # Build a map from base image number to label for quick lookup (not needed for simple ablation mode)
    if not (args.no_controlnet and args.simple):
        # Include training, validation, test, and unlabeled images in the label map
        all_orig_paths = train_imgs + valid_imgs + test_imgs + unlabeled_imgs
        all_orig_labels = np.concatenate((train_labels, valid_labels, test_labels, unlabeled_labels))
        base_to_label = {
            os.path.splitext(os.path.basename(p))[0]: lbl
            for p, lbl in zip(all_orig_paths, all_orig_labels)
        }
        
        if verbose:
            print(f"Populated base_to_label dictionary with {len(base_to_label)} entries.")
            example_keys = list(base_to_label.keys())[:5]
            print(f"Example keys in base_to_label: {example_keys}")

    # Sampling loop
    for filename in selected_filenames:
        extension = ".png" if args.no_controlnet else ".jpg"
        if args.no_controlnet and args.simple:
            # In simple ablation mode, infer label from filename prefix: 'yes'->1, 'no'->0
            prefix = filename.split('_')[0].lower()
            lbl = 1 if prefix == 'yes' else 0
            image_path = str(Path(args.gen_image_dir) / f"{filename}{extension}")
            modified_image_paths.append(image_path)
            modified_image_labels.append(lbl)
            if verbose:
                print(f"Added {image_path} with label {lbl}")
            continue
        # Existing ablation or non-ablation logic follows...
        # Determine the correct extension based on ablation flag
        # Get the image file path...
        image_path = str(Path(args.gen_image_dir) / f"{filename}{extension}")
        if not args.no_controlnet and not os.path.exists(image_path):
            for candidate_extension in (".png", ".jpeg", ".jpg"):
                candidate_path = Path(args.gen_image_dir) / f"{filename}{candidate_extension}"
                if candidate_path.exists():
                    image_path = str(candidate_path)
                    break
        if not os.path.exists(image_path):
            print(f"Warning: Modified image {image_path} not found.")
            continue
        base = None
        if args.no_controlnet:
            parts = filename.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                base = parts[-1]
            else:
                print(f"Warning: Could not extract base number from ablation filename {filename}")
                continue
        else:
            base = _extract_celeba_base_id(filename, attributes)
            if base is None:
                print(f"Warning: Could not extract base number from generated filename {filename}")
                continue
        if args.no_controlnet and base is None:
            print(f"Debug: Ablation filename {filename} resulted in base=None")
            continue
        lbl = base_to_label.get(base)
        if lbl is not None:
            modified_image_paths.append(image_path)
            modified_image_labels.append(lbl)
            if verbose:
                print(f"Added {image_path} with label {lbl}")
        else:
            if args.no_controlnet:
                print(f"Debug: Failed lookup for ablation file '{filename}'. Extracted base='{base}'. Key not in base_to_label.")
                print(f"Warning: Could not find original image label for base number {base} (from ablation file {filename})")
            else:
                print(f"Warning: Could not find original image for {filename}")

    if verbose:
        print("Finish sampling for generated images")

    # Print information about the added images
    if gen_use_generated:
        print(
            f"Using generated images ({gen_mode}): total generated samples considered = {len(modified_image_labels)}"
        )
    else:
        added_count = len(modified_image_labels)
        print(f"Added {added_count} generated {attributes} images, "
              f"before training set has {len(train_labels)} images, "
              f"now {len(train_labels) + added_count}")
            
    # Load unlabeled predictions - moved outside conditional block
    unlabeled_predictions = np.load("data/celebA/predictions.npy")[idxs]
    unlabeled_predictions = np.array([unlabeled_predictions[i] for i in range(len(all_datas)) if split[i] == UNLABELED])
    if verbose:
        print(f"Unlabeled predictions length: {len(unlabeled_predictions)}")

    # Construct training data based on gen_only flag
    if gen_use_generated:
        # Use only generated images: split into 80/20 or 80/10/10
        indices = list(range(len(modified_image_paths)))
        random.shuffle(indices)
        if gen_three_split:
            train_end = int(0.8 * len(indices))
            val_end = int(0.9 * len(indices))
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        else:
            # 8:1 split for train_only
            train_end = max(1, (len(indices) * 8) // 9)
            if train_end >= len(indices):
                train_end = max(len(indices) - 1, 0)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:]
            test_indices = []
        all_train_imgs = [modified_image_paths[i] for i in train_indices]
        all_train_labels = np.array(modified_image_labels)[train_indices]
        valid_imgs = [modified_image_paths[i] for i in val_indices]
        valid_labels = np.array(modified_image_labels)[val_indices]
        if gen_three_split:
            test_imgs = [modified_image_paths[i] for i in test_indices]
            test_labels = np.array(modified_image_labels)[test_indices]
    elif modified_image_paths:
        # Handle combined Active Learning or original approach for CelebA
        if args.al_combined and args.use_active_learning:
            # Combined approach: Apply AL on combined dataset (original + generated)
            orig_count = len(train_imgs)
            gen_count = len(modified_image_paths)
            
            # Create temporary datasets for combined Active Learning
            temp_train_dataset = CelebADataset(train_imgs, train_labels, val_transform)
            temp_gen_dataset = CelebADataset(modified_image_paths, np.array(modified_image_labels), val_transform)
            temp_combined_dataset = torch.utils.data.ConcatDataset([temp_train_dataset, temp_gen_dataset])
            temp_valid_dataset = CelebADataset(valid_imgs, valid_labels, val_transform)
            
            combined_count = len(temp_combined_dataset)
            combined_target_count = int(combined_count * args.combined_ratio)
            
            if verbose:
                print(f"Combined AL approach (CelebA): selecting {combined_target_count} samples from {combined_count} (original: {orig_count}, generated: {gen_count})")
            
            # Define device for Active Learning
            al_device = _resolve_runtime_device(args.gpu)
            
            # Use Active Learning for data selection on combined dataset (timed)
            al_select_start = time.time()
            selected_indices = active_learning_selection(temp_combined_dataset, temp_valid_dataset, args, al_device, verbose)
            al_duration = time.time() - al_select_start
            al_total_time += al_duration
            if verbose:
                print(f"[TIMING] Active Learning selection (Combined CelebA) took {al_duration:.2f} seconds")
            
            # Ensure we don't exceed the target count
            if len(selected_indices) > combined_target_count:
                random.seed(args.seed)
                selected_indices = random.sample(selected_indices, combined_target_count)
            
            # Analyze selection: count original vs generated
            orig_selected = sum(1 for idx in selected_indices if idx < orig_count)
            gen_selected = len(selected_indices) - orig_selected
            print(f"AL selected from combined CelebA dataset: {orig_selected} original + {gen_selected} generated = {len(selected_indices)} total")
            
            # Extract selected images and labels
            all_train_imgs = []
            all_train_labels = []
            for idx in selected_indices:
                if idx < orig_count:
                    all_train_imgs.append(train_imgs[idx])
                    all_train_labels.append(train_labels[idx])
                else:
                    gen_idx = idx - orig_count
                    all_train_imgs.append(modified_image_paths[gen_idx])
                    all_train_labels.append(modified_image_labels[gen_idx])
            all_train_labels = np.array(all_train_labels)
            
            selection_method = f"Combined Active Learning ({args.al_strategy})"
            print(f"CelebA: {selection_method} selected {len(all_train_imgs)} samples ({orig_selected} original + {gen_selected} generated)")
            
        else:
            # Original approach: Apply AL/random on original data, then add all generated
            orig_count = len(train_imgs)
            sub_count = int(orig_count * args.sub_ratio)
            
            if args.use_active_learning and args.sub_ratio < 1.0:
                if verbose:
                    print(f"Using Active Learningstrategy to select {sub_count} samples from {orig_count} for CelebA")
                # Define device for Active Learning if not already defined
                al_device = _resolve_runtime_device(args.gpu)
                    
                # Create temporary datasets for Active Learning
                temp_train_dataset = CelebADataset(train_imgs, train_labels, val_transform)
                temp_valid_dataset = CelebADataset(valid_imgs, valid_labels, val_transform)
                
                # Use Active Learning for data selection (timed)
                al_select_start = time.time()
                selected_indices = active_learning_selection(temp_train_dataset, temp_valid_dataset, args, al_device, verbose)
                al_duration = time.time() - al_select_start
                al_total_time += al_duration
                if verbose:
                    print(f"[TIMING] Active Learning selection (CelebA) took {al_duration:.2f} seconds")
                # Ensure we don't exceed the target count
                if len(selected_indices) > sub_count:
                    random.seed(args.seed)
                    selected_indices = random.sample(selected_indices, sub_count)
            elif args.use_llm_selection and args.sub_ratio < 1.0:
                if verbose:
                    print(f"Using LLM-based selection ({args.llm_strategy}) to select {sub_count} samples from {orig_count} for CelebA")
                
                # Use LLM-based selection (timed)
                llm_select_start = time.time()
                llm_data_selection_fn = _require_llm_data_selection()
                selected_indices = llm_data_selection_fn(
                    image_paths=train_imgs,
                    labels=train_labels,
                    sub_ratio=args.sub_ratio,
                    dataset_type='celeba',
                    lvlm_model=args.llm_model,
                    strategy=args.llm_strategy,
                    device=_resolve_runtime_device(args.gpu),
                    random_seed=args.seed,
                    verbose=verbose
                )
                llm_duration = time.time() - llm_select_start
                al_total_time += llm_duration  # Add to AL time for comparison
                if verbose:
                    print(f"[TIMING] LLM selection (CelebA) took {llm_duration:.2f} seconds")
            else:
                if verbose:
                    print(f"Using random sampling to select {sub_count} samples from {orig_count} for CelebA")
                # Use random sampling (original behavior)
                random.seed(args.seed)
                selected_indices = random.sample(range(orig_count), sub_count)
                
            sub_train_imgs = [train_imgs[i] for i in selected_indices]
            sub_train_labels = train_labels[selected_indices]
            all_train_imgs = sub_train_imgs + modified_image_paths
            all_train_labels = np.concatenate((sub_train_labels, np.array(modified_image_labels)))
            selection_method = ("Active Learning" if args.use_active_learning and args.sub_ratio < 1.0 
                               else f"LLM-based selection ({args.llm_strategy})" if args.use_llm_selection and args.sub_ratio < 1.0
                               else "Random sampling")
            print(f"CelebA: Combined {len(sub_train_imgs)} selected samples with {len(modified_image_paths)} generated images using {selection_method}")
            
        combined_labels = np.concatenate((all_train_labels, valid_labels, test_labels, unlabeled_labels))
        # Store subsampling info for later padding of predictions
        if not args.al_combined:
            args.selected_indices = selected_indices
            args.orig_train_count = orig_count
        else:
            args.selected_indices = None  # Combined mode doesn't need this
            args.orig_train_count = None
    else:
        all_train_imgs = train_imgs
        all_train_labels = train_labels
        combined_labels = np.concatenate((train_labels, valid_labels, test_labels, unlabeled_labels))

    # Save combined indices and labels when using generated images
    if modified_image_paths:
        modified_filenames = [os.path.basename(path).split('.')[0] for path in modified_image_paths]
        if gen_use_generated:
            combined_train_idx = np.array(modified_filenames, dtype=object)
        else:
            combined_train_idx = np.array([*train_idx, *modified_filenames], dtype=object)
        # Determine naming for simple vs attribute-based cases
        name_key = "simple" if args.no_controlnet and args.simple else attributes
        if verbose:
            train_idx_path = str(canonical_save_path(f"idx/train_idx_{name_key}_{args.num_added_img}imgs.npy", create_parent=True))
            labels_path = str(canonical_save_path(f"labels/labels_{name_key}_{args.num_added_img}imgs.npy", create_parent=True))
            if not os.path.exists(train_idx_path):
                np.save(train_idx_path, combined_train_idx)
                print(f"Saved combined train indices to {train_idx_path}")
            else:
                if verbose:
                    print(f"{train_idx_path} already exists, skipping save.")
            if not os.path.exists(labels_path):
                np.save(labels_path, combined_labels)
                print(f"Saved combined labels to {labels_path}")
            else:
                if verbose:
                    print(f"{labels_path} already exists, skipping save.")

    if verbose:
        print(f"Training images: {len(all_train_imgs)}, Labels: {len(all_train_labels)}")
        print(f"Validation images: {len(valid_imgs)}, Labels: {len(valid_labels)}")
        # 84050 20000
    # Create datasets using the combined data
    train_dataset = CelebADataset(all_train_imgs, all_train_labels, train_transform)
    valid_dataset = CelebADataset(valid_imgs, valid_labels, val_transform)
    if len(test_imgs) > 0:
        test_dataset = CelebADataset(test_imgs, test_labels, val_transform)
        test_loader = make_loader(test_dataset, shuffle=False)
    else:
        test_dataset = None
        test_loader = None

    train_loader = make_loader(train_dataset, shuffle=True)
    valid_loader = make_loader(valid_dataset, shuffle=False)
    num_classes = 2


# Define model, loss, and optimizer
if torch.cuda.is_available():
    runtime_gpu_index = _resolve_cuda_device_index(args.gpu)
    torch.cuda.set_device(runtime_gpu_index)
    device = torch.device(f"cuda:{runtime_gpu_index}")
else:
    device = torch.device("cpu")
# if verbose and device.type == "cuda":
#     print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

if args.model == 'resnet':
    # Use full ResNet-18 for multiclass ImageNet, simplified version for binary tasks (CelebA)
    if args.dataset == 'imagenet':
        # Load randomly initialized ResNet-18 and adjust final layer for multiclass
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Simplified ResNet for binary classification
        if args.pretrained:
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet18(weights=None)
        base_model.layer3 = nn.Identity()
        base_model.layer4 = nn.Identity()
        base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model = base_model
        model.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(128, 1)
        )
elif args.model == 'vit':
    # Initialize ViT base model
    model = create_model(
        'vit_base_patch16_224',
        pretrained=args.pretrained,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    )
    num_head_features = model.head.in_features
    if args.dataset == 'imagenet':
        # Multiclass head for ImageNet subset
        model.head = nn.Sequential(
            nn.Dropout(p=0.5),  # Added Dropout layer
            nn.Linear(num_head_features, num_classes)
        )
    else:
        # Binary classification head for CelebA
        model.head = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(num_head_features, 1)
        )
elif args.model == 'clip':
    # Initialize CLIP-based classifier with appropriate output size
    model = CLIPForBinary(
        dropout_rate=0.5,
        pretrained=args.pretrained,
        num_classes=num_classes if args.dataset=='imagenet' else 1
    )

model = model.to(device)

# Load saved checkpoint if --resume_model is provided
if args.resume_model:
    args.resume_model = str(resolve_artifact_path(args.resume_model))
    if not os.path.exists(args.resume_model):
        raise FileNotFoundError(f"Resume model not found: {args.resume_model}")
    ckpt = torch.load(args.resume_model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    resumed_val_acc = ckpt.get('best_val_acc', 'N/A')
    resumed_test_acc, resume_log_path = _lookup_resume_test_acc(args.resume_model)
    resume_msg = f"Resumed model from {args.resume_model} (best val acc: {resumed_val_acc}"
    if resumed_test_acc is not None:
        resume_msg += f", base test acc: {resumed_test_acc:.4f}"
    resume_msg += ")"
    print(resume_msg)
    if resumed_test_acc is not None and resume_log_path is not None:
        print(f"Recovered base test acc from log: {resume_log_path}")

    # Backbone freezing controlled by --freeze_backbone
    pass
else:
    print(f"Using {args.model} model with {'pretrained' if args.pretrained else 'random'} weights")

# Freeze early layers when fine-tuning to prevent catastrophic forgetting
if args.freeze_backbone:
    frozen_count = 0
    if args.model == 'resnet':
        # Freeze everything except layer4 and fc
        for name, param in model.named_parameters():
            if not any(k in name for k in ('layer4', 'fc')):
                param.requires_grad = False
                frozen_count += 1
    elif args.model == 'vit':
        # Freeze all except last 2 blocks and head
        for name, param in model.named_parameters():
            if 'head' not in name and 'blocks.11' not in name and 'blocks.10' not in name:
                param.requires_grad = False
                frozen_count += 1
    elif args.model == 'clip':
        # Freeze CLIP backbone, only fine-tune classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_count += 1
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Froze {frozen_count} parameter groups, {trainable_count} remain trainable")

# For consistency, handle pos_weight and loss for both datasets
if args.dataset == 'imagenet':
    # Multi-class classification: use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
else:
    num_negative = (all_train_labels == 0).sum()
    num_positive = (all_train_labels == 1).sum()
    pos_weight = torch.tensor([num_negative / num_positive]).to(device)
    # Use smoothed BCE loss since label_smoothing param is unavailable
    criterion = SmoothBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=0.1)

# Optimize only trainable parameters (respects freeze_backbone)
trainable_params = [p for p in model.parameters() if p.requires_grad]
if args.model == 'clip':
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
elif args.model == 'resnet':
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-3)
elif args.model == 'vit':
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-3)

# Use StepLR for pretrained CLIP/ViT on ImageNet, otherwise CosineAnnealingLR for CLIP/ViT, StepLR for ResNet
if args.dataset == 'imagenet' and args.pretrained and args.model in ('clip', 'vit'):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
elif args.model == 'clip':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
elif args.model == 'vit':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
elif args.model == 'resnet':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

def evaluate(model, data_loader, save_preds=False):
    model.eval()
    # start_time = time.time()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            elif args.dataset == 'celeba':
                # For CelebA, ensure labels are for binary classification
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_correct += (preds == labels.int()).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    # elapsed_time = time.time() - start_time
    # print(f"Evaluation time: {elapsed_time:.2f} seconds")
    if save_preds:
        return acc, np.array(all_preds), np.array(all_labels)
    return acc, avg_loss

# After training, collect predictions for diagnostics and optionally save them.
def save_predictions(model, train_loader, valid_loader, test_loader=None, write_file=True):
    if write_file:
        print("Generating and saving predictions...")
    else:
        print("Generating predictions in memory (--save_preds False).")

    # Get predictions for training set
    _, train_preds, _ = evaluate(model, train_loader, save_preds=True)

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)

    test_preds = None
    if test_loader is not None:
        test_acc, test_preds, _ = evaluate(model, test_loader, save_preds=True)
        # print(f"Held-out test accuracy: {test_acc:.4f}")

    if args.dataset == 'imagenet':
        original_count = len(train_idx)
        # Number of subsampled original predictions available
        selected_indices = getattr(args, "selected_indices", None)
        if selected_indices is not None:
            sub_count = len(selected_indices)
        else:
            sub_count = 0  # Combined mode or no subsampling
        # Subsampled original preds come first in train_preds
        sub_preds = train_preds[:sub_count] if sub_count > 0 else []
        if getattr(args, 'pad_missing', False) and selected_indices is not None:
            # Initialize full-length original predictions with -1 (int) instead of NaN
            full_orig_preds = np.full((original_count,), -1, dtype=int)
            # Fill in the subsampled predictions at correct indices
            for idx, pred in zip(selected_indices, sub_preds):
                full_orig_preds[idx] = pred
            combined_preds = np.concatenate((full_orig_preds.flatten(), valid_preds.flatten()))
        else:
            # Convert list to numpy array before flattening
            sub_preds_array = np.array(sub_preds) if isinstance(sub_preds, list) else sub_preds
            combined_preds = np.concatenate((sub_preds_array.flatten(), valid_preds.flatten()))

        if test_preds is not None:
            combined_preds = np.concatenate((combined_preds, test_preds.flatten()))
        if verbose:
            print(
                f"Prediction segments -> train: {len(train_preds)}, val: {len(valid_preds)}, "
                f"test: {len(test_preds) if test_preds is not None else 0}"
            )
    else:
        # Handle celeba or other cases
        if len(modified_image_paths) > 0:
            gen_cnt = len(modified_image_labels)
            if getattr(args, "pad_missing", False):
                # Pad missing original training predictions with -1 (int) instead of NaN
                full_orig_preds = np.full((args.orig_train_count,), -1, dtype=int)
                sub_preds = train_preds[:len(args.selected_indices)]
                for idx, pred in zip(args.selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                modified_preds = train_preds[len(args.selected_indices):]
                segs = [
                    full_orig_preds.flatten(),
                    valid_preds.flatten()
                ]
                if test_preds is not None:
                    segs.append(test_preds.flatten())
                segs.append(modified_preds.flatten())
                segs.append(unlabeled_predictions)
                combined_preds = np.concatenate(segs)
            else:
                original_train_preds = train_preds[:-gen_cnt]
                modified_preds = train_preds[-gen_cnt:]
                segs = [
                    original_train_preds.flatten(),
                    valid_preds.flatten()
                ]
                if test_preds is not None:
                    segs.append(test_preds.flatten())
                if gen_cnt > 0:
                    segs.append(modified_preds.flatten())
                segs.append(unlabeled_predictions)
                combined_preds = np.concatenate(segs)
        else:
            original_train_preds = train_preds
            segs = [
                original_train_preds.flatten(),
                valid_preds.flatten()
            ]
            if test_preds is not None:
                segs.append(test_preds.flatten())
            segs.append(unlabeled_predictions)
            combined_preds = np.concatenate(segs)

    try:
        print(f"Combined predictions length: {len(combined_preds)} (should match combined_labels length: {len(combined_labels)})")
    except NameError:
        print(f"Combined predictions length: {len(combined_preds)}")
    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if gen_use_generated:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h" # Add 'h' for ablation/no_controlnet

    # Use simple naming if in simple ablation mode
    pred_key = "simple" if args.no_controlnet and args.simple else attributes
    if write_file:
        pred_dir = canonical_save_path("pred")
        os.makedirs(pred_dir, exist_ok=True)
        np.save(
            os.path.join(
                pred_dir,
                f"predictions_{args.dataset}_{args.model}_{pred_key}_{args.num_added_img}imgs{suffix}_{time_now}.npy",
            ),
            combined_preds
        )

    return combined_preds

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=20, test_loader=None):
    global predictions_for_diagnostics
    # Track best validation performance
    best_val_acc = 0.0
    best_model_state = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # For pretrained ViT and CLIP on ImageNet, freeze backbone for initial epochs
    freeze_epochs = 10
    if args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
        for name, param in model.named_parameters():
            # keep only classification head trainable
            if args.model == 'vit':
                if "head" not in name:
                    param.requires_grad = False
            else:  # clip
                if "classifier" not in name:
                    param.requires_grad = False
        if verbose:
            print(f"Froze {args.model} backbone parameters for initial {freeze_epochs} epochs")

    for epoch in range(num_epochs):
        # Unfreeze backbone after initial freeze_epochs for pretrained ImageNet CLIP/ViT
        if epoch == freeze_epochs and args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
            for param in model.parameters():
                param.requires_grad = True
            if verbose:
                print(f"Unfroze all {args.model} backbone parameters; now fine-tuning entire model")
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
            else:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                total_correct += ((torch.sigmoid(outputs) > 0.5).int() == labels.int()).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        val_acc, val_loss = evaluate(model, valid_loader)
        # Update best model if this epoch's validation accuracy is higher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            # if verbose:
            #     print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    if args.plot:
        epochs = range(1, num_epochs+1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0,0].plot(epochs, train_loss_history, marker='o')
        axes[0,0].set_title("Train Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")

        axes[0,1].plot(epochs, train_acc_history, marker='o')
        axes[0,1].set_title("Train Accuracy")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Accuracy")

        axes[1,0].plot(epochs, val_loss_history, marker='o')
        axes[1,0].set_title("Validation Loss")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")

        axes[1,1].plot(epochs, val_acc_history, marker='o')
        axes[1,1].set_title("Validation Accuracy")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Accuracy")

        plt.tight_layout()
        plot_path = canonical_save_path(
            f"plot/plot_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png",
            create_parent=True,
        )
        plt.savefig(plot_path)
        if args.dataset == 'imagenet':
            plt.savefig(f"./plot_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png")
        plt.close()

    # Load best or last model before saving predictions
    if args.save_pred_model == 'best' and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model checkpoint if requested
    if args.save_model:
        model_path = canonical_save_path(
            f"models/model_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.pth",
            create_parent=True,
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'args': vars(args),
        }, model_path)
        print(f"Model saved to {model_path}")

    # Evaluate on test set (when available) before saving predictions
    if test_loader is not None:
        test_acc, test_loss = evaluate(model, test_loader)
        print(f"Model Test ACC: {test_acc:.4f}, test loss: {test_loss:.4f}")

    predictions_for_diagnostics = save_predictions(
        model,
        train_loader,
        valid_loader,
        test_loader=test_loader,
        write_file=args.save_preds,
    )
    # Return training history for external use
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Function to parse log file (adapted from compare_acc.py)
def parse_log(log_path):
    """
    Parse the log file to extract (attribute, description, accuracy) tuples.
    """
    # 1：ACC: 0.893557  ... attribute "beard", description "No"
    pattern1 = re.compile(
        r'ACC:\s*(?P<acc>\d+\.\d+).*attribute\s+"(?P<attr>[^"]+)",\s*description\s+"(?P<desc>[^"]+)"'
    )
    # 2：The accuracy with "hair color" = "white" is
    pattern2 = re.compile(
        r'The accuracy with\s+"(?P<attr>[^"]+)"\s*=\s*"(?P<desc>[^"]+)"\s*is\s*(?P<acc>\d+\.\d+)'
    )
    pattern_overall = re.compile(
        r'(?:Model Test ACC|Held-out test accuracy|Model Validation ACC):\s*(?P<acc>\d+\.\d+)'
    )
    overall_acc = None

    results = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if overall_acc is None:
                    m0 = pattern_overall.search(line)
                    if m0:
                        overall_acc = float(m0.group('acc'))
                m1 = pattern1.search(line)
                if m1:
                    groups = m1.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
                    continue

                m2 = pattern2.search(line)
                if m2:
                    groups = m2.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
    except FileNotFoundError:
        print(f"Warning: Comparison log file not found at {log_path}")
        return [], None

    return results, overall_acc

# Start training with scheduler
# Print training start message and record total training time
# print(f"Starting training for {args.num_epochs} epochs...")
total_start_time = time.time()
train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=args.num_epochs,
    test_loader=test_loader,
)
total_elapsed_time = time.time() - total_start_time
print(f"Total training time for {args.num_epochs} epochs: {total_elapsed_time:.2f} seconds")
selection_time_label = "LLM selection" if args.use_llm_selection else "Active Learning selection"
print(f"Total {selection_time_label} time: {al_total_time:.2f} seconds")
overall_elapsed_time = time.time() - overall_start_time
print(f"End-to-end wall time (data prep + selection + training + setup): {overall_elapsed_time:.2f} seconds")



# Run the model diagnostics after predictions are saved
print("\nRunning model diagnostics on generated predictions...")

# Define thresholds for diagnostics (same for both datasets)
acc_diff_threshold = 0.01
distribution_diff_threshold = 0.3
dominant_label_in_failure_threshold = 0.8
dominant_label_in_prediction_threshold = 0.8
rare_label_threshold = 0.93
dominant_error_threshold = 0.3
error_cover_threshold = 0.9
top_error = 3

# Determine filename suffix for predictions
suffix = ""
if gen_use_generated:
    suffix += "_g"
if args.no_controlnet:
    suffix += "_h"  # Add 'h' for ablation/no_controlnet

# Load the predictions we just saved
prediction_file = str(
    canonical_save_path(
        f"pred/predictions_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}.npy"
    )
)
if not os.path.exists(prediction_file):
    if predictions_for_diagnostics is not None:
        print("Using in-memory predictions for diagnostics (--save_preds False).")
        prediction_file = None
        run_diagnostics = True
    else:
        print(f"Skipping model diagnostics because prediction file was not found: {prediction_file}")
        prediction_file = None
        run_diagnostics = False
else:
    run_diagnostics = True

if run_diagnostics:
    if prediction_file is not None:
        predictions = np.load(prediction_file)
    else:
        predictions = predictions_for_diagnostics

    # Prepare labels and split for diagnostics based on dataset
    if args.dataset == 'imagenet':
        # Prefer generated-mode labels if present, otherwise fall back to original split
        if diag_labels_imagenet is not None and diag_split_imagenet is not None:
            labels_to_diagnose = diag_labels_imagenet
            split = diag_split_imagenet
        else:
            full_ds = ImageFolder(data_dir)
            labels_all = np.array(full_ds.targets)
            expected_train = len(train_idx)
            expected_val = len(val_idx)
            expected_test = len(test_idx) if 'test_idx' in locals() else 0
            total_predictions = len(predictions)
            total_without_test = expected_train + expected_val
            total_with_test = total_without_test + expected_test

            if expected_test and total_predictions == total_with_test:
                labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx], labels_all[test_idx]])
                split = [TRAIN] * expected_train + [VALID] * expected_val + [TEST] * expected_test
            elif total_predictions == total_without_test:
                labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx]])
                split = [TRAIN] * expected_train + [VALID] * expected_val
                if expected_test:
                    print(
                        "Warning: Predictions file omits test samples (expected total "
                        f"{total_with_test}, got {total_predictions}). Diagnostics run on train+val only."
                    )
            else:
                print(
                    "Warning: Predictions length ({}) does not match expected counts ({}/{}).".format(
                        total_predictions, total_without_test, total_with_test
                    )
                )
                labels_to_diagnose = np.concatenate([labels_all[train_idx], labels_all[val_idx]])
                split = [TRAIN] * expected_train + [VALID] * expected_val
    elif args.dataset == 'celeba':
        # CelebA attribute metadata only covers the original images, not generated samples.
        # Saved predictions may include a generated-image segment, so strip it back out before diagnostics.
        orig_train_count = len(train_labels)
        valid_count = len(valid_labels)
        test_count = len(test_labels)
        unlabeled_count = len(unlabeled_labels)
        generated_count = len(modified_image_labels)

        labels_to_diagnose = np.concatenate((train_labels, valid_labels, test_labels, unlabeled_labels))
        split = (
            [TRAIN] * orig_train_count +
            [VALID] * valid_count +
            [TEST] * test_count +
            [UNLABELED] * unlabeled_count
        )

        expected_original_total = len(labels_to_diagnose)
        expected_with_generated = expected_original_total + generated_count

        if generated_count > 0 and len(predictions) == expected_with_generated:
            train_end = orig_train_count
            valid_end = train_end + valid_count
            test_end = valid_end + test_count
            generated_end = test_end + generated_count
            predictions = np.concatenate((predictions[:test_end], predictions[generated_end:]))
        elif len(predictions) != expected_original_total:
            print(
                "Warning: Skipping CelebA diagnostics because predictions length "
                f"({len(predictions)}) does not match expected original-only length "
                f"({expected_original_total}) or generated-inclusive length ({expected_with_generated})."
            )
            run_diagnostics = False

# Run diagnostics for both datasets (skip for any generated-only mode or combined AL mode)
if run_diagnostics and not gen_use_generated and not args.al_combined:
    # Instantiate ModelDiagnose and run diagnostics
    MD = ModelDiagnose(
        labels=labels_to_diagnose,
        predictions=predictions,
        split=split,
        attribute_dict_path=args.attribute_file,
        print=args.verbose,
        dataset=args.dataset,
        acc_diff_threshold=acc_diff_threshold,
        distribution_diff_threshold=distribution_diff_threshold,
        dominant_label_in_failure_threshold=dominant_label_in_failure_threshold,
        dominant_label_in_prediction_threshold=dominant_label_in_prediction_threshold,
        rare_label_threshold=rare_label_threshold,
        dominant_error_threshold=dominant_error_threshold,
        error_cover_threshold=error_cover_threshold,
        top_error=top_error
    )

    # Run diagnostics based on dataset
    if args.dataset == 'imagenet':
        # print("Start detect_failure_by_label\n")
        # MD.detect_failure_by_label()
        print("Start detect_prediction_correlation\n")
        MD.detect_prediction_correlation()
        # print("Start detect_failure_prediction_correlation\n")
        # MD.detect_failure_prediction_correlation()
    elif args.dataset == 'celeba':
        print("Start detect_failure_by_label\n")
        MD.detect_failure_by_label()
        print("Start detect_prediction_correlation\n")
        MD.detect_prediction_correlation()
        print("Start detect_failure_prediction_correlation\n")
        MD.detect_failure_prediction_correlation()
        print("Model diagnostics complete.")

    # Optionally save a report
    if args.save_report:
        report_dir = canonical_save_path("reports")
        os.makedirs(report_dir, exist_ok=True)
        MD.generate_report(f"{report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")
        print(f"Diagnostic report saved to {report_dir}/{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs{suffix}_{time_now}")

    # --- Accuracy Comparison Logic ---
    if gen_use_generated:
        print("Skipping accuracy comparison due to generated-only mode")
    else:
        if compare_log_path:
            print(f"\nComparing accuracy with log file: {compare_log_path}")
            orig_entries, orig_overall = parse_log(compare_log_path)
            if orig_entries is not None:
                orig_map = {(e['attribute'], e['description']): e['accuracy'] for e in orig_entries}
                current_overall = MD.valid_acc
                if orig_overall is not None and current_overall is not None:
                    if current_overall > orig_overall:
                        overall_status = 'improved'
                    elif current_overall < orig_overall:
                        overall_status = 'declined'
                    else:
                        overall_status = 'same'
                    print(f'Overall accuracy: {orig_overall:.6f} -> {current_overall:.6f}: {overall_status}')
                elif current_overall is not None:
                     print(f'Current overall accuracy: {current_overall:.6f} (Original overall accuracy not found in log)')
                else:
                     print(f'Original overall accuracy: {orig_overall:.6f} (Current overall accuracy not available)')

                print("\nAttribute-specific accuracy comparison:")
                current_compared_count = 0
                for (attribute, description), orig_acc in orig_map.items():
                    current_acc = None
                    if attribute in MD.attributes and description in MD.attributes[attribute]['valid acc']:
                        current_acc = MD.attributes[attribute]['valid acc'][description]
                    if current_acc is None:
                        continue
                    if current_acc > orig_acc:
                        status = 'improved'
                    elif current_acc < orig_acc:
                        status = 'declined'
                    else:
                        status = 'same'
                    print(f'{attribute}, {description}, {orig_acc:.6f} -> {current_acc:.6f}: {status}')
                    current_compared_count += 1

                print(f"\nCompared {current_compared_count} attribute-specific accuracies.")
            else:
                print(f"Could not parse or find the comparison log file: {compare_log_path}")
        else:
            print("\nSkipping accuracy comparison: No comparison log file specified for the selected model or file not found.")
else:
    if args.al_combined:
        print("Skipping model diagnostics due to --al_combined mode (data indices are misaligned)")
    else:
        print("Skipping model diagnostics due to generated-only mode")
    print("Skipping accuracy comparison due to skipped diagnostics")

# Continue with the rest of the script regardless of diagnostics
if not gen_use_generated and not args.al_combined:

        # --- Active Learning vs Random Selection Comparison ---
        if args.use_active_learning and args.sub_ratio < 1.0:
            # Construct the random selection log file path based on current configuration

            log_name1 = f"AL_{args.model}_{args.dataset}_random_{args.num_added_img}_{args.sub_ratio}.log"
            log_name2 = f"AL_random_{args.model}_{args.dataset}_{args.num_added_img}_ratio{args.sub_ratio}.log"
            # AL_random_resnet_imagenet_1000_ratio${SUB_RATIO}.log



            log_name1 = f"AL_{args.model}_{args.dataset}_random_{args.num_added_img}.log"
            log_name2 = f"AL_random_{args.model}_{args.dataset}_{args.num_added_img}_ratio{args.sub_ratio}.log"

            if args.dataset == "imagenet":
                log_dir = f"logs_{args.dataset}"
            elif args.dataset == "celeba":
                log_dir = "logs"

            log_path1 = os.path.join(log_dir, log_name1)
            log_path2 = os.path.join(log_dir, log_name2)

            random_log_path = None
            if os.path.exists(log_path2):
                random_log_path = log_path2
            elif os.path.exists(log_path1):
                random_log_path = log_path1
            else:
                print(
                    "Skipping Active Learning vs Random comparison: "
                    f"no log file found at {log_path1} or {log_path2}"
                )

            if random_log_path is not None:
                print(f"Using log file: {random_log_path}")
                print(f"\n" + "="*80)
                print(f"ACTIVE LEARNING vs RANDOM SELECTION COMPARISON")
                print(f"="*80)
                print(f"Comparing current Active Learning results with random selection: {random_log_path}")
                
                # Parse random selection log
                random_entries, random_overall = parse_log(random_log_path)
                
                if random_entries is not None:
                    random_map = {(e['attribute'], e['description']): e['accuracy'] for e in random_entries}
                    # Use current running results from MD (not from log file)
                    current_overall = MD.valid_acc
                    
                    # Overall accuracy comparison
                    if random_overall is not None and current_overall is not None:
                        overall_diff = current_overall - random_overall
                        if current_overall > random_overall:
                            overall_status = 'ACTIVE LEARNING WINS'
                        elif current_overall < random_overall:
                            overall_status = 'RANDOM SELECTION WINS'
                        else:
                            overall_status = 'TIE'
                        print(f'Overall accuracy: Random {random_overall:.6f} -> Active Learning {current_overall:.6f} (Δ={overall_diff:+.6f}): {overall_status}')
                    elif current_overall is not None:
                        print(f'Current Active Learning overall accuracy: {current_overall:.6f} (Random overall accuracy not found)')
                    else:
                        print(f'Random overall accuracy: {random_overall:.6f} (Current Active Learning accuracy not available)')

                    # Attribute-specific comparison
                    print("\nAttribute-specific Active Learning vs Random Selection comparison:")
                    al_wins = 0
                    random_wins = 0
                    ties = 0
                    al_vs_random_count = 0
                    
                    for (attribute, description), random_acc in random_map.items():
                        # Extract current AL result from MD.attributes (not from log file)
                        current_acc = None
                        if attribute in MD.attributes and description in MD.attributes[attribute]['valid acc']:
                            current_acc = MD.attributes[attribute]['valid acc'][description]
                        if current_acc is None:
                            continue
                        
                        diff = current_acc - random_acc
                        if current_acc > random_acc:
                            status = 'AL WINS'
                            al_wins += 1
                        elif current_acc < random_acc:
                            status = 'RANDOM WINS'
                            random_wins += 1
                        else:
                            status = 'TIE'
                            ties += 1
                        
                        print(f'{attribute}, {description}, Random {random_acc:.6f} -> AL {current_acc:.6f} (Δ={diff:+.6f}): {status}')
                        al_vs_random_count += 1

                    print(f"\n" + "-"*60)
                    print(f"ACTIVE LEARNING vs RANDOM SELECTION SUMMARY:")
                    print(f"Total compared attributes: {al_vs_random_count}")
                    print(f"Active Learning wins: {al_wins}")
                    print(f"Random Selection wins: {random_wins}")
                    print(f"Ties: {ties}")
                    if al_vs_random_count > 0:
                        al_win_rate = al_wins / al_vs_random_count * 100
                        print(f"Active Learning win rate: {al_win_rate:.1f}%")
                    print(f"-"*60)
                else:
                    print(f"Could not parse the random selection log file: {random_log_path}")
            else:
                print(f"\nRandom selection comparison skipped: {random_log_path} not found")
else:
    # Special case for generated-only mode or al_combined mode
    print("Skipping model diagnostics")
    if len(val_acc_history) > 0:
        final_val_acc = val_acc_history[-1]
        print("\n" + "="*60)
        print(f"Final Epoch Validation Accuracy: {final_val_acc:.4f}")
        print("="*60 + "\n")

print("Current time:", time_now)
