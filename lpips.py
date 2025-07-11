import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys, os
# Ensure the installed lpips package is imported, not this script itself
sys.path.remove(os.path.dirname(__file__))
import warnings
# Suppress torchvision pretrained-weights deprecation warnings
warnings.filterwarnings("ignore", 
    message=".*The parameter 'pretrained' is deprecated.*", 
    category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", 
    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*", 
    category=UserWarning, module="torchvision.models._utils")

import contextlib
import io

import lpips
from pytorch_fid.fid_score import calculate_frechet_distance
import clip
from pytorch_fid.inception import InceptionV3
from numpy.linalg import pinv, slogdet
import torch.nn.functional as F
import argparse

# Helper to collect all images recursively
def collect_image_files(root_dir, exts=('png','jpg','jpeg','bmp')):
    files = []
    for root, _, fnames in os.walk(root_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return files


# Parse command-line arguments for dataset selection
parser = argparse.ArgumentParser(description="Compute diversity and distribution metrics")
parser.add_argument("--dataset", type=str, choices=["imagenet", "celeba"], default="imagenet",
                    help="Dataset to evaluate: 'imagenet' or 'celeba'")
parser.add_argument("--attributes", type=str, default=None,
                    help="Underscore-separated attribute values for SafeFix folder (e.g., 'pinkcolor_fabrictexture')")
args = parser.parse_args()

# Directories for generated images based on dataset selection
if args.dataset == "imagenet":
    hibug_dir = "exampleData/imagenet_simple"
    if args.attributes:
        safefix_dir = f"exampleData/imagenet_{args.attributes}"
    else:
        safefix_dir = "exampleData/imagenet_pinkcolor_fabrictexture"
    real_dir = "imagenet_subset"
elif args.dataset == "celeba":
    hibug_dir = "exampleData/hibug_simple"
    if args.attributes:
        safefix_dir = f"exampleData/{args.attributes}"
    else:
        safefix_dir = "exampleData/redhair_brownskin_sademotion"
    real_dir = "/data/oxx220000/img_align_celeba"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate FID model
fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

def load_images(path, limit=None):
    imgs = []
    files = collect_image_files(path)
    for i, p in enumerate(sorted(files)):
        if limit and i >= limit:
            break
        try:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img).unsqueeze(0).to(device))
        except:
            continue
    return imgs

# Load a subset for LPIPS
imgs_hibug = load_images(hibug_dir, limit=50)
imgs_safefix = load_images(safefix_dir, limit=50)

# 1. Diversity: LPIPS
with contextlib.redirect_stdout(io.StringIO()):
    lpips_fn = lpips.LPIPS(net='alex').to(device)

def avg_lpips(imgs):
    n = len(imgs)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(lpips_fn(imgs[i], imgs[j]).item())
    return np.mean(dists)

div_h = avg_lpips(imgs_hibug)
div_s = avg_lpips(imgs_safefix)
print(f"LPIPS Diversity - HiBug: {div_h:.4f}, SafeFix: {div_s:.4f}")

# Inception preprocessing for FID
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor, Compose

inception_transform = Compose([
    Resize(299),
    CenterCrop(299),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
])

def compute_activations(paths, model, device):
    activations = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            img_t = inception_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                act = model(img_t)[0].cpu().numpy().flatten()
            activations.append(act)
        except:
            continue
    acts = np.stack(activations, axis=0)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

# 2. FID: Real vs. generated sets
# Compute FID between real images and each generated set


# 3. Attribute consistency (CLIP)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
def clip_consistency(path, attribute, limit=100):
    scores = []
    text = clip.tokenize([f"A photo of an object with {attribute}"]).to(device)
    txt_emb = clip_model.encode_text(text).float()
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    files = collect_image_files(path)
    for i, p in enumerate(sorted(files)):
        if i >= limit:
            break
        img = Image.open(p).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = clip_model.encode_image(img_t).float()
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            # cosine similarity between image and text embeddings
            score = F.cosine_similarity(img_emb, txt_emb).item()
        scores.append(score)
    return np.mean(scores)

cons_h = clip_consistency(hibug_dir, "hibug color")
cons_s = clip_consistency(safefix_dir, "safefix color")
print(f"CLIP Consistency - HiBug: {cons_h:.4f}, SafeFix: {cons_s:.4f}")

# 4. Feature distribution KL divergence
def feat_stats(path, limit=200):
    feats = []
    files = collect_image_files(path)
    if not files:
        # No images found; return zero mean and identity covariance of correct feature dimension
        dim = clip_model.visual.output_dim
        return np.zeros(dim), np.eye(dim)
    for i, p in enumerate(sorted(files)):
        if i >= limit:
            break
        try:
            img = Image.open(p).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats.append(clip_model.encode_image(img_t).cpu().numpy().flatten())
        except:
            continue
    feats = np.stack(feats)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)

mu_h, cov_h = feat_stats(hibug_dir)
mu_s, cov_s = feat_stats(safefix_dir)

 # 5. Real images baseline
# Collect real image paths and compute Inception activations
files_r = collect_image_files(real_dir)
mu_r_incep, cov_r_incep = compute_activations(files_r, fid_model, device)
print(f"FID baseline computed on real images: mean dim {mu_r_incep.shape[0]}")

# FID: real vs. HiBug
files_h = collect_image_files(hibug_dir)
m_h, s_h = compute_activations(files_h, fid_model, device)
fid_h = calculate_frechet_distance(mu_r_incep, cov_r_incep, m_h, s_h)
print(f"FID (Real->HiBug): {fid_h:.4f}")

# FID: real vs. SafeFix
files_s = collect_image_files(safefix_dir)
m_s, s_s = compute_activations(files_s, fid_model, device)
fid_s = calculate_frechet_distance(mu_r_incep, cov_r_incep, m_s, s_s)
print(f"FID (Real->SafeFix): {fid_s:.4f}")


# 6. Real images CLIP-based stats for KL
mu_r_clip, cov_r_clip = feat_stats(real_dir)
print(f"CLIP stats dims: mean {mu_r_clip.shape[0]}, cov {cov_r_clip.shape}")

# Compare KL divergence between real and generated feature distributions
def kl_div(mu1, cov1, mu2, cov2, eps=1e-6):
    # Regularize covariances
    dim = mu1.shape[0]
    cov1_reg = cov1 + eps * np.eye(dim)
    cov2_reg = cov2 + eps * np.eye(dim)
    # Compute pseudo-inverse of cov2
    inv_cov2 = pinv(cov2_reg)
    diff = mu2 - mu1
    # Trace term
    term1 = np.trace(inv_cov2 @ cov1_reg)
    # Quadratic term
    term2 = diff.T @ inv_cov2 @ diff
    # Log-determinant term via slogdet
    sign1, logdet1 = slogdet(cov1_reg)
    sign2, logdet2 = slogdet(cov2_reg)
    term3 = logdet2 - logdet1
    k = dim
    return 0.5 * (term1 + term2 - k + term3)
    
kl_real_hibug = kl_div(mu_r_clip, cov_r_clip, mu_h, cov_h)
kl_real_safefix = kl_div(mu_r_clip, cov_r_clip, mu_s, cov_s)
print(f"KL Divergence (Real->HiBug): {kl_real_hibug:.4f}")
print(f"KL Divergence (Real->SafeFix): {kl_real_safefix:.4f}")


kl_value = kl_div(mu_h, cov_h, mu_s, cov_s)
print(f"KL Divergence: {kl_value:.4f}")

# Print results
# print(f"LPIPS Diversity - HiBug: {div_h:.4f}, SafeFix: {div_s:.4f}")
# print(f"FID between HiBug and SafeFix: {fid_value:.4f}")
# print(f"CLIP Consistency - HiBug: {cons_h:.4f}, SafeFix: {cons_s:.4f}")
# print(f"KL Divergence: {kl_value:.4f}")