"""
Common imports for the safefix project.
This module consolidates all frequently used imports to reduce redundancy.
"""

# Standard Library
import os
import json
import argparse
import datetime
import time
import random
import re
import copy
import warnings
from pathlib import Path

# Configure warnings for better user experience
warnings.filterwarnings(
    "ignore",
    "`resume_download` is deprecated and will be removed in version 1.0.0",
    FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
    module="torchvision.models._utils",
)
warnings.filterwarnings(
    "ignore",
    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning,
    module="torchvision.models._utils",
)

# Core PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

# Transformers and Vision Models
from transformers import (
    ViTModel, 
    ViTFeatureExtractor,
    CLIPImageProcessor,
    CLIPModel,
    CLIPConfig
)
try:
    from timm import create_model
except ModuleNotFoundError:
    create_model = None

# Computer Vision and Image Processing
from PIL import Image

# Data Science and ML
import numpy as np
# Machine Learning and Data Science
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from itertools import combinations

# Plotting and Visualization
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Project-specific imports
from utils.defines import *
from utils.stat import ModelDiagnose
from processData.report import ReportGenerator

# Suppress common warnings
warnings.filterwarnings(
    "ignore",
    "The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers",
    FutureWarning
)
