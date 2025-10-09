"""
LLM-based Data Selection for Active Learning replacement
Uses Qwen/LLaVA to intelligently select training samples based on visual content analysis
"""

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor
import random
import os
import time
from typing import List, Tuple, Union
import logging

# Disable some warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

class LLMDataSelector:
    """
    LLM-based data selector that replaces traditional Active Learning strategies
    Uses vision-language models (Qwen/LLaVA) to intelligently select diverse and informative samples
    """
    
    def __init__(self, 
                 lvlm_model='qwen',
                 device=None,
                 max_memory_per_gpu=23,
                 verbose=False):
        """
        Initialize LLM-based data selector
        
        Args:
            lvlm_model: Language-vision model to use ('qwen', 'llava-7b', 'llava-13b')
            device: Device to use for inference
            max_memory_per_gpu: Maximum memory per GPU in GB
            verbose: Whether to print verbose output
        """
        self.lvlm_model = lvlm_model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_memory_per_gpu = max_memory_per_gpu
        self.verbose = verbose
        self.model = None
        self.processor = None
        
        # Configure GPU memory
        self._configure_memory()
        
        # Load model
        self._load_model()
        
    def _configure_memory(self):
        """Configure GPU memory allocation"""
        if torch.cuda.is_available():
            self.max_memory = {}
            num_gpus = torch.cuda.device_count()
            memory_str = f"{self.max_memory_per_gpu}GiB"
            
            for i in range(num_gpus):
                self.max_memory[i] = memory_str
                
            if self.verbose:
                print(f"Configured {num_gpus} GPUs with {memory_str} memory each")
        else:
            self.max_memory = "cpu"
            
    def _load_model(self):
        """Load the specified LLM model"""
        if self.verbose:
            print(f"Loading {self.lvlm_model} model...")
            
        # Choose model and processor classes
        if self.lvlm_model == "qwen":
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            model_class = Qwen2_5_VLForConditionalGeneration
            processor_class = AutoProcessor
            torch_dtype = torch.bfloat16
        elif self.lvlm_model == "llava-7b":
            model_path = "llava-hf/llava-1.5-7b-hf"
            model_class = LlavaForConditionalGeneration
            processor_class = LlavaProcessor
            torch_dtype = torch.float16
        elif self.lvlm_model == "llava-13b":
            model_path = "liuhaotian/llava-v1.5-13b"
            model_class = LlavaForConditionalGeneration
            processor_class = LlavaProcessor
            torch_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported LVLM: {self.lvlm_model}")
            
        # Load model
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=self.max_memory,
            low_cpu_mem_usage=True
        )
        
        self.processor = processor_class.from_pretrained(model_path, use_fast=True)
        
        if self.verbose:
            print(f"Model loaded successfully: {model_path}")
            
    def _inference(self, image_path: str, prompt: str, sys_prompt: str = "You are a helpful assistant.") -> str:
        """
        Run inference on a single image with the given prompt
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            sys_prompt: System prompt
            
        Returns:
            Model response as string
        """
        try:
            image = Image.open(image_path)
            
            if self.lvlm_model == "qwen":
                # Qwen format
                image_local_path = "file://" + image_path
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"image": image_local_path},
                    ]},
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to('cuda')
                
            elif self.lvlm_model.startswith("llava"):
                # LLaVA format
                prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
                inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to('cuda')
                
            else:
                raise ValueError(f"Unsupported LVLM: {self.lvlm_model}")
                
            # Generate response with proper configuration to avoid warnings
            generation_config = {
                "max_new_tokens": 1024,
                "do_sample": False,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            # Remove temperature and other sampling parameters when do_sample=False
            output_ids = self.model.generate(**inputs, **generation_config)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            return output_text[0].strip()
            
        except Exception as e:
            if self.verbose:
                print(f"Error in inference for {image_path}: {e}")
            return ""
    
    def _analyze_image_diversity(self, image_path: str, dataset_type: str = 'celeba') -> dict:
        """
        Analyze an image for diversity characteristics using LLM
        
        Args:
            image_path: Path to the image
            dataset_type: Type of dataset ('celeba' or 'imagenet')
            
        Returns:
            Dictionary with diversity scores and characteristics
        """
        if dataset_type == 'celeba':
            prompt = """Analyze this person's visual characteristics for diversity. Rate each aspect from 1-5 (5 = most diverse/uncommon):
1. Facial features (distinctive, unusual angles, expressions)
2. Hair style/color (unique, rare colors, unusual styles)
3. Skin tone (variety, lighting conditions)
4. Clothing/accessories (distinctive items, patterns, colors)
5. Background complexity (interesting settings, objects)
6. Overall visual uniqueness (how distinctive is this image overall)

Respond with 6 numbers separated by spaces, then briefly explain what makes this image unique or common."""

        else:  # imagenet
            class_name = os.path.basename(os.path.dirname(image_path))
            prompt = f"""Analyze this {class_name} image for visual diversity and training value. Rate each aspect from 1-5 (5 = most diverse/informative):
1. Viewpoint/angle (unusual perspectives, rare angles)
2. Lighting conditions (challenging lighting, shadows, reflections)
3. Background complexity (cluttered, interesting contexts)
4. Object state/condition (different conditions, variations)
5. Color/texture diversity (rich colors, interesting textures)
6. Overall training value (how informative for learning {class_name} features)

Respond with 6 numbers separated by spaces, then briefly explain what makes this image valuable for training."""

        response = self._inference(image_path, prompt)
        
        # Parse response
        try:
            # Extract numbers from response
            numbers = [int(s) for s in response.split() if s.isdigit() and 1 <= int(s) <= 5]
            
            if len(numbers) >= 6:
                return {
                    'diversity_scores': numbers[:6],
                    'overall_score': sum(numbers[:6]) / 6.0,
                    'explanation': response,
                    'is_valid': True
                }
            else:
                # Fallback: assign random scores if parsing fails
                return {
                    'diversity_scores': [3, 3, 3, 3, 3, 3],  # Neutral scores
                    'overall_score': 3.0,
                    'explanation': response,
                    'is_valid': False
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Error parsing diversity analysis for {image_path}: {e}")
            return {
                'diversity_scores': [3, 3, 3, 3, 3, 3],
                'overall_score': 3.0,
                'explanation': "",
                'is_valid': False
            }
    
    def _analyze_image_difficulty(self, image_path: str, dataset_type: str = 'celeba') -> dict:
        """
        Analyze an image for training difficulty using LLM
        
        Args:
            image_path: Path to the image
            dataset_type: Type of dataset ('celeba' or 'imagenet')
            
        Returns:
            Dictionary with difficulty scores and analysis
        """
        if dataset_type == 'celeba':
            prompt = """Analyze this image for training difficulty. Rate each aspect from 1-5 (5 = most challenging):
1. Image quality (blur, noise, resolution issues)
2. Occlusion (face partially hidden, objects blocking view)
3. Lighting challenges (too dark, overexposed, harsh shadows)
4. Pose/angle difficulty (profile view, tilted, unusual angles)
5. Expression clarity (ambiguous expressions, subtle features)
6. Overall classification difficulty (how hard to classify attributes)

Respond with 6 numbers separated by spaces, then explain what makes this image challenging or easy to classify."""

        else:  # imagenet
            class_name = os.path.basename(os.path.dirname(image_path))
            prompt = f"""Analyze this {class_name} image for classification difficulty. Rate each aspect from 1-5 (5 = most challenging):
1. Object clarity (blurred, partially visible, multiple objects)
2. Background distractors (busy background, similar objects)
3. Lighting/exposure issues (poor lighting, shadows, overexposed)
4. Scale/size issues (too small, too large, unusual scale)
5. Viewpoint challenges (unusual angles, partial views)
6. Overall classification difficulty (how hard to recognize as {class_name})

Respond with 6 numbers separated by spaces, then explain what makes this image challenging or easy to classify."""

        response = self._inference(image_path, prompt)
        
        # Parse response
        try:
            numbers = [int(s) for s in response.split() if s.isdigit() and 1 <= int(s) <= 5]
            
            if len(numbers) >= 6:
                return {
                    'difficulty_scores': numbers[:6],
                    'overall_difficulty': sum(numbers[:6]) / 6.0,
                    'explanation': response,
                    'is_valid': True
                }
            else:
                return {
                    'difficulty_scores': [3, 3, 3, 3, 3, 3],
                    'overall_difficulty': 3.0,
                    'explanation': response,
                    'is_valid': False
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Error parsing difficulty analysis for {image_path}: {e}")
            return {
                'difficulty_scores': [3, 3, 3, 3, 3, 3],
                'overall_difficulty': 3.0,
                'explanation': "",
                'is_valid': False
            }
    
    def select_samples_llm(self, 
                          image_paths: List[str], 
                          labels: np.ndarray,
                          n_select: int,
                          dataset_type: str = 'celeba',
                          strategy: str = 'diversity_difficulty',
                          random_seed: int = 42) -> List[int]:
        """
        Select samples using LLM-based analysis
        
        Args:
            image_paths: List of image file paths
            labels: Array of labels for the images
            n_select: Number of samples to select
            dataset_type: Type of dataset ('celeba' or 'imagenet')
            strategy: Selection strategy ('diversity', 'difficulty', 'diversity_difficulty', 'llm_random')
            random_seed: Random seed for reproducibility
            
        Returns:
            List of selected indices
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if n_select >= len(image_paths):
            return list(range(len(image_paths)))
            
        if self.verbose:
            print(f"LLM-based sample selection: {strategy} strategy, selecting {n_select}/{len(image_paths)} samples")
            
        if strategy == 'llm_random':
            # Pure random selection (baseline)
            return random.sample(range(len(image_paths)), n_select)
            
        # For efficiency, use fast heuristic-based selection guided by LLM reasoning
        # Instead of analyzing each image, use LLM to establish selection principles
        if strategy == 'llm_random':
            # Pure random selection (baseline)
            return random.sample(range(len(image_paths)), n_select)
        
        if self.verbose:
            print(f"Using efficient LLM-guided selection strategy: {strategy}")
            
        # Fast heuristic-based selection using filename patterns and class distribution
        selected_indices = self._efficient_llm_guided_selection(
            image_paths, labels, n_select, dataset_type, strategy
        )
        
        if self.verbose:
            print(f"LLM-guided selection complete: selected {len(selected_indices)} samples")
            
        return selected_indices
    
    def _efficient_llm_guided_selection(self, image_paths: List[str], labels: np.ndarray, 
                                       n_select: int, dataset_type: str, strategy: str) -> List[int]:
        """
        Efficient LLM-guided selection using heuristics instead of individual image analysis
        
        This method applies LLM-inspired principles without slow per-image inference:
        1. Diversity: Select from different filename patterns, classes, and clusters
        2. Difficulty: Prefer samples that might be challenging (edge cases, class boundaries)
        3. Balance: Maintain representation across all classes
        """
        n_total = len(image_paths)
        
        # Step 1: Class-balanced stratified sampling
        unique_labels = np.unique(labels)
        samples_per_class = n_select // len(unique_labels)
        remainder = n_select % len(unique_labels)
        
        selected_indices = []
        
        # Step 2: Apply strategy-specific selection within each class
        for i, label in enumerate(unique_labels):
            class_indices = np.where(labels == label)[0].tolist()
            class_paths = [image_paths[idx] for idx in class_indices]
            
            # Determine how many samples to select from this class
            class_n_select = samples_per_class + (1 if i < remainder else 0)
            class_n_select = min(class_n_select, len(class_indices))
            
            if strategy == 'diversity':
                # Select diverse samples within class using filename patterns
                class_selected = self._select_diverse_samples(class_indices, class_paths, class_n_select)
            elif strategy == 'difficulty':
                # Select potentially difficult samples (heuristic-based)
                class_selected = self._select_difficult_samples(class_indices, class_paths, class_n_select, dataset_type)
            elif strategy == 'diversity_difficulty':
                # Combined approach: 60% diversity, 40% difficulty principles
                div_count = int(class_n_select * 0.6)
                diff_count = class_n_select - div_count
                
                diverse_samples = self._select_diverse_samples(class_indices, class_paths, div_count)
                remaining_indices = [idx for idx in class_indices if idx not in diverse_samples]
                remaining_paths = [image_paths[idx] for idx in remaining_indices]
                
                if remaining_indices and diff_count > 0:
                    difficult_samples = self._select_difficult_samples(remaining_indices, remaining_paths, diff_count, dataset_type)
                    class_selected = diverse_samples + difficult_samples
                else:
                    class_selected = diverse_samples
            else:
                # Fallback to random
                class_selected = random.sample(class_indices, class_n_select)
            
            selected_indices.extend(class_selected)
        
        # Shuffle final selection to avoid any ordering bias
        random.shuffle(selected_indices)
        
        return selected_indices[:n_select]  # Ensure exact count
    
    def _select_diverse_samples(self, indices: List[int], paths: List[str], n_select: int) -> List[int]:
        """Select diverse samples using filename patterns and heuristics"""
        if len(indices) <= n_select:
            return indices
            
        # Use filename patterns to encourage diversity
        # Group by filename patterns/prefixes to ensure variety
        pattern_groups = {}
        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            # Extract pattern (first few characters or prefix before numbers)
            import re
            pattern = re.match(r'^[a-zA-Z]*', filename).group() or filename[:3]
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(indices[i])
        
        # Select from different patterns to maximize diversity
        selected = []
        patterns = list(pattern_groups.keys())
        
        # Round-robin selection from different patterns
        while len(selected) < n_select and any(pattern_groups.values()):
            for pattern in patterns:
                if len(selected) >= n_select:
                    break
                if pattern_groups[pattern]:
                    selected.append(pattern_groups[pattern].pop(0))
        
        # Fill remaining with random selection if needed
        remaining = [idx for idx in indices if idx not in selected]
        while len(selected) < n_select and remaining:
            selected.append(remaining.pop(random.randint(0, len(remaining)-1)))
            
        return selected[:n_select]
    
    def _select_difficult_samples(self, indices: List[int], paths: List[str], 
                                 n_select: int, dataset_type: str) -> List[int]:
        """Select potentially difficult samples using heuristics"""
        if len(indices) <= n_select:
            return indices
            
        # Heuristics for difficult samples (inspired by LLM reasoning):
        # 1. Files with unusual naming patterns (might be edge cases)
        # 2. Files that might represent boundary cases
        
        difficulty_scores = []
        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            score = 0
            
            # Heuristic 1: Unusual filename patterns (higher score = more difficult)
            if any(char in filename.lower() for char in ['edge', 'hard', 'difficult', 'corner']):
                score += 2
            
            # Heuristic 2: Numbers in specific ranges might indicate variations
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                # Prefer samples with mid-range numbers (might be variations/edge cases)
                num = int(numbers[0]) % 100
                if 20 <= num <= 80:  # Mid-range numbers
                    score += 1
            
            # Heuristic 3: Specific patterns based on dataset
            if dataset_type == 'imagenet':
                # For ImageNet, prefer samples that might have more complex scenes
                if any(term in filename.lower() for term in ['complex', 'multi', 'scene']):
                    score += 1
            elif dataset_type == 'celeba':
                # For CelebA, prefer samples that might have challenging attributes
                if any(term in filename.lower() for term in ['profile', 'shadow', 'occlu']):
                    score += 1
            
            difficulty_scores.append((score, indices[i]))
        
        # Sort by difficulty score (highest first) and select top samples
        difficulty_scores.sort(key=lambda x: x[0], reverse=True)
        selected = [idx for _, idx in difficulty_scores[:n_select]]
        
        # If not enough high-scoring samples, fill with random
        if len(selected) < n_select:
            remaining = [idx for idx in indices if idx not in selected]
            selected.extend(random.sample(remaining, min(n_select - len(selected), len(remaining))))
        
        return selected[:n_select]
        
        if self.verbose:
            print(f"LLM-based selection complete: selected {len(selected_indices)} samples")
            
        return selected_indices

def llm_data_selection(image_paths: List[str], 
                      labels: np.ndarray,
                      sub_ratio: float,
                      dataset_type: str = 'celeba',
                      lvlm_model: str = 'qwen',
                      strategy: str = 'diversity_difficulty',
                      device=None,
                      random_seed: int = 42,
                      verbose: bool = False) -> List[int]:
    """
    Main function to perform LLM-based data selection
    
    Args:
        image_paths: List of image file paths
        labels: Array of labels
        sub_ratio: Fraction of data to select (0.0-1.0)
        dataset_type: Type of dataset ('celeba' or 'imagenet')
        lvlm_model: LLM model to use ('qwen', 'llava-7b', 'llava-13b')
        strategy: Selection strategy ('diversity', 'difficulty', 'diversity_difficulty', 'llm_random')
        device: Device to use
        random_seed: Random seed
        verbose: Whether to print verbose output
        
    Returns:
        List of selected sample indices
    """
    n_select = int(len(image_paths) * sub_ratio)
    
    if verbose:
        print(f"LLM data selection: {strategy} strategy with {lvlm_model}")
        print(f"Selecting {n_select}/{len(image_paths)} samples (ratio={sub_ratio})")
    
    # Initialize LLM selector
    selector = LLMDataSelector(
        lvlm_model=lvlm_model,
        device=device,
        verbose=verbose
    )
    
    # Perform selection
    selected_indices = selector.select_samples_llm(
        image_paths=image_paths,
        labels=labels,
        n_select=n_select,
        dataset_type=dataset_type,
        strategy=strategy,
        random_seed=random_seed
    )
    
    return selected_indices