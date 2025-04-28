"""
sd15.py
Stable Diffusion v1.5 inference script using Hugging Face diffusers.
All code comments are in English.
"""

from diffusers import StableDiffusionPipeline
import torch
import itertools
import random
import datetime

def generate_images(prompt: str, num_images: int = 1, output_prefix: str = "output"):
    """
    Generate images using Stable Diffusion v1.5.

    Args:
        prompt (str): The text prompt for image generation.
        num_images (int): Number of images to generate.
        output_prefix (str): Prefix for output image filenames.
    """
    # Load the Stable Diffusion v1.5 pipeline with fp16 precision (revision removed)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    # Authentication is managed via `huggingface-cli login` or HF_TOKEN env var; no need for explicit use_auth_token
    pipeline = pipeline.to("cuda")

    # Perform image generation
    result = pipeline(
        prompt,
        num_inference_steps=50,         # Number of diffusion steps
        num_images_per_prompt=num_images
    )

    # Save the single generated image
    img = result.images[0]
    filename = f"{output_prefix}.png"
    img.save(filename)
    print(f"Saved image: {filename}")

# Define attribute tags for prompt combinations
attributes = {
    "gender": ["female", "male"],
    "hair color": [
        "white", "yellow", "blonde", "black", "gray", "brown", "red"
    ],
    "skin": ["white", "black", "brown"],
    "lipstick": ["wearing lipstick", "without lipstick"]
}

def get_random_ordered_combinations(attrs: dict) -> list:
    """
    Generate all attribute tag combinations and return them in a random order.
    """
    keys = list(attrs.keys())
    # Create Cartesian product of all tag lists
    combos = list(itertools.product(*(attrs[k] for k in keys)))
    # Shuffle to randomize order
    random.shuffle(combos)
    # Convert each tuple to a dict mapping attribute names to tags
    return [dict(zip(keys, combo)) for combo in combos]

if __name__ == "__main__":
    # Generate a random sequence of attribute combinations
    combos = get_random_ordered_combinations(attributes)
    # Select the first combination from the randomized list
    combo = combos[0]
    # Build prompt from selected tags, including lipstick presence/absence
    user_prompt = (
        f"A {combo['gender']} with {combo['hair color']} hair "
        f"and {combo['skin']} skin {combo['lipstick']}"
    )
    # Get current timestamp
    time_now = datetime.datetime.now().strftime("%m%d%H%M%S")
    print("Current time:", time_now)
    # Sanitize lipstick tag for filenames
    lipstick_tag = combo['lipstick'].replace(" ", "_")
    # Build output prefix including attributes and timestamp
    output_prefix = f"{combo['gender']}_{combo['hair color']}_{combo['skin']}_{lipstick_tag}_{time_now}"
    batch_size = 1  # default batch size
    generate_images(user_prompt, num_images=batch_size, output_prefix=output_prefix)
