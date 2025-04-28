#!/usr/bin/env python3
import argparse, os, torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import re # Import the regular expression module for filename sanitization

def sanitize_filename(text):
    """Removes or replaces characters invalid for filenames."""
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    # Replace spaces with underscores (optional, but common)
    sanitized = sanitized.replace(" ", "_")
    # Limit length if necessary (optional)
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized

def parse_args():
    p = argparse.ArgumentParser(description="Generate images using a fine-tuned Stable Diffusion UNet.")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model ID.")
    p.add_argument("--ckpt_root", required=True, help="Root directory containing checkpoint folders (e.g., 'path/to/lora_outputs').")
    p.add_argument("--ckpt", type=int, required=True, help="Checkpoint step number to load (e.g., 1000).")
    p.add_argument("--prompts", nargs="+", default=["a photo of an astronaut riding a horse on the moon"], help="One or more prompts to generate images for.")
    p.add_argument("--out_dir", default="outputs", help="Directory to save the generated images.")
    p.add_argument("--steps", type=int, default=50, help="Number of diffusion inference steps.")
    # --- Added Argument ---
    p.add_argument("--num_images", type=int, default=20, help="Number of images to generate per prompt.")
    # --- GPU selection ---
    p.add_argument("--device", default="cuda", help="Device to run inference on (e.g., 'cuda', 'cpu').")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load the fine-tuned UNet for the requested step
    unet_path = os.path.join(args.ckpt_root, f"checkpoint-{args.ckpt}", "unet")
    if not os.path.isdir(unet_path):
        print(f"Error: UNet directory not found at {unet_path}")
        return

    print(f"Loading UNet from: {unet_path}")
    try:
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
    except Exception as e:
        print(f"Error loading UNet: {e}")
        return

    # 2) Build a full pipeline from the *base* model but inject the new UNet
    print(f"Loading base model: {args.base_model}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model,
            unet=unet,
            torch_dtype=torch.float16     # or bfloat16 if GPUs allow
        ).to(args.device)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # Optional: Enable memory-efficient attention if xformers is installed
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except ImportError:
        print("xFormers not installed. Running without memory efficient attention.")
    except Exception as e:
         print(f"Could not enable xFormers: {e}. Running without memory efficient attention.")


    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # --- Modified Loop ---
    for prompt in args.prompts:
        print(f"\nGenerating {args.num_images} images for prompt: '{prompt}'...")
        # Generate multiple images by passing num_images_per_prompt
        # Use a generator for potentially lower memory usage if generating many images
        # Note: generator=torch.Generator(device=args.device).manual_seed(SOME_SEED) # for reproducibility if needed
        try:
            result = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                num_images_per_prompt=args.num_images # Generate the requested number of images
            )
            images = result.images # This is now a list of PIL Images
        except Exception as e:
            print(f"Error during image generation for prompt '{prompt}': {e}")
            continue # Skip to the next prompt

        # Sanitize the prompt for use in filenames
        safe_prompt_name = sanitize_filename(prompt)

        # Iterate through the generated images and save each one
        for i, img in enumerate(images):
            # Create a unique filename using the checkpoint, sanitized prompt, and image index
            filename = f"ckpt-{args.ckpt:05d}-{safe_prompt_name}-{i:03d}.png" # Added index {i:03d} (e.g., 000, 001)
            filepath = os.path.join(args.out_dir, filename)
            try:
                img.save(filepath)
                print(f"✓ Saved {filepath}")
            except Exception as e:
                print(f"Error saving image {filepath}: {e}")

    print("\nImage generation complete.")

if __name__ == "__main__":
    main()