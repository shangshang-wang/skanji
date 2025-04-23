import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import os

# --- Configuration ---
# 1. Path to your checkpoint directory (the one containing 'unet', 'unet_ema', etc.)
checkpoint_dir = "/project/neiswang_1391/shangsha/sakana/skanji/outputs/train_from_scratch/checkpoint-13400" # Or the full path like "/path/to/your/checkpoint-13400"

# 2. Choose which UNet weights to load ('unet' or 'unet_ema')
#    'unet_ema' (Exponential Moving Average) is generally recommended for inference.
unet_subfolder = "unet_ema"
unet_path = os.path.join(checkpoint_dir, unet_subfolder)

# 3. Specify the base model identifier (Must be compatible with your fine-tuning!)
#    Common examples: "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base", etc.
base_model_id = "CompVis/stable-diffusion-v1-4"

# 4. Define your text prompt(s)
prompts = [
    "Fish",
]

# 5. Generation Parameters
num_samples_per_prompt = 2 # How many images to generate for EACH prompt
num_inference_steps = 30   # Number of denoising steps (20-50 is common)
guidance_scale = 7.5       # How strongly the prompt guides generation (5-10 is common)
output_dir = "generated_images" # Directory to save the images

# --- Setup ---
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    # Use float16 for memory efficiency on GPU if supported
    torch_dtype = torch.float16
    print("Using GPU (CUDA).")
else:
    device = "cpu"
    torch_dtype = torch.float32 # CPU usually works better with float32
    print("Using CPU.")

# --- Load the Custom UNet ---
print(f"Loading custom UNet weights from: {unet_path}")
try:
    # The UNet class might differ based on the base model (e.g., for SDXL)
    # UNet2DConditionModel is standard for SD 1.x/2.x
    unet = UNet2DConditionModel.from_pretrained(
        unet_path,
        torch_dtype=torch_dtype,
        # If you encounter errors, you might need low_cpu_mem_usage=False
        # low_cpu_mem_usage=False
    )
    print("Custom UNet loaded successfully.")
except Exception as e:
    print(f"Error loading UNet: {e}")
    print("Ensure the path is correct and contains 'config.json' and model weights ('diffusion_pytorch_model.safetensors' or '.bin').")
    exit() # Exit if UNet loading fails

# --- Load the Base Pipeline ---
print(f"Loading base pipeline: {base_model_id}")
try:
    # Load the rest of the pipeline components (text encoder, VAE, scheduler) from the base model
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet, # Inject the custom UNet!
        torch_dtype=torch_dtype,
        # Add safety_checker=None if you want to disable it (at your own risk)
        # safety_checker=None,
        # Add revision="fp16" if using float16 and the base model supports it explicitly
        # revision="fp16" if torch_dtype == torch.float16 else None,
    )
    print("Base pipeline loaded successfully with custom UNet.")
except Exception as e:
    print(f"Error loading base pipeline: {e}")
    print(f"Ensure the base model ID '{base_model_id}' is correct and compatible with your UNet.")
    exit()

# Move pipeline to the selected device (GPU or CPU)
pipeline = pipeline.to(device)
print(f"Pipeline moved to device: {device}")

# Optional: Enable optimizations if you have xformers installed (pip install xformers)
# try:
#     pipeline.enable_xformers_memory_efficient_attention()
#     print("Enabled xformers memory efficient attention.")
# except ImportError:
#     print("xformers not installed. Running without memory efficient attention.")
#     pass


# --- Generate Images ---
print("\nStarting image generation...")
image_counter = 0
for i, prompt in enumerate(prompts):
    print(f"\nGenerating images for prompt {i+1}/{len(prompts)}: '{prompt}'")

    # Generate images (can return multiple images per prompt directly)
    # Use a generator for reproducibility if desired
    seed = torch.Generator(device=device).manual_seed(42 + i) # Different seed per prompt

    generated_images = pipeline(
        prompt,
        num_images_per_prompt=num_samples_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=seed,
    ).images

    print(f"Generated {len(generated_images)} images for this prompt.")

    # --- Save Images ---
    for j, img in enumerate(generated_images):
        image_counter += 1
        # Sanitize prompt for filename (optional)
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in prompt).replace(' ', '_')[:50]
        filename = os.path.join(output_dir, f"img_{image_counter:03d}_prompt_{i+1}_{j+1}_{safe_prompt}.png")
        img.save(filename)
        print(f"Saved: {filename}")

print(f"\nImage generation complete. Total images saved: {image_counter} in '{output_dir}'")