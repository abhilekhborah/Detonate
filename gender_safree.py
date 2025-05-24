
import pandas as pd
from pathlib import Path
import torch
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    EulerDiscreteScheduler
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

# Check CUDA availability and device count
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")

# Model ID for SDXL
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Initialize the modified pipeline
def load_modified_pipeline():
    print("Loading modified SDXL pipeline...")
    
    # Load individual components
    vae = AutoencoderKL.from_pretrained(
        model_id, 
        subfolder="vae",
        cache_dir="cache",
        torch_dtype=torch.float16
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, 
        subfolder="text_encoder",
        cache_dir="cache",
        torch_dtype=torch.float16
    )
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, 
        subfolder="text_encoder_2",
        cache_dir="cache",
        torch_dtype=torch.float16
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, 
        subfolder="tokenizer",
        cache_dir="cache"
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id, 
        subfolder="tokenizer_2",
        cache_dir="cache"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        model_id, 
        subfolder="unet",
        cache_dir="cache",
        torch_dtype=torch.float16
    )
    
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, 
        subfolder="scheduler",
        cache_dir="cache"
    )
    
    # Optional components for IP-Adapter (if needed)
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir="cache",
            torch_dtype=torch.float16
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir="cache"
        )
    except:
        print("Could not load image encoder and feature extractor, setting to None")
        image_encoder = None
        feature_extractor = None
    
    # Create the modified pipeline
    pipe = ModifiedStableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        force_zeros_for_empty_prompt=True
    )
    
    return pipe

# Load the pipeline
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    pipe = load_modified_pipeline()
    # Move to GPU
    pipe = pipe.to("cuda")
else:
    print("Using single device")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = load_modified_pipeline()
    print(f"Pipeline loaded on device: {device}")
    pipe = pipe.to(device)

# Load prompts from CSV files
prompt_files = ['gender_hateful.csv']

gender_prompts = []

# Read each file and extract prompts
for file in prompt_files:
    df = pd.read_csv(file)
    # Extract first 10 prompts starting from row 2 (index 1)
    prompts = df.iloc[1:11, 0].tolist()  # Assuming prompts are in the first column
    
    # Save to appropriate list based on filename

    gender_prompts = prompts


print(f"Loaded {len(gender_prompts)} gender prompts")

def modified_sdxl_gen(prompt, safree_config=None):
    """
    Generate image using modified SDXL pipeline
    
    Args:
        prompt: Text prompt for generation
        safree_config: Dictionary containing SafeFree configuration
                      Example: {
                          "safree": True,
                          "alpha": 0.5,
                          "re_attn_t": [0, 10],
                          "logger": None
                      }
    """
    # Default SafeFree configuration
    if safree_config is None:
        safree_config = {
            "safree": False,
            "alpha": 0.0,
            "re_attn_t": [0, 10],
            "logger": None
        }
    
    # Generate image
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            height=1024,
            width=1024,
            negative_prompt="",
            negative_prompt_space= [
    "Misogyny", "Misandry", "Sexism", "Gender-Based Hate",
    "Gender Identity Attacks", "Anti-Feminist", "Homophobic Slurs",
    "Transphobia", "Discrimination Based on Gender",
    "Violence Against Women", "Anti-LGBTQ+", "Toxic Masculinity",
    "Sexual Harassment", "Feminazi", "Stereotyping Women",
    "Offensive Language Toward Genders", "Hate Speech About Gays",
    "Derogatory Language About Trans People", "Gender-Based Insults"
],  # For SafeFree functionality
            safree_dict=safree_config,
            return_dict=True
        )
    
    return result.images[0]

# Create the base directory
base_dir = Path("modified_sdxl_outputs")
base_dir.mkdir(exist_ok=True)

# Create category directories and generate images for each prompt
categories = {
    "gender": gender_prompts
}

# Configuration options for different runs
configs = {
    "vanilla": {
        "safree": False,
        "alpha": 0.0,
        "re_attn_t": [0, 10],
        "logger": None
    },
    "safree_low": {
        "safree": True,
        "alpha": 0.3,
        "re_attn_t": [0, 15],
        "logger": None
    },
    "safree_high": {
        "safree": True,
        "alpha": 0.7,
        "re_attn_t": [0, 20],
        "logger": None
    }
}

# Choose which configuration to use (change this as needed)
current_config = "safree_high"  # Change to "safree_low" or "safree_high" to test SafeFree
safree_config = configs[current_config]

print(f"Using configuration: {current_config}")
print(f"SafeFree enabled: {safree_config['safree']}")

for category, prompts in categories.items():
    # Create category directory
    category_dir = base_dir / category
    category_dir.mkdir(exist_ok=True)
    
    # Process each prompt
    for i, prompt in enumerate(prompts, 1):
        try:
            # Create prompt-specific directory
            prompt_dir = category_dir / f"prompt{i}_{current_config}"
            prompt_dir.mkdir(exist_ok=True)
            
            print(f"Generating image for {category} prompt {i}: {prompt[:50]}...")
            
            # Generate image using modified SDXL
            image_content = modified_sdxl_gen(prompt, safree_config)
            
            # Save the image
            image_path = prompt_dir / f"image{i}.png"
            image_content.save(image_path)
            
            # Save prompt text for reference
            prompt_file = prompt_dir / "prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"Saved image at {image_path}")
            
        except Exception as e:
            print(f"Error generating image for {category} prompt {i}: {str(e)}")
            continue

print("All images generated and saved successfully.")

# # Optional: Generate comparison images with different SafeFree settings
# def generate_comparison_images(sample_prompts, num_samples=3):
#     """Generate comparison images with different SafeFree configurations"""
#     comparison_dir = base_dir / "comparisons"
#     comparison_dir.mkdir(exist_ok=True)
    
#     for i, prompt in enumerate(sample_prompts[:num_samples], 1):
#         prompt_dir = comparison_dir / f"comparison_{i}"
#         prompt_dir.mkdir(exist_ok=True)
        
#         # Save prompt
#         with open(prompt_dir / "prompt.txt", 'w', encoding='utf-8') as f:
#             f.write(prompt)
        
#         # Generate with different configs
#         for config_name, config in configs.items():
#             try:
#                 print(f"Generating comparison {i} with {config_name}...")
#                 image = modified_sdxl_gen(prompt, config)
#                 image.save(prompt_dir / f"image_{config_name}.png")
#             except Exception as e:
#                 print(f"Error in comparison generation: {str(e)}")

# Uncomment to generate comparison images
# sample_prompts = race_prompts[:2] + disability_prompts[:1] + gender_prompts[:1]
# generate_comparison_images(sample_prompts)