import pandas as pd
from pathlib import Path
import torch
# Check CUDA availability and device count
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
from diffusers import StableDiffusion3Pipeline
import torch.nn as nn

id = "stabilityai/stable-diffusion-xl-base-1.0"
# id = "stabilityai/stable-diffusion-3.5-large"
# Check if CUDA is available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with weight sharing")
    
    # Load model on CPU first
    pipe = StableDiffusion3Pipeline.from_pretrained(
        id, 
        cache_dir="cache", 
        torch_dtype=torch.float16  # Using float16 for better memory efficiency
    )
    
    # Distribute model across GPUs
    device_map = "auto"  # Let accelerate handle the optimal device mapping
    pipe.to("cuda", device_map=device_map)
else:
    print("Not enough GPUs available, using a single device")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusion3Pipeline.from_pretrained(
        id, 
        cache_dir="cache", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

# Rest of your code remains unchanged
prompt_files = ['race_hateful.csv','disability_hateful.csv', 'gender_hateful.csv']
# Lists to store prompts for each category
race_prompts = []
disability_prompts = []
gender_prompts = []

# Read each file and extract prompts
for file in prompt_files:
    df = pd.read_csv(file)
    # Extract first 10 prompts starting from row 2 (index 1)
    prompts = df.iloc[1:11, 0].tolist()  # Assuming prompts are in the first column
    
    # Save to appropriate list based on filename
    if 'race' in file:
        race_prompts = prompts
    elif 'disability' in file:
        disability_prompts = prompts
    elif 'gender' in file:
        gender_prompts = prompts

print(f"Loaded {len(race_prompts)} race prompts")
print(f"Loaded {len(disability_prompts)} disability prompts")
print(f"Loaded {len(gender_prompts)} gender prompts")

def sdl_gen(prompt):
    image = pipe(prompt,
        num_inference_steps=30,
        guidance_scale=7.0).images[0]
    return image

# Create the base directory
base_dir = Path("sdl_vanilla")
base_dir.mkdir(exist_ok=True)

# Create category directories and generate images for each prompt
categories = {
    "race": race_prompts,
    "disability": disability_prompts,
    "gender": gender_prompts
}

for category, prompts in categories.items():
    # Create category directory
    category_dir = base_dir / category
    category_dir.mkdir(exist_ok=True)
    
    # Process each prompt
    for i, prompt in enumerate(prompts, 1):
        # Create prompt-specific directory
        prompt_dir = category_dir / f"prompt{i}"
        prompt_dir.mkdir(exist_ok=True)
        
        # Generate image using SDL
        image_content = sdl_gen(prompt)
        
        # Save the image
        image_path = prompt_dir / f"image{i}.png"
        image_content.save(image_path)
        
        print(f"Saved image at {image_path}")

print("All images generated and saved successfully.")