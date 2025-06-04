import os
import json
import shutil
import random
from pathlib import Path

def extract_test_data():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    imagen_real_path = Path("./raw_data/IMAGEN_dataset/real")
    test_path = Path("./dataset/test")
    json_file = Path("./raw_data/json_files/imagen_json.json")
    
    # Create test directory
    os.makedirs(test_path, exist_ok=True)
    
    # Get 100 random real images
    real_images = [f for f in os.listdir(imagen_real_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(real_images, min(100, len(real_images)))
    
    # Copy selected images to test folder
    selected_filenames = set()
    for i, img in enumerate(selected_images):
        src = imagen_real_path / img
        dst = test_path / f"REAL_{i:03d}.png"
        shutil.copy2(src, dst)
        selected_filenames.add(img)
    
    # Load JSON and extract prompts (excluding selected images)
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Collect prompts not from selected images
    available_prompts = []
    for entry in data:
        if entry['real_image_file_name'] not in selected_filenames:
            available_prompts.append(entry['prompts'])
    
    # Select 100 random prompts
    selected_prompts = random.sample(available_prompts, min(100, len(available_prompts)))
    
    # Save prompts to JSON
    with open('./prompts.json', 'w') as f:
        json.dump(selected_prompts, f, indent=2)

if __name__ == "__main__":
    extract_test_data() 