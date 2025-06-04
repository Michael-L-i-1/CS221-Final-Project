import os
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict

def create_dataset():
    # Define paths
    raw_data_path = Path("./raw_data")
    dataset_path = Path("./dataset")
    
    # Source directories
    dalle_path = raw_data_path / "DALLE_dataset"
    sd_path = raw_data_path / "SD_dataset"
    
    # Create dataset directories
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    # Collect all images
    fake_images = []
    real_images = []
    
    # Get images from DALL-E dataset
    dalle_fake_path = dalle_path / "fake"
    dalle_real_path = dalle_path / "real"
    
    if dalle_fake_path.exists():
        fake_images.extend([(dalle_fake_path / f, "DALLE") for f in os.listdir(dalle_fake_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if dalle_real_path.exists():
        real_images.extend([(dalle_real_path / f, "DALLE") for f in os.listdir(dalle_real_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Get images from Stable Diffusion dataset
    sd_fake_path = sd_path / "fake"
    sd_real_path = sd_path / "real"
    
    if sd_fake_path.exists():
        fake_images.extend([(sd_fake_path / f, "SD") for f in os.listdir(sd_fake_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if sd_real_path.exists():
        real_images.extend([(sd_real_path / f, "SD") for f in os.listdir(sd_real_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(fake_images)} fake images and {len(real_images)} real images")
    
    # Shuffle the lists
    random.shuffle(fake_images)
    random.shuffle(real_images)
    
    # Check if we have enough images
    total_needed = 1200  # 1000 train + 200 val
    fake_needed = total_needed // 2  # 600 fake
    real_needed = total_needed // 2  # 600 real
    
    if len(fake_images) < fake_needed or len(real_images) < real_needed:
        print(f"Warning: Not enough images. Need {fake_needed} fake and {real_needed} real.")
        print(f"Available: {len(fake_images)} fake, {len(real_images)} real")
        # Adjust to available amounts
        available_pairs = min(len(fake_images), len(real_images))
        fake_needed = min(fake_needed, available_pairs)
        real_needed = min(real_needed, available_pairs)
    
    # Split into train and validation
    # Train: 500 fake + 500 real = 1000
    # Val: 100 fake + 100 real = 200
    train_fake_count = int(fake_needed * 0.833)  # ~83.3% for train
    train_real_count = int(real_needed * 0.833)
    
    train_fake = fake_images[:train_fake_count]
    val_fake = fake_images[train_fake_count:train_fake_count + (fake_needed - train_fake_count)]
    
    train_real = real_images[:train_real_count]
    val_real = real_images[train_real_count:train_real_count + (real_needed - train_real_count)]
    
    # Create labels dictionary
    labels_data = {
        "train": [],
        "val": []
    }
    
    # Copy training images and create labels
    print("Copying training images...")
    for i, (src_path, source) in enumerate(train_fake):
        dst_name = f"train_fake_{i}_{source}_{src_path.name}"
        dst_path = train_path / dst_name
        shutil.copy2(src_path, dst_path)
        labels_data["train"].append({
            "filename": dst_name,
            "label": "fake",
            "source": source
        })
    
    for i, (src_path, source) in enumerate(train_real):
        dst_name = f"train_real_{i}_{source}_{src_path.name}"
        dst_path = train_path / dst_name
        shutil.copy2(src_path, dst_path)
        labels_data["train"].append({
            "filename": dst_name,
            "label": "real",
            "source": source
        })
    
    # Copy validation images and create labels
    print("Copying validation images...")
    for i, (src_path, source) in enumerate(val_fake):
        dst_name = f"val_fake_{i}_{source}_{src_path.name}"
        dst_path = val_path / dst_name
        shutil.copy2(src_path, dst_path)
        labels_data["val"].append({
            "filename": dst_name,
            "label": "fake",
            "source": source
        })
    
    for i, (src_path, source) in enumerate(val_real):
        dst_name = f"val_real_{i}_{source}_{src_path.name}"
        dst_path = val_path / dst_name
        shutil.copy2(src_path, dst_path)
        labels_data["val"].append({
            "filename": dst_name,
            "label": "real",
            "source": source
        })
    
    # Shuffle the labels within each split
    random.shuffle(labels_data["train"])
    random.shuffle(labels_data["val"])
    
    # Save labels as JSON
    labels_file = dataset_path / "labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels_data, f, indent=2)
    
    # Create simple text labels file as well
    labels_txt = dataset_path / "labels.txt"
    with open(labels_txt, 'w') as f:
        f.write("# Dataset Labels\n")
        f.write("# Format: split/filename,label,source\n\n")
        
        for item in labels_data["train"]:
            f.write(f"train/{item['filename']},{item['label']},{item['source']}\n")
        
        for item in labels_data["val"]:
            f.write(f"val/{item['filename']},{item['label']},{item['source']}\n")
    
    # Print summary
    print("\nDataset creation complete!")
    print(f"Training images: {len(labels_data['train'])}")
    print(f"  - Fake: {sum(1 for x in labels_data['train'] if x['label'] == 'fake')}")
    print(f"  - Real: {sum(1 for x in labels_data['train'] if x['label'] == 'real')}")
    print(f"Validation images: {len(labels_data['val'])}")
    print(f"  - Fake: {sum(1 for x in labels_data['val'] if x['label'] == 'fake')}")
    print(f"  - Real: {sum(1 for x in labels_data['val'] if x['label'] == 'real')}")
    print(f"\nLabels saved to:")
    print(f"  - {labels_file}")
    print(f"  - {labels_txt}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    create_dataset() 