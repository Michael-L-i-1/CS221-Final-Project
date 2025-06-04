import os
import torch
import clip
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm

def classify_with_clip():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Define text prompts
    text_prompts = [
        "a real image", 
        "a fake, AI generated image"
    ]
    
    # Tokenize text prompts
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Get text features
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Process test images
    test_path = Path("./dataset/test")
    results = []
    correct = 0
    total = 0
    
    image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Classifying Images"):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load and preprocess image
            image_path = test_path / img_file
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Get image features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (image_features @ text_features.T).squeeze(0)
            predicted_idx = similarities.argmax().item()
            
            # Determine prediction and ground truth
            predicted_label = "real" if predicted_idx == 0 else "fake"
            true_label = "real" if img_file.startswith("REAL_") else "fake"
            
            # Store result
            results.append({
                "filename": img_file,
                "predicted": predicted_label,
                "true_label": true_label,
                "confidence": similarities[predicted_idx].item(),
                "similarities": similarities.tolist()
            })
            
            # Update accuracy
            if predicted_label == true_label:
                correct += 1
            total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    output = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }
    
    with open("./clip_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"CLIP Classification Results:")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to clip_results.json")

if __name__ == "__main__":
    classify_with_clip()
