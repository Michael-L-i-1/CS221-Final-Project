import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from transformers import ViTForImageClassification, ViTImageProcessor
import csv
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import random

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

class FlatFolderDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.labels[idx]
        image_path = os.path.join(self.root_dir, entry['filename'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label: 'real' -> 0, 'fake' -> 1 (following baseline2.py)
        label = 0 if entry['label'] == 'real' else 1
        return image, label

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2, architecture='resnet18'):
        super(EnsembleModel, self).__init__()
        
        # ResNet branch - try different architectures
        if architecture == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif architecture == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet34(pretrained=True)
            
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # ViT branch  
        self.vit = models.vit_b_16(pretrained=True)
        vit_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()
        
        # combined classifier - simplified to reduce overfitting
        combined_features = resnet_features + vit_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Make freezing optional
        self.freeze_early = True
        if self.freeze_early:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        # Less aggressive freezing - unfreeze more layers
        for name, param in self.resnet.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        # Unfreeze more ViT layers
        for name, param in self.vit.named_parameters():
            if any(layer in name for layer in ['encoder.layers.8', 'encoder.layers.9', 
                                               'encoder.layers.10', 'encoder.layers.11', 'heads']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        # ResNet features
        resnet_features = self.resnet(x)
        
        # ViT features
        vit_features = self.vit(x)
        
        # Combine features
        combined = torch.cat([resnet_features, vit_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        return output

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def train(config=None):
    set_seed(42)
    
    # setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # load training labels from JSON
    with open('./dataset/labels.json', 'r') as f:
        labels = json.load(f)
    train_labels = labels['train']
    val_labels = labels['val']

    # data loading
    train_transform, test_transform = get_transforms()
    train_dataset = FlatFolderDataset('./dataset/train', train_labels, train_transform)
    val_dataset = FlatFolderDataset('./dataset/val', val_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']*2, 
                           shuffle=False, num_workers=2)
    
    # model setup
    model = EnsembleModel(num_classes=2, dropout_rate=config['dropout_rate'], 
                         architecture=config['architecture'])
    model = model.to(device)
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'], 
        patience=config['scheduler_patience'], verbose=True)
    
    print(f"Training with config: {config}")
    
    # training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # track losses for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # learning rate scheduling
        scheduler.step(val_loss)
        
        # early stopping and model saving
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_ensemble_model.pth')
            print("  Model saved!")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")
            
        if patience_counter >= config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # plot and save training curves
    plt.figure(figsize=(15, 5))
    
    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved as 'training_curves.png'")

def test(config=None):
    set_seed(42)
    
    # use same config as training
    if config is None:
        config = {
            'dropout_rate': 0.2,
            'architecture': 'resnet50',
        }
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # load test labels
    test_labels = []
    with open('./dataset/test_labels.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_labels.append({
                'filename': row['filename'],
                'label': row['label']
            })
    
    # Data loading
    _, test_transform = get_transforms()
    test_dataset = FlatFolderDataset('./dataset/test', test_labels, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # model setup - USE SAME CONFIG AS TRAINING
    model = EnsembleModel(num_classes=2, dropout_rate=config['dropout_rate'], 
                         architecture=config['architecture'])
    model.load_state_dict(torch.load('best_ensemble_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # testing with per-class accuracy
    correct = 0
    total = 0
    real_total = 0
    real_correct = 0
    fake_total = 0
    fake_correct = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store for detailed analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                if label == 0:  # real
                    real_total += 1
                    if pred == label:
                        real_correct += 1
                else:  # fake
                    fake_total += 1
                    if pred == label:
                        fake_correct += 1
    
    # results
    acc = correct / total
    real_acc = real_correct / real_total if real_total > 0 else 0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    
    print(f"\n=== Test Results ===")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Real images: {real_correct}/{real_total} correct ({real_acc:.4f})")
    print(f"Fake images: {fake_correct}/{fake_total} correct ({fake_acc:.4f})")
    print(f"Balanced Accuracy: {(real_acc + fake_acc) / 2:.4f}")

if __name__ == "__main__":
    set_seed(42)
    
    print("Training ensemble model...")
    
    config = {
        'learning_rate': 0.0001,
        'batch_size': 64,
        'dropout_rate': 0.1,
        'weight_decay': 5e-05,
        'architecture': 'resnet50',
        'num_epochs': 10,
        'patience': 10,
        'scheduler_patience': 5,
        'scheduler_factor': 0.3,
    }
    
    train(config)
    print("\nTesting ensemble model...")
    test(config)
