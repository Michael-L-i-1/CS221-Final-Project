import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import json
from PIL import Image
import csv

# Paths
train_dir = './dataset/train'
val_dir = './dataset/val'
test_dir = './dataset/test'
labels_json_path = './dataset/labels.json'
model_save_path = './baseline2_resnet.pth'

# Transforms (default ImageNet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# FlatFolderDataset class to load images and labels
class FlatFolderDataset(Dataset):
    def __init__(self, img_dir, label_list, transform):
        self.img_dir = img_dir
        self.samples = label_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = os.path.join(self.img_dir, entry['filename'])
        image = Image.open(img_path).convert('RGB')
        label = 0 if entry['label'] == 'real' else 1
        image = self.transform(image)
        return image, label

def get_model(device):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: real, fake
    model = model.to(device)
    return model

def train():
    with open(labels_json_path, 'r') as f:
        labels = json.load(f)
    train_labels = labels['train']
    val_labels = labels['val']

    train_dataset = FlatFolderDataset(train_dir, train_labels, transform)
    val_dataset = FlatFolderDataset(val_dir, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    criterion = nn.CrossEntropyLoss()  # basic loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # basic learning rate and optimizer

    # training
    best_acc = 0.0
    for epoch in range(10):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1} Validation Accuracy: {acc:.2%}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)

    print(f"Best validation accuracy: {best_acc:.2%}")
    print(f"Best model saved to {model_save_path}")

def test():
    # Use test_labels.csv for test set labels
    test_labels = []
    with open('./dataset/test_labels.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_labels.append({
                'filename': row['filename'],
                'label': row['label']
            })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Use FlatFolderDataset for test set
    test_dataset = FlatFolderDataset(test_dir, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    correct = 0
    total = 0

    # Track per-class stats
    real_total = 0
    real_correct = 0
    fake_total = 0
    fake_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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

    acc = correct / total
    real_acc = real_correct / real_total if real_total > 0 else 0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    print(f"Test Accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Real images: {real_correct}/{real_total} correct ({real_acc:.2%})")
    print(f"Fake images: {fake_correct}/{fake_total} correct ({fake_acc:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()

    if args.train:
        train()
    if args.test:
        test()
