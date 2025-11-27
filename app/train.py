import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
import os

# Common training function
def train_model(model, train_loader, val_loader, save_path, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total * 100:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model training complete. Saved as {save_path}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define training tasks
def train_all():
    tasks = [
        ("data/xray/train", "data/xray/val", "xray_model.pth"),
        ("data/lung/train", "data/lung/val", "lung_model.pth"),
        ("data/chest_xray/train", "data/chest_xray/val", "pneumonia_model.pth")
    ]

    for train_root, val_root, model_path in tasks:
        print(f"\n🔧 Training model: {model_path}")
        train_data = datasets.ImageFolder(root=train_root, transform=transform)
        val_data = datasets.ImageFolder(root=val_root, transform=transform)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)

        train_model(model, train_loader, val_loader, save_path=model_path, epochs=2)

if __name__ == "__main__":
    train_all()