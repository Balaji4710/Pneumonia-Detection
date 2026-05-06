import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ALERT_THRESHOLD = config['email_settings']['threshold_percentage']
except Exception as e:
    print(f"Error loading config.yaml: {e}")
    ALERT_THRESHOLD = 80
print("Downloading dataset...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(path, 'chest_xray', 'train')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) # Set to 0 for Windows stability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model():
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)
def get_severity(prob):
    if prob < 0.3:
        return "Low/Normal", "green"
    elif prob < 0.6:
        return "Moderate", "orange"
    else:
        return "Severe", "red"
def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    print(f"Starting training on {device}...")

    for epoch in range(2):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    model_instance = get_model()
    train(model_instance)
    torch.save(model_instance.state_dict(), "pneumonia_model.pth")
    print("✅ Model saved as pneumonia_model.pth. You can now run Pneumonia.py")