import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

print("Downloading dataset...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(path, 'chest_xray', 'train')
test_dir = os.path.join(path, 'chest_xray', 'test')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_model():
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


model = get_model()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()

        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        score = output[:, class_idx]

        self.model.zero_grad()
        score.backward()

        if self.gradients is None:
            return np.zeros((224, 224))  # Fallback if gradient capture fails


        alpha = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        weighted_activations = alpha * self.activations
        heatmap = torch.sum(weighted_activations, dim=1).squeeze()


        heatmap = torch.maximum(heatmap, torch.tensor(0))

        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)

        return heatmap.detach().cpu().numpy()

def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    print(f"Starting training on {device}...")
    for epoch in range(2):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")


def visualize_result(img_path):
    raw_img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probabilities, 1)

        class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']
        prediction_label = class_names[pred.item()]
        confidence_score = conf.item() * 100

    cam = GradCAM(model, model.layer4[-1])
    heatmap = cam.generate_heatmap(input_tensor, class_idx=pred.item())
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original X-Ray\n(Actual: {os.path.basename(os.path.dirname(img_path))})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)

    plt.title(f"AI Prediction: {prediction_label}\nConfidence: {confidence_score:.2f}%")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()

    categories = ['NORMAL', 'PNEUMONIA']

    for cat in categories:
        cat_path = os.path.join(test_dir, cat)
        if os.path.exists(cat_path):
            sample_img = os.listdir(cat_path)[0]
            img_full_path = os.path.join(cat_path, sample_img)
            print(f"Processing category: {cat}")
            visualize_result(img_full_path)