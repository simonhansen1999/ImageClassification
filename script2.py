import numpy as np
from PIL import Image
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    torch.manual_seed(42)

    image, label = train_data[0]
    print(image.size())

    net = NeuralNet()

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch + 1} loss: {running_loss / len(train_loader):.4f}")

    print("Finished Training")

    torch.save(net.state_dict(), 'script2_trained_net.pth')

    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

def load_image(image_path):
    new_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(image_path).convert('RGB')
    image = new_transform(image)
    image = image.unsqueeze(0)

    return image

def estimate():
    net = NeuralNet()

    net.to(device)

    net.load_state_dict(torch.load('script2_trained_net.pth'))

    image_dir = "images"
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images = [(path, load_image(os.path.join(image_dir, path))) for path in image_paths]

    net.eval()

    cols = 5  # Number of images per row
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(cols * 4, rows * 4))  # Adjust size as needed

    with torch.no_grad():
        for i, (filename, image) in enumerate(images):
            image = image.to(device)
            outputs = net(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            print(f"Predicted: {filename} -> {class_names[predicted.item()]} with {confidence.item() * 100:.2f}% confidence")

            plt.subplot(rows, cols, i + 1)
            image = Image.open("images/" + filename).convert("RGB")
            plt.imshow(image)
            plt.axis('off')
            plt.title(class_names[predicted.item()] + f" ({confidence.item() * 100:.2f}%)")

    plt.tight_layout()
    plt.show()

def main():
    while True:
        choice = input("Commands: 'train', 'est' or 'exit'): ").strip().lower()
        if choice == "train":
            train()
        elif choice == "est":
            estimate()
        elif choice == "exit":
            print("Exiting the program.")
            break
        else:
            print("Unknown command. Use 'train', 'est' or 'exit'.")

if __name__ == "__main__":
    main()