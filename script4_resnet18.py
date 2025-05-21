import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

# CIFAR-100 class names
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'cactus', 'camel', 'can', 'castle', 'caterpillar',
    'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
    'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
    'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# ResNet18 expects 224x224 images, so resize here
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    torch.manual_seed(42)

    net = resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 100)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    num_epochs = 20
    for epoch in range(num_epochs):
        net.train()
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
    torch.save(net.state_dict(), 'script4_trained_net.pth')

    # Evaluate
    net.eval()
    correct = 0
    total = 0
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
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(image_path).convert('RGB')
    image = new_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def estimate():
    net = resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 100)
    net.to(device)
    net.load_state_dict(torch.load('script4_trained_net.pth'))

    image_dir = "images2"
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
            image = Image.open("images2/" + filename).convert("RGB")
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
