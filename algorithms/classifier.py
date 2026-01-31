"""
Classifieur CNN pour MNIST et CIFAR-10
Support des modèles pré-entraînés (ResNet-50, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, TRAINING


class SimpleCNN(nn.Module):
    """
    CNN simple pour la classification d'images
    Compatible avec MNIST (28x28, 1 canal) et CIFAR-10 (32x32, 3 canaux)
    """
    def __init__(self, num_classes=10, input_channels=1, image_size=28):
        super(SimpleCNN, self).__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculer la taille de sortie après convolutions et pooling
        # 3 pooling layers réduisent la taille par 8
        final_size = image_size // 8
        
        # Couches fully connected
        self.fc1 = nn.Linear(128 * final_size * final_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.5)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


def train_classifier(model, train_loader, criterion, optimizer, device=DEVICE):
    """
    Entraîne le classifieur pour une époque
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU/GPU)
    
    Returns:
        loss_avg: Perte moyenne sur l'époque
        accuracy: Précision moyenne sur l'époque
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    loss_avg = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return loss_avg, accuracy


def evaluate_classifier(model, test_loader, criterion, device=DEVICE):
    """
    Évalue le classifieur sur un ensemble de test
    
    Args:
        model: Modèle à évaluer
        test_loader: DataLoader pour les données de test
        criterion: Fonction de perte
        device: Device (CPU/GPU)
    
    Returns:
        loss_avg: Perte moyenne
        accuracy: Précision
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    loss_avg = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return loss_avg, accuracy


def save_model(model, path):
    """Sauvegarde le modèle"""
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé : {path}")


def load_model(model, path, device=DEVICE):
    """Charge le modèle"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Modèle chargé : {path}")
    return model


def get_pretrained_model(model_name="resnet50", num_classes=10, input_channels=3, pretrained=True):
    """
    Charge un modèle pré-entraîné et adapte pour le dataset
    
    Args:
        model_name: "resnet18", "resnet50", "vgg16", "efficientnet_b0"
        num_classes: Nombre de classes de sortie
        input_channels: Nombre de canaux d'entrée (1 pour MNIST, 3 pour CIFAR)
        pretrained: Utiliser les poids pré-entraînés ImageNet
    
    Returns:
        model: Modèle adapté
    """
    # Charger le modèle pré-entraîné
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    # Adapter la première couche pour MNIST (1 canal)
    if input_channels == 1:
        if model_name.startswith("resnet"):
            # Adapter conv1 pour 1 canal
            original_conv1 = model.conv1
            model.conv1 = nn.Conv2d(
                1, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            # Moyenner les poids sur les 3 canaux
            if pretrained:
                model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        elif model_name == "vgg16":
            original_conv = model.features[0]
            model.features[0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding
            )
            if pretrained:
                model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        elif model_name.startswith("efficientnet"):
            # EfficientNet : première couche dans features[0][0]
            original_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            if pretrained:
                model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    
    return model


def get_model(model_type="simple_cnn", dataset="mnist", pretrained=True):
    """
    Fonction helper pour obtenir un modèle selon le type et dataset
    
    Args:
        model_type: "simple_cnn", "resnet18", "resnet50", "vgg16", "efficientnet_b0"
        dataset: "mnist" ou "cifar10"
        pretrained: Utiliser les poids pré-entraînés (pour modèles ImageNet)
    
    Returns:
        model: Modèle initialisé
    """
    num_classes = 10
    
    if dataset == "mnist":
        input_channels = 1
        image_size = 28
    else:  # cifar10
        input_channels = 3
        image_size = 32
    
    if model_type == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels, image_size=image_size)
    else:
        model = get_pretrained_model(
            model_name=model_type,
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained
        )
    
    return model


if __name__ == "__main__":
    # Test du modèle
    print(f"Device: {DEVICE}")
    
    # Test CNN Simple
    print("\n=== CNN Simple ===")
    model_mnist = SimpleCNN(num_classes=10, input_channels=1, image_size=28)
    model_mnist.to(DEVICE)
    x_mnist = torch.randn(4, 1, 28, 28).to(DEVICE)
    out_mnist = model_mnist(x_mnist)
    print(f"MNIST - Input: {x_mnist.shape}, Output: {out_mnist.shape}")
    total_params_simple = sum(p.numel() for p in model_mnist.parameters())
    print(f"Nombre de paramètres: {total_params_simple:,}")
    
    # Test ResNet-50
    print("\n=== ResNet-50 ===")
    model_resnet = get_model("resnet50", "cifar10", pretrained=False)
    model_resnet.to(DEVICE)
    x_cifar = torch.randn(4, 3, 32, 32).to(DEVICE)
    out_resnet = model_resnet(x_cifar)
    print(f"CIFAR-10 - Input: {x_cifar.shape}, Output: {out_resnet.shape}")
    total_params_resnet = sum(p.numel() for p in model_resnet.parameters())
    print(f"Nombre de paramètres: {total_params_resnet:,}")
    
    # Test ResNet-50 pour MNIST
    print("\n=== ResNet-50 (MNIST) ===")
    model_resnet_mnist = get_model("resnet50", "mnist", pretrained=False)
    model_resnet_mnist.to(DEVICE)
    x_mnist = torch.randn(4, 1, 28, 28).to(DEVICE)
    out_resnet_mnist = model_resnet_mnist(x_mnist)
    print(f"MNIST - Input: {x_mnist.shape}, Output: {out_resnet_mnist.shape}")
