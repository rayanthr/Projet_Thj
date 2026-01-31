"""
Utilitaires pour le chargement de données
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, DATASETS, TRAINING


def get_mnist_loaders(batch_size=TRAINING["batch_size"], num_workers=0):
    """
    Charge les données MNIST
    
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASETS["mnist"]["mean"], DATASETS["mnist"]["std"])
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=TRAINING["batch_size"], num_workers=0):
    """
    Charge les données CIFAR-10
    
    Returns:
        train_loader, test_loader
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATASETS["cifar10"]["mean"], DATASETS["cifar10"]["std"])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASETS["cifar10"]["mean"], DATASETS["cifar10"]["std"])
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_sample_batch(dataset_name="mnist", batch_size=100):
    """
    Récupère un échantillon de données pour les expérimentations
    
    Args:
        dataset_name: "mnist" ou "cifar10"
        batch_size: Taille de l'échantillon
    
    Returns:
        images, labels: Batch d'images et labels
    """
    if dataset_name == "mnist":
        _, test_loader = get_mnist_loaders(batch_size=batch_size)
    elif dataset_name == "cifar10":
        _, test_loader = get_cifar10_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Dataset non reconnu: {dataset_name}")
    
    # Récupérer le premier batch
    images, labels = next(iter(test_loader))
    
    return images, labels


def denormalize_image(image, dataset_name="mnist"):
    """
    Dénormalise une image pour l'affichage
    
    Args:
        image: Image normalisée (tensor)
        dataset_name: "mnist" ou "cifar10"
    
    Returns:
        image: Image dénormalisée [0, 1]
    """
    mean = torch.tensor(DATASETS[dataset_name]["mean"]).view(-1, 1, 1)
    std = torch.tensor(DATASETS[dataset_name]["std"]).view(-1, 1, 1)
    
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    return image


if __name__ == "__main__":
    print("Test du chargement des données")
    
    # Test MNIST
    print("\n=== MNIST ===")
    train_loader, test_loader = get_mnist_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test CIFAR-10
    print("\n=== CIFAR-10 ===")
    train_loader, test_loader = get_cifar10_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test échantillon
    print("\n=== Échantillon ===")
    sample_imgs, sample_labels = get_sample_batch("mnist", batch_size=50)
    print(f"Sample shape: {sample_imgs.shape}")
