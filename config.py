"""
Configuration globale du projet THJ
"""
import torch
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Création des dossiers s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Device (GPU si disponible, sinon CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres des datasets
DATASETS = {
    "mnist": {
        "num_classes": 10,
        "image_size": (1, 28, 28),
        "mean": (0.1307,),
        "std": (0.3081,)
    },
    "cifar10": {
        "num_classes": 10,
        "image_size": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010)
    }
}

# Hyperparamètres d'entraînement
TRAINING = {
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam"
}

# Paramètres des attaques
ATTACKS = {
    "fgsm": {
        "epsilon": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        "description": "Fast Gradient Sign Method"
    },
    "pgd": {
        "epsilon": 0.3,
        "alpha": 0.01,
        "num_iter": 40,
        "description": "Projected Gradient Descent"
    },
    "bim": {
        "epsilon": 0.3,
        "alpha": 0.01,
        "num_iter": 10,
        "description": "Basic Iterative Method"
    }
}

# Paramètres de défense
DEFENSES = {
    "passive": {
        "description": "Aucune défense",
        "params": {}
    },
    "gaussian_noise": {
        "description": "Ajout de bruit gaussien",
        "params": {"std": 0.05}
    },
    "median_filter": {
        "description": "Filtre médian",
        "params": {"kernel_size": 3}
    },
    "jpeg_compression": {
        "description": "Compression JPEG",
        "params": {"quality": 75}
    },
    "bit_depth_reduction": {
        "description": "Réduction profondeur de bits",
        "params": {"bits": 4}
    },
    "input_transform": {
        "description": "Transformations aléatoires",
        "params": {"rotation_range": 5, "translation_range": 2}
    },
    "ensemble": {
        "description": "Ensemble de défenses",
        "params": {}
    },
    "randomized_smoothing": {
        "description": "Lissage aléatoire",
        "params": {"sigma": 0.12, "n_samples": 10}
    }
}

# Paramètres du jeu
GAME_THEORY = {
    "payoff_correct": 1,      # Gain pour J2 si classification correcte
    "payoff_incorrect": -1,   # Perte pour J2 si classification incorrecte
    "max_perturbation": 0.3,  # Perturbation maximale autorisée
    "num_strategies": 10      # Nombre de stratégies discrètes pour chaque joueur
}

# Paramètres de visualisation
VIZ = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid"
}

# Modèles disponibles
MODELS = {
    "simple_cnn": {
        "name": "Simple CNN",
        "description": "CNN personnalisé léger",
        "params": "~500K"
    },
    "resnet18": {
        "name": "ResNet-18",
        "description": "ResNet 18 couches (pré-entraîné ImageNet)",
        "params": "~11M"
    },
    "resnet50": {
        "name": "ResNet-50",
        "description": "ResNet 50 couches (pré-entraîné ImageNet)",
        "params": "~25M"
    },
    "vgg16": {
        "name": "VGG-16",
        "description": "VGG 16 couches (pré-entraîné ImageNet)",
        "params": "~138M"
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "description": "EfficientNet B0 (pré-entraîné ImageNet)",
        "params": "~5M"
    }
}
