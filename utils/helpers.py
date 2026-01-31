"""
Fonctions helper diverses
"""
import torch
import numpy as np
import json
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, MODELS_DIR, RESULTS_DIR


def set_seed(seed=42):
    """
    Fixe les seeds pour la reproductibilité
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    """
    Compte le nombre de paramètres d'un modèle
    
    Returns:
        total: Nombre total de paramètres
        trainable: Nombre de paramètres entraînables
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def save_experiment_results(results, experiment_name):
    """
    Sauvegarde les résultats d'une expérience
    
    Args:
        results: Dictionnaire de résultats
        experiment_name: Nom de l'expérience
    """
    filepath = RESULTS_DIR / f"{experiment_name}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Résultats sauvegardés: {filepath}")


def load_experiment_results(experiment_name):
    """
    Charge les résultats d'une expérience
    
    Args:
        experiment_name: Nom de l'expérience
    
    Returns:
        results: Dictionnaire de résultats
    """
    filepath = RESULTS_DIR / f"{experiment_name}.json"
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def get_device_info():
    """
    Affiche les informations sur le device utilisé
    """
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"CUDA disponible: Oui")
        print(f"Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"GPU actuel: {torch.cuda.current_device()}")
        print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
        
        # Mémoire GPU
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Mémoire allouée: {memory_allocated:.2f} GB")
        print(f"Mémoire réservée: {memory_cached:.2f} GB")
    else:
        print("CUDA disponible: Non")


def compute_l2_distance(img1, img2):
    """
    Calcule la distance L2 entre deux images
    
    Args:
        img1, img2: Tensors d'images
    
    Returns:
        distance: Distance L2
    """
    return torch.norm(img1 - img2, p=2).item()


def compute_linf_distance(img1, img2):
    """
    Calcule la distance L-infini entre deux images
    
    Args:
        img1, img2: Tensors d'images
    
    Returns:
        distance: Distance L-infini
    """
    return torch.max(torch.abs(img1 - img2)).item()


def format_time(seconds):
    """
    Formate un temps en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
    
    Returns:
        time_str: Temps formaté
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_epsilon_schedule(start, end, num_steps, schedule_type="linear"):
    """
    Crée un schedule d'epsilon pour les attaques itératives
    
    Args:
        start: Valeur initiale
        end: Valeur finale
        num_steps: Nombre de pas
        schedule_type: "linear", "exponential", "cosine"
    
    Returns:
        schedule: Liste de valeurs d'epsilon
    """
    if schedule_type == "linear":
        return np.linspace(start, end, num_steps)
    elif schedule_type == "exponential":
        return np.exp(np.linspace(np.log(start + 1e-8), np.log(end + 1e-8), num_steps))
    elif schedule_type == "cosine":
        t = np.linspace(0, np.pi, num_steps)
        return start + (end - start) * (1 - np.cos(t)) / 2
    else:
        raise ValueError(f"Schedule type non reconnu: {schedule_type}")


class AverageMeter:
    """
    Classe pour calculer et stocker la moyenne et la valeur actuelle
    Utile pour le tracking pendant l'entraînement
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    print("=== Test des helpers ===")
    
    # Device info
    get_device_info()
    
    # Seed
    set_seed(42)
    print("\nSeed fixé à 42")
    
    # Distance
    img1 = torch.randn(1, 28, 28)
    img2 = img1 + torch.randn(1, 28, 28) * 0.1
    
    l2_dist = compute_l2_distance(img1, img2)
    linf_dist = compute_linf_distance(img1, img2)
    
    print(f"\nDistances:")
    print(f"  L2: {l2_dist:.4f}")
    print(f"  L-inf: {linf_dist:.4f}")
    
    # Format time
    print(f"\nFormat time:")
    print(f"  3661s = {format_time(3661)}")
    print(f"  125s = {format_time(125)}")
    
    # Epsilon schedule
    schedule = create_epsilon_schedule(0, 0.3, 10, "cosine")
    print(f"\nEpsilon schedule (cosine): {schedule}")
    
    # Average meter
    meter = AverageMeter()
    for i in range(5):
        meter.update(i)
    print(f"\nAverage meter: {meter.avg:.2f}")
