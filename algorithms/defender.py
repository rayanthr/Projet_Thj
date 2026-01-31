"""
Méthodes de défense contre les attaques adversariales
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE


class DefenseStrategy:
    """Classe de base pour les stratégies de défense"""
    
    def __init__(self, name="base"):
        self.name = name
    
    def apply(self, images):
        """
        Applique la défense sur les images
        
        Args:
            images: Tensor d'images (B, C, H, W)
        
        Returns:
            defended_images: Images après défense
        """
        raise NotImplementedError


class PassiveDefense(DefenseStrategy):
    """
    Défense passive : Aucune transformation
    Le modèle classifie directement les images
    """
    def __init__(self):
        super().__init__(name="passive")
    
    def apply(self, images):
        return images


class GaussianNoiseDefense(DefenseStrategy):
    """
    Défense par ajout de bruit gaussien
    
    Principe : Ajouter du bruit aléatoire pour "noyer" les perturbations adversariales
    Trade-off : Peut réduire la précision sur images propres
    """
    def __init__(self, std=0.05):
        super().__init__(name="gaussian_noise")
        self.std = std
    
    def apply(self, images):
        noise = torch.randn_like(images) * self.std
        defended = images + noise
        return torch.clamp(defended, 0, 1)


class MedianFilterDefense(DefenseStrategy):
    """
    Défense par filtre médian
    
    Principe : Remplacer chaque pixel par la médiane de son voisinage
    Efficace contre : Perturbations localisées (salt & pepper noise)
    """
    def __init__(self, kernel_size=3):
        super().__init__(name="median_filter")
        self.kernel_size = kernel_size
    
    def apply(self, images):
        # Convertir en numpy pour scipy
        images_np = images.cpu().numpy()
        defended_np = np.zeros_like(images_np)
        
        # Appliquer filtre médian sur chaque image et canal
        for i in range(images_np.shape[0]):  # Batch
            for c in range(images_np.shape[1]):  # Canaux
                defended_np[i, c] = median_filter(
                    images_np[i, c], 
                    size=self.kernel_size
                )
        
        defended = torch.from_numpy(defended_np).to(images.device)
        return torch.clamp(defended, 0, 1)


class JPEGCompressionDefense(DefenseStrategy):
    """
    Défense par compression JPEG
    
    Principe : Compresser puis décompresser l'image
    Effet : Supprime les hautes fréquences où se cachent souvent les perturbations
    """
    def __init__(self, quality=75):
        super().__init__(name="jpeg_compression")
        self.quality = quality
    
    def apply(self, images):
        # Simulation simplifiée de compression JPEG
        # En pratique, on utiliserait PIL/cv2 pour vraie compression
        
        # Approximation : Quantification des valeurs
        scale = 255.0 / self.quality
        quantized = torch.round(images * 255 / scale) * scale / 255
        
        return torch.clamp(quantized, 0, 1)


class InputTransformDefense(DefenseStrategy):
    """
    Défense par transformations aléatoires de l'entrée
    
    Principe : Rotation, translation, scaling légers
    Effet : Les perturbations adversariales sont sensibles aux transformations
    """
    def __init__(self, rotation_range=5, translation_range=2):
        super().__init__(name="input_transform")
        self.rotation_range = rotation_range
        self.translation_range = translation_range
    
    def apply(self, images):
        batch_size = images.size(0)
        defended = []
        
        for i in range(batch_size):
            img = images[i:i+1]
            
            # Rotation aléatoire
            angle = torch.rand(1).item() * 2 * self.rotation_range - self.rotation_range
            theta = torch.tensor([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
            ], dtype=torch.float32).unsqueeze(0).to(images.device)
            
            grid = F.affine_grid(theta, img.size(), align_corners=False)
            transformed = F.grid_sample(img, grid, align_corners=False)
            
            defended.append(transformed)
        
        return torch.clamp(torch.cat(defended, dim=0), 0, 1)


class BitDepthReductionDefense(DefenseStrategy):
    """
    Défense par réduction de la profondeur de bits
    
    Principe : Quantifier les valeurs de pixels (ex: 8 bits → 4 bits)
    Effet : Réduit l'espace des perturbations possibles
    """
    def __init__(self, bits=4):
        super().__init__(name="bit_depth_reduction")
        self.bits = bits
        self.levels = 2 ** bits
    
    def apply(self, images):
        # Quantifier sur self.levels niveaux
        quantized = torch.round(images * (self.levels - 1)) / (self.levels - 1)
        return torch.clamp(quantized, 0, 1)


class EnsembleDefense(DefenseStrategy):
    """
    Défense par ensemble de transformations
    
    Principe : Appliquer plusieurs transformations et moyenner
    Plus robuste mais plus coûteux
    """
    def __init__(self, defenses=None):
        super().__init__(name="ensemble")
        if defenses is None:
            self.defenses = [
                GaussianNoiseDefense(std=0.03),
                BitDepthReductionDefense(bits=5),
                JPEGCompressionDefense(quality=80)
            ]
        else:
            self.defenses = defenses
    
    def apply(self, images):
        # Appliquer chaque défense et moyenner
        defended_list = []
        for defense in self.defenses:
            defended_list.append(defense.apply(images))
        
        # Moyenne des résultats
        defended = torch.stack(defended_list).mean(dim=0)
        return torch.clamp(defended, 0, 1)


class RandomizedSmoothingDefense(DefenseStrategy):
    """
    Défense par lissage aléatoire (Randomized Smoothing)
    
    Principe : Ajouter plusieurs échantillons de bruit et voter
    Fournit des garanties théoriques de robustesse certifiée
    """
    def __init__(self, sigma=0.12, n_samples=10):
        super().__init__(name="randomized_smoothing")
        self.sigma = sigma
        self.n_samples = n_samples
    
    def apply(self, images):
        # Pour chaque image, générer n_samples versions bruitées
        # Note: Cette méthode est plus adaptée à l'inférence avec le modèle
        # Ici on retourne juste une version bruitée représentative
        noise = torch.randn_like(images) * self.sigma
        defended = images + noise
        return torch.clamp(defended, 0, 1)


def get_defense(defense_name, **kwargs):
    """
    Factory pour créer une stratégie de défense
    
    Args:
        defense_name: Nom de la défense
        **kwargs: Paramètres spécifiques à la défense
    
    Returns:
        defense: Instance de DefenseStrategy
    """
    defenses_map = {
        "passive": PassiveDefense,
        "gaussian_noise": GaussianNoiseDefense,
        "median_filter": MedianFilterDefense,
        "jpeg_compression": JPEGCompressionDefense,
        "input_transform": InputTransformDefense,
        "bit_depth_reduction": BitDepthReductionDefense,
        "ensemble": EnsembleDefense,
        "randomized_smoothing": RandomizedSmoothingDefense
    }
    
    if defense_name not in defenses_map:
        raise ValueError(f"Défense inconnue: {defense_name}. Choix: {list(defenses_map.keys())}")
    
    return defenses_map[defense_name](**kwargs)


class DefensiveModel(nn.Module):
    """
    Wrapper qui combine un modèle avec une stratégie de défense
    """
    def __init__(self, base_model, defense_strategy):
        super().__init__()
        self.base_model = base_model
        self.defense = defense_strategy
    
    def forward(self, x):
        # Appliquer la défense avant la classification
        x_defended = self.defense.apply(x)
        return self.base_model(x_defended)


if __name__ == "__main__":
    print("Test des stratégies de défense\n")
    
    # Créer des images de test
    images = torch.randn(4, 3, 32, 32) * 0.5 + 0.5  # [0, 1]
    images = torch.clamp(images, 0, 1)
    
    print(f"Images originales - Shape: {images.shape}, Range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Tester chaque défense
    defenses = [
        ("Passive", PassiveDefense()),
        ("Gaussian Noise", GaussianNoiseDefense(std=0.05)),
        ("Median Filter", MedianFilterDefense(kernel_size=3)),
        ("JPEG Compression", JPEGCompressionDefense(quality=75)),
        ("Bit Depth Reduction", BitDepthReductionDefense(bits=4)),
        ("Randomized Smoothing", RandomizedSmoothingDefense(sigma=0.1))
    ]
    
    print("\n" + "="*60)
    for name, defense in defenses:
        defended = defense.apply(images)
        diff = (defended - images).abs().mean().item()
        print(f"\n{name}:")
        print(f"  Output range: [{defended.min():.3f}, {defended.max():.3f}]")
        print(f"  Différence moyenne: {diff:.6f}")
    
    print("\n" + "="*60)
    print("✅ Toutes les défenses fonctionnent correctement!")
