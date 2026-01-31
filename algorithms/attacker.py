"""
Attaques adversariales : FGSM, PGD, BIM
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, ATTACKS


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM)
    Goodfellow et al., 2014
    
    Génère une perturbation adversariale en une seule étape dans la direction
    du gradient de la fonction de perte.
    """
    def __init__(self, model, epsilon=0.3):
        """
        Args:
            model: Modèle à attaquer
            epsilon: Magnitude de la perturbation (valeur entre 0 et 1)
        """
        self.model = model
        self.epsilon = epsilon
        
    def attack(self, images, labels):
        """
        Génère des exemples adversariaux avec FGSM
        
        Args:
            images: Images originales (tensor)
            labels: Labels vrais (tensor)
        
        Returns:
            perturbed_images: Images perturbées
            perturbation: Perturbation ajoutée
        """
        images = images.clone().detach().to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Calcul de la perte
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # Backward pass pour obtenir le gradient
        self.model.zero_grad()
        loss.backward()
        
        # Génération de la perturbation
        # Signe du gradient multiplié par epsilon
        data_grad = images.grad.data
        perturbation = self.epsilon * data_grad.sign()
        
        # Application de la perturbation
        perturbed_images = images + perturbation
        
        # Clipping pour garder les valeurs dans [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach(), perturbation.detach()
    
    def evaluate(self, model, data_loader, device=DEVICE):
        """
        Évalue l'efficacité de l'attaque
        
        Returns:
            accuracy: Précision sur les images adversariales
            success_rate: Taux de réussite de l'attaque
        """
        model.eval()
        correct = 0
        total = 0
        attack_success = 0
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Prédictions originales
            with torch.no_grad():
                original_outputs = model(images)
                _, original_pred = original_outputs.max(1)
            
            # Génération d'exemples adversariaux
            perturbed_images, _ = self.attack(images, labels)
            
            # Prédictions sur images perturbées
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # L'attaque réussit si la prédiction change et devient incorrecte
            originally_correct = original_pred.eq(labels)
            now_incorrect = ~predicted.eq(labels)
            attack_success += (originally_correct & now_incorrect).sum().item()
        
        accuracy = 100. * correct / total
        success_rate = 100. * attack_success / total
        
        return accuracy, success_rate


class PGDAttack:
    """
    Projected Gradient Descent (PGD)
    Madry et al., 2017
    
    Version itérative de FGSM avec projection sur une boule L-infini
    """
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_iter=40):
        """
        Args:
            model: Modèle à attaquer
            epsilon: Magnitude maximale de la perturbation
            alpha: Taille du pas à chaque itération
            num_iter: Nombre d'itérations
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        
    def attack(self, images, labels):
        """
        Génère des exemples adversariaux avec PGD
        
        Args:
            images: Images originales (tensor)
            labels: Labels vrais (tensor)
        
        Returns:
            perturbed_images: Images perturbées
            perturbation: Perturbation totale ajoutée
        """
        images = images.clone().detach().to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)
        
        # Initialisation aléatoire dans la boule epsilon
        perturbed_images = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.num_iter):
            perturbed_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(perturbed_images)
            
            # Calcul de la perte
            loss = criterion(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Mise à jour avec gradient ascent
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + self.alpha * data_grad.sign()
            
            # Projection sur la boule epsilon autour de l'image originale
            perturbation = torch.clamp(perturbed_images - images, -self.epsilon, self.epsilon)
            perturbed_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        perturbation = perturbed_images - images
        return perturbed_images, perturbation
    
    def evaluate(self, model, data_loader, device=DEVICE):
        """Évalue l'efficacité de l'attaque"""
        model.eval()
        correct = 0
        total = 0
        attack_success = 0
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Prédictions originales
            with torch.no_grad():
                original_outputs = model(images)
                _, original_pred = original_outputs.max(1)
            
            # Génération d'exemples adversariaux
            perturbed_images, _ = self.attack(images, labels)
            
            # Prédictions sur images perturbées
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            originally_correct = original_pred.eq(labels)
            now_incorrect = ~predicted.eq(labels)
            attack_success += (originally_correct & now_incorrect).sum().item()
        
        accuracy = 100. * correct / total
        success_rate = 100. * attack_success / total
        
        return accuracy, success_rate


class BIMAttack:
    """
    Basic Iterative Method (BIM)
    Version itérative de FGSM sans initialisation aléatoire
    """
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_iter=10):
        """
        Args:
            model: Modèle à attaquer
            epsilon: Magnitude maximale de la perturbation
            alpha: Taille du pas à chaque itération
            num_iter: Nombre d'itérations
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        
    def attack(self, images, labels):
        """
        Génère des exemples adversariaux avec BIM
        """
        images = images.clone().detach().to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)
        
        perturbed_images = images.clone()
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.num_iter):
            perturbed_images.requires_grad = True
            
            outputs = self.model(perturbed_images)
            loss = criterion(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + self.alpha * data_grad.sign()
            
            # Projection
            perturbation = torch.clamp(perturbed_images - images, -self.epsilon, self.epsilon)
            perturbed_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        perturbation = perturbed_images - images
        return perturbed_images, perturbation
    
    def evaluate(self, model, data_loader, device=DEVICE):
        """Évalue l'efficacité de l'attaque"""
        model.eval()
        correct = 0
        total = 0
        attack_success = 0
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                original_outputs = model(images)
                _, original_pred = original_outputs.max(1)
            
            perturbed_images, _ = self.attack(images, labels)
            
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            originally_correct = original_pred.eq(labels)
            now_incorrect = ~predicted.eq(labels)
            attack_success += (originally_correct & now_incorrect).sum().item()
        
        accuracy = 100. * correct / total
        success_rate = 100. * attack_success / total
        
        return accuracy, success_rate


if __name__ == "__main__":
    # Test des attaques
    from classifier import SimpleCNN
    
    model = SimpleCNN(num_classes=10, input_channels=1, image_size=28)
    model.to(DEVICE)
    model.eval()
    
    # Images de test
    images = torch.randn(4, 1, 28, 28).to(DEVICE)
    labels = torch.randint(0, 10, (4,)).to(DEVICE)
    
    print("Test des attaques:")
    print(f"Device: {DEVICE}")
    print(f"Images shape: {images.shape}")
    
    # Test FGSM
    fgsm = FGSMAttack(model, epsilon=0.3)
    perturbed_fgsm, pert_fgsm = fgsm.attack(images, labels)
    print(f"\nFGSM - Perturbation max: {pert_fgsm.abs().max().item():.4f}")
    
    # Test PGD
    pgd = PGDAttack(model, epsilon=0.3, alpha=0.01, num_iter=10)
    perturbed_pgd, pert_pgd = pgd.attack(images, labels)
    print(f"PGD - Perturbation max: {pert_pgd.abs().max().item():.4f}")
    
    # Test BIM
    bim = BIMAttack(model, epsilon=0.3, alpha=0.01, num_iter=10)
    perturbed_bim, pert_bim = bim.attack(images, labels)
    print(f"BIM - Perturbation max: {pert_bim.abs().max().item():.4f}")
