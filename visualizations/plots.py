"""
Module de visualisation des résultats
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import VIZ, RESULTS_DIR


def plot_adversarial_examples(original_images, perturbed_images, perturbations, 
                              predictions_orig, predictions_adv, labels, 
                              num_examples=5, save_path=None):
    """
    Affiche des exemples d'attaques adversariales
    
    Args:
        original_images: Images originales
        perturbed_images: Images perturbées
        perturbations: Perturbations appliquées
        predictions_orig: Prédictions sur images originales
        predictions_adv: Prédictions sur images adversariales
        labels: Vrais labels
        num_examples: Nombre d'exemples à afficher
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(num_examples, 4, figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        # Image originale
        ax = axes[i, 0] if num_examples > 1 else axes[0]
        img_orig = original_images[i].cpu()
        # Convertir (C, H, W) -> (H, W, C) pour matplotlib
        if img_orig.ndim == 3 and img_orig.shape[0] == 3:
            img_orig = img_orig.permute(1, 2, 0)
        else:
            img_orig = img_orig.squeeze()
        img_orig = np.clip(img_orig.numpy(), 0, 1)
        ax.imshow(img_orig, cmap='gray' if img_orig.ndim == 2 else None)
        ax.set_title(f'Original\nTrue: {labels[i]}\nPred: {predictions_orig[i]}')
        ax.axis('off')
        
        # Perturbation
        ax = axes[i, 1] if num_examples > 1 else axes[1]
        pert = perturbations[i].cpu()
        # Pour perturbation RGB, moyenner les canaux
        if pert.ndim == 3 and pert.shape[0] == 3:
            pert = pert.mean(dim=0)
        else:
            pert = pert.squeeze()
        pert = pert.numpy()
        ax.imshow(pert, cmap='seismic', vmin=-0.3, vmax=0.3)
        ax.set_title(f'Perturbation\nL∞: {np.abs(pert).max():.3f}')
        ax.axis('off')
        
        # Image adversariale
        ax = axes[i, 2] if num_examples > 1 else axes[2]
        img_adv = perturbed_images[i].cpu()
        # Convertir (C, H, W) -> (H, W, C) pour matplotlib
        if img_adv.ndim == 3 and img_adv.shape[0] == 3:
            img_adv = img_adv.permute(1, 2, 0)
        else:
            img_adv = img_adv.squeeze()
        img_adv = np.clip(img_adv.numpy(), 0, 1)
        ax.imshow(img_adv, cmap='gray' if img_adv.ndim == 2 else None)
        ax.set_title(f'Adversarial\nPred: {predictions_adv[i]}')
        ax.axis('off')
        
        # Différence amplifiée
        ax = axes[i, 3] if num_examples > 1 else axes[3]
        diff_orig = original_images[i].cpu()
        diff_adv = perturbed_images[i].cpu()
        if diff_orig.ndim == 3 and diff_orig.shape[0] == 3:
            # Moyenner les canaux pour la différence RGB
            diff = (diff_adv - diff_orig).mean(dim=0).numpy() * 10
        else:
            diff = (diff_adv.squeeze() - diff_orig.squeeze()).numpy() * 10
        ax.imshow(diff, cmap='seismic')
        ax.set_title('Différence (×10)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


def plot_attack_success_vs_epsilon(epsilons, accuracies, attack_success_rates, 
                                   attack_name="FGSM", save_path=None):
    """
    Affiche la précision et le taux de réussite de l'attaque en fonction d'epsilon
    
    Args:
        epsilons: Liste des valeurs d'epsilon
        accuracies: Précisions correspondantes
        attack_success_rates: Taux de réussite de l'attaque
        attack_name: Nom de l'attaque
        save_path: Chemin pour sauvegarder la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Précision
    ax1.plot(epsilons, accuracies, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax1.set_xlabel('Epsilon (ε)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Model Accuracy vs Epsilon\n({attack_name})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Taux de réussite de l'attaque
    ax2.plot(epsilons, attack_success_rates, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Epsilon (ε)', fontsize=12)
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax2.set_title(f'Attack Success Rate vs Epsilon\n({attack_name})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


def plot_payoff_matrix(payoff_matrix, attacker_strategies, defender_strategies,
                      save_path=None):
    """
    Affiche la matrice des gains du jeu
    
    Args:
        payoff_matrix: Matrice des gains
        attacker_strategies: Labels des stratégies de l'attaquant
        defender_strategies: Labels des stratégies du défenseur
        save_path: Chemin pour sauvegarder la figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer la heatmap
    sns.heatmap(payoff_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0, ax=ax, cbar_kws={'label': 'Payoff (Defender)'})
    
    # Labels
    ax.set_xlabel('Defender Strategies', fontsize=12)
    ax.set_ylabel('Attacker Strategies', fontsize=12)
    ax.set_title('Game Payoff Matrix (Defender Perspective)', fontsize=14)
    
    # Stratégies - gérer à la fois les floats (epsilon) et les strings (noms d'attaque)
    if len(attacker_strategies) <= 20:
        if isinstance(attacker_strategies[0], str):
            y_labels = attacker_strategies
        else:
            y_labels = [f'{s:.3f}' for s in attacker_strategies]
        ax.set_yticklabels(y_labels, rotation=0)
    
    if len(defender_strategies) <= 20:
        ax.set_xticklabels(defender_strategies, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


def plot_nash_equilibrium(nash_strategies_attacker, nash_strategies_defender,
                         attacker_strategies, defender_strategies, save_path=None):
    """
    Affiche l'équilibre de Nash (stratégies mixtes)
    
    Args:
        nash_strategies_attacker: Distribution de probabilité (attaquant)
        nash_strategies_defender: Distribution de probabilité (défenseur)
        attacker_strategies: Labels des stratégies de l'attaquant
        defender_strategies: Labels des stratégies du défenseur
        save_path: Chemin pour sauvegarder la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stratégie mixte de l'attaquant
    x_pos = np.arange(len(attacker_strategies))
    ax1.bar(x_pos, nash_strategies_attacker, color='#e74c3c', alpha=0.7)
    # Label adapté selon le type de stratégie
    if isinstance(attacker_strategies[0], str):
        ax1.set_xlabel('Attack Type', fontsize=12)
        ax1.set_xticklabels(attacker_strategies, rotation=45)
    else:
        ax1.set_xlabel('Epsilon (ε)', fontsize=12)
        ax1.set_xticklabels([f'{s:.2f}' for s in attacker_strategies], rotation=45)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Nash Equilibrium - Attacker Mixed Strategy', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Stratégie mixte du défenseur
    x_pos = np.arange(len(defender_strategies))
    ax2.bar(x_pos, nash_strategies_defender, color='#2ecc71', alpha=0.7)
    ax2.set_xlabel('Defense Strategy', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Nash Equilibrium - Defender Mixed Strategy', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(defender_strategies, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """
    Affiche l'historique d'entraînement
    
    Args:
        train_losses: Pertes d'entraînement par époque
        train_accs: Précisions d'entraînement par époque
        val_losses: Pertes de validation par époque
        val_accs: Précisions de validation par époque
        save_path: Chemin pour sauvegarder la figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, train_losses, 'o-', label='Train', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 's-', label='Validation', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'o-', label='Train', linewidth=2, markersize=6)
    ax2.plot(epochs, val_accs, 's-', label='Validation', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


def plot_comparison_attacks(results_dict, metric='accuracy', save_path=None):
    """
    Compare plusieurs attaques
    
    Args:
        results_dict: Dictionnaire {attack_name: {'epsilons': [...], metric: [...]}}
        metric: 'accuracy' ou 'success_rate'
        save_path: Chemin pour sauvegarder la figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (attack_name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(results['epsilons'], results[metric], 'o-', 
               label=attack_name, linewidth=2, markersize=8, color=color)
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ylabel = 'Accuracy (%)' if metric == 'accuracy' else 'Attack Success Rate (%)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Comparison of Attacks - {ylabel}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ["dpi"], bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Test du module de visualisation")
    
    # Test données
    epsilons = np.linspace(0, 0.3, 7)
    accuracies = 100 - epsilons * 200  # Simulation
    success_rates = epsilons * 200  # Simulation
    
    # Test plot
    fig = plot_attack_success_vs_epsilon(
        epsilons, 
        accuracies, 
        success_rates, 
        "FGSM",
        save_path=RESULTS_DIR / "test_plot.png"
    )
    print("Test de visualisation terminé")
