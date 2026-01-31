"""
Script de comparaison des modèles pour le rapport
Génère des graphiques comparatifs entre CNN simple et modèles pré-entraînés
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from config import DEVICE, MODELS_DIR, RESULTS_DIR
from algorithms.classifier import get_model, train_classifier, evaluate_classifier, save_model
from algorithms.attacker import FGSMAttack, PGDAttack, BIMAttack
from utils.data_loader import get_mnist_loaders, get_cifar10_loaders, get_sample_batch
from utils.helpers import set_seed, format_time, count_parameters
import time


def compare_model_architectures(models_list, dataset_name="cifar10"):
    """
    Compare les architectures de différents modèles
    
    Args:
        models_list: Liste des noms de modèles à comparer
        dataset_name: "mnist" ou "cifar10"
    
    Returns:
        results: Dictionnaire avec les statistiques
    """
    print(f"\n{'='*60}")
    print(f"Comparaison des architectures de modèles - {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for model_name in models_list:
        print(f"\n📊 Analyse de {model_name}...")
        
        pretrained = (model_name != "simple_cnn")
        model = get_model(model_type=model_name, dataset=dataset_name, pretrained=False)
        
        # Compter les paramètres
        total_params, trainable_params = count_parameters(model)
        
        # Taille approximative en mémoire (en MB)
        param_size = total_params * 4 / (1024 * 1024)  # 4 bytes par paramètre (float32)
        
        results[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': param_size
        }
        
        print(f"  Paramètres totaux: {total_params:,}")
        print(f"  Paramètres entraînables: {trainable_params:,}")
        print(f"  Taille approximative: {param_size:.2f} MB")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_names = [m.replace('_', ' ').title() for m in models_list]
    params = [results[m]['total_params'] / 1e6 for m in models_list]  # En millions
    sizes = [results[m]['size_mb'] for m in models_list]
    
    # Graphique 1: Nombre de paramètres
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_list)))
    bars1 = axes[0].bar(model_names, params, color=colors, alpha=0.8)
    axes[0].set_ylabel('Paramètres (Millions)', fontsize=12)
    axes[0].set_title('Nombre de Paramètres par Modèle', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Annotations
    for bar, param in zip(bars1, params):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:.1f}M',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Graphique 2: Taille mémoire
    bars2 = axes[1].bar(model_names, sizes, color=colors, alpha=0.8)
    axes[1].set_ylabel('Taille Mémoire (MB)', fontsize=12)
    axes[1].set_title('Taille Mémoire des Modèles', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Annotations
    for bar, size in zip(bars2, sizes):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:.1f} MB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = RESULTS_DIR / f"comparison_architectures_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: {save_path}")
    
    return results, fig


def compare_training_performance(models_list, dataset_name="cifar10", epochs=5):
    """
    Compare les performances d'entraînement des modèles
    """
    print(f"\n{'='*60}")
    print(f"Comparaison des performances d'entraînement - {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders()
    else:
        train_loader, test_loader = get_cifar10_loaders()
    
    results = {}
    
    for model_name in models_list:
        print(f"\n🚀 Entraînement de {model_name}...")
        
        # Créer le modèle
        pretrained = (model_name != "simple_cnn")
        model = get_model(model_type=model_name, dataset=dataset_name, pretrained=pretrained)
        model.to(DEVICE)
        
        # Configuration
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if model_name == "simple_cnn" else 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Historique
        train_accs = []
        val_accs = []
        train_times = []
        
        # Entraînement
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_classifier(model, train_loader, criterion, optimizer)
            train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc = evaluate_classifier(model, test_loader, criterion)
            val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            train_times.append(epoch_time)
            
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        
        # Sauvegarder le modèle
        model_path = MODELS_DIR / f"{dataset_name}_{model_name}_model.pth"
        save_model(model, model_path)
        
        results[model_name] = {
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'final_val_acc': val_accs[-1],
            'train_times': train_times,
            'total_time': total_time,
            'avg_epoch_time': np.mean(train_times)
        }
        
        print(f"  ✅ Précision finale: {val_accs[-1]:.2f}%")
        print(f"  ⏱️ Temps total: {format_time(total_time)}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_list)))
    
    # Graphique 1: Évolution de la précision (entraînement)
    for idx, model_name in enumerate(models_list):
        model_label = model_name.replace('_', ' ').title()
        axes[0, 0].plot(range(1, epochs+1), results[model_name]['train_accuracies'], 
                       'o-', label=model_label, linewidth=2, markersize=8, color=colors[idx])
    axes[0, 0].set_xlabel('Époque', fontsize=12)
    axes[0, 0].set_ylabel('Précision Train (%)', fontsize=12)
    axes[0, 0].set_title('Évolution de la Précision (Entraînement)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 105])
    
    # Graphique 2: Évolution de la précision (validation)
    for idx, model_name in enumerate(models_list):
        model_label = model_name.replace('_', ' ').title()
        axes[0, 1].plot(range(1, epochs+1), results[model_name]['val_accuracies'], 
                       's-', label=model_label, linewidth=2, markersize=8, color=colors[idx])
    axes[0, 1].set_xlabel('Époque', fontsize=12)
    axes[0, 1].set_ylabel('Précision Validation (%)', fontsize=12)
    axes[0, 1].set_title('Évolution de la Précision (Validation)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # Graphique 3: Précision finale
    model_names = [m.replace('_', ' ').title() for m in models_list]
    final_accs = [results[m]['final_val_acc'] for m in models_list]
    bars = axes[1, 0].bar(model_names, final_accs, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Précision Validation (%)', fontsize=12)
    axes[1, 0].set_title('Précision Finale des Modèles', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Graphique 4: Temps d'entraînement
    train_times = [results[m]['total_time'] for m in models_list]
    bars = axes[1, 1].bar(model_names, train_times, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Temps (secondes)', fontsize=12)
    axes[1, 1].set_title('Temps d\'Entraînement Total', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       format_time(t),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = RESULTS_DIR / f"comparison_training_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: {save_path}")
    
    return results, fig


def compare_adversarial_robustness(models_list, dataset_name="cifar10", attack_type="fgsm"):
    """
    Compare la robustesse adversariale des modèles
    """
    print(f"\n{'='*60}")
    print(f"Comparaison de la robustesse adversariale - {attack_type.upper()}")
    print(f"{'='*60}\n")
    
    if dataset_name == "mnist":
        _, test_loader = get_mnist_loaders()
    else:
        _, test_loader = get_cifar10_loaders()
    
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = {}
    
    for model_name in models_list:
        print(f"\n⚔️ Test de {model_name}...")
        
        # Charger le modèle
        model_path = MODELS_DIR / f"{dataset_name}_{model_name}_model.pth"
        if not model_path.exists():
            print(f"  ⚠️ Modèle non trouvé: {model_path}")
            continue
        
        pretrained = (model_name != "simple_cnn")
        model = get_model(model_type=model_name, dataset=dataset_name, pretrained=pretrained)
        model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        accuracies = []
        success_rates = []
        
        for eps in epsilons:
            print(f"  Epsilon = {eps:.3f}...", end=" ")
            
            if eps == 0:
                criterion = nn.CrossEntropyLoss()
                _, acc = evaluate_classifier(model, test_loader, criterion)
                accuracies.append(acc)
                success_rates.append(0.0)
            else:
                if attack_type == "fgsm":
                    attacker = FGSMAttack(model, epsilon=eps)
                elif attack_type == "pgd":
                    attacker = PGDAttack(model, epsilon=eps, alpha=0.01, num_iter=10)
                else:
                    attacker = BIMAttack(model, epsilon=eps, alpha=0.01, num_iter=10)
                
                acc, sr = attacker.evaluate(model, test_loader)
                accuracies.append(acc)
                success_rates.append(sr)
            
            print(f"Acc: {acc:.2f}%")
        
        results[model_name] = {
            'epsilons': epsilons,
            'accuracies': accuracies,
            'success_rates': success_rates
        }
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_list)))
    
    # Graphique 1: Précision vs Epsilon
    for idx, model_name in enumerate(models_list):
        if model_name not in results:
            continue
        model_label = model_name.replace('_', ' ').title()
        axes[0].plot(results[model_name]['epsilons'], 
                    results[model_name]['accuracies'],
                    'o-', label=model_label, linewidth=3, markersize=10, color=colors[idx])
    
    axes[0].set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Précision (%)', fontsize=14, fontweight='bold')
    axes[0].set_title(f'Robustesse Face aux Attaques {attack_type.upper()}', 
                     fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(True, alpha=0.3, linewidth=1.5)
    axes[0].set_ylim([0, 105])
    axes[0].tick_params(labelsize=11)
    
    # Graphique 2: Taux de réussite de l'attaque
    for idx, model_name in enumerate(models_list):
        if model_name not in results:
            continue
        model_label = model_name.replace('_', ' ').title()
        axes[1].plot(results[model_name]['epsilons'], 
                    results[model_name]['success_rates'],
                    's-', label=model_label, linewidth=3, markersize=10, color=colors[idx])
    
    axes[1].set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Taux de Réussite de l\'Attaque (%)', fontsize=14, fontweight='bold')
    axes[1].set_title(f'Efficacité de l\'Attaque {attack_type.upper()}', 
                     fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12, loc='lower right')
    axes[1].grid(True, alpha=0.3, linewidth=1.5)
    axes[1].set_ylim([0, 105])
    axes[1].tick_params(labelsize=11)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / f"comparison_robustness_{attack_type}_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: {save_path}")
    
    return results, fig


def generate_comparison_report(models_list, dataset_name="cifar10"):
    """
    Génère un rapport complet de comparaison
    """
    print(f"\n{'='*60}")
    print("GÉNÉRATION DU RAPPORT DE COMPARAISON")
    print(f"{'='*60}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'models': models_list,
        'results': {}
    }
    
    # 1. Architecture
    print("\n📊 1. Comparaison des architectures...")
    arch_results, _ = compare_model_architectures(models_list, dataset_name)
    report['results']['architecture'] = arch_results
    
    # 2. Performance d'entraînement (avec peu d'époques pour la démo)
    print("\n🚀 2. Comparaison des performances d'entraînement...")
    train_results, _ = compare_training_performance(models_list, dataset_name, epochs=3)
    report['results']['training'] = train_results
    
    # 3. Robustesse adversariale
    for attack in ['fgsm', 'pgd']:
        print(f"\n⚔️ 3. Comparaison de la robustesse ({attack.upper()})...")
        robust_results, _ = compare_adversarial_robustness(models_list, dataset_name, attack)
        report['results'][f'robustness_{attack}'] = robust_results
    
    # Sauvegarder le rapport JSON
    report_path = RESULTS_DIR / f"comparison_report_{dataset_name}_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✅ Rapport JSON sauvegardé: {report_path}")
    
    # Créer un résumé visuel
    create_summary_report(report, dataset_name)
    
    print("\n" + "="*60)
    print("✅ RAPPORT DE COMPARAISON TERMINÉ")
    print("="*60)
    
    return report


def create_summary_report(report, dataset_name):
    """
    Crée un graphique résumé de toutes les comparaisons
    """
    models_list = report['models']
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_list)))
    model_names = [m.replace('_', ' ').title() for m in models_list]
    
    # 1. Nombre de paramètres
    ax1 = fig.add_subplot(gs[0, 0])
    params = [report['results']['architecture'][m]['total_params'] / 1e6 for m in models_list]
    ax1.barh(model_names, params, color=colors, alpha=0.8)
    ax1.set_xlabel('Paramètres (Millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Taille des Modèles', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Précision finale
    ax2 = fig.add_subplot(gs[0, 1])
    final_accs = [report['results']['training'][m]['final_val_acc'] for m in models_list]
    bars = ax2.bar(range(len(models_list)), final_accs, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(models_list)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Précision (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Précision de Classification', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Temps d'entraînement
    ax3 = fig.add_subplot(gs[0, 2])
    times = [report['results']['training'][m]['total_time'] for m in models_list]
    ax3.barh(model_names, times, color=colors, alpha=0.8)
    ax3.set_xlabel('Temps (secondes)', fontsize=11, fontweight='bold')
    ax3.set_title('Temps d\'Entraînement', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Robustesse FGSM (ε=0.3)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'robustness_fgsm' in report['results']:
        robust_accs = []
        for m in models_list:
            if m in report['results']['robustness_fgsm']:
                accs = report['results']['robustness_fgsm'][m]['accuracies']
                # Prendre la précision à epsilon=0.3 (dernière valeur)
                robust_accs.append(accs[-1])
            else:
                robust_accs.append(0)
        
        bars = ax4.bar(range(len(models_list)), robust_accs, color=colors, alpha=0.8)
        ax4.set_xticks(range(len(models_list)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.set_ylabel('Précision (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Robustesse FGSM (ε=0.3)', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 105])
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, robust_accs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Comparaison Précision Clean vs Adversariale
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(models_list))
    width = 0.35
    
    clean_accs = [report['results']['training'][m]['final_val_acc'] for m in models_list]
    if 'robustness_fgsm' in report['results']:
        adv_accs = []
        for m in models_list:
            if m in report['results']['robustness_fgsm']:
                adv_accs.append(report['results']['robustness_fgsm'][m]['accuracies'][-1])
            else:
                adv_accs.append(0)
    
        bars1 = ax5.bar(x - width/2, clean_accs, width, label='Clean', color='#2ecc71', alpha=0.8)
        bars2 = ax5.bar(x + width/2, adv_accs, width, label='Adversarial (ε=0.3)', color='#e74c3c', alpha=0.8)
        
        ax5.set_ylabel('Précision (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Clean vs Adversarial', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names, rotation=45, ha='right')
        ax5.legend(fontsize=10)
        ax5.set_ylim([0, 105])
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. Trade-off: Précision vs Robustesse
    ax6 = fig.add_subplot(gs[1, 2])
    if 'robustness_fgsm' in report['results']:
        for idx, m in enumerate(models_list):
            if m in report['results']['robustness_fgsm']:
                clean_acc = report['results']['training'][m]['final_val_acc']
                robust_acc = report['results']['robustness_fgsm'][m]['accuracies'][-1]
                ax6.scatter(clean_acc, robust_acc, s=300, color=colors[idx], 
                          alpha=0.7, edgecolors='black', linewidth=2, label=model_names[idx])
        
        ax6.set_xlabel('Précision Clean (%)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Précision Adversariale (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Trade-off Précision-Robustesse', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
    
    fig.suptitle(f'Rapport de Comparaison - {dataset_name.upper()}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    save_path = RESULTS_DIR / f"comparison_summary_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Résumé visuel sauvegardé: {save_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparaison de modèles pour le rapport")
    parser.add_argument("--models", nargs='+', default=["simple_cnn", "resnet50"],
                       help="Liste des modèles à comparer")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["mnist", "cifar10"],
                       help="Dataset à utiliser")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Générer le rapport complet
    report = generate_comparison_report(args.models, args.dataset)
    
    print("\n📊 Tous les graphiques ont été générés dans:", RESULTS_DIR)
    print("\nGraphiques disponibles:")
    print(f"  1. comparison_architectures_{args.dataset}.png")
    print(f"  2. comparison_training_{args.dataset}.png")
    print(f"  3. comparison_robustness_fgsm_{args.dataset}.png")
    print(f"  4. comparison_robustness_pgd_{args.dataset}.png")
    print(f"  5. comparison_summary_{args.dataset}.png")


if __name__ == "__main__":
    main()
