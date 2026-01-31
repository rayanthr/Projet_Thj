"""
Script principal pour exécuter les expérimentations
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path

from config import DEVICE, MODELS_DIR, RESULTS_DIR, TRAINING, MODELS
from algorithms.classifier import SimpleCNN, train_classifier, evaluate_classifier, save_model, get_model
from algorithms.attacker import FGSMAttack, PGDAttack, BIMAttack
from algorithms.game_model import AdversarialGame
from utils.data_loader import get_mnist_loaders, get_cifar10_loaders, get_sample_batch
from utils.helpers import set_seed, save_experiment_results, get_device_info
from visualizations.plots import (
    plot_adversarial_examples,
    plot_attack_success_vs_epsilon,
    plot_training_history
)


def train_model(dataset_name="mnist", model_type="simple_cnn", epochs=10):
    """
    Entraîne un modèle CNN sur MNIST ou CIFAR-10
    
    Args:
        dataset_name: "mnist" ou "cifar10"
        model_type: "simple_cnn", "resnet50", etc.
        epochs: Nombre d'époques d'entraînement
    """
    print(f"\n{'='*60}")
    print(f"Entraînement du modèle {MODELS.get(model_type, {}).get('name', model_type)} sur {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # Chargement des données
    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders()
    else:
        train_loader, test_loader = get_cifar10_loaders()
    
    # Création du modèle
    pretrained = (model_type != "simple_cnn")
    model = get_model(model_type=model_type, dataset=dataset_name, pretrained=pretrained)
    model.to(DEVICE)
    
    # Configuration de l'entraînement
    criterion = nn.CrossEntropyLoss()
    # Learning rate adapté
    lr = 0.001 if model_type == "simple_cnn" else 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Historique
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Entraînement
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_classifier(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc = evaluate_classifier(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Sauvegarde
    model_path = MODELS_DIR / f"{dataset_name}_{model_type}_model.pth"
    save_model(model, model_path)
    
    # Visualisation
    fig = plot_training_history(train_losses, train_accs, val_losses, val_accs,
                                save_path=RESULTS_DIR / f"{dataset_name}_training.png")
    
    return model


def evaluate_attack(model, dataset_name="mnist", attack_type="fgsm", epsilon=0.3):
    """
    Évalue une attaque adversariale
    
    Args:
        model: Modèle à attaquer
        dataset_name: "mnist" ou "cifar10"
        attack_type: "fgsm", "pgd", ou "bim"
        epsilon: Magnitude de la perturbation
    """
    print(f"\n{'='*60}")
    print(f"Évaluation de l'attaque {attack_type.upper()} avec epsilon={epsilon}")
    print(f"{'='*60}\n")
    
    # Chargement des données
    if dataset_name == "mnist":
        _, test_loader = get_mnist_loaders()
    else:
        _, test_loader = get_cifar10_loaders()
    
    # Création de l'attaque
    if attack_type == "fgsm":
        attacker = FGSMAttack(model, epsilon=epsilon)
    elif attack_type == "pgd":
        attacker = PGDAttack(model, epsilon=epsilon, alpha=0.01, num_iter=40)
    else:  # bim
        attacker = BIMAttack(model, epsilon=epsilon, alpha=0.01, num_iter=10)
    
    # Évaluation
    accuracy, success_rate = attacker.evaluate(model, test_loader)
    
    print(f"\nRésultats:")
    print(f"  Précision sur images adversariales: {accuracy:.2f}%")
    print(f"  Taux de réussite de l'attaque: {success_rate:.2f}%")
    
    # Visualisation de quelques exemples
    images, labels = get_sample_batch(dataset_name, batch_size=5)
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    with torch.no_grad():
        outputs_orig = model(images)
        _, pred_orig = outputs_orig.max(1)
    
    perturbed_images, perturbations = attacker.attack(images, labels)
    
    with torch.no_grad():
        outputs_adv = model(perturbed_images)
        _, pred_adv = outputs_adv.max(1)
    
    fig = plot_adversarial_examples(
        images, perturbed_images, perturbations,
        pred_orig.cpu().numpy(), pred_adv.cpu().numpy(),
        labels.cpu().numpy(),
        num_examples=5,
        save_path=RESULTS_DIR / f"{attack_type}_examples.png"
    )
    
    return accuracy, success_rate


def analyze_epsilon_impact(model, dataset_name="mnist", attack_type="fgsm"):
    """
    Analyse l'impact d'epsilon sur l'efficacité de l'attaque
    """
    print(f"\n{'='*60}")
    print(f"Analyse de l'impact d'epsilon pour {attack_type.upper()}")
    print(f"{'='*60}\n")
    
    # Chargement des données
    if dataset_name == "mnist":
        _, test_loader = get_mnist_loaders()
    else:
        _, test_loader = get_cifar10_loaders()
    
    # Test pour différentes valeurs d'epsilon
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []
    success_rates = []
    
    for eps in epsilons:
        print(f"\nÉvaluation avec epsilon = {eps}")
        
        if attack_type == "fgsm":
            attacker = FGSMAttack(model, epsilon=eps)
        elif attack_type == "pgd":
            attacker = PGDAttack(model, epsilon=eps, alpha=0.01, num_iter=40)
        else:
            attacker = BIMAttack(model, epsilon=eps, alpha=0.01, num_iter=10)
        
        if eps == 0:
            # Pas d'attaque
            criterion = nn.CrossEntropyLoss()
            _, acc = evaluate_classifier(model, test_loader, criterion)
            accuracies.append(acc)
            success_rates.append(0.0)
        else:
            acc, sr = attacker.evaluate(model, test_loader)
            accuracies.append(acc)
            success_rates.append(sr)
        
        print(f"Précision: {acc:.2f}%, Taux de réussite: {sr:.2f}%")
    
    # Visualisation
    fig = plot_attack_success_vs_epsilon(
        epsilons, accuracies, success_rates, attack_type.upper(),
        save_path=RESULTS_DIR / f"{attack_type}_epsilon_analysis.png"
    )
    
    # Sauvegarde des résultats
    results = {
        "dataset": dataset_name,
        "attack": attack_type,
        "epsilons": epsilons,
        "accuracies": accuracies,
        "success_rates": success_rates
    }
    save_experiment_results(results, f"{attack_type}_epsilon_analysis")
    
    return results


def game_theory_analysis(model, dataset_name="mnist", attack_type="fgsm"):
    """
    Analyse de théorie des jeux
    """
    print(f"\n{'='*60}")
    print("Analyse de Théorie des Jeux")
    print(f"{'='*60}\n")
    
    # Échantillon de données
    images, labels = get_sample_batch(dataset_name, batch_size=100)
    
    # Créer le jeu
    if attack_type == "fgsm":
        attack_class = FGSMAttack
    elif attack_type == "pgd":
        attack_class = PGDAttack
    else:
        attack_class = BIMAttack
    
    game = AdversarialGame(
        model=model,
        attack_class=attack_class,
        dataset_sample=(images, labels),
        num_strategies=10,
        max_epsilon=0.3
    )
    
    # Calcul de la matrice des gains
    payoff_matrix = game.compute_payoff_matrix()
    
    # Analyse
    game.analyze_strategies()
    
    # Équilibre de Nash
    equilibria = game.find_nash_equilibrium()
    
    # Sauvegarde
    results = game.get_payoff_summary()
    results["equilibria"] = [
        {"attacker": eq[0].tolist(), "defender": eq[1].tolist()}
        for eq in equilibria
    ]
    save_experiment_results(results, f"game_theory_{attack_type}")
    
    return results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Projet THJ - Attaques Adversariales")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "attack", "analyze", "game"],
                       help="Mode d'exécution")
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar10"],
                       help="Dataset à utiliser")
    parser.add_argument("--model", type=str, default="simple_cnn",
                       choices=list(MODELS.keys()),
                       help="Type de modèle à utiliser")
    parser.add_argument("--attack", type=str, default="fgsm",
                       choices=["fgsm", "pgd", "bim"],
                       help="Type d'attaque")
    parser.add_argument("--epsilon", type=float, default=0.3,
                       help="Valeur d'epsilon pour l'attaque")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Nombre d'époques pour l'entraînement")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Configuration
    set_seed(args.seed)
    get_device_info()
    
    # Exécution selon le mode
    if args.mode == "train":
        train_model(args.dataset, args.model, args.epochs)
    
    elif args.mode == "attack":
        # Charger le modèle
        model_path = MODELS_DIR / f"{args.dataset}_{args.model}_model.pth"
        if not model_path.exists():
            print("⚠️ Modèle non trouvé. Entraînement nécessaire.")
            model = train_model(args.dataset, args.model, args.epochs)
        else:
            pretrained = (args.model != "simple_cnn")
            model = get_model(model_type=args.model, dataset=args.dataset, pretrained=pretrained)
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Modèle chargé: {model_path}")
        
        evaluate_attack(model, args.dataset, args.attack, args.epsilon)
    
    elif args.mode == "analyze":
        # Charger le modèle
        model_path = MODELS_DIR / f"{args.dataset}_{args.model}_model.pth"
        if not model_path.exists():
            print("⚠️ Modèle non trouvé. Entraînement nécessaire.")
            model = train_model(args.dataset, args.model, args.epochs)
        else:
            pretrained = (args.model != "simple_cnn")
            model = get_model(model_type=args.model, dataset=args.dataset, pretrained=pretrained)
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Modèle chargé: {model_path}")
        
        analyze_epsilon_impact(model, args.dataset, args.attack)
    
    elif args.mode == "game":
        # Charger le modèle
        model_path = MODELS_DIR / f"{args.dataset}_{args.model}_model.pth"
        if not model_path.exists():
            print("⚠️ Modèle non trouvé. Entraînement nécessaire.")
            model = train_model(args.dataset, args.model, args.epochs)
        else:
            pretrained = (args.model != "simple_cnn")
            model = get_model(model_type=args.model, dataset=args.dataset, pretrained=pretrained)
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Modèle chargé: {model_path}")
        
        game_theory_analysis(model, args.dataset, args.attack)
    
    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
