"""
Interface graphique Streamlit pour le projet THJ
Visualisation interactive des attaques adversariales
"""
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent))

from config import DEVICE, DATASETS, MODELS_DIR, MODELS
from algorithms.classifier import SimpleCNN, load_model, save_model, train_classifier, evaluate_classifier, get_model
from algorithms.attacker import FGSMAttack, PGDAttack, BIMAttack
from algorithms.game_model import AdversarialGame
from utils.data_loader import get_mnist_loaders, get_cifar10_loaders, get_sample_batch
from utils.helpers import set_seed
from visualizations.plots import (
    plot_adversarial_examples,
    plot_attack_success_vs_epsilon,
    plot_payoff_matrix,
    plot_nash_equilibrium
)

# Configuration de la page
st.set_page_config(
    page_title="Jeu d'Attaque Adversariale",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model(dataset_name, model_type="simple_cnn"):
    """Charge un modèle (sans widget interactif)"""
    model_path = MODELS_DIR / f"{dataset_name}_{model_type}_model.pth"
    
    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders()
    else:  # cifar10
        train_loader, test_loader = get_cifar10_loaders()
    
    # Créer le modèle
    pretrained = (model_type != "simple_cnn")
    model = get_model(model_type=model_type, dataset=dataset_name, pretrained=pretrained)
    model.to(DEVICE)
    
    # Charger le modèle existant
    if model_path.exists():
        model = load_model(model, model_path)
        return model, test_loader, True  # True = modèle chargé
    else:
        return model, test_loader, False  # False = modèle non entraîné


def train_model_interactive(model, dataset_name, model_type, train_loader):
    """Entraîne le modèle de manière interactive"""
    model_path = MODELS_DIR / f"{dataset_name}_{model_type}_model.pth"
    epochs = 5
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001 if model_type == "simple_cnn" else 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        status_text.text(f"Epoch {epoch+1}/{epochs}")
        loss, acc = train_classifier(model, train_loader, criterion, optimizer)
        progress_bar.progress((epoch + 1) / epochs)
        st.write(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")
    
    save_model(model, model_path)
    st.success("✅ Entraînement terminé!")
    return model


def main():
    """Fonction principale de l'application Streamlit"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🎯 Jeu d\'Attaque Adversariale</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
        Exploration des attaques adversariales en classification d'images <br>
        à travers la théorie des jeux
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Barre latérale
    st.sidebar.title("⚙️ Configuration")
    
    # Choix du dataset (AVANT le modèle pour filtrer selon le dataset)
    dataset_name = st.sidebar.selectbox(
        "📊 Dataset",
        ["mnist", "cifar10"],
        help="Choisir entre MNIST (chiffres) ou CIFAR-10 (objets)"
    )
    
    # Choix du modèle (filtré selon le dataset)
    st.sidebar.markdown("### 🤖 Modèle")
    
    # Filtrer les modèles disponibles selon le dataset
    available_models = list(MODELS.keys())
    if dataset_name == "mnist":
        # VGG-16 ne fonctionne pas avec MNIST
        available_models = [m for m in available_models if m != "vgg16"]
        st.sidebar.warning("⚠️ VGG-16 n'est pas compatible avec MNIST")
    
    model_type = st.sidebar.selectbox(
        "🎯 Architecture",
        available_models,
        format_func=lambda x: f"{MODELS[x]['name']} ({MODELS[x]['params']})",
        help="Choisir entre CNN simple ou modèles pré-entraînés ImageNet"
    )
    
    # Info sur le modèle sélectionné
    st.sidebar.info(f"📊 {MODELS[model_type]['description']}")
    
    # Choix de l'attaque
    attack_type = st.sidebar.selectbox(
        "⚔️ Type d'attaque",
        ["FGSM", "PGD", "BIM"],
        help="Fast Gradient Sign Method, Projected Gradient Descent, Basic Iterative Method"
    )
    
    # Paramètres de l'attaque
    st.sidebar.markdown("### 🎛️ Paramètres de l'attaque")
    
    epsilon = st.sidebar.slider(
        "Epsilon (ε)",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Magnitude de la perturbation"
    )
    
    if attack_type in ["PGD", "BIM"]:
        num_iter = st.sidebar.slider(
            "Nombre d'itérations",
            min_value=1,
            max_value=50,
            value=10,
            help="Nombre d'itérations pour l'attaque itérative"
        )
        
        alpha = st.sidebar.slider(
            "Alpha (taille du pas)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            help="Taille du pas à chaque itération"
        )
    
    # Seed pour reproductibilité
    seed = st.sidebar.number_input("🌱 Random Seed", value=42, min_value=0)
    set_seed(seed)
    
    # Charger le modèle
    st.markdown('<h2 class="sub-header">🤖 Chargement du modèle</h2>', unsafe_allow_html=True)
    
    try:
        model, test_loader, model_loaded = load_or_train_model(dataset_name, model_type)
        
        if model_loaded:
            st.success(f"✅ Modèle {MODELS.get(model_type, {}).get('name', model_type)} chargé avec succès!")
        else:
            st.warning(f"⚠️ Modèle non trouvé. Entraînement nécessaire.")
            if st.button("🚀 Entraîner le modèle"):
                with st.spinner("Entraînement en cours..."):
                    # Charger les données d'entraînement
                    if dataset_name == "mnist":
                        train_loader, _ = get_mnist_loaders()
                    else:
                        train_loader, _ = get_cifar10_loaders()
                    
                    model = train_model_interactive(model, dataset_name, model_type, train_loader)
                    
                    # Forcer le rechargement du cache
                    load_or_train_model.clear()
                    st.success("🎉 Modèle entraîné et sauvegardé avec succès!")
                    st.info("🔄 Rechargement de la page...")
                    st.rerun()
            else:
                st.info("👆 Cliquez sur le bouton pour entraîner le modèle")
                return
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Évaluation du modèle
    with st.spinner("Évaluation du modèle..."):
        criterion = torch.nn.CrossEntropyLoss()
        loss, accuracy = evaluate_classifier(model, test_loader, criterion)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Précision du modèle", f"{accuracy:.2f}%")
    with col2:
        st.metric("📉 Loss", f"{loss:.4f}")
    with col3:
        st.metric("🖥️ Device", str(DEVICE))
    with col4:
        st.metric("🤖 Modèle", MODELS[model_type]['name'])
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["🔍 Exemples Adversariaux", "📈 Analyse Epsilon", "🎲 Théorie des Jeux"])
    
    # TAB 1: Exemples adversariaux
    with tab1:
        st.markdown('<h2 class="sub-header">🔍 Visualisation d\'Exemples Adversariaux</h2>', unsafe_allow_html=True)
        
        num_examples = st.slider("Nombre d'exemples à afficher", 1, 10, 5)
        
        if st.button("🎲 Générer des exemples"):
            with st.spinner("Génération d'exemples adversariaux..."):
                # Récupérer un échantillon
                images, labels = get_sample_batch(dataset_name, batch_size=num_examples)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Prédictions originales
                with torch.no_grad():
                    outputs_orig = model(images)
                    _, pred_orig = outputs_orig.max(1)
                
                # Créer l'attaque
                if attack_type == "FGSM":
                    attacker = FGSMAttack(model, epsilon=epsilon)
                elif attack_type == "PGD":
                    attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
                else:  # BIM
                    attacker = BIMAttack(model, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
                
                # Générer les exemples adversariaux
                perturbed_images, perturbations = attacker.attack(images, labels)
                
                # Prédictions adversariales
                with torch.no_grad():
                    outputs_adv = model(perturbed_images)
                    _, pred_adv = outputs_adv.max(1)
                
                # Visualisation
                fig = plot_adversarial_examples(
                    images, perturbed_images, perturbations,
                    pred_orig.cpu().numpy(), pred_adv.cpu().numpy(),
                    labels.cpu().numpy(),
                    num_examples=num_examples
                )
                
                st.pyplot(fig)
                
                # Statistiques
                attack_success = (pred_orig.eq(labels) & ~pred_adv.eq(labels)).sum().item()
                success_rate = 100 * attack_success / num_examples
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Attaques réussies", f"{attack_success}/{num_examples}")
                with col2:
                    st.metric("📊 Taux de réussite", f"{success_rate:.1f}%")
    
    # TAB 2: Analyse en fonction d'epsilon
    with tab2:
        st.markdown('<h2 class="sub-header">📈 Analyse de l\'Impact d\'Epsilon</h2>', unsafe_allow_html=True)
        
        epsilon_range = st.slider(
            "Plage d'epsilon",
            min_value=0.0,
            max_value=0.5,
            value=(0.0, 0.3),
            step=0.05
        )
        
        num_points = st.slider("Nombre de points", 3, 15, 7)
        
        if st.button("🚀 Lancer l'analyse"):
            epsilons = np.linspace(epsilon_range[0], epsilon_range[1], num_points)
            accuracies = []
            success_rates = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, eps in enumerate(epsilons):
                status_text.text(f"Évaluation epsilon = {eps:.3f} ({idx+1}/{num_points})")
                
                # Créer l'attaque
                if attack_type == "FGSM":
                    attacker = FGSMAttack(model, epsilon=eps)
                elif attack_type == "PGD":
                    attacker = PGDAttack(model, epsilon=eps, alpha=alpha, num_iter=num_iter)
                else:
                    attacker = BIMAttack(model, epsilon=eps, alpha=alpha, num_iter=num_iter)
                
                acc, sr = attacker.evaluate(model, test_loader)
                accuracies.append(acc)
                success_rates.append(sr)
                
                progress_bar.progress((idx + 1) / num_points)
            
            status_text.text("✅ Analyse terminée!")
            
            # Visualisation
            fig = plot_attack_success_vs_epsilon(
                epsilons, accuracies, success_rates, attack_type
            )
            st.pyplot(fig)
            
            # Tableau de résultats
            st.markdown("### 📊 Résultats détaillés")
            import pandas as pd
            df = pd.DataFrame({
                'Epsilon': epsilons,
                'Accuracy (%)': accuracies,
                'Success Rate (%)': success_rates
            })
            st.dataframe(df.style.format({'Epsilon': '{:.3f}', 'Accuracy (%)': '{:.2f}', 'Success Rate (%)': '{:.2f}'}))
    
    # TAB 3: Théorie des jeux
    with tab3:
        st.markdown('<h2 class="sub-header">🎲 Analyse par Théorie des Jeux</h2>', unsafe_allow_html=True)
        
        st.info("""
        Cette section modélise l'interaction entre l'attaquant et le classifieur comme un jeu à somme nulle
        et calcule l'équilibre de Nash.
        
        ⚠️ **Note importante** : Les attaques adversariales sont asymétriquement puissantes. 
        Avec des epsilons élevés (>0.2) et des défenses basiques, l'attaquant gagnera souvent.
        Pour favoriser le défenseur, réduisez l'epsilon maximum ou utilisez un modèle plus robuste.
        """)
        
        num_strategies = st.slider("Nombre de stratégies de l'attaquant", 3, 15, 10)
        max_epsilon_game = st.slider("Epsilon maximum", 0.05, 0.5, 0.15, 0.05,
                                     help="⚠️ Valeurs élevées (>0.2) favorisent l'attaquant. Essayez 0.10-0.15 pour un jeu plus équilibré.")
        
        if st.button("🎯 Calculer l'équilibre de Nash"):
            with st.spinner("Calcul de la matrice des gains et de l'équilibre de Nash..."):
                # Échantillon pour le jeu
                images, labels = get_sample_batch(dataset_name, batch_size=100)
                
                # Créer le jeu
                if attack_type == "FGSM":
                    attack_class = FGSMAttack
                elif attack_type == "PGD":
                    attack_class = PGDAttack
                else:
                    attack_class = BIMAttack
                
                game = AdversarialGame(
                    model=model,
                    attack_class=attack_class,
                    dataset_sample=(images, labels),
                    num_strategies=num_strategies,
                    max_epsilon=max_epsilon_game
                )
                
                # Calcul de la matrice des gains
                payoff_matrix = game.compute_payoff_matrix()
                
                # Affichage de la matrice
                st.markdown("### 🗺️ Matrice des Gains")
                fig_matrix = plot_payoff_matrix(
                    payoff_matrix,
                    game.attacker_strategies,
                    game.defender_strategies
                )
                st.pyplot(fig_matrix)
                
                # === NOUVELLE SECTION : ANALYSE DE DOMINANCE ===
                st.markdown("### 🎯 Analyse de Dominance")
                
                dominance_results = game.analyze_dominance()
                
                # Créer deux colonnes pour attaquant et défenseur
                col_att, col_def = st.columns(2)
                
                with col_att:
                    st.markdown("#### ⚔️ Attaquant")
                    st.caption("Objectif : minimiser le payoff du défenseur")
                    
                    # Stratégie strictement dominante
                    if dominance_results['attacker']['strictly_dominant'] is not None:
                        idx = dominance_results['attacker']['strictly_dominant']
                        eps = game.attacker_strategies[idx]
                        st.success(f"✅ **Stratégie strictement dominante**\n\nε = {eps:.3f}\n\n→ Cette stratégie bat toutes les autres !")
                    else:
                        st.info("❌ Aucune stratégie strictement dominante")
                    
                    # Stratégies dominées
                    if dominance_results['attacker']['strictly_dominated']:
                        st.warning(f"🚫 **Stratégies strictement dominées** (à éliminer)")
                        dominated_eps = [f"ε = {game.attacker_strategies[idx]:.3f}" 
                                       for idx in dominance_results['attacker']['strictly_dominated']]
                        st.write("\n".join([f"- {eps}" for eps in dominated_eps]))
                    else:
                        st.success("✓ Aucune stratégie strictement dominée")
                    
                    # Stratégies faiblement dominantes
                    if dominance_results['attacker']['weakly_dominant']:
                        with st.expander("⚠️ Stratégies faiblement dominantes"):
                            weak_eps = [f"ε = {game.attacker_strategies[idx]:.3f}" 
                                      for idx in dominance_results['attacker']['weakly_dominant']]
                            st.write("\n".join([f"- {eps}" for eps in weak_eps]))
                
                with col_def:
                    st.markdown("#### 🛡️ Défenseur")
                    st.caption("Objectif : maximiser son payoff")
                    
                    # Stratégie strictement dominante
                    if dominance_results['defender']['strictly_dominant'] is not None:
                        idx = dominance_results['defender']['strictly_dominant']
                        strat = game.defender_strategies[idx]
                        st.success(f"✅ **Stratégie strictement dominante**\n\n{strat}\n\n→ Cette stratégie bat toutes les autres !")
                    else:
                        st.info("❌ Aucune stratégie strictement dominante")
                    
                    # Stratégies dominées
                    if dominance_results['defender']['strictly_dominated']:
                        st.warning(f"🚫 **Stratégies strictement dominées** (à éliminer)")
                        dominated_strats = [game.defender_strategies[idx] 
                                          for idx in dominance_results['defender']['strictly_dominated']]
                        st.write("\n".join([f"- {s}" for s in dominated_strats]))
                    else:
                        st.success("✓ Aucune stratégie strictement dominée")
                    
                    # Stratégies faiblement dominantes
                    if dominance_results['defender']['weakly_dominant']:
                        with st.expander("⚠️ Stratégies faiblement dominantes"):
                            weak_strats = [game.defender_strategies[idx] 
                                         for idx in dominance_results['defender']['weakly_dominant']]
                            st.write("\n".join([f"- {s}" for s in weak_strats]))
                
                st.markdown("---")
                # === FIN ANALYSE DE DOMINANCE ===
                
                # Analyse des stratégies pures
                st.markdown("### 📊 Analyse des Stratégies Pures")
                
                strategy_analysis = game.analyze_strategies()
                
                col_pure_att, col_pure_def = st.columns(2)
                
                with col_pure_att:
                    st.markdown("#### ⚔️ Attaquant")
                    st.metric("Meilleure stratégie pure", 
                             f"ε = {strategy_analysis['attacker']['best_epsilon']:.4f}")
                    st.metric("Gain espéré", 
                             f"{strategy_analysis['attacker']['expected_gain']:.4f}")
                    st.caption("Stratégie qui minimise le payoff du défenseur")
                
                with col_pure_def:
                    st.markdown("#### 🛡️ Défenseur")
                    st.metric("Stratégie passive", 
                             strategy_analysis['defender']['passive_strategy'])
                    st.metric("Gain minimum garanti", 
                             f"{strategy_analysis['defender']['worst_case']:.4f}")
                    st.caption("Pire cas pour le défenseur sans défense")
                
                # Valeur du jeu
                st.info(f"**🎯 Valeur approximative du jeu** : {strategy_analysis['game_value']:.4f}")
                
                st.markdown("---")
                
                # Équilibre de Nash
                st.markdown("### ⚖️ Équilibre de Nash")
                equilibria = game.find_nash_equilibrium()
                
                if equilibria:
                    for idx, (attacker_mix, defender_mix) in enumerate(equilibria):
                        st.write(f"**Équilibre {idx + 1}:**")
                        
                        # Afficher les stratégies mixtes avec format clair
                        st.markdown("#### 🎯 Stratégies Mixtes à l'Équilibre de Nash")
                        
                        col_strat1, col_strat2 = st.columns(2)
                        
                        with col_strat1:
                            st.markdown("**⚔️ Stratégie Mixte Attaquant:**")
                            # Afficher le vecteur brut
                            st.code(f"{np.array2string(attacker_mix, precision=4, separator=', ')}", language="python")
                            # Détailler chaque probabilité
                            st.markdown("**Détail des probabilités:**")
                            for i, (eps, prob) in enumerate(zip(game.attacker_strategies, attacker_mix)):
                                if prob > 0.001:
                                    st.write(f"• ε = {eps:.3f} → **{prob*100:.1f}%**")
                        
                        with col_strat2:
                            st.markdown("**🛡️ Stratégie Mixte Défenseur:**")
                            # Afficher le vecteur brut
                            st.code(f"{np.array2string(defender_mix, precision=4, separator=', ')}", language="python")
                            # Détailler chaque probabilité
                            st.markdown("**Détail des probabilités:**")
                            for j, (defense, prob) in enumerate(zip(game.defender_strategies, defender_mix)):
                                if prob > 0.001:
                                    st.write(f"• {defense} → **{prob*100:.1f}%**")
                        
                        # Epsilon moyen optimal
                        epsilon_mean = np.dot(attacker_mix, game.attacker_strategies)
                        st.metric("🎯 Epsilon moyen optimal", f"{epsilon_mean:.4f}")
                        
                        # Calculer le payoff à l'équilibre
                        equilibrium_payoff_defender = attacker_mix @ payoff_matrix @ defender_mix
                        equilibrium_accuracy = (equilibrium_payoff_defender + 1) / 2  # Conversion [-1,1] → [0,1]
                        
                        # Déterminer le gagnant
                        st.markdown("### 🏆 Résultat du Jeu")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            if equilibrium_accuracy > 0.5:
                                st.success(f"""
                                ### 🛡️ **VICTOIRE DU DÉFENSEUR** 🛡️
                                
                                **Précision à l'équilibre : {equilibrium_accuracy*100:.2f}%**
                                
                                Le défenseur parvient à maintenir une précision supérieure à 50% 
                                malgré les attaques adversariales optimales.
                                
                                ✅ Les stratégies de défense sont efficaces contre ce niveau d'attaque.
                                """)
                            elif equilibrium_accuracy < 0.5:
                                st.error(f"""
                                ### ⚔️ **VICTOIRE DE L'ATTAQUANT** ⚔️
                                
                                **Précision à l'équilibre : {equilibrium_accuracy*100:.2f}%**
                                
                                L'attaquant réussit à réduire la précision du modèle en dessous de 50%
                                avec sa stratégie optimale.
                                
                                💡 **C'est normal !** Les attaques adversariales sont très puissantes.
                                Pour favoriser le défenseur : réduisez epsilon max à 0.10-0.15 ou utilisez un modèle plus robuste.
                                """)
                            else:
                                st.warning(f"""
                                ### ⚖️ **ÉGALITÉ PARFAITE** ⚖️
                                
                                **Précision à l'équilibre : {equilibrium_accuracy*100:.2f}%**
                                
                                Les deux joueurs obtiennent un résultat équilibré à 50%.
                                """)
                        
                        # Visualisation et analyse
                        # Détecter si c'est une stratégie pure (une seule probabilité > 95%)
                        is_attacker_pure = np.max(attacker_mix) > 0.95
                        is_defender_pure = np.max(defender_mix) > 0.95
                        
                        # === TOUJOURS AFFICHER L'ANALYSE DES STRATÉGIES ===
                        st.markdown("### 📊 Analyse de l'Équilibre")
                        
                        if is_attacker_pure and is_defender_pure:
                            st.info("**Type d'équilibre** : Stratégies Pures (une seule stratégie par joueur)")
                        else:
                            st.success("**Type d'équilibre** : Stratégies Mixtes (plusieurs stratégies actives)")
                        
                        # Analyse approfondie
                        mixed_analysis = game.analyze_mixed_strategy_equilibrium(attacker_mix, defender_mix)
                        
                        # Deux colonnes : Attaquant et Défenseur
                        col_att_mix, col_def_mix = st.columns(2)
                        
                        with col_att_mix:
                            st.markdown("#### ⚔️ Attaquant")
                            
                            att_analysis = mixed_analysis['attacker']
                            
                            # Support
                            st.write(f"**📊 Support : {att_analysis['support_size']} stratégie(s)**")
                            for idx in att_analysis['support']:
                                eps = game.attacker_strategies[idx]
                                prob = attacker_mix[idx]
                                payoff = att_analysis['expected_payoffs'][idx]
                                if prob > 0.01:  # Afficher seulement si probabilité significative
                                    st.write(f"• ε={eps:.3f} : {prob*100:.1f}% (payoff: {payoff:.4f})")
                            
                            # Propriété d'indifférence
                            if att_analysis['support_size'] > 1:
                                st.write("**🎲 Propriété d'indifférence**")
                                if att_analysis['is_indifferent']:
                                    st.success(f"✅ Vérifié (variance: {att_analysis['variance_in_support']:.6f})")
                                    st.caption("→ L'attaquant est indifférent entre ses options")
                                else:
                                    st.warning(f"⚠️ Variance non nulle ({att_analysis['variance_in_support']:.6f})")
                            else:
                                st.caption("Stratégie pure : un seul choix optimal")
                            
                            # Stratégies hors support
                            outside_support = [i for i in range(len(game.attacker_strategies)) 
                                             if i not in att_analysis['support']]
                            if outside_support and len(outside_support) <= 10:
                                with st.expander(f"📈 Stratégies hors support ({len(outside_support)})"):
                                    for i in outside_support[:10]:  # Limiter à 10
                                        eps = game.attacker_strategies[i]
                                        payoff = att_analysis['expected_payoffs'][i]
                                        st.write(f"• ε={eps:.3f} : payoff={payoff:.4f}")
                                    if att_analysis['dominated_outside']:
                                        st.success("✅ Toutes dominées")
                                    else:
                                        st.error("⚠️ Anomalie détectée!")
                        
                        with col_def_mix:
                            st.markdown("#### 🛡️ Défenseur")
                            
                            def_analysis = mixed_analysis['defender']
                            
                            # Support
                            st.write(f"**📊 Support : {def_analysis['support_size']} stratégie(s)**")
                            for idx in def_analysis['support']:
                                strat = game.defender_strategies[idx]
                                prob = defender_mix[idx]
                                payoff = def_analysis['expected_payoffs'][idx]
                                if prob > 0.01:
                                    st.write(f"• {strat} : {prob*100:.1f}% (payoff: {payoff:.4f})")
                            
                            # Propriété d'indifférence
                            if def_analysis['support_size'] > 1:
                                st.write("**🎲 Propriété d'indifférence**")
                                if def_analysis['is_indifferent']:
                                    st.success(f"✅ Vérifié (variance: {def_analysis['variance_in_support']:.6f})")
                                    st.caption("→ Le défenseur est indifférent entre ses options")
                                else:
                                    st.warning(f"⚠️ Variance non nulle ({def_analysis['variance_in_support']:.6f})")
                            else:
                                st.caption("Stratégie pure : un seul choix optimal")
                            
                            # Stratégies hors support
                            outside_support = [j for j in range(len(game.defender_strategies)) 
                                             if j not in def_analysis['support']]
                            if outside_support:
                                with st.expander(f"📈 Stratégies hors support ({len(outside_support)})"):
                                    for j in outside_support:
                                        strat = game.defender_strategies[j]
                                        payoff = def_analysis['expected_payoffs'][j]
                                        st.write(f"• {strat} : payoff={payoff:.4f}")
                                    if def_analysis['dominated_outside']:
                                        st.success("✅ Toutes dominées")
                                    else:
                                        st.error("⚠️ Anomalie détectée!")
                        
                        # Validation globale
                        st.markdown("---")
                        if mixed_analysis['is_valid_nash']:
                            st.success("""
                            **✅ ÉQUILIBRE DE NASH VALIDE**
                            - Stratégies du support sont optimales
                            - Stratégies hors support sont dominées  
                            - Aucun joueur ne peut améliorer son payoff en déviant
                            """)
                        else:
                            st.error("⚠️ ATTENTION : Équilibre potentiellement invalide !")
                        
                        # Graphiques (seulement si stratégie mixte)
                        if not (is_attacker_pure and is_defender_pure):
                            st.markdown("### 📊 Distribution des Probabilités")
                            fig_nash = plot_nash_equilibrium(
                                attacker_mix, defender_mix,
                                game.attacker_strategies,
                                game.defender_strategies
                            )
                            st.pyplot(fig_nash)
                        
                        # === FIN ANALYSE ===
    
   

if __name__ == "__main__":
    main()
