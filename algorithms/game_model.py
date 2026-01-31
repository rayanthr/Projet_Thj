"""
Modélisation du jeu adversarial avec théorie des jeux
Calcul de l'équilibre de Nash et analyse des stratégies
"""
import numpy as np
import torch
import nashpy as nash
from itertools import product
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, GAME_THEORY
from .defender import get_defense


class AdversarialGame:
    """
    Modélisation du jeu à somme nulle entre Attaquant et Classifieur
    
    Joueur 1 (Attaquant): Choisit l'intensité de perturbation epsilon
    Joueur 2 (Classifieur): Peut avoir différentes stratégies de défense
    """
    
    def __init__(self, model, attack_class, dataset_sample, num_strategies=10, max_epsilon=0.3, 
                 defense_strategies=None):
        """
        Args:
            model: Modèle CNN du classifieur
            attack_class: Classe d'attaque (FGSM, PGD, etc.)
            dataset_sample: Échantillon de données pour évaluer les stratégies
            num_strategies: Nombre de stratégies discrètes pour l'attaquant
            max_epsilon: Perturbation maximale
            defense_strategies: Liste des stratégies de défense à utiliser
        """
        self.model = model
        self.attack_class = attack_class
        self.dataset_sample = dataset_sample
        self.num_strategies = num_strategies
        self.max_epsilon = max_epsilon
        
        # Stratégies de l'attaquant: différents niveaux d'epsilon
        self.attacker_strategies = np.linspace(0, max_epsilon, num_strategies)
        
        # Stratégies du défenseur: plusieurs options de défense
        if defense_strategies is None:
            # Par défaut: 5 stratégies de défense
            self.defender_strategies = [
                "passive",               # Aucune défense
                "gaussian_noise",        # Ajout de bruit
                "bit_depth_reduction",   # Quantification
                "median_filter",         # Filtre médian
                "jpeg_compression"       # Compression JPEG
            ]
        else:
            self.defender_strategies = defense_strategies
        
        # Matrice des gains (sera calculée)
        self.payoff_matrix = None
        self.nash_equilibrium = None
        
    def compute_payoff_matrix(self):
        """
        Calcule la matrice des gains du jeu
        
        Gain pour le Défenseur (J2):
            +1 si classification correcte
            -1 si classification incorrecte
        
        Le jeu est à somme nulle, donc gain J1 = -gain J2
        
        Returns:
            payoff_defender: Matrice des gains du défenseur (num_attacker_strat x num_defender_strat)
        """
        print("Calcul de la matrice des gains...")
        
        num_attacker_strat = len(self.attacker_strategies)
        num_defender_strat = len(self.defender_strategies)
        
        payoff_defender = np.zeros((num_attacker_strat, num_defender_strat))
        
        self.model.eval()
        images, labels = self.dataset_sample
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Pour chaque stratégie de l'attaquant
        for i, epsilon in enumerate(self.attacker_strategies):
            print(f"Évaluation epsilon = {epsilon:.3f} ({i+1}/{num_attacker_strat})")
            
            # Pour chaque stratégie du défenseur
            for j, defense in enumerate(self.defender_strategies):
                # Génération d'attaque avec cet epsilon
                if epsilon == 0:
                    # Pas d'attaque
                    perturbed_images = images
                else:
                    attacker = self.attack_class(self.model, epsilon=epsilon)
                    perturbed_images, _ = attacker.attack(images, labels)
                
                # Application de la défense (si applicable)
                defended_images = self._apply_defense(perturbed_images, defense)
                
                # Évaluation de la précision
                with torch.no_grad():
                    outputs = self.model(defended_images)
                    _, predicted = outputs.max(1)
                    correct = predicted.eq(labels).sum().item()
                    accuracy = correct / labels.size(0)
                
                # Gain du défenseur: proportion de classifications correctes (transformée en [-1, 1])
                payoff_defender[i, j] = 2 * accuracy - 1
        
        self.payoff_matrix = payoff_defender
        return payoff_defender
    
    def _apply_defense(self, images, defense_type):
        """
        Applique une stratégie de défense
        
        Args:
            images: Images (possiblement perturbées)
            defense_type: Type de défense à appliquer
        
        Returns:
            defended_images: Images après défense
        """
        try:
            defense = get_defense(defense_type)
            return defense.apply(images)
        except ValueError:
            # Si la défense n'existe pas, retourner images inchangées
            print(f"Attention: Défense '{defense_type}' inconnue, utilisation passive")
            return images
    
    def find_nash_equilibrium(self):
        """
        Trouve l'équilibre de Nash du jeu
        
        Returns:
            nash_eq: Liste des équilibres de Nash (stratégies mixtes)
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        # Jeu à somme nulle: gain J1 = -gain J2
        payoff_attacker = -self.payoff_matrix
        payoff_defender = self.payoff_matrix
        
        # Création du jeu avec nashpy
        game = nash.Game(payoff_attacker, payoff_defender)
        
        # Calcul des équilibres de Nash
        equilibria = list(game.support_enumeration())
        
        self.nash_equilibrium = equilibria
        
        print(f"\n{len(equilibria)} équilibre(s) de Nash trouvé(s):")
        for idx, (attacker_mix, defender_mix) in enumerate(equilibria):
            print(f"\nÉquilibre {idx + 1}:")
            print(f"  Stratégie mixte Attaquant: {attacker_mix}")
            print(f"  Stratégie mixte Défenseur: {defender_mix}")
            
            # Epsilon moyen utilisé par l'attaquant
            epsilon_mean = np.dot(attacker_mix, self.attacker_strategies)
            print(f"  Epsilon moyen optimal: {epsilon_mean:.4f}")
        
        return equilibria
    
    def analyze_strategies(self):
        """
        Analyse des stratégies pures et dominantes
        
        Returns:
            dict: Résultats de l'analyse
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        # Pour l'attaquant (maximise son gain = minimise gain du défenseur)
        best_attacker_strategy = np.argmin(self.payoff_matrix[:, 0])
        best_epsilon = self.attacker_strategies[best_attacker_strategy]
        attacker_gain = -self.payoff_matrix[best_attacker_strategy, 0]
        
        # Pour le défenseur
        worst_case = np.min(self.payoff_matrix[:, 0])
        
        # Valeur du jeu
        game_value = np.mean(self.payoff_matrix)
        
        results = {
            'attacker': {
                'best_pure_strategy_idx': best_attacker_strategy,
                'best_epsilon': best_epsilon,
                'expected_gain': attacker_gain
            },
            'defender': {
                'worst_case': worst_case,
                'passive_strategy': 'passive'
            },
            'game_value': game_value
        }
        
        # Affichage console (optionnel)
        print("\n=== Analyse des stratégies ===")
        print("\nAttaquant (J1):")
        print(f"  Meilleure stratégie pure: epsilon = {best_epsilon:.4f}")
        print(f"  Gain attendu: {attacker_gain:.4f}")
        print("\nDéfenseur (J2):")
        print(f"  Stratégie passive")
        print(f"  Gain minimum garanti: {worst_case:.4f}")
        print(f"\n Valeur approximative du jeu: {game_value:.4f}")
        
        return results
        
    def compute_best_response(self, opponent_strategy, player="attacker"):
        """
        Calcule la meilleure réponse à une stratégie adverse
        
        Args:
            opponent_strategy: Stratégie mixte de l'adversaire (vecteur de probabilités)
            player: "attacker" ou "defender"
        
        Returns:
            best_response: Meilleure stratégie mixte en réponse
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        if player == "attacker":
            # L'attaquant veut minimiser le gain du défenseur
            expected_payoffs = self.payoff_matrix @ opponent_strategy
            best_strategy_idx = np.argmin(expected_payoffs)
            best_response = np.zeros(len(self.attacker_strategies))
            best_response[best_strategy_idx] = 1.0
        else:  # defender
            # Le défenseur veut maximiser son gain
            expected_payoffs = opponent_strategy @ self.payoff_matrix
            best_strategy_idx = np.argmax(expected_payoffs)
            best_response = np.zeros(len(self.defender_strategies))
            best_response[best_strategy_idx] = 1.0
        
        return best_response
    
    def get_payoff_summary(self):
        """
        Retourne un résumé de la matrice des gains
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        summary = {
            "min_payoff_defender": np.min(self.payoff_matrix),
            "max_payoff_defender": np.max(self.payoff_matrix),
            "mean_payoff_defender": np.mean(self.payoff_matrix),
            "attacker_strategies": self.attacker_strategies.tolist(),
            "defender_strategies": self.defender_strategies,
            "payoff_matrix": self.payoff_matrix.tolist()
        }
        
        return summary
    
    def analyze_dominance(self):
        """
        Analyse la dominance des stratégies pour les deux joueurs
        
        Returns:
            dict: Résultats de l'analyse de dominance {
                'attacker': {
                    'strictly_dominant': indices ou None,
                    'weakly_dominant': liste d'indices,
                    'strictly_dominated': liste d'indices,
                    'weakly_dominated': liste d'indices
                },
                'defender': {...}
            }
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        results = {
            'attacker': {
                'strictly_dominant': None,
                'weakly_dominant': [],
                'strictly_dominated': [],
                'weakly_dominated': []
            },
            'defender': {
                'strictly_dominant': None,
                'weakly_dominant': [],
                'strictly_dominated': [],
                'weakly_dominated': []
            }
        }
        
        # Analyse pour l'ATTAQUANT (minimise le payoff du défenseur)
        num_att = len(self.attacker_strategies)
        for i in range(num_att):
            is_strictly_dominant = True
            is_weakly_dominant = True
            dominates_count = 0
            
            for j in range(num_att):
                if i == j:
                    continue
                
                # Comparer stratégie i vs stratégie j
                # i domine j si payoff[i,:] <= payoff[j,:] pour tout défenseur
                comparison = self.payoff_matrix[i, :] <= self.payoff_matrix[j, :]
                strict_comparison = self.payoff_matrix[i, :] < self.payoff_matrix[j, :]
                
                if np.all(comparison) and np.any(strict_comparison):
                    # i domine strictement j
                    dominates_count += 1
                elif not np.all(comparison):
                    is_strictly_dominant = False
                    is_weakly_dominant = False
            
            if dominates_count == num_att - 1:
                results['attacker']['strictly_dominant'] = i
            elif is_weakly_dominant and dominates_count > 0:
                results['attacker']['weakly_dominant'].append(i)
        
        # Détecter les stratégies dominées de l'attaquant
        for i in range(num_att):
            is_strictly_dominated = False
            is_weakly_dominated = False
            
            for j in range(num_att):
                if i == j:
                    continue
                
                # j domine i ?
                comparison = self.payoff_matrix[j, :] <= self.payoff_matrix[i, :]
                strict_comparison = self.payoff_matrix[j, :] < self.payoff_matrix[i, :]
                
                if np.all(comparison) and np.any(strict_comparison):
                    is_strictly_dominated = True
                    break
                elif np.all(comparison):
                    is_weakly_dominated = True
            
            if is_strictly_dominated:
                results['attacker']['strictly_dominated'].append(i)
            elif is_weakly_dominated:
                results['attacker']['weakly_dominated'].append(i)
        
        # Analyse pour le DÉFENSEUR (maximise son payoff)
        num_def = len(self.defender_strategies)
        for i in range(num_def):
            is_strictly_dominant = True
            is_weakly_dominant = True
            dominates_count = 0
            
            for j in range(num_def):
                if i == j:
                    continue
                
                # i domine j si payoff[:,i] >= payoff[:,j] pour tout attaquant
                comparison = self.payoff_matrix[:, i] >= self.payoff_matrix[:, j]
                strict_comparison = self.payoff_matrix[:, i] > self.payoff_matrix[:, j]
                
                if np.all(comparison) and np.any(strict_comparison):
                    dominates_count += 1
                elif not np.all(comparison):
                    is_strictly_dominant = False
                    is_weakly_dominant = False
            
            if dominates_count == num_def - 1:
                results['defender']['strictly_dominant'] = i
            elif is_weakly_dominant and dominates_count > 0:
                results['defender']['weakly_dominant'].append(i)
        
        # Détecter les stratégies dominées du défenseur
        for i in range(num_def):
            is_strictly_dominated = False
            is_weakly_dominated = False
            
            for j in range(num_def):
                if i == j:
                    continue
                
                # j domine i ?
                comparison = self.payoff_matrix[:, j] >= self.payoff_matrix[:, i]
                strict_comparison = self.payoff_matrix[:, j] > self.payoff_matrix[:, i]
                
                if np.all(comparison) and np.any(strict_comparison):
                    is_strictly_dominated = True
                    break
                elif np.all(comparison):
                    is_weakly_dominated = True
            
            if is_strictly_dominated:
                results['defender']['strictly_dominated'].append(i)
            elif is_weakly_dominated:
                results['defender']['weakly_dominated'].append(i)
        
        return results
    
    def print_dominance_analysis(self):
        """
        Affiche l'analyse de dominance de manière lisible
        """
        results = self.analyze_dominance()
        
        print("\n" + "="*60)
        print("ANALYSE DE DOMINANCE")
        print("="*60)
        
        # ATTAQUANT
        print("\n🎯 ATTAQUANT (minimise le payoff du défenseur):")
        print("-" * 60)
        
        if results['attacker']['strictly_dominant'] is not None:
            idx = results['attacker']['strictly_dominant']
            eps = self.attacker_strategies[idx]
            print(f"  ✅ Stratégie STRICTEMENT DOMINANTE: ε = {eps:.3f}")
            print(f"     → Cette stratégie bat toutes les autres !")
        else:
            print("  ❌ Aucune stratégie strictement dominante")
        
        if results['attacker']['weakly_dominant']:
            print(f"\n  ⚠️  Stratégies FAIBLEMENT DOMINANTES:")
            for idx in results['attacker']['weakly_dominant']:
                eps = self.attacker_strategies[idx]
                print(f"     - ε = {eps:.3f}")
        
        if results['attacker']['strictly_dominated']:
            print(f"\n  🚫 Stratégies STRICTEMENT DOMINÉES (à éliminer):")
            for idx in results['attacker']['strictly_dominated']:
                eps = self.attacker_strategies[idx]
                print(f"     - ε = {eps:.3f}")
        else:
            print("\n  ✓ Aucune stratégie strictement dominée")
        
        # DÉFENSEUR
        print("\n\n🛡️  DÉFENSEUR (maximise son payoff):")
        print("-" * 60)
        
        if results['defender']['strictly_dominant'] is not None:
            idx = results['defender']['strictly_dominant']
            strat = self.defender_strategies[idx]
            print(f"  ✅ Stratégie STRICTEMENT DOMINANTE: {strat}")
            print(f"     → Cette stratégie bat toutes les autres !")
        else:
            print("  ❌ Aucune stratégie strictement dominante")
        
        if results['defender']['weakly_dominant']:
            print(f"\n  ⚠️  Stratégies FAIBLEMENT DOMINANTES:")
            for idx in results['defender']['weakly_dominant']:
                strat = self.defender_strategies[idx]
                print(f"     - {strat}")
        
        if results['defender']['strictly_dominated']:
            print(f"\n  🚫 Stratégies STRICTEMENT DOMINÉES (à éliminer):")
            for idx in results['defender']['strictly_dominated']:
                strat = self.defender_strategies[idx]
                print(f"     - {strat}")
        else:
            print("\n  ✓ Aucune stratégie strictement dominée")
        
        print("\n" + "="*60)
    
    def analyze_mixed_strategy_equilibrium(self, attacker_mix, defender_mix):
        """
        Analyse approfondie d'un équilibre en stratégies mixtes
        
        Args:
            attacker_mix: Distribution de probabilités de l'attaquant
            defender_mix: Distribution de probabilités du défenseur
        
        Returns:
            dict: Analyse complète de l'équilibre mixte
        """
        if self.payoff_matrix is None:
            self.compute_payoff_matrix()
        
        epsilon = 1e-5  # Seuil pour considérer qu'une probabilité est > 0
        
        # 1. SUPPORT DE L'ÉQUILIBRE
        attacker_support = np.where(attacker_mix > epsilon)[0]
        defender_support = np.where(defender_mix > epsilon)[0]
        
        # 2. PAYOFFS ESPÉRÉS pour chaque stratégie pure
        # Payoff espéré si l'attaquant joue stratégie i contre la stratégie mixte du défenseur
        attacker_expected_payoffs = -self.payoff_matrix @ defender_mix  # Attaquant veut minimiser payoff défenseur
        
        # Payoff espéré si le défenseur joue stratégie j contre la stratégie mixte de l'attaquant
        defender_expected_payoffs = attacker_mix @ self.payoff_matrix
        
        # 3. VÉRIFICATION DE LA PROPRIÉTÉ D'INDIFFÉRENCE
        # Toutes les stratégies dans le support doivent avoir le même payoff espéré
        attacker_support_payoffs = attacker_expected_payoffs[attacker_support]
        defender_support_payoffs = defender_expected_payoffs[defender_support]
        
        # Variance des payoffs dans le support (devrait être ≈ 0)
        attacker_variance = np.var(attacker_support_payoffs) if len(attacker_support) > 1 else 0
        defender_variance = np.var(defender_support_payoffs) if len(defender_support) > 1 else 0
        
        # 4. PAYOFF D'ÉQUILIBRE
        equilibrium_payoff_attacker = np.dot(attacker_expected_payoffs, attacker_mix)
        equilibrium_payoff_defender = np.dot(defender_expected_payoffs, defender_mix)
        
        # 5. VÉRIFICATION: stratégies hors support doivent avoir payoff ≤ support
        attacker_max_payoff_in_support = np.max(attacker_support_payoffs) if len(attacker_support) > 0 else -np.inf
        defender_max_payoff_in_support = np.max(defender_support_payoffs) if len(defender_support) > 0 else -np.inf
        
        attacker_dominated_outside = True
        defender_dominated_outside = True
        
        for i in range(len(self.attacker_strategies)):
            if i not in attacker_support:
                if attacker_expected_payoffs[i] > attacker_max_payoff_in_support + epsilon:
                    attacker_dominated_outside = False
        
        for j in range(len(self.defender_strategies)):
            if j not in defender_support:
                if defender_expected_payoffs[j] > defender_max_payoff_in_support + epsilon:
                    defender_dominated_outside = False
        
        results = {
            'attacker': {
                'support': attacker_support,
                'support_size': len(attacker_support),
                'expected_payoffs': attacker_expected_payoffs,
                'support_payoffs': attacker_support_payoffs,
                'variance_in_support': attacker_variance,
                'is_indifferent': attacker_variance < 1e-4,  # Propriété d'indifférence
                'dominated_outside': attacker_dominated_outside,
                'equilibrium_payoff': equilibrium_payoff_attacker
            },
            'defender': {
                'support': defender_support,
                'support_size': len(defender_support),
                'expected_payoffs': defender_expected_payoffs,
                'support_payoffs': defender_support_payoffs,
                'variance_in_support': defender_variance,
                'is_indifferent': defender_variance < 1e-4,
                'dominated_outside': defender_dominated_outside,
                'equilibrium_payoff': equilibrium_payoff_defender
            },
            'is_pure_strategy': (len(attacker_support) == 1 and len(defender_support) == 1),
            'is_valid_nash': (attacker_dominated_outside and defender_dominated_outside)
        }
        
        return results
    
    def print_mixed_strategy_analysis(self, attacker_mix, defender_mix):
        """
        Affiche l'analyse d'un équilibre en stratégies mixtes de manière lisible
        """
        analysis = self.analyze_mixed_strategy_equilibrium(attacker_mix, defender_mix)
        
        print("\n" + "="*70)
        print("ANALYSE DE L'ÉQUILIBRE EN STRATÉGIES MIXTES")
        print("="*70)
        
        if analysis['is_pure_strategy']:
            print("\n⚠️  Cet équilibre est en STRATÉGIES PURES (une seule stratégie par joueur)")
        else:
            print("\n✅ Cet équilibre est en STRATÉGIES MIXTES (plusieurs stratégies actives)")
        
        # ATTAQUANT
        print("\n" + "-"*70)
        print("🎯 ATTAQUANT")
        print("-"*70)
        
        att_analysis = analysis['attacker']
        print(f"\n📊 Support de l'équilibre: {att_analysis['support_size']} stratégie(s)")
        for idx in att_analysis['support']:
            eps = self.attacker_strategies[idx]
            prob = attacker_mix[idx]
            payoff = att_analysis['expected_payoffs'][idx]
            print(f"   • ε = {eps:.3f} → Probabilité: {prob:.4f} | Payoff espéré: {payoff:.4f}")
        
        if att_analysis['support_size'] > 1:
            print(f"\n🎲 Propriété d'indifférence:")
            print(f"   Variance des payoffs dans le support: {att_analysis['variance_in_support']:.6f}")
            if att_analysis['is_indifferent']:
                print(f"   ✅ VÉRIFIÉ: Toutes les stratégies du support donnent le même payoff espéré")
                print(f"      → L'attaquant est indifférent entre ses options (c'est pourquoi il joue aléatoirement)")
            else:
                print(f"   ⚠️  ATTENTION: Variance non nulle (équilibre approximatif)")
        
        print(f"\n📈 Stratégies hors support:")
        outside_count = 0
        for i in range(len(self.attacker_strategies)):
            if i not in att_analysis['support']:
                eps = self.attacker_strategies[i]
                payoff = att_analysis['expected_payoffs'][i]
                print(f"   • ε = {eps:.3f} → Payoff espéré: {payoff:.4f} (non joué)")
                outside_count += 1
        if outside_count == 0:
            print(f"   Toutes les stratégies sont dans le support")
        
        if att_analysis['dominated_outside']:
            print(f"   ✅ Stratégies hors support sont dominées (payoff ≤ support)")
        else:
            print(f"   ⚠️  ANOMALIE: Certaines stratégies hors support ont meilleur payoff!")
        
        # DÉFENSEUR
        print("\n" + "-"*70)
        print("🛡️  DÉFENSEUR")
        print("-"*70)
        
        def_analysis = analysis['defender']
        print(f"\n📊 Support de l'équilibre: {def_analysis['support_size']} stratégie(s)")
        for idx in def_analysis['support']:
            strat = self.defender_strategies[idx]
            prob = defender_mix[idx]
            payoff = def_analysis['expected_payoffs'][idx]
            print(f"   • {strat} → Probabilité: {prob:.4f} | Payoff espéré: {payoff:.4f}")
        
        if def_analysis['support_size'] > 1:
            print(f"\n🎲 Propriété d'indifférence:")
            print(f"   Variance des payoffs dans le support: {def_analysis['variance_in_support']:.6f}")
            if def_analysis['is_indifferent']:
                print(f"   ✅ VÉRIFIÉ: Toutes les stratégies du support donnent le même payoff espéré")
                print(f"      → Le défenseur est indifférent entre ses options (c'est pourquoi il joue aléatoirement)")
            else:
                print(f"   ⚠️  ATTENTION: Variance non nulle (équilibre approximatif)")
        
        print(f"\n📈 Stratégies hors support:")
        outside_count = 0
        for j in range(len(self.defender_strategies)):
            if j not in def_analysis['support']:
                strat = self.defender_strategies[j]
                payoff = def_analysis['expected_payoffs'][j]
                print(f"   • {strat} → Payoff espéré: {payoff:.4f} (non joué)")
                outside_count += 1
        if outside_count == 0:
            print(f"   Toutes les stratégies sont dans le support")
        
        if def_analysis['dominated_outside']:
            print(f"   ✅ Stratégies hors support sont dominées (payoff ≤ support)")
        else:
            print(f"   ⚠️  ANOMALIE: Certaines stratégies hors support ont meilleur payoff!")
        
        # VALIDATION GLOBALE
        print("\n" + "-"*70)
        print("✅ VALIDATION DE L'ÉQUILIBRE DE NASH")
        print("-"*70)
        
        if analysis['is_valid_nash']:
            print("✅ Équilibre de Nash VALIDE:")
            print("   • Stratégies du support sont optimales")
            print("   • Stratégies hors support sont dominées")
            print("   • Aucun joueur ne peut améliorer son payoff en déviant")
        else:
            print("⚠️  ATTENTION: Équilibre potentiellement invalide")
        
        print(f"\n🎯 Payoff d'équilibre:")
        print(f"   Attaquant: {att_analysis['equilibrium_payoff']:.4f}")
        print(f"   Défenseur: {def_analysis['equilibrium_payoff']:.4f}")
        print(f"   Précision du modèle: {(def_analysis['equilibrium_payoff'] + 1) / 2 * 100:.2f}%")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # Test du module
    from classifier import SimpleCNN
    from attacker import FGSMAttack
    
    print("Test de la modélisation du jeu")
    
    # Création d'un modèle simple
    model = SimpleCNN(num_classes=10, input_channels=1, image_size=28)
    model.to(DEVICE)
    model.eval()
    
    # Données de test
    images = torch.randn(50, 1, 28, 28)
    labels = torch.randint(0, 10, (50,))
    dataset_sample = (images, labels)
    
    # Création du jeu
    game = AdversarialGame(
        model=model,
        attack_class=FGSMAttack,
        dataset_sample=dataset_sample,
        num_strategies=5,
        max_epsilon=0.3
    )
    
    # Calcul de la matrice des gains
    payoff = game.compute_payoff_matrix()
    print(f"\nMatrice des gains (Défenseur):\n{payoff}")
    
    # Analyse
    game.analyze_strategies()
    
    # Équilibre de Nash
    game.find_nash_equilibrium()
