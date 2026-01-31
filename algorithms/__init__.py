"""
Module contenant les algorithmes principaux du projet
"""
from .classifier import SimpleCNN, train_classifier, evaluate_classifier, get_model, get_pretrained_model
from .attacker import FGSMAttack, PGDAttack, BIMAttack
from .game_model import AdversarialGame
from .defender import (
    DefenseStrategy, PassiveDefense, GaussianNoiseDefense, 
    MedianFilterDefense, JPEGCompressionDefense, InputTransformDefense,
    BitDepthReductionDefense, EnsembleDefense, RandomizedSmoothingDefense,
    get_defense, DefensiveModel
)

__all__ = [
    'SimpleCNN',
    'get_model',
    'get_pretrained_model',
    'train_classifier',
    'evaluate_classifier',
    'FGSMAttack',
    'PGDAttack',
    'BIMAttack',
    'AdversarialGame',
    'DefenseStrategy',
    'PassiveDefense',
    'GaussianNoiseDefense',
    'MedianFilterDefense',
    'JPEGCompressionDefense',
    'InputTransformDefense',
    'BitDepthReductionDefense',
    'EnsembleDefense',
    'RandomizedSmoothingDefense',
    'get_defense',
    'DefensiveModel'
]
