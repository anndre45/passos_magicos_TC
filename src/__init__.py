"""
Passos Mágicos - Machine Learning Project
Predição de Defasagem Escolar

Módulo principal com imports facilitados
"""

# Imports principais para facilitar uso
from .utils import Config, logger
from .preprocessing import DataPreprocessor, preprocess_pipeline
from .features import FeatureEngineer, engineer_features_pipeline
from .train import ModelTrainer, train_pipeline
from .model_test import ModelPredictor, load_model_for_api
from .evaluation import ModelEvaluator

__all__ = [
    'Config',
    'logger',
    'DataPreprocessor',
    'preprocess_pipeline',
    'FeatureEngineer',
    'engineer_features_pipeline',
    'ModelTrainer',
    'train_pipeline',
    'ModelPredictor',
    'load_model_for_api',
    'ModelEvaluator'
]