"""
train.py
Módulo de treinamento do modelo para o projeto Passos Mágicos
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import logging
from datetime import datetime
from utils import Config, save_joblib, save_json
from preprocessing import preprocess_pipeline
from features import engineer_features_pipeline
from evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Classe para treinamento de modelos"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.training_history = []
    
    def initialize_models(self):
        """
        Inicializa todos os modelos a serem testados
        
        Returns:
            Dicionário com modelos
        """
        logger.info("Inicializando modelos...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced',
                verbose=-1
            )
        }
        
        logger.info(f"{len(self.models)} modelos inicializados")
        return self.models
    
    def train_single_model(self, model, X_train, y_train, model_name):
        """
        Treina um único modelo
        
        Args:
            model: Modelo a ser treinado
            X_train: Features de treino
            y_train: Target de treino
            model_name: Nome do modelo
            
        Returns:
            Modelo treinado
        """
        logger.info(f"Treinando modelo: {model_name}")
        
        try:
            model.fit(X_train, y_train)
            logger.info(f"Modelo {model_name} treinado com sucesso")
            return model
        except Exception as e:
            logger.error(f"Erro ao treinar {model_name}: {e}")
            return None
    
    def cross_validate_model(self, model, X_train, y_train, cv=None):
        """
        Realiza validação cruzada do modelo
        
        Args:
            model: Modelo a ser validado
            X_train: Features de treino
            y_train: Target de treino
            cv: Número de folds (padrão: Config.CV_FOLDS)
            
        Returns:
            Scores da validação cruzada
        """
        if cv is None:
            cv = Config.CV_FOLDS
        
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        return scores
    
    def train_all_models(self, X_train, y_train, use_cv=True):
        """
        Treina todos os modelos e seleciona o melhor
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            use_cv: Se True, usa validação cruzada
            
        Returns:
            Dicionário com modelos treinados e scores
        """
        logger.info("Iniciando treinamento de todos os modelos...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Treinando: {name}")
            logger.info(f"{'='*50}")
            
            # Treina o modelo
            trained_model = self.train_single_model(model, X_train, y_train, name)
            
            if trained_model is None:
                continue
            
            # Validação cruzada
            if use_cv:
                cv_scores = self.cross_validate_model(trained_model, X_train, y_train)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                logger.info(f"CV F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
                
                results[name] = {
                    'model': trained_model,
                    'cv_scores': cv_scores,
                    'mean_score': mean_score,
                    'std_score': std_score
                }
                
                # Atualiza melhor modelo
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = trained_model
                    self.best_model_name = name
            else:
                results[name] = {
                    'model': trained_model
                }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Melhor modelo: {self.best_model_name}")
        logger.info(f"Score: {self.best_score:.4f}")
        logger.info(f"{'='*50}\n")
        
        return results
    
    def hyperparameter_tuning(self, model, X_train, y_train, param_grid, model_name):
        """
        Realiza ajuste de hiperparâmetros
        
        Args:
            model: Modelo base
            X_train: Features de treino
            y_train: Target de treino
            param_grid: Grade de parâmetros
            model_name: Nome do modelo
            
        Returns:
            Melhor modelo encontrado
        """
        logger.info(f"Ajustando hiperparâmetros para: {model_name}")
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        logger.info(f"Melhor score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_best_model(self, X_train, y_train):
        """
        Ajusta hiperparâmetros do melhor modelo
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo com hiperparâmetros otimizados
        """
        logger.info("Iniciando ajuste de hiperparâmetros do melhor modelo...")
        
        # Define grades de parâmetros para cada modelo
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 70]
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Recria o modelo base
            base_model = self.models[self.best_model_name]
            
            # Ajusta hiperparâmetros
            tuned_model = self.hyperparameter_tuning(
                base_model, X_train, y_train, param_grid, self.best_model_name
            )
            
            self.best_model = tuned_model
            logger.info("Hiperparâmetros ajustados com sucesso")
            
            return tuned_model
        else:
            logger.info("Grade de parâmetros não definida para este modelo")
            return self.best_model
    
    def save_model(self, filepath=None):
        """
        Salva o melhor modelo
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if filepath is None:
            filepath = Config.MODELS_DIR / Config.MODEL_FILE
        
        save_joblib(self.best_model, filepath)
        
        # Salva metadados do modelo
        metadata = {
            'model_name': self.best_model_name,
            'best_score': float(self.best_score),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_params': str(self.best_model.get_params())
        }
        
        metadata_path = Config.MODELS_DIR / 'model_metadata.json'
        save_json(metadata, metadata_path)
        
        logger.info(f"Modelo salvo em: {filepath}")
        logger.info(f"Metadados salvos em: {metadata_path}")


def train_pipeline(data_path, tune_hyperparams=False, save_model=True):
    """
    Pipeline completo de treinamento
    
    Args:
        data_path: Caminho dos dados
        tune_hyperparams: Se True, ajusta hiperparâmetros
        save_model: Se True, salva o modelo treinado
        
    Returns:
        Tupla (trainer, evaluator, results)
    """
    logger.info("="*70)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO")
    logger.info("="*70)
    
    # 1. Pré-processamento
    logger.info("\n[1/5] Pré-processamento dos dados...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        data_path, 
        save_processed=True
    )
    
    # 2. Inicializa trainer
    logger.info("\n[2/5] Inicializando trainer...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # 3. Treina todos os modelos
    logger.info("\n[3/5] Treinando modelos...")
    results = trainer.train_all_models(X_train, y_train, use_cv=True)
    
    # 4. Ajuste de hiperparâmetros (opcional)
    if tune_hyperparams:
        logger.info("\n[4/5] Ajustando hiperparâmetros...")
        trainer.tune_best_model(X_train, y_train)
    else:
        logger.info("\n[4/5] Pulando ajuste de hiperparâmetros...")
    
    # 5. Avaliação final
    logger.info("\n[5/5] Avaliação final do modelo...")
    evaluator = ModelEvaluator()
    
    # Predições
    y_train_pred = trainer.best_model.predict(X_train)
    y_test_pred = trainer.best_model.predict(X_test)
    
    # Avalia
    train_metrics = evaluator.evaluate(y_train, y_train_pred, "Treino")
    test_metrics = evaluator.evaluate(y_test, y_test_pred, "Teste")
    
    # Relatório completo
    evaluator.generate_report(
        y_test, 
        y_test_pred, 
        trainer.best_model_name,
        save_path=Config.MODELS_DIR / 'evaluation_report.txt'
    )
    
    # Salva modelo
    if save_model:
        trainer.save_model()
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE DE TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("="*70)
    
    return trainer, evaluator, results


if __name__ == "__main__":
    # Configuração
    Config.create_directories()
    
    # Caminho dos dados
    data_path = Config.RAW_DATA_DIR / Config.RAW_DATA_FILE
    
    # Executa pipeline de treinamento
    trainer, evaluator, results = train_pipeline(
        data_path,
        tune_hyperparams=True,  # Ajustar hiperparâmetros
        save_model=True  # Salvar modelo
    )
    
    logger.info("\nTreinamento finalizado!")
    logger.info(f"Melhor modelo: {trainer.best_model_name}")
    logger.info(f"F1-Score: {trainer.best_score:.4f}")