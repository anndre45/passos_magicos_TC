"""
test_train.py
Testes unitários para o módulo train.py
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.abspath("src"))

from train import ModelTrainer, train_pipeline


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """Dados simples para testes rápidos"""
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(30),
        "feat2": np.random.randn(30),
    })
    y = pd.Series(np.random.choice([0, 1, 2], size=30))
    return X, y


@pytest.fixture
def trainer():
    return ModelTrainer()


# ─────────────────────────────────────────────
# ModelTrainer.__init__
# ─────────────────────────────────────────────

class TestModelTrainerInit:
    def test_initial_state(self, trainer):
        assert trainer.models == {}
        assert trainer.best_model is None
        assert trainer.best_model_name is None
        assert trainer.best_score == 0

    def test_training_history_empty(self, trainer):
        assert trainer.training_history == []


# ─────────────────────────────────────────────
# initialize_models
# ─────────────────────────────────────────────

class TestInitializeModels:
    def test_returns_dict(self, trainer):
        models = trainer.initialize_models()
        assert isinstance(models, dict)

    def test_has_expected_models(self, trainer):
        models = trainer.initialize_models()
        assert "Random Forest" in models
        assert "Logistic Regression" in models

    def test_models_count(self, trainer):
        models = trainer.initialize_models()
        assert len(models) >= 4


# ─────────────────────────────────────────────
# train_single_model
# ─────────────────────────────────────────────

class TestTrainSingleModel:
    def test_returns_trained_model(self, trainer, simple_data):
        X, y = simple_data
        model = LogisticRegression(max_iter=200)
        result = trainer.train_single_model(model, X, y, "LR")
        assert result is not None

    def test_returns_none_on_error(self, trainer, simple_data):
        X, y = simple_data
        bad_model = MagicMock()
        bad_model.fit.side_effect = Exception("fit error")
        result = trainer.train_single_model(bad_model, X, y, "BadModel")
        assert result is None

    def test_model_can_predict_after_train(self, trainer, simple_data):
        X, y = simple_data
        model = LogisticRegression(max_iter=200)
        trained = trainer.train_single_model(model, X, y, "LR")
        preds = trained.predict(X)
        assert len(preds) == len(y)


# ─────────────────────────────────────────────
# cross_validate_model
# ─────────────────────────────────────────────

class TestCrossValidateModel:
    def test_returns_array_of_scores(self, trainer, simple_data):
        X, y = simple_data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        scores = trainer.cross_validate_model(model, X, y, cv=3)
        assert len(scores) == 3

    def test_scores_between_0_and_1(self, trainer, simple_data):
        X, y = simple_data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        scores = trainer.cross_validate_model(model, X, y, cv=3)
        assert all(0 <= s <= 1 for s in scores)

    def test_uses_default_cv_folds(self, trainer, simple_data):
        X, y = simple_data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        from utils import Config
        scores = trainer.cross_validate_model(model, X, y)
        assert len(scores) == Config.CV_FOLDS


# ─────────────────────────────────────────────
# train_all_models
# ─────────────────────────────────────────────

class TestTrainAllModels:
    def test_returns_non_empty_dict(self, trainer, simple_data):
        X, y = simple_data
        trainer.initialize_models()
        results = trainer.train_all_models(X, y, use_cv=False)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_best_model_set_after_cv(self, trainer, simple_data):
        X, y = simple_data
        # Use apenas LR para ser rápido
        trainer.models = {
            "Logistic Regression": LogisticRegression(max_iter=200)
        }
        trainer.train_all_models(X, y, use_cv=True)
        assert trainer.best_model is not None

    def test_results_contain_model_key(self, trainer, simple_data):
        X, y = simple_data
        trainer.models = {
            "Logistic Regression": LogisticRegression(max_iter=200)
        }
        results = trainer.train_all_models(X, y, use_cv=False)
        assert "Logistic Regression" in results
        assert "model" in results["Logistic Regression"]


# ─────────────────────────────────────────────
# train_pipeline (com mocks completos)
# ─────────────────────────────────────────────

class TestTrainPipeline:
    def _make_mock_preprocess(self, simple_data):
        """Retorna um mock de preprocess_pipeline"""
        X, y = simple_data
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        mock_preprocessor = MagicMock()
        return X_train, X_test, y_train, y_test, mock_preprocessor

    def test_pipeline_returns_tuple(self, simple_data, tmp_path):
        preprocess_result = self._make_mock_preprocess(simple_data)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {}

        with patch("train.preprocess_pipeline", return_value=preprocess_result), \
             patch("train.engineer_features_pipeline", side_effect=lambda X, *a, **kw: X), \
             patch("train.ModelEvaluator", return_value=mock_evaluator), \
             patch("train.Config.MODELS_DIR", tmp_path), \
             patch("train.save_joblib"), \
             patch("train.save_json"):

            trainer_obj = ModelTrainer()
            trainer_obj.models = {
                "Logistic Regression": LogisticRegression(max_iter=200)
            }

            with patch("train.ModelTrainer", return_value=trainer_obj):
                result = train_pipeline("fake.xlsx", tune_hyperparams=False, save_model=False)

        assert result is not None
        assert len(result) == 3

    def test_pipeline_trainer_has_best_model(self, simple_data, tmp_path):
        preprocess_result = self._make_mock_preprocess(simple_data)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {}

        with patch("train.preprocess_pipeline", return_value=preprocess_result), \
             patch("train.engineer_features_pipeline", side_effect=lambda X, *a, **kw: X), \
             patch("train.ModelEvaluator", return_value=mock_evaluator), \
             patch("train.Config.MODELS_DIR", tmp_path), \
             patch("train.save_joblib"), \
             patch("train.save_json"):

            trainer_obj = ModelTrainer()
            trainer_obj.models = {
                "Logistic Regression": LogisticRegression(max_iter=200)
            }

            with patch("train.ModelTrainer", return_value=trainer_obj):
                trainer_result, _, _ = train_pipeline(
                    "fake.xlsx", tune_hyperparams=False, save_model=False
                )

        assert trainer_result.best_model is not None
