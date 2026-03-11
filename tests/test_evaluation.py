import sys
import os

sys.path.append(os.path.abspath("src"))

from evaluation import ModelEvaluator


def test_evaluator_creation():
    """Testa se o avaliador pode ser criado"""
    evaluator = ModelEvaluator()
    assert evaluator is not None
