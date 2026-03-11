"""
predictor.py
Classe para carregar modelo e fazer predições
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Adiciona o diretório src ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from src.utils import load_joblib, Config, decode_defasagem_target

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Classe para realizar predições com o modelo treinado"""

    def __init__(self):
        """Inicializa o preditor carregando modelo, scaler e encoders"""
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.is_loaded = False

        # Mapeamento de classes
        self.class_mapping = {
            0: "Em fase",
            1: "Moderada",
            2: "Severa"
        }

        self._load_artifacts()

    def _load_artifacts(self):
        """Carrega modelo, scaler e encoders"""
        try:
            models_dir = Config.MODELS_DIR

            # Carrega modelo
            model_path = models_dir / Config.MODEL_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

            self.model = load_joblib(model_path)
            logger.info(f"Modelo carregado: {model_path}")

            # Carrega scaler
            scaler_path = models_dir / Config.SCALER_FILE
            if scaler_path.exists():
                self.scaler = load_joblib(scaler_path)
                logger.info(f"Scaler carregado: {scaler_path}")
            else:
                logger.warning(f"Scaler não encontrado: {scaler_path}")

            # Carrega label encoders
            encoder_path = models_dir / Config.LABEL_ENCODER_FILE
            if encoder_path.exists():
                self.label_encoders = load_joblib(encoder_path)
                logger.info(f"Label encoders carregados: {encoder_path}")
            else:
                logger.warning(f"Label encoders não encontrados: {encoder_path}")

            # Obtém nomes das features do modelo
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            else:
                self.feature_names = [
                    'Ano ingresso', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN',
                    'RA', 'Fase', 'Turma', 'Gênero', 'Instituição de ensino'
                ]

            logger.info(f"Features esperadas: {self.feature_names}")

            self.is_loaded = True
            logger.info("Artefatos carregados com sucesso")

        except Exception as e:
            logger.error(f"Erro ao carregar artefatos: {e}")
            raise

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """
        Pré-processa os dados de entrada
        """

        try:

            feature_mapping = {
                'ano_ingresso': 'Ano ingresso',
                'IAA': 'IAA',
                'IEG': 'IEG',
                'IPS': 'IPS',
                'IDA': 'IDA',
                'IPV': 'IPV',
                'IAN': 'IAN',
                'RA': 'RA',
                'Fase': 'Fase',
                'Turma': 'Turma',
                'Genero': 'Gênero',
                'Instituicao_ensino': 'Instituição de ensino'
            }

            mapped_data = {feature_mapping[k]: v for k, v in input_data.items()}
            df = pd.DataFrame([mapped_data])

            df = df[self.feature_names]

            numeric_features = ['Ano ingresso', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']
            categorical_features = ['RA', 'Fase', 'Turma', 'Gênero', 'Instituição de ensino']

            # Codificação
            if self.label_encoders:
                for col in categorical_features:
                    if col in self.label_encoders:

                        le = self.label_encoders[col]
                        value = str(df[col].iloc[0])

                        if value in le.classes_:
                            df[col] = le.transform([value])[0]
                        else:
                            df[col] = -1
                            logger.warning(f"Valor '{value}' não visto no treino para '{col}'")

            # Normalização
            if self.scaler:
                df[numeric_features] = self.scaler.transform(df[numeric_features])

            return df

        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            raise

    def predict(self, input_data: dict) -> dict:
        """
        Realiza predição
        """

        try:

            if not self.is_loaded:
                raise RuntimeError("Modelo não está carregado")

            df_processed = self.preprocess_input(input_data)

            prediction = self.model.predict(df_processed)[0]

            if hasattr(self.model, 'predict_proba'):

                probabilities = self.model.predict_proba(df_processed)[0]
                confidence = float(np.max(probabilities))

                prob_dict = {
                    self.class_mapping[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }

            else:

                confidence = 1.0

                prob_dict = {
                    self.class_mapping[i]: 1.0 if i == prediction else 0.0
                    for i in range(3)
                }

            result = {
                "prediction": self.class_mapping[int(prediction)],
                "prediction_code": int(prediction),
                "confidence": confidence,
                "probabilities": prob_dict
            }

            return result

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise

    def predict_batch(self, input_list: list) -> list:
        """
        Realiza predições em lote
        """

        results = []

        for input_data in input_list:

            try:

                result = self.predict(input_data)
                results.append(result)

            except Exception as e:

                logger.error(f"Erro ao processar registro: {e}")

                results.append({
                    "error": str(e),
                    "prediction": None
                })

        return results


# Instância global
predictor = None


def get_predictor() -> ModelPredictor:
    """
    Retorna instância singleton do predictor
    """

    global predictor

    if predictor is None:
        predictor = ModelPredictor()

    return predictor