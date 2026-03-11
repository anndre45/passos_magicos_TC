"""
test.py
Módulo de teste do modelo para o projeto Passos Mágicos
Interface para a API (Pessoa 2)
VERSÃO FINAL - Funciona com 12 features (7 numéricas + 5 categóricas)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from utils import (
    Config, 
    load_joblib, 
    decode_defasagem_target,
    logger
)

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Classe para fazer previsões com o modelo treinado
    Interface para a API
    """
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """
        Inicializa o preditor carregando modelo e preprocessadores
        """
        if model_path is None:
            model_path = Config.MODELS_DIR / Config.MODEL_FILE
        if scaler_path is None:
            scaler_path = Config.MODELS_DIR / Config.SCALER_FILE
        if encoder_path is None:
            encoder_path = Config.MODELS_DIR / Config.LABEL_ENCODER_FILE
        
        logger.info("Carregando modelo e preprocessadores...")
        self.model = load_joblib(model_path)
        self.scaler = load_joblib(scaler_path)
        self.label_encoders = load_joblib(encoder_path)
        
        # Features que o SCALER normaliza (apenas numéricas)
        self.scaler_features = [
            'Ano ingresso',
            'IAA',
            'IEG',
            'IPS',
            'IDA',
            'IPV',
            'IAN'
        ]
        
        # Features categóricas (codificadas, não normalizadas)
        self.categorical_features = [
            'RA',
            'Fase',
            'Turma',
            'Gênero',
            'Instituição de ensino'
        ]
        
        # TODAS as features do modelo (ordem importa!)
        self.all_features = self.scaler_features + self.categorical_features
        
        logger.info(f"Features numéricas (scaler): {self.scaler_features}")
        logger.info(f"Features categóricas: {self.categorical_features}")
        logger.info("Modelo carregado e pronto para predições!")
    
    def _map_input_to_features(self, input_dict):
        """Mapeia input do usuário para features do modelo"""
        feature_mapping = {
            'ANOS_NA_PM_2020': 'Ano ingresso',
            'ANOS_NA_PM_2021': 'Ano ingresso',
            'ANOS_NA_PM_2022': 'Ano ingresso',
            'ANOS_NA_PM': 'Ano ingresso',
            'IAA_2020': 'IAA',
            'IAA_2021': 'IAA',
            'IAA_2022': 'IAA',
            'IAA': 'IAA',
            'IEG_2020': 'IEG',
            'IEG_2021': 'IEG',
            'IEG_2022': 'IEG',
            'IEG': 'IEG',
            'IPS_2020': 'IPS',
            'IPS_2021': 'IPS',
            'IPS_2022': 'IPS',
            'IPS': 'IPS',
            'IDA_2020': 'IDA',
            'IDA_2021': 'IDA',
            'IDA_2022': 'IDA',
            'IDA': 'IDA',
            'IPV_2020': 'IPV',
            'IPV_2021': 'IPV',
            'IPV_2022': 'IPV',
            'IPV': 'IPV',
            'IAN_2020': 'IAN',
            'IAN_2021': 'IAN',
            'IAN_2022': 'IAN',
            'IAN': 'IAN'
        }
        
        mapped_data = {}
        for input_key, input_value in input_dict.items():
            if input_key in feature_mapping:
                feature_name = feature_mapping[input_key]
                mapped_data[feature_name] = input_value
        
        # Adiciona valores padrão para features faltantes
        for feature in self.scaler_features:
            if feature not in mapped_data:
                if feature == 'Ano ingresso':
                    mapped_data[feature] = 2
                else:
                    mapped_data[feature] = 5.0
        
        return mapped_data
    
    def preprocess_input(self, data):
        """
        Pré-processa dados de entrada para predição
        IMPORTANTE: Normaliza apenas numéricas, codifica categóricas
        """
        if isinstance(data, dict):
            mapped_data = self._map_input_to_features(data)
            df = pd.DataFrame([mapped_data])
        else:
            mapped_rows = [self._map_input_to_features(row.to_dict()) for _, row in data.iterrows()]
            df = pd.DataFrame(mapped_rows)
        
        logger.info(f"Pré-processando {len(df)} registro(s)...")
        
        # === PARTE 1: Features NUMÉRICAS ===
        # Garante que existem
        for col in self.scaler_features:
            if col not in df.columns:
                df[col] = 5.0 if col != 'Ano ingresso' else 2
        
        # Converte para numérico
        df_numeric = df[self.scaler_features].copy()
        for col in self.scaler_features:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(5.0)
        
        # NORMALIZA com scaler
        df_numeric_scaled = pd.DataFrame(
            self.scaler.transform(df_numeric),
            columns=self.scaler_features,
            index=df.index
        )
        
        # === PARTE 2: Features CATEGÓRICAS ===
        defaults = {
            'RA': 0,
            'Fase': 1,
            'Turma': 0,
            'Gênero': 'Não informado',
            'Instituição de ensino': 'Não informado'
        }
        
        df_categorical = pd.DataFrame(index=df.index)
        
        for col in self.categorical_features:
            if col in df.columns:
                df_categorical[col] = df[col]
            else:
                df_categorical[col] = defaults.get(col, 0)
            
            # Converte para string
            df_categorical[col] = df_categorical[col].astype(str)
            
            # CODIFICA (se tiver encoder)
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df_categorical[col] = df_categorical[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                # Tenta converter para numérico
                df_categorical[col] = pd.to_numeric(df_categorical[col], errors='coerce').fillna(0)
        
        # Garante que é numérico
        df_categorical = df_categorical.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        
        # === PARTE 3: JUNTA TUDO ===
        df_final = pd.concat([df_numeric_scaled, df_categorical], axis=1)
        
        # Garante ordem correta
        df_final = df_final[self.all_features]
        
        logger.info(f"✅ Shape final: {df_final.shape} | Colunas: {list(df_final.columns)}")
        
        return df_final
    
    def predict(self, data, return_proba=False):
        """Faz predição de defasagem escolar"""
        df_processed = self.preprocess_input(data)
        predictions = self.model.predict(df_processed)
        predictions_decoded = decode_defasagem_target(pd.Series(predictions))
        
        if return_proba:
            probabilities = self.model.predict_proba(df_processed)
            return predictions_decoded.tolist(), probabilities.tolist()
        
        return predictions_decoded.tolist()
    
    def predict_single(self, input_dict):
        """Faz predição para um único estudante - Método para API"""
        if not isinstance(input_dict, dict):
            raise ValueError("Input deve ser um dicionário")
        
        try:
            prediction, probabilities = self.predict(input_dict, return_proba=True)
            
            result = {
                'defasagem_prevista': prediction[0],
                'probabilidades': {
                    'Em fase': float(probabilities[0][0]),
                    'Moderada': float(probabilities[0][1]),
                    'Severa': float(probabilities[0][2])
                },
                'confianca': float(max(probabilities[0]))
            }
            
            logger.info(f"✅ Predição: {result['defasagem_prevista']} (confiança: {result['confianca']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_batch(self, data_list):
        """Faz predições em lote"""
        df = pd.DataFrame(data_list)
        predictions, probabilities = self.predict(df, return_proba=True)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'id': i,
                'defasagem_prevista': pred,
                'probabilidades': {
                    'Em fase': float(probabilities[i][0]),
                    'Moderada': float(probabilities[i][1]),
                    'Severa': float(probabilities[i][2])
                },
                'confianca': float(max(probabilities[i]))
            }
            results.append(result)
        
        logger.info(f"✅ {len(results)} predições em lote realizadas")
        return results
    
    def get_feature_importance(self, top_n=10):
        """Retorna importância das features"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': self.all_features[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                
                return importance_df.to_dict('records')
            else:
                return {"message": "Modelo não suporta feature importance"}
        except Exception as e:
            logger.error(f"Erro ao obter feature importance: {e}")
            return {"error": str(e)}


def create_example_input():
    """Cria exemplo de input para teste"""
    return {
        'IAN_2022': 7.5,
        'IDA_2022': 8.0,
        'IEG_2022': 7.8,
        'IAA_2022': 8.5,
        'IPS_2022': 8.2,
        'IPV_2022': 8.1,
        'ANOS_NA_PM_2022': 3
    }


def test_model():
    """Testa o modelo com exemplos"""
    logger.info("Iniciando teste do modelo...")
    
    try:
        predictor = ModelPredictor()
        
        # Teste 1: Predição única
        logger.info("\n" + "="*50)
        logger.info("TESTE 1: Predição Única")
        logger.info("="*50)
        
        example = create_example_input()
        logger.info(f"Input: {example}")
        
        result = predictor.predict_single(example)
        logger.info(f"\n📊 Resultado:")
        logger.info(f"  Defasagem: {result['defasagem_prevista']}")
        logger.info(f"  Confiança: {result['confianca']:.2%}")
        logger.info(f"  Probabilidades:")
        for classe, prob in result['probabilidades'].items():
            logger.info(f"    {classe}: {prob:.2%}")
        
        # Teste 2: Cenários diferentes
        logger.info("\n" + "="*50)
        logger.info("TESTE 2: Diferentes Cenários")
        logger.info("="*50)
        
        scenarios = [
            {"nome": "🟢 Em fase", "IAN_2022": 10, "IDA_2022": 9, "IEG_2022": 9, "IAA_2022": 9, "IPS_2022": 9, "IPV_2022": 9, "ANOS_NA_PM_2022": 4},
            {"nome": "🟡 Moderada", "IAN_2022": 6, "IDA_2022": 6.5, "IEG_2022": 6, "IAA_2022": 6.5, "IPS_2022": 6, "IPV_2022": 6.5, "ANOS_NA_PM_2022": 2},
            {"nome": "🔴 Severa", "IAN_2022": 3, "IDA_2022": 4, "IEG_2022": 3.5, "IAA_2022": 4, "IPS_2022": 3.5, "IPV_2022": 4, "ANOS_NA_PM_2022": 1}
        ]
        
        for scenario in scenarios:
            nome = scenario.pop("nome")
            res = predictor.predict_single(scenario)
            logger.info(f"\n{nome}: {res['defasagem_prevista']} ({res['confianca']:.1%})")
        
        # Teste 3: Batch
        logger.info("\n" + "="*50)
        logger.info("TESTE 3: Predição em Lote")
        logger.info("="*50)
        
        batch = [
            {"IAN_2022": 8, "IDA_2022": 8.5, "IEG_2022": 8, "IAA_2022": 8.5, "IPS_2022": 8, "IPV_2022": 8.5, "ANOS_NA_PM_2022": 3},
            {"IAN_2022": 5, "IDA_2022": 5.5, "IEG_2022": 5, "IAA_2022": 5.5, "IPS_2022": 5, "IPV_2022": 5.5, "ANOS_NA_PM_2022": 2}
        ]
        
        batch_results = predictor.predict_batch(batch)
        for res in batch_results:
            logger.info(f"  [{res['id']+1}] {res['defasagem_prevista']} ({res['confianca']:.1%})")
        
        # Teste 4: Feature Importance
        logger.info("\n" + "="*50)
        logger.info("TESTE 4: Feature Importance")
        logger.info("="*50)
        
        importance = predictor.get_feature_importance(top_n=7)
        if isinstance(importance, list):
            logger.info("\n📊 Features mais importantes:")
            for item in importance:
                logger.info(f"  {item['feature']:20s}: {item['importance']:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("✅ TODOS OS TESTES CONCLUÍDOS!")
        logger.info("="*50)
        
        return predictor
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_model_for_api():
    """Carrega modelo para API - Função para Pessoa 2"""
    try:
        predictor = ModelPredictor()
        logger.info("✅ Modelo carregado para API")
        return predictor
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise


if __name__ == "__main__":
    Config.create_directories()
    predictor = test_model()
    
    print("\n" + "="*70)
    print("📚 EXEMPLO PARA API (PESSOA 2)")
    print("="*70)
    print("\n🐍 Código:")
    print("""
from test import load_model_for_api

predictor = load_model_for_api()

def predict_endpoint(request_data):
    return predictor.predict_single(request_data)
""")
    
    print("\n📥 JSON de entrada:")
    import json
    print(json.dumps(create_example_input(), indent=2))
    
    print("\n📤 JSON de saída:")
    try:
        output = predictor.predict_single(create_example_input())
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Erro: {e}")
    
    print("\n✅ Pronto para API!")