"""
utils.py
Funções auxiliares para o projeto Passos Mágicos
"""

import os
import pickle
import joblib
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Configuração de logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configurações do projeto"""
    
    # Diretórios
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Arquivos
    RAW_DATA_FILE = 'Base fiap.xlsx'
    TRAIN_DATA_FILE = 'train_data.csv'
    TEST_DATA_FILE = 'test_data.csv'
    MODEL_FILE = 'model.pkl'
    SCALER_FILE = 'scaler.pkl'
    LABEL_ENCODER_FILE = 'label_encoder.pkl'
    
    # Parâmetros do modelo
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Classificação de defasagem baseada no IAN
    DEFASAGEM_CLASSES = {
        'Em fase': 0,      # IAN >= 10
        'Moderada': 1,     # 5 <= IAN < 10
        'Severa': 2        # IAN < 5
    }
    
    # Classificação PEDRA
    PEDRA_RANGES = {
        'Quartzo': (3.0, 6.1),
        'Ágata': (6.2, 7.1),
        'Ametista': (7.2, 8.1),
        'Topázio': (8.2, 9.2)
    }
    
    @classmethod
    def create_directories(cls):
        """Cria os diretórios necessários"""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, 
                         cls.PROCESSED_DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Diretórios criados/verificados com sucesso")


def save_pickle(obj, filepath):
    """
    Salva objeto usando pickle
    
    Args:
        obj: Objeto a ser salvo
        filepath: Caminho do arquivo
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Objeto salvo com sucesso em: {filepath}")
    except Exception as e:
        logger.error(f"Erro ao salvar objeto: {e}")
        raise


def load_pickle(filepath):
    """
    Carrega objeto usando pickle
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Objeto carregado
    """
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Objeto carregado com sucesso de: {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Erro ao carregar objeto: {e}")
        raise


def save_joblib(obj, filepath):
    """
    Salva objeto usando joblib (mais eficiente para modelos sklearn)
    
    Args:
        obj: Objeto a ser salvo
        filepath: Caminho do arquivo
    """
    try:
        joblib.dump(obj, filepath)
        logger.info(f"Objeto salvo com sucesso em: {filepath}")
    except Exception as e:
        logger.error(f"Erro ao salvar objeto: {e}")
        raise


def load_joblib(filepath):
    """
    Carrega objeto usando joblib
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Objeto carregado
    """
    try:
        obj = joblib.load(filepath)
        logger.info(f"Objeto carregado com sucesso de: {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Erro ao carregar objeto: {e}")
        raise


def save_json(data, filepath):
    """
    Salva dados em formato JSON
    
    Args:
        data: Dados a serem salvos
        filepath: Caminho do arquivo
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"JSON salvo com sucesso em: {filepath}")
    except Exception as e:
        logger.error(f"Erro ao salvar JSON: {e}")
        raise


def load_json(filepath):
    """
    Carrega dados de arquivo JSON
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dados carregados
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON carregado com sucesso de: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar JSON: {e}")
        raise


def classify_defasagem(ian_value):
    """
    Classifica a defasagem baseada no valor do IAN
    
    Args:
        ian_value: Valor do IAN
        
    Returns:
        Classificação da defasagem (Em fase, Moderada, Severa)
    """
    if pd.isna(ian_value):
        return None
    
    if ian_value >= 10:
        return 'Em fase'
    elif ian_value >= 5:
        return 'Moderada'
    else:
        return 'Severa'


def classify_pedra(inde_value):
    """
    Classifica a PEDRA baseada no valor do INDE
    
    Args:
        inde_value: Valor do INDE
        
    Returns:
        Classificação da PEDRA (Quartzo, Ágata, Ametista, Topázio)
    """
    if pd.isna(inde_value):
        return None
    
    for pedra, (min_val, max_val) in Config.PEDRA_RANGES.items():
        if min_val <= inde_value <= max_val:
            return pedra
    
    return 'Quartzo'  # Default


def get_timestamp():
    """
    Retorna timestamp atual formatado
    
    Returns:
        String com timestamp
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def log_dataframe_info(df, name="DataFrame"):
    """
    Loga informações sobre o DataFrame
    
    Args:
        df: DataFrame pandas
        name: Nome do DataFrame para o log
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Informações do {name}")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Colunas: {df.columns.tolist()}")
    logger.info(f"Tipos de dados:\n{df.dtypes}")
    logger.info(f"Valores faltantes:\n{df.isnull().sum()}")
    logger.info(f"{'='*50}\n")


def calculate_ian_from_phases(fase_efetiva, fase_ideal):
    """
    Calcula o IAN baseado nas fases
    
    Args:
        fase_efetiva: Fase atual do estudante
        fase_ideal: Fase ideal conforme idade
        
    Returns:
        Valor do IAN
    """
    try:
        d = float(fase_efetiva) - float(fase_ideal)
        
        if d >= 0:
            return 10
        elif d > -2:
            return 5
        else:
            return 2.5
    except:
        return None


def encode_defasagem_target(defasagem_series):
    """
    Codifica a variável target de defasagem
    
    Args:
        defasagem_series: Series com classificação de defasagem
        
    Returns:
        Series codificada (0, 1, 2)
    """
    mapping = Config.DEFASAGEM_CLASSES
    return defasagem_series.map(mapping)


def decode_defasagem_target(encoded_series):
    """
    Decodifica a variável target de defasagem
    
    Args:
        encoded_series: Series com valores codificados (0, 1, 2)
        
    Returns:
        Series com classificação (Em fase, Moderada, Severa)
    """
    reverse_mapping = {v: k for k, v in Config.DEFASAGEM_CLASSES.items()}
    return encoded_series.map(reverse_mapping)


def get_memory_usage(df):
    """
    Calcula uso de memória do DataFrame
    
    Args:
        df: DataFrame pandas
        
    Returns:
        String formatada com uso de memória
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    return f"{memory_mb:.2f} MB"


def reduce_memory_usage(df):
    """
    Reduz uso de memória do DataFrame otimizando tipos de dados
    
    Args:
        df: DataFrame pandas
        
    Returns:
        DataFrame otimizado
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Uso de memória inicial: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Uso de memória final: {end_mem:.2f} MB")
    logger.info(f"Redução: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


if __name__ == "__main__":
    # Teste das funções
    Config.create_directories()
    logger.info("Módulo utils.py testado com sucesso!")