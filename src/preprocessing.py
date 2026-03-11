"""
preprocessing.py
Módulo de pré-processamento dos dados do projeto Passos Mágicos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from utils import (
    Config, 
    classify_defasagem, 
    encode_defasagem_target,
    log_dataframe_info,
    save_joblib
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pré-processamento dos dados"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath):
        """
        Carrega dados do arquivo Excel
        
        Args:
            filepath: Caminho do arquivo Excel
            
        Returns:
            DataFrame consolidado
        """
        logger.info(f"Carregando dados de: {filepath}")
        
        try:
            # Lê todas as abas do Excel
            excel_file = pd.ExcelFile(filepath)
            dfs = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                dfs[sheet_name] = df
                logger.info(f"Aba '{sheet_name}' carregada: {df.shape}")
            
            # Consolida os dados
            df_consolidated = self._consolidate_data(dfs)
            
            log_dataframe_info(df_consolidated, "Dados Consolidados")
            return df_consolidated
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def _consolidate_data(self, dfs):
        """
        Consolida dados de múltiplos anos
        
        Args:
            dfs: Dicionário com DataFrames de cada ano
            
        Returns:
            DataFrame consolidado
        """
        logger.info("Consolidando dados de múltiplos anos...")
        
        # Identifica o ano mais recente para usar como base
        years = sorted([int(k) for k in dfs.keys() if k.isdigit()], reverse=True)
        
        if not years:
            # Se não houver anos numéricos, usa a primeira aba
            return list(dfs.values())[0]
        
        # Usa o ano mais recente como base
        latest_year = str(years[0])
        df_base = dfs[latest_year].copy()
        
        logger.info(f"Usando dados de {latest_year} como base")
        
        # Adiciona features de anos anteriores se disponíveis
        if len(years) > 1:
            prev_year = str(years[1])
            df_prev = dfs[prev_year]
            
            # Merge com ano anterior se houver coluna NOME
            if 'NOME' in df_base.columns and 'NOME' in df_prev.columns:
                df_base = self._merge_with_previous_year(df_base, df_prev, latest_year, prev_year)
        
        return df_base
    
    def _merge_with_previous_year(self, df_current, df_prev, year_current, year_prev):
        """
        Faz merge com dados do ano anterior
        
        Args:
            df_current: DataFrame do ano atual
            df_prev: DataFrame do ano anterior
            year_current: Ano atual
            year_prev: Ano anterior
            
        Returns:
            DataFrame com merge
        """
        # Seleciona colunas importantes do ano anterior
        cols_to_merge = []
        for col in df_prev.columns:
            if any(indicator in col for indicator in ['IAN', 'IDA', 'IEG', 'INDE', 'IPV']):
                cols_to_merge.append(col)
        
        if 'NOME' not in cols_to_merge:
            cols_to_merge.insert(0, 'NOME')
        
        df_prev_selected = df_prev[cols_to_merge].copy()
        
        # Renomeia colunas do ano anterior
        rename_dict = {col: f"{col}_PREV" for col in cols_to_merge if col != 'NOME'}
        df_prev_selected.rename(columns=rename_dict, inplace=True)
        
        # Merge
        df_merged = df_current.merge(df_prev_selected, on='NOME', how='left')
        
        logger.info(f"Merge realizado: {df_merged.shape}")
        return df_merged
    
    def create_target(self, df, ian_column='IAN_2022'):
        """
        Cria a variável target baseada no IAN
        
        Args:
            df: DataFrame
            ian_column: Nome da coluna IAN
            
        Returns:
            DataFrame com target criado
        """
        logger.info("Criando variável target...")
        
        # Identifica a coluna IAN mais recente se não especificado
        if ian_column not in df.columns:
            ian_cols = [col for col in df.columns if 'IAN' in col and 'PREV' not in col]
            if ian_cols:
                ian_column = sorted(ian_cols)[-1]
                logger.info(f"Usando coluna: {ian_column}")
        
        # Cria classificação de defasagem
        df['DEFASAGEM_CLASSE'] = df[ian_column].apply(classify_defasagem)
        
        # Codifica para numérico
        df['TARGET'] = encode_defasagem_target(df['DEFASAGEM_CLASSE'])
        
        # Remove registros sem target
        df_clean = df.dropna(subset=['TARGET']).copy()
        
        logger.info(f"Distribuição do target:")
        logger.info(f"\n{df_clean['DEFASAGEM_CLASSE'].value_counts()}")
        logger.info(f"\nTotal de registros: {len(df_clean)}")
        
        return df_clean
    
    def identify_feature_types(self, df, exclude_cols=None):
        """
        Identifica tipos de features (numéricas e categóricas)
        
        Args:
            df: DataFrame
            exclude_cols: Colunas a excluir
            
        Returns:
            Tupla (numeric_features, categorical_features)
        """
        if exclude_cols is None:
            exclude_cols = ['NOME', 'TARGET', 'DEFASAGEM_CLASSE']
        
        # Features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # Features categóricas
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        categorical_features = [col for col in categorical_features if col not in exclude_cols]
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        logger.info(f"Features numéricas identificadas: {len(numeric_features)}")
        logger.info(f"Features categóricas identificadas: {len(categorical_features)}")
        
        return numeric_features, categorical_features
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Trata valores faltantes
        
        Args:
            df: DataFrame
            strategy: Estratégia de imputação ('median', 'mean', 'mode')
            
        Returns:
            DataFrame sem valores faltantes
        """
        logger.info("Tratando valores faltantes...")
        
        df_clean = df.copy()
        
        # Trata features numéricas
        for col in self.numeric_features:
            if df_clean[col].isnull().sum() > 0:
                if strategy == 'median':
                    fill_value = df_clean[col].median()
                elif strategy == 'mean':
                    fill_value = df_clean[col].mean()
                else:
                    fill_value = 0
                
                df_clean[col].fillna(fill_value, inplace=True)
                logger.info(f"Coluna '{col}': preenchida com {strategy}")
        
        # Trata features categóricas
        for col in self.categorical_features:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna('Desconhecido', inplace=True)
                logger.info(f"Coluna '{col}': preenchida com 'Desconhecido'")
        
        logger.info("Valores faltantes tratados com sucesso")
        return df_clean
    
    def encode_categorical_features(self, df, fit=True):
        """
        Codifica features categóricas
        
        Args:
            df: DataFrame
            fit: Se True, ajusta os encoders. Se False, usa encoders existentes
            
        Returns:
            DataFrame com features codificadas
        """
        logger.info("Codificando features categóricas...")
        
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Trata valores não vistos durante treino
                        le = self.label_encoders[col]
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                
                logger.info(f"Coluna '{col}' codificada")
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """
        Normaliza features numéricas
        
        Args:
            df: DataFrame
            fit: Se True, ajusta o scaler. Se False, usa scaler existente
            
        Returns:
            DataFrame com features normalizadas
        """
        logger.info("Normalizando features numéricas...")
        
        df_scaled = df.copy()
        
        if self.numeric_features:
            if fit:
                df_scaled[self.numeric_features] = self.scaler.fit_transform(
                    df_scaled[self.numeric_features]
                )
            else:
                df_scaled[self.numeric_features] = self.scaler.transform(
                    df_scaled[self.numeric_features]
                )
            
            logger.info(f"{len(self.numeric_features)} features normalizadas")
        
        return df_scaled
    
    def prepare_features_and_target(self, df):
        """
        Prepara features (X) e target (y)
        
        Args:
            df: DataFrame processado
            
        Returns:
            Tupla (X, y)
        """
        feature_cols = self.numeric_features + self.categorical_features
        
        X = df[feature_cols].copy()
        y = df['TARGET'].copy()
        
        logger.info(f"Features preparadas: {X.shape}")
        logger.info(f"Target preparado: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=None, random_state=None):
        """
        Divide dados em treino e teste
        
        Args:
            X: Features
            y: Target
            test_size: Proporção de teste
            random_state: Seed aleatória
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = Config.TEST_SIZE
        if random_state is None:
            random_state = Config.RANDOM_STATE
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Dados divididos - Treino: {X_train.shape}, Teste: {X_test.shape}")
        logger.info(f"Distribuição treino:\n{y_train.value_counts()}")
        logger.info(f"Distribuição teste:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath=None):
        """
        Salva o preprocessador (scaler e encoders)
        
        Args:
            filepath: Caminho base para salvar
        """
        if filepath is None:
            filepath = Config.MODELS_DIR
        
        # Salva scaler
        scaler_path = filepath / Config.SCALER_FILE
        save_joblib(self.scaler, scaler_path)
        
        # Salva label encoders
        encoder_path = filepath / Config.LABEL_ENCODER_FILE
        save_joblib(self.label_encoders, encoder_path)
        
        logger.info("Preprocessador salvo com sucesso")


def preprocess_pipeline(filepath, save_processed=True):
    """
    Pipeline completo de pré-processamento
    
    Args:
        filepath: Caminho do arquivo de dados
        save_processed: Se True, salva dados processados
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test, preprocessor)
    """
    logger.info("Iniciando pipeline de pré-processamento...")
    
    # Inicializa preprocessador
    preprocessor = DataPreprocessor()
    
    # Carrega dados
    df = preprocessor.load_data(filepath)
    
    # Cria target
    df = preprocessor.create_target(df)
    
    # Identifica tipos de features
    preprocessor.identify_feature_types(df)
    
    # Trata valores faltantes
    df = preprocessor.handle_missing_values(df)
    
    # Codifica features categóricas
    df = preprocessor.encode_categorical_features(df, fit=True)
    
    # Normaliza features
    df = preprocessor.scale_features(df, fit=True)
    
    # Prepara X e y
    X, y = preprocessor.prepare_features_and_target(df)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Salva dados processados
    if save_processed:
        train_df = X_train.copy()
        train_df['TARGET'] = y_train
        test_df = X_test.copy()
        test_df['TARGET'] = y_test
        
        train_path = Config.PROCESSED_DATA_DIR / Config.TRAIN_DATA_FILE
        test_path = Config.PROCESSED_DATA_DIR / Config.TEST_DATA_FILE
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Dados processados salvos em: {Config.PROCESSED_DATA_DIR}")
    
    # Salva preprocessador
    preprocessor.save_preprocessor()
    
    logger.info("Pipeline de pré-processamento concluído!")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Teste do módulo
    Config.create_directories()
    
    data_path = Config.RAW_DATA_DIR / Config.RAW_DATA_FILE
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(data_path)
    
    logger.info("Módulo preprocessing.py testado com sucesso!")