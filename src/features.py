"""
features.py
Módulo de Feature Engineering para o projeto Passos Mágicos
"""

import pandas as pd
import numpy as np
import logging
from utils import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self):
        self.created_features = []
    
    def create_all_features(self, df):
        """
        Cria todas as features de engenharia
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame com novas features
        """
        logger.info("Iniciando feature engineering...")
        
        df_eng = df.copy()
        
        # Features de desempenho acadêmico
        df_eng = self.create_academic_features(df_eng)
        
        # Features de engajamento e comportamento
        df_eng = self.create_engagement_features(df_eng)
        
        # Features de adequação e progresso
        df_eng = self.create_progress_features(df_eng)
        
        # Features de evolução temporal
        df_eng = self.create_temporal_features(df_eng)
        
        # Features de agregação
        df_eng = self.create_aggregation_features(df_eng)
        
        # Features de interação
        df_eng = self.create_interaction_features(df_eng)
        
        logger.info(f"Feature engineering concluído. {len(self.created_features)} novas features criadas")
        logger.info(f"Features criadas: {self.created_features}")
        
        return df_eng
    
    def create_academic_features(self, df):
        """
        Cria features relacionadas ao desempenho acadêmico
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com novas features acadêmicas
        """
        logger.info("Criando features acadêmicas...")
        
        # Média geral de desempenho (se houver notas)
        nota_cols = [col for col in df.columns if 'NOTA_' in col]
        if nota_cols:
            df['MEDIA_NOTAS'] = df[nota_cols].mean(axis=1)
            self.created_features.append('MEDIA_NOTAS')
            
            # Desvio padrão das notas (consistência)
            df['STD_NOTAS'] = df[nota_cols].std(axis=1)
            self.created_features.append('STD_NOTAS')
        
        # Performance em indicadores de aprendizagem
        ida_cols = [col for col in df.columns if 'IDA' in col and 'PREV' not in col]
        if ida_cols:
            df['MEDIA_IDA'] = df[ida_cols].mean(axis=1)
            self.created_features.append('MEDIA_IDA')
        
        # Score composto de desempenho
        if 'IDA_2022' in df.columns and 'IEG_2022' in df.columns:
            df['SCORE_DESEMPENHO'] = (df['IDA_2022'] * 0.6 + df['IEG_2022'] * 0.4)
            self.created_features.append('SCORE_DESEMPENHO')
        
        return df
    
    def create_engagement_features(self, df):
        """
        Cria features relacionadas ao engajamento
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com novas features de engajamento
        """
        logger.info("Criando features de engajamento...")
        
        # Média de engajamento
        ieg_cols = [col for col in df.columns if 'IEG' in col and 'PREV' not in col]
        if ieg_cols:
            df['MEDIA_IEG'] = df[ieg_cols].mean(axis=1)
            self.created_features.append('MEDIA_IEG')
        
        # Engajamento acima da média (flag)
        if 'IEG_2022' in df.columns:
            media_ieg = df['IEG_2022'].median()
            df['ENGAJAMENTO_ALTO'] = (df['IEG_2022'] > media_ieg).astype(int)
            self.created_features.append('ENGAJAMENTO_ALTO')
        
        # Score de participação
        if 'QTDE_AVAL_2022' in df.columns:
            df['QTDE_AVAL_NORMALIZADA'] = df['QTDE_AVAL_2022'] / df['QTDE_AVAL_2022'].max()
            self.created_features.append('QTDE_AVAL_NORMALIZADA')
        
        return df
    
    def create_progress_features(self, df):
        """
        Cria features relacionadas ao progresso e adequação
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com novas features de progresso
        """
        logger.info("Criando features de progresso...")
        
        # Adequação ao nível
        ian_cols = [col for col in df.columns if 'IAN' in col and 'PREV' not in col]
        if ian_cols:
            df['MEDIA_IAN'] = df[ian_cols].mean(axis=1)
            self.created_features.append('MEDIA_IAN')
        
        # Indicador de ponto de virada
        ipv_cols = [col for col in df.columns if 'IPV' in col and 'PREV' not in col]
        if ipv_cols:
            df['MEDIA_IPV'] = df[ipv_cols].mean(axis=1)
            self.created_features.append('MEDIA_IPV')
        
        # Ponto de virada atingido (flag)
        pv_cols = [col for col in df.columns if 'PONTO_VIRADA' in col]
        if pv_cols:
            df['PONTO_VIRADA_FLAG'] = df[pv_cols].max(axis=1).astype(int)
            self.created_features.append('PONTO_VIRADA_FLAG')
        
        # Anos na instituição
        if 'ANOS_NA_PM_2020' in df.columns:
            df['ANOS_EXPERIENCIA'] = df['ANOS_NA_PM_2020']
            self.created_features.append('ANOS_EXPERIENCIA')
        
        return df
    
    def create_temporal_features(self, df):
        """
        Cria features de evolução temporal
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com features temporais
        """
        logger.info("Criando features temporais...")
        
        # Evolução de IDA
        if 'IDA_2022' in df.columns and 'IDA_2021' in df.columns:
            df['EVOLUCAO_IDA'] = df['IDA_2022'] - df['IDA_2021']
            self.created_features.append('EVOLUCAO_IDA')
        
        # Evolução de IEG
        if 'IEG_2022' in df.columns and 'IEG_2021' in df.columns:
            df['EVOLUCAO_IEG'] = df['IEG_2022'] - df['IEG_2021']
            self.created_features.append('EVOLUCAO_IEG')
        
        # Evolução de INDE
        if 'INDE_2022' in df.columns and 'INDE_2021' in df.columns:
            df['EVOLUCAO_INDE'] = df['INDE_2022'] - df['INDE_2021']
            self.created_features.append('EVOLUCAO_INDE')
            
            # Taxa de crescimento INDE
            df['TAXA_CRESCIMENTO_INDE'] = (
                (df['INDE_2022'] - df['INDE_2021']) / (df['INDE_2021'] + 1e-5)
            )
            self.created_features.append('TAXA_CRESCIMENTO_INDE')
        
        # Evolução de IAN
        if 'IAN_2022' in df.columns and 'IAN_2021' in df.columns:
            df['EVOLUCAO_IAN'] = df['IAN_2022'] - df['IAN_2021']
            self.created_features.append('EVOLUCAO_IAN')
            
            # Melhoria na adequação (flag)
            df['MELHOROU_ADEQUACAO'] = (df['EVOLUCAO_IAN'] > 0).astype(int)
            self.created_features.append('MELHOROU_ADEQUACAO')
        
        # Tendência geral de progresso
        evolucao_cols = [col for col in df.columns if 'EVOLUCAO_' in col]
        if len(evolucao_cols) >= 2:
            df['TENDENCIA_PROGRESSO'] = df[evolucao_cols].mean(axis=1)
            self.created_features.append('TENDENCIA_PROGRESSO')
        
        return df
    
    def create_aggregation_features(self, df):
        """
        Cria features de agregação de múltiplos indicadores
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com features agregadas
        """
        logger.info("Criando features agregadas...")
        
        # Score psicossocial composto
        psi_cols = [col for col in df.columns if 'IPS' in col or 'IPP' in col]
        if psi_cols:
            df['SCORE_PSICOSSOCIAL'] = df[psi_cols].mean(axis=1)
            self.created_features.append('SCORE_PSICOSSOCIAL')
        
        # Score de autoavaliação
        iaa_cols = [col for col in df.columns if 'IAA' in col and 'PREV' not in col]
        if iaa_cols:
            df['MEDIA_IAA'] = df[iaa_cols].mean(axis=1)
            self.created_features.append('MEDIA_IAA')
        
        # Score holístico (combinação de múltiplos indicadores)
        indicator_cols = []
        for prefix in ['IDA', 'IEG', 'IAN', 'IAA', 'IPS', 'IPP', 'IPV']:
            cols = [col for col in df.columns if prefix in col and '2022' in col]
            if cols:
                indicator_cols.extend(cols)
        
        if indicator_cols:
            df['SCORE_HOLISTICO'] = df[indicator_cols].mean(axis=1)
            self.created_features.append('SCORE_HOLISTICO')
        
        # Índice de desenvolvimento geral
        if 'INDE_2022' in df.columns:
            # Normaliza INDE para 0-1
            df['INDE_NORMALIZADO'] = (df['INDE_2022'] - df['INDE_2022'].min()) / (
                df['INDE_2022'].max() - df['INDE_2022'].min()
            )
            self.created_features.append('INDE_NORMALIZADO')
        
        return df
    
    def create_interaction_features(self, df):
        """
        Cria features de interação entre variáveis
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com features de interação
        """
        logger.info("Criando features de interação...")
        
        # Interação desempenho x engajamento
        if 'IDA_2022' in df.columns and 'IEG_2022' in df.columns:
            df['IDA_X_IEG'] = df['IDA_2022'] * df['IEG_2022']
            self.created_features.append('IDA_X_IEG')
        
        # Interação adequação x desempenho
        if 'IAN_2022' in df.columns and 'IDA_2022' in df.columns:
            df['IAN_X_IDA'] = df['IAN_2022'] * df['IDA_2022']
            self.created_features.append('IAN_X_IDA')
        
        # Interação psicossocial x pedagógico
        if 'IPS_2022' in df.columns and 'IPP_2022' in df.columns:
            df['IPS_X_IPP'] = df['IPS_2022'] * df['IPP_2022']
            self.created_features.append('IPS_X_IPP')
        
        # Razão desempenho/adequação
        if 'IDA_2022' in df.columns and 'IAN_2022' in df.columns:
            df['RAZAO_IDA_IAN'] = df['IDA_2022'] / (df['IAN_2022'] + 1e-5)
            self.created_features.append('RAZAO_IDA_IAN')
        
        # Consistência entre autoavaliação e desempenho
        if 'IAA_2022' in df.columns and 'IDA_2022' in df.columns:
            df['DIFF_IAA_IDA'] = abs(df['IAA_2022'] - df['IDA_2022'])
            self.created_features.append('DIFF_IAA_IDA')
        
        return df
    
    def create_risk_indicators(self, df):
        """
        Cria indicadores de risco de defasagem
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame com indicadores de risco
        """
        logger.info("Criando indicadores de risco...")
        
        # Risco por baixo IAN
        if 'IAN_2022' in df.columns:
            df['RISCO_IAN'] = (df['IAN_2022'] < 5).astype(int)
            self.created_features.append('RISCO_IAN')
        
        # Risco por baixo IDA
        if 'IDA_2022' in df.columns:
            threshold_ida = df['IDA_2022'].quantile(0.25)
            df['RISCO_IDA'] = (df['IDA_2022'] < threshold_ida).astype(int)
            self.created_features.append('RISCO_IDA')
        
        # Risco por baixo engajamento
        if 'IEG_2022' in df.columns:
            threshold_ieg = df['IEG_2022'].quantile(0.25)
            df['RISCO_ENGAJAMENTO'] = (df['IEG_2022'] < threshold_ieg).astype(int)
            self.created_features.append('RISCO_ENGAJAMENTO')
        
        # Score de risco geral
        risk_cols = [col for col in df.columns if 'RISCO_' in col]
        if risk_cols:
            df['SCORE_RISCO_GERAL'] = df[risk_cols].sum(axis=1)
            self.created_features.append('SCORE_RISCO_GERAL')
        
        return df
    
    def select_features(self, df, top_n=None):
        """
        Seleciona features mais importantes (placeholder para feature selection)
        
        Args:
            df: DataFrame
            top_n: Número de top features a selecionar
            
        Returns:
            DataFrame com features selecionadas
        """
        # Este método pode ser expandido com técnicas como:
        # - SelectKBest
        # - RFE (Recursive Feature Elimination)
        # - Feature importance de tree-based models
        
        logger.info("Seleção de features não implementada nesta versão")
        return df
    
    def get_feature_names(self):
        """
        Retorna lista de features criadas
        
        Returns:
            Lista de nomes de features
        """
        return self.created_features


def engineer_features_pipeline(df):
    """
    Pipeline completo de feature engineering
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com features engenheiradas
    """
    logger.info("Iniciando pipeline de feature engineering...")
    
    engineer = FeatureEngineer()
    
    # Cria todas as features
    df_engineered = engineer.create_all_features(df)
    
    # Cria indicadores de risco
    df_engineered = engineer.create_risk_indicators(df_engineered)
    
    logger.info(f"Pipeline de feature engineering concluído!")
    logger.info(f"Shape final: {df_engineered.shape}")
    
    return df_engineered, engineer


if __name__ == "__main__":
    # Teste do módulo
    logger.info("Módulo features.py carregado com sucesso!")