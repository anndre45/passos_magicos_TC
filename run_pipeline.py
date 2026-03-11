"""
run_pipeline.py
Script para executar o pipeline completo de Machine Learning
"""

import sys
import logging
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils import Config, logger
from src.train import train_pipeline
from src.model_test import test_model


def main():
    """
    Executa o pipeline completo de treinamento e teste
    """
    print("="*80)
    print("🚀 INICIANDO PIPELINE COMPLETO - PASSOS MÁGICOS")
    logger.info("Pipeline iniciado")
    print("="*80)
    print()
    
    try:
        # Cria diretórios necessários
        logger.info("📁 Criando estrutura de diretórios...")
        Config.create_directories()
        
        # Verifica se arquivo de dados existe
        data_path = Config.RAW_DATA_DIR / Config.RAW_DATA_FILE
        if not data_path.exists():
            logger.error(f"❌ Arquivo de dados não encontrado: {data_path}")
            logger.info("📝 Por favor, coloque o arquivo 'Base fiap.xlsx' em data/raw/")
            return
        
        logger.info(f"✅ Arquivo de dados encontrado: {data_path}")
        print()
        
        # ETAPA 1: Treinamento
        print("="*80)
        print("📊 ETAPA 1/2: TREINAMENTO DO MODELO")
        print("="*80)
        print()
        
        trainer, evaluator, results = train_pipeline(
            data_path,
            tune_hyperparams=True,  # Ajustar hiperparâmetros
            save_model=True         # Salvar modelo
        )
        
        print()
        print("="*80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"🏆 Melhor modelo: {trainer.best_model_name}")
        print(f"📈 F1-Score: {trainer.best_score:.4f}")
        print()
        
        # ETAPA 2: Teste
        print("="*80)
        print("🧪 ETAPA 2/2: TESTE DO MODELO")
        print("="*80)
        print()
        
        predictor = test_model()
        
        print()
        print("="*80)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print()
        
        # Resumo Final
        print("="*80)
        print("📋 RESUMO FINAL")
        print("="*80)
        print()
        print("✅ Arquivos gerados:")
        print(f"   📦 Modelo: {Config.MODELS_DIR / Config.MODEL_FILE}")
        print(f"   📦 Scaler: {Config.MODELS_DIR / Config.SCALER_FILE}")
        print(f"   📦 Encoders: {Config.MODELS_DIR / Config.LABEL_ENCODER_FILE}")
        print(f"   📄 Metadados: {Config.MODELS_DIR / 'model_metadata.json'}")
        print(f"   📊 Relatório: {Config.MODELS_DIR / 'evaluation_report.txt'}")
        print()
        print("✅ Dados processados:")
        print(f"   📊 Treino: {Config.PROCESSED_DATA_DIR / Config.TRAIN_DATA_FILE}")
        print(f"   📊 Teste: {Config.PROCESSED_DATA_DIR / Config.TEST_DATA_FILE}")
        print()
        print("📝 Próximos passos:")
        print("   1. Revisar o relatório de avaliação")
        print("   2. Integrar modelo com a API (Pessoa 2)")
        print("   3. Implementar testes unitários (Pessoa 3)")
        print("   4. Configurar monitoramento (Pessoa 4)")
        print()
        print("="*80)
        print("🎉 PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução do pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()