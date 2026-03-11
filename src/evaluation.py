"""
evaluation.py
Módulo de avaliação do modelo para o projeto Passos Mágicos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import logging
from utils import Config, decode_defasagem_target

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe para avaliação de modelos"""
    
    def __init__(self):
        self.metrics_history = []
        self.class_names = ['Em fase', 'Moderada', 'Severa']
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calcula todas as métricas de avaliação
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            Dicionário com métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def evaluate(self, y_true, y_pred, dataset_name="Dataset"):
        """
        Avalia modelo e exibe métricas
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            dataset_name: Nome do dataset (Treino/Teste)
            
        Returns:
            Dicionário com métricas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AVALIAÇÃO - {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Calcula métricas
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Exibe métricas
        logger.info(f"\nMétricas Gerais:")
        logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
        logger.info(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
        logger.info(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nMatriz de Confusão:")
        logger.info(f"\n{cm}")
        
        # Classification Report
        logger.info(f"\nRelatório de Classificação:")
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        logger.info(f"\n{report}")
        
        # Salva histórico
        self.metrics_history.append({
            'dataset': dataset_name,
            'metrics': metrics,
            'confusion_matrix': cm
        })
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plota matriz de confusão
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            save_path: Caminho para salvar a figura
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Contagem'}
        )
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.ylabel('Valor Real', fontsize=12)
        plt.xlabel('Valor Predito', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusão salva em: {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, metrics_train, metrics_test, save_path=None):
        """
        Plota comparação de métricas entre treino e teste
        
        Args:
            metrics_train: Métricas de treino
            metrics_test: Métricas de teste
            save_path: Caminho para salvar a figura
        """
        metrics_names = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        train_values = [metrics_train[m] for m in metrics_names]
        test_values = [metrics_test[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, train_values, width, label='Treino', color='steelblue')
        bars2 = ax.bar(x + width/2, test_values, width, label='Teste', color='coral')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparação de Métricas: Treino vs Teste', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparação de métricas salva em: {save_path}")
        
        plt.close()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """
        Plota distribuição de classes (real vs predito)
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            save_path: Caminho para salvar a figura
        """
        # Decodifica valores se necessário
        if not isinstance(y_true.iloc[0] if isinstance(y_true, pd.Series) else y_true[0], str):
            y_true_decoded = decode_defasagem_target(pd.Series(y_true))
            y_pred_decoded = decode_defasagem_target(pd.Series(y_pred))
        else:
            y_true_decoded = y_true
            y_pred_decoded = y_pred
        
        # Conta distribuições
        true_counts = pd.Series(y_true_decoded).value_counts()
        pred_counts = pd.Series(y_pred_decoded).value_counts()
        
        # Cria DataFrame para plotagem
        df_plot = pd.DataFrame({
            'Real': true_counts,
            'Predito': pred_counts
        }).fillna(0)
        
        # Plota
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot.plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
        
        ax.set_title('Distribuição de Classes: Real vs Predito', fontsize=16, fontweight='bold')
        ax.set_xlabel('Classe de Defasagem', fontsize=12)
        ax.set_ylabel('Contagem', fontsize=12)
        ax.legend(['Real', 'Predito'])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribuição de classes salva em: {save_path}")
        
        plt.close()
    
    def generate_report(self, y_true, y_pred, model_name, save_path=None):
        """
        Gera relatório completo de avaliação
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            save_path: Caminho para salvar o relatório
        """
        report = []
        report.append("="*70)
        report.append(f"RELATÓRIO DE AVALIAÇÃO DO MODELO")
        report.append("="*70)
        report.append(f"\nModelo: {model_name}")
        report.append(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "="*70)
        
        # Métricas
        metrics = self.calculate_metrics(y_true, y_pred)
        report.append("\nMÉTRICAS DE DESEMPENHO:")
        report.append("-"*70)
        for metric_name, metric_value in metrics.items():
            report.append(f"{metric_name.replace('_', ' ').title():.<40} {metric_value:.4f}")
        
        # Matriz de Confusão
        cm = confusion_matrix(y_true, y_pred)
        report.append("\n" + "="*70)
        report.append("MATRIZ DE CONFUSÃO:")
        report.append("-"*70)
        report.append(f"\n{cm}")
        
        # Classification Report
        report.append("\n" + "="*70)
        report.append("RELATÓRIO DETALHADO POR CLASSE:")
        report.append("-"*70)
        class_report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        report.append(f"\n{class_report}")
        
        # Análise de erros
        report.append("\n" + "="*70)
        report.append("ANÁLISE DE ERROS:")
        report.append("-"*70)
        errors = y_true != y_pred
        error_rate = errors.sum() / len(y_true)
        report.append(f"Taxa de erro total: {error_rate:.2%}")
        report.append(f"Total de erros: {errors.sum()} de {len(y_true)} predições")
        
        # Análise por classe
        report.append("\nErros por classe:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_true == i
            class_errors = errors[class_mask].sum() if class_mask.sum() > 0 else 0
            class_total = class_mask.sum()
            if class_total > 0:
                class_error_rate = class_errors / class_total
                report.append(f"  {class_name}: {class_errors}/{class_total} ({class_error_rate:.2%})")
        
        # Recomendações
        report.append("\n" + "="*70)
        report.append("RECOMENDAÇÕES:")
        report.append("-"*70)
        
        if metrics['f1_weighted'] >= 0.80:
            report.append("✓ Modelo apresenta desempenho EXCELENTE (F1 >= 0.80)")
            report.append("  Recomendado para produção com monitoramento contínuo.")
        elif metrics['f1_weighted'] >= 0.70:
            report.append("✓ Modelo apresenta desempenho BOM (F1 >= 0.70)")
            report.append("  Pode ser colocado em produção com monitoramento atento.")
        elif metrics['f1_weighted'] >= 0.60:
            report.append("⚠ Modelo apresenta desempenho MODERADO (F1 >= 0.60)")
            report.append("  Considere ajustes antes de produção.")
        else:
            report.append("✗ Modelo apresenta desempenho BAIXO (F1 < 0.60)")
            report.append("  NÃO recomendado para produção. Necessário retreinamento.")
        
        report.append("\n" + "="*70)
        report.append("FIM DO RELATÓRIO")
        report.append("="*70)
        
        # Junta o relatório
        full_report = "\n".join(report)
        
        # Exibe no log
        logger.info(f"\n{full_report}")
        
        # Salva em arquivo
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            logger.info(f"\nRelatório salvo em: {save_path}")
        
        return full_report
    
    def evaluate_model_confidence(self, y_true, y_pred_proba):
        """
        Avalia confiança das predições
        
        Args:
            y_true: Valores reais
            y_pred_proba: Probabilidades preditas
            
        Returns:
            Dicionário com estatísticas de confiança
        """
        max_probas = np.max(y_pred_proba, axis=1)
        
        stats = {
            'mean_confidence': np.mean(max_probas),
            'std_confidence': np.std(max_probas),
            'min_confidence': np.min(max_probas),
            'max_confidence': np.max(max_probas),
            'median_confidence': np.median(max_probas)
        }
        
        logger.info("\nEstatísticas de Confiança:")
        for key, value in stats.items():
            logger.info(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        return stats
    
    def compare_models(self, models_results):
        """
        Compara múltiplos modelos
        
        Args:
            models_results: Dicionário {nome_modelo: metrics}
            
        Returns:
            DataFrame com comparação
        """
        comparison_df = pd.DataFrame(models_results).T
        comparison_df = comparison_df.sort_values('f1_weighted', ascending=False)
        
        logger.info("\n" + "="*70)
        logger.info("COMPARAÇÃO DE MODELOS")
        logger.info("="*70)
        logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df


if __name__ == "__main__":
    # Teste do módulo
    logger.info("Módulo evaluation.py carregado com sucesso!")