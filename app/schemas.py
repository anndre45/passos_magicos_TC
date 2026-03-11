"""
Schemas de validação para a API usando Pydantic
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
from datetime import datetime

class StudentInput(BaseModel):
    """Schema para entrada de dados do estudante"""
    
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "RA": "12345",
                "Ano ingresso": 2020,
                "Fase": "Fase 3",
                "Turma": "Turma A",
                "Gênero": "Feminino",
                "Instituição de ensino": "Escola Municipal",
                "IAA": 7.5,
                "IEG": 8.0,
                "IPS": 6.5,
                "IDA": 7.0,
                "IPV": 8.5,
                "IAN": 7.8
            }
        }
    )
    
    # Campos com aliases para aceitar nomes com espaços e acentos
    RA: str = Field(..., description="Registro do Aluno", alias="RA")
    ano_ingresso: int = Field(..., description="Ano de ingresso", alias="Ano ingresso")
    Fase: str = Field(..., description="Fase do aluno")
    Turma: str = Field(..., description="Turma do aluno")
    Genero: str = Field(..., description="Gênero do aluno", alias="Gênero")
    Instituicao_ensino: str = Field(..., description="Instituição de ensino", alias="Instituição de ensino")
    
    # Indicadores
    IAA: float = Field(..., description="Indicador de Autoavaliação")
    IEG: float = Field(..., description="Indicador de Engajamento")
    IPS: float = Field(..., description="Indicador Psicossocial")
    IDA: float = Field(..., description="Indicador de Desempenho Acadêmico")
    IPV: float = Field(..., description="Indicador de Ponto de Virada")
    IAN: float = Field(..., description="Indicador de Adequação de Nível")

class PredictionOutput(BaseModel):
    """Schema para resposta de predição"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Em fase",
                "prediction_code": 0,
                "confidence": 0.85,
                "probabilities": {
                    "Em fase": 0.85,
                    "Moderada": 0.10,
                    "Severa": 0.05
                }
            }
        }
    )
    
    prediction: str = Field(..., description="Classe predita")
    prediction_code: int = Field(..., description="Código da classe (0, 1, 2)")
    confidence: float = Field(..., description="Confiança da predição")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades por classe")

class PredictionResponse(BaseModel):
    """Schema para resposta de predição com metadados"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Em fase",
                "prediction_code": 0,
                "probabilities": {
                    "Em fase": 0.85,
                    "Moderada": 0.10,
                    "Severa": 0.05
                },
                "risk_level": "Baixo",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )
    
    prediction: str = Field(..., description="Classe predita")
    prediction_code: int = Field(..., description="Código da classe (0, 1, 2)")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades por classe")
    risk_level: str = Field(..., description="Nível de risco (Baixo, Médio, Alto)")
    timestamp: str = Field(..., description="Timestamp da predição")

class BatchPredictionInput(BaseModel):
    """Schema para predição em lote"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "students": [
                    {
                        "RA": "12345",
                        "Ano ingresso": 2020,
                        "Fase": "Fase 3",
                        "Turma": "Turma A",
                        "Gênero": "Feminino",
                        "Instituição de ensino": "Escola Municipal",
                        "IAA": 7.5,
                        "IEG": 8.0,
                        "IPS": 6.5,
                        "IDA": 7.0,
                        "IPV": 8.5,
                        "IAN": 7.8
                    }
                ]
            }
        }
    )
    
    students: list[StudentInput] = Field(..., description="Lista de estudantes")

class BatchPredictionResponse(BaseModel):
    """Schema para resposta de predição em lote"""
    predictions: list[PredictionResponse] = Field(..., description="Lista de predições")
    total_students: int = Field(..., description="Total de estudantes processados")
    timestamp: str = Field(..., description="Timestamp do processamento")

class HealthResponse(BaseModel):
    """Schema para resposta de health check"""
    status: str = Field(..., description="Status da API")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    version: str = Field(..., description="Versão da API")

class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo"""
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    model_type: str = Field(..., description="Tipo do modelo")
    features: list[str] = Field(..., description="Features esperadas")
    num_features: int = Field(..., description="Número de features")
    classes: Dict[str, str] = Field(..., description="Classes de predição")
    scaler_loaded: bool = Field(..., description="Se o scaler está carregado")
    encoders_loaded: bool = Field(..., description="Se os encoders estão carregados")

class ErrorResponse(BaseModel):
    """Schema para resposta de erro"""
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes do erro")
    timestamp: str = Field(..., description="Timestamp do erro")