"""
main.py
API FastAPI para predição de defasagem escolar
Projeto: Passos Mágicos - Predição de Defasagem Escolar
"""
import csv

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager
from app.schemas import StudentInput, PredictionOutput, HealthResponse
from app.predictor import ModelPredictor, get_predictor  # ✅ ADICIONAR get_predictor
from pathlib import Path

# Configuração de logging
Path("logs").mkdir(exist_ok=True)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("passos_magicos_api")
# Gerenciador de ciclo de vida (DEVE vir ANTES da inicialização do FastAPI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    # Startup
    try:
        logger.info("🚀 Iniciando API...")
        predictor = get_predictor()
        logger.info("✅ API iniciada com sucesso!")
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar API: {e}")
        raise
    
    yield  # API está rodando
    
    # Shutdown (se necessário fazer cleanup)
    logger.info("🛑 Encerrando API...")

# Inicializa FastAPI (AGORA o lifespan já está definido)
app = FastAPI(
    title="Passos Mágicos - API de Predição",
    description="API para predição de risco de defasagem escolar",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configuração CORS (permite requisições de qualquer origem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informações básicas da API"""
    return {
        "message": "API Passos Mágicos - Predição de Defasagem Escolar",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check - verifica se a API está funcionando
    
    Returns:
        Status da API e do modelo
    """
    try:
        predictor = get_predictor()
        
        return HealthResponse(
            status="healthy",
            model_loaded=predictor.is_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Serviço indisponível: {str(e)}"
        )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(student: StudentInput):
    """
    Endpoint de predição - recebe dados do estudante e retorna predição
    
    Args:
        student: Dados do estudante (StudentInput schema)
        
    Returns:
        Predição de defasagem escolar (PredictionOutput schema)
        
    Raises:
        HTTPException: Se houver erro na predição
    """
    try:
        logger.info(f"📥 Recebendo requisição de predição para RA: {student.RA}")
        
        # Obtém preditor
        predictor = get_predictor()
        
        # Converte input para dicionário
        input_data = student.model_dump()  # Pydantic V2 usa model_dump() ao invés de dict()
        
        # Realiza predição
        result = predictor.predict(input_data)
        
        logger.info(
            f"Prediction | RA={student.RA} | "
            f"class={result['prediction']} | "
            f"confidence={result['confidence']:.2f}"
        )       
        
        log_file = Path("logs/predictions.csv")

        write_header = not log_file.exists()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["timestamp", "RA", "prediction", "confidence"])

            writer.writerow([
                datetime.now().isoformat(),
                student.RA,
                result["prediction"],
                result["confidence"]
            ])
        
        return PredictionOutput(**result)
        
    except ValueError as e:
        logger.error(f"❌ Erro de validação: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erro de validação: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )

@app.post("/predict/batch", response_model=List[PredictionOutput], tags=["Prediction"])
async def predict_batch(students: List[StudentInput]):
    """
    Endpoint de predição em lote - recebe lista de estudantes
    
    Args:
        students: Lista de dados dos estudantes
        
    Returns:
        Lista de predições
        
    Raises:
        HTTPException: Se houver erro nas predições
    """
    try:
        logger.info(f"📥 Recebendo requisição de predição em lote: {len(students)} estudantes")
        
        # Obtém preditor
        predictor = get_predictor()
        
        # Converte inputs para lista de dicionários
        input_list = [student.model_dump() for student in students]  # Pydantic V2
        
        # Realiza predições
        results = predictor.predict_batch(input_list)
        
        logger.info(f"✅ {len(results)} predições realizadas")
        
        return [PredictionOutput(**result) for result in results if "error" not in result]
        
    except Exception as e:
        logger.error(f"❌ Erro na predição em lote: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Retorna informações sobre o modelo carregado
    
    Returns:
        Informações do modelo
    """
    try:
        predictor = get_predictor()
        
        info = {
            "model_loaded": predictor.is_loaded,
            "model_type": type(predictor.model).__name__ if predictor.model else None,
            "features": predictor.feature_names,
            "num_features": len(predictor.feature_names) if predictor.feature_names else 0,
            "classes": predictor.class_mapping,
            "scaler_loaded": predictor.scaler is not None,
            "encoders_loaded": predictor.label_encoders is not None
        }
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter informações do modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )

# Tratamento de erros global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para exceções não tratadas"""
    logger.error(f"❌ Erro não tratado: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Erro interno do servidor",
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Roda o servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )