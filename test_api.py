"""
Script de teste da API - Passos Mágicos
Pessoa 2: Natasha
"""

import requests
import json

# URL da API
BASE_URL = "http://localhost:5000"

def test_health():
    """Testa o endpoint de health check"""
    print("=" * 50)
    print("🔍 TESTANDO HEALTH CHECK")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict():
    """Testa o endpoint de predição"""
    print("=" * 50)
    print("🔮 TESTANDO PREDIÇÃO")
    print("=" * 50)
    
    # Exemplo 1: Estudante em fase
    data_em_fase = {
        "PEDRA_2022": 70,
        "IAA_2022": 7.5,
        "IEG_2022": 8.0,
        "IPS_2022": 6.5,
        "IDA_2022": 7.0,
        "IPP_2022": 7.5,
        "IPV_2022": 8.5,
        "IAN_2022": 7.8,
        "INDE_2022": 7.2,
        "PEDRA_2023": 75,
        "IAA_2023": 8.0,
        "IEG_2023": 8.5,
        "IPS_2023": 7.0,
        "IDA_2023": 7.5,
        "IPP_2023": 8.0,
        "IPV_2023": 9.0,
        "IAN_2023": 8.2,
        "INDE_2023": 7.8,
        "PEDRA_2024": 80,
        "IAA_2024": 8.5,
        "IEG_2024": 9.0,
        "IPS_2024": 7.5,
        "IDA_2024": 8.0,
        "IPP_2024": 8.5,
        "IPV_2024": 9.5,
        "IAN_2024": 8.8,
        "INDE_2024": 8.4
    }
    
    print("\n📊 EXEMPLO 1: Estudante com bom desempenho")
    print("-" * 50)
    response = requests.post(f"{BASE_URL}/predict", json=data_em_fase)
    print(f"Status Code: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # Exemplo 2: Estudante com risco moderado
    data_moderada = {
        "PEDRA_2022": 50,
        "IAA_2022": 5.0,
        "IEG_2022": 5.5,
        "IPS_2022": 4.5,
        "IDA_2022": 5.0,
        "IPP_2022": 5.5,
        "IPV_2022": 6.0,
        "IAN_2022": 5.2,
        "INDE_2022": 5.0,
        "PEDRA_2023": 55,
        "IAA_2023": 5.5,
        "IEG_2023": 6.0,
        "IPS_2023": 5.0,
        "IDA_2023": 5.5,
        "IPP_2023": 6.0,
        "IPV_2023": 6.5,
        "IAN_2023": 5.8,
        "INDE_2023": 5.5,
        "PEDRA_2024": 60,
        "IAA_2024": 6.0,
        "IEG_2024": 6.5,
        "IPS_2024": 5.5,
        "IDA_2024": 6.0,
        "IPP_2024": 6.5,
        "IPV_2024": 7.0,
        "IAN_2024": 6.2,
        "INDE_2024": 6.0
    }
    
    print("\n📊 EXEMPLO 2: Estudante com desempenho moderado")
    print("-" * 50)
    response = requests.post(f"{BASE_URL}/predict", json=data_moderada)
    print(f"Status Code: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    print()

def test_error():
    """Testa tratamento de erros"""
    print("=" * 50)
    print("⚠️ TESTANDO TRATAMENTO DE ERROS")
    print("=" * 50)
    
    # Dados incompletos
    data_invalida = {
        "PEDRA_2022": 70,
        "IAA_2022": 7.5
        # Faltam campos obrigatórios
    }
    
    print("\n❌ Enviando dados incompletos:")
    print("-" * 50)
    response = requests.post(f"{BASE_URL}/predict", json=data_invalida)
    print(f"Status Code: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 INICIANDO TESTES DA API")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_predict()
        test_error()
        
        print("=" * 50)
        print("✅ TODOS OS TESTES CONCLUÍDOS!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando em http://localhost:5000")
        print("\nPara iniciar a API, execute:")
        print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 5000")