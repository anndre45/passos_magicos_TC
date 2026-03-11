import joblib
from pathlib import Path
import json

# Caminhos dos arquivos
models_dir = Path("models")

model_path = models_dir / "model.pkl"
scaler_path = models_dir / "scaler.pkl"
encoder_path = models_dir / "label_encoder.pkl"
metadata_path = models_dir / "model_metadata.json"

# Carrega os objetos
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoders = joblib.load(encoder_path)

print("="*80)
print("📦 MODELO")
print("="*80)
print(f"Tipo: {type(model)}")
print("Parâmetros:", getattr(model, "get_params", lambda: "N/A")())
print("Features usadas:", getattr(model, "feature_names_in_", "Não disponível"))
print()

print("="*80)
print("📏 SCALER")
print("="*80)
print(f"Tipo: {type(scaler)}")
print("Médias:", getattr(scaler, "mean_", "N/A"))
print("Desvios padrão:", getattr(scaler, "scale_", "N/A"))
print()

print("="*80)
print("🔤 ENCODERS")
print("="*80)
for col, le in encoders.items():
    print(f"Coluna: {col} → Classes: {list(le.classes_)}")
print()

print("="*80)
print("🧾 METADADOS")
print("="*80)
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
else:
    print("Arquivo model_metadata.json não encontrado.")
