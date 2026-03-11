import joblib
import json

# Carregar o modelo
model = joblib.load('models/model.pkl')

# Carregar metadados
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print("=" * 70)
print("INFORMAÇÕES DO MODELO")
print("=" * 70)
print(f"\nTipo do modelo: {type(model).__name__}")
print(f"\nMetadados: {json.dumps(metadata, indent=2)}")

# Tentar descobrir as features
if hasattr(model, 'feature_names_in_'):
    print(f"\n✅ Features esperadas ({len(model.feature_names_in_)}):")
    for i, feature in enumerate(model.feature_names_in_, 1):
        print(f"  {i}. {feature}")
else:
    print("\n⚠️ Modelo não tem 'feature_names_in_'. Verificando n_features_in_...")
    if hasattr(model, 'n_features_in_'):
        print(f"  Número de features: {model.n_features_in_}")

print("\n" + "=" * 70)