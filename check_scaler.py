import joblib

# Carregar o scaler
scaler = joblib.load('models/scaler.pkl')

print("=" * 70)
print("SCALER")
print("=" * 70)
print(f"\nTipo: {type(scaler)}")

if hasattr(scaler, 'feature_names_in_'):
    print(f"\nFeatures que foram normalizadas:")
    for feature in scaler.feature_names_in_:
        print(f"  - {feature}")

if hasattr(scaler, 'mean_'):
    print(f"\nNúmero de features numéricas: {len(scaler.mean_)}")

print("=" * 70)