import joblib

# Carregar o label encoder
label_encoder = joblib.load('models/label_encoder.pkl')

print("=" * 70)
print("LABEL ENCODER")
print("=" * 70)
print(f"\nTipo: {type(label_encoder)}")
print(f"\nConteúdo completo:")
print(label_encoder)

if isinstance(label_encoder, dict):
    print("\n📋 É um dicionário! Chaves disponíveis:")
    for key, value in label_encoder.items():
        print(f"\n  Chave: {key}")
        print(f"  Tipo: {type(value)}")
        if hasattr(value, 'classes_'):
            print(f"  Classes: {value.classes_}")

print("=" * 70)