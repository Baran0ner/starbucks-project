import pickle
import os
from model import load_data, preprocess_data, train_model

def save_model():
    print("📥 Veriler yükleniyor...")
    data = load_data("starbucks.csv")

    print("🔧 Veri ön işleniyor...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)

    print("🤖 Model eğitiliyor...")
    model = train_model(X_train, y_train)

    # Klasörü oluştur (varsa hata vermez)
    os.makedirs("models", exist_ok=True)

    # Modeli kaydet
    with open("models/starbucks_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model kaydedildi → models/starbucks_model.pkl")

    # Scaler'ı kaydet
    with open("models/starbucks_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Scaler kaydedildi → models/starbucks_scaler.pkl")

    # Feature isimlerini kaydet
    with open("models/starbucks_feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("✅ Feature isimleri kaydedildi → models/starbucks_feature_names.pkl")

if __name__ == "__main__":
    save_model()
