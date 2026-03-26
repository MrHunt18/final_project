# predict.py
import pickle
import pandas as pd

fields = ['age','sex','cp','trestbps','chol','fbs',
          'restecg','thalach','exang','oldpeak','slope','ca','thal']

# Load model and preprocessor
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

mode = input("Enter mode (file/input): ").strip().lower()

if mode == "file":
    file_path = input("Enter CSV file path: ").strip()
    df = pd.read_csv(file_path)
    df = df[fields]
    X_encoded = preprocessor.transform(df)
    probas = model.predict_proba(X_encoded)[:, 1]
    preds = (probas >= 0.5).astype(int)

    for i, (pred, proba) in enumerate(zip(preds, probas)):
        print(f"Patient {i+1}: {'Yes' if pred == 1 else 'No'} (Probability: {proba:.2f})")

elif mode == "input":
    print("Enter values for the following features:")
    input_data = {}
    for field in fields:
        val = input(f"{field}: ")
        input_data[field] = float(val) if field not in ['cp', 'restecg', 'slope', 'thal'] else int(val)

    df = pd.DataFrame([input_data])
    X_encoded = preprocessor.transform(df)
    proba = model.predict_proba(X_encoded)[0][1]
    pred = 1 if proba >= 0.5 else 0

    print(f"✅ Probability: {proba:.2f}")
    print("✅ Heart Disease Prediction:", "Yes" if pred == 1 else "No")

else:
    print("❌ Invalid mode. Use 'file' or 'input'.")
