# validate_random.py
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Load test set
df = pd.read_csv("test_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Scale and predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
acc = accuracy_score(y, y_pred)

print(f"✅ Test Accuracy on unseen data: {acc * 100:.2f}%")
