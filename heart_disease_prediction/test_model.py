# test_model.py
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

X_encoded = preprocessor.transform(X)
probas = model.predict_proba(X_encoded)[:, 1]
y_pred = (probas >= 0.5).astype(int)

acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, probas)
cm = confusion_matrix(y, y_pred)

print(f"✅ Accuracy: {acc * 100:.2f}%")
print(f"✅ ROC AUC Score: {auc:.2f}")
print("📊 Confusion Matrix:")
print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
