# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Define categorical and numerical columns
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing: One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Transform features
X_encoded = preprocessor.fit_transform(X)

# Apply SMOTE after encoding
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_encoded, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
scores = cross_val_score(model, X_res, y_res, cv=5, scoring='accuracy')
print(f"✅ XGBoost Cross-validated accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

# Save model and transformer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("✅ Model and preprocessor saved.")
