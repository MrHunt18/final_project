import os
import pickle
import xgboost as xgb

ROOT = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(ROOT, "model.pkl")
refit_path = os.path.join(ROOT, "model_resaved.pkl")
booster_path = os.path.join(ROOT, "model_booster.json")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Loaded model from model.pkl")

if hasattr(model, "get_booster"):
    booster = model.get_booster()
    booster.save_model(booster_path)
    print(f"Saved booster as {booster_path}")
    # Re-pickle model in current environment
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Re-saved model pickle to {model_path}")

    # Optional: also save as xgb model file
    model.save_model(os.path.join(ROOT, "model_xgb.json"))
    print("Saved XGB model with model.save_model() as model_xgb.json")

else:
    print("Loaded model does not support get_booster(); cannot refresh")
