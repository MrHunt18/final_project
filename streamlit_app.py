import os
import gzip
import io
import streamlit as st
import torch
import pandas as pd
import gdown
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
HEART_DIR = os.path.join(ROOT, "heart_disease_prediction")
MODEL_PKL = os.path.join(HEART_DIR, "model.pkl")
PREPROCESSOR_PKL = os.path.join(HEART_DIR, "preprocessor.pkl")
LOCAL_MODEL_PATH = os.path.join(ROOT, "best_model_state_dict.pth.gz")
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"

# Load heart disease model and preprocessor once
heart_model = None
heart_preprocessor = None

if os.path.exists(MODEL_PKL) and os.path.exists(PREPROCESSOR_PKL):
    with open(MODEL_PKL, "rb") as f:
        heart_model = pd.read_pickle(f)
    with open(PREPROCESSOR_PKL, "rb") as f:
        heart_preprocessor = pd.read_pickle(f)

st.set_page_config(page_title="Combined ECG + Heart Disease App", layout="wide")

st.title("Combined Heart Disease & ECG Detection")
st.markdown("This app integrates ECG image detection and tabular heart disease prediction.")

@st.cache_data(show_spinner=False)
def download_model_from_drive(file_id: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        return dest_path

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    return dest_path

@st.cache_resource
def load_ecg_model(model_path: str):
    with gzip.open(model_path, "rb") as f:
        buffer = io.BytesIO(f.read())

    state_dict = torch.load(buffer, map_location="cpu")
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()
    return model

ecg_model = None
if GDRIVE_FILE_ID == "YOUR_FILE_ID_HERE":
    st.warning("Set GDRIVE_FILE_ID in streamlit_app.py after uploading the compressed model to Google Drive.")
else:
    try:
        model_path = download_model_from_drive(GDRIVE_FILE_ID, LOCAL_MODEL_PATH)
        ecg_model = load_ecg_model(model_path)
    except Exception as error:
        ecg_model = None
        st.error("Unable to load ECG model from Google Drive. Check the file ID and network access.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class_names = ["abnormal", "history_mi", "mi", "normal"]

tab1, tab2 = st.tabs(["ECG Detection", "Heart Disease Tabular"])

with tab1:
    st.header("ECG Image Classification")
    uploaded = st.file_uploader("Upload ECG image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded ECG", use_column_width=True)

        if st.button("Run ECG Prediction"):
            if ecg_model is None:
                st.error("ECG model is not loaded.")
            else:
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = ecg_model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).squeeze()
                    pred = probs.argmax().item()

                st.write(f"Prediction: **{class_names[pred]}**")
                st.write(f"Confidence: **{probs[pred].item() * 100:.1f}%**")

with tab2:
    st.header("Heart Disease Prediction (Tabular)")
    if heart_model is None or heart_preprocessor is None:
        st.warning("Heart disease model/preprocessor not found in heart_disease_prediction folder.")
    else:
        fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        user_vals = {}
        cols = st.columns(3)
        for i, fld in enumerate(fields):
            if fld in ["age","trestbps","chol","thalach","oldpeak"]:
                user_vals[fld] = cols[i % 3].number_input(
                    fld,
                    value=0.0 if fld == "oldpeak" else 0,
                    format="%f" if fld == "oldpeak" else "%d"
                )
            else:
                user_vals[fld] = cols[i % 3].number_input(fld, value=0, step=1)

        if st.button("Predict Heart Disease"):
            df_in = pd.DataFrame([user_vals])
            try:
                x_enc = heart_preprocessor.transform(df_in)
                prob = heart_model.predict_proba(x_enc)[:, 1][0]
                pred = int(prob >= 0.5)
                st.write(f"Probability of heart disease: {prob:.4f}")
                st.write("Prediction: ✅ Heart disease" if pred == 1 else "Prediction: ❌ No heart disease")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown("---")
st.write("Built from existing ECG detection and heart_disease_prediction modules.")
