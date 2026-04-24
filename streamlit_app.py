import os
import pickle

import streamlit as st
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from drive_utils import download_from_gdrive
from model_loader import load_ecg_model

ROOT = os.path.abspath(os.path.dirname(__file__))
HEART_DIR = os.path.join(ROOT, "heart_disease_prediction")
MODEL_PKL = os.path.join(HEART_DIR, "model.pkl")
PREPROCESSOR_PKL = os.path.join(HEART_DIR, "preprocessor.pkl")
LOCAL_MODEL_PATH = os.path.join(ROOT, "best_model_state_dict.pth.gz")
GDRIVE_FILE_ID = "1bzQQSQi96Ed8HUQA3Zzyh6ONmTydHmsH"


def load_heart_models():
    if os.path.exists(MODEL_PKL) and os.path.exists(PREPROCESSOR_PKL):
        with open(MODEL_PKL, "rb") as f:
            heart_model = pickle.load(f)
        with open(PREPROCESSOR_PKL, "rb") as f:
            heart_preprocessor = pickle.load(f)
        return heart_model, heart_preprocessor
    return None, None


@st.cache_resource
def get_ecg_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        return load_ecg_model(LOCAL_MODEL_PATH)

    model_path = download_from_gdrive(GDRIVE_FILE_ID, LOCAL_MODEL_PATH)
    return load_ecg_model(model_path)


@st.cache_resource
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


heart_model, heart_preprocessor = load_heart_models()

st.set_page_config(page_title="Combined ECG + Heart Disease App", layout="wide")

st.title("Combined Heart Disease & ECG Detection")
st.markdown("This app integrates ECG image detection and tabular heart disease prediction.")

tab1, tab2 = st.tabs(["ECG Detection", "Heart Disease Tabular"])

with tab1:
    st.header("ECG Image Classification")
    uploaded_file = st.file_uploader("Upload ECG image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded ECG", use_column_width=True)

        try:
            ecg_model = get_ecg_model()
            transform = get_image_transform()
        except Exception as e:
            ecg_model = None
            transform = None
            st.error(f"Unable to load ECG model: {e}")

        if st.button("Run ECG Prediction"):
            if ecg_model is None or transform is None:
                st.error("ECG model is not available. Please check your Google Drive file ID and connection.")
            else:
                try:
                    input_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        outputs = ecg_model(input_tensor)
                        probs = torch.softmax(outputs, dim=1).squeeze()
                        predicted_index = int(probs.argmax().item())

                    class_names = ["abnormal", "history_mi", "mi", "normal"]
                    prediction = class_names[predicted_index]
                    confidence = float(probs[predicted_index].item() * 100)

                    st.success(f"Prediction: **{prediction}**")
                    st.write(f"Confidence: **{confidence:.1f}%**")

                    st.markdown("**Class probabilities:**")
                    for name, score in zip(class_names, probs.tolist()):
                        st.write(f"- {name}: {score * 100:.2f}%")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with tab2:
    st.header("Heart Disease Prediction (Tabular)")
    if heart_model is None or heart_preprocessor is None:
        st.warning("Heart disease model/preprocessor not found in heart_disease_prediction folder.")
    else:
        fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        user_vals = {}
        cols = st.columns(3)
        for i, fld in enumerate(fields):
            if fld in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
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
                #st.write(f"Probability of heart disease: {prob:.4f}")
                st.write("Prediction: ✅ there is a risk of Heart disease" if pred == 1 else "Prediction: ❌ No heart disease risk")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown("---")
st.write("this is just an insight, please consult the professionals.")
