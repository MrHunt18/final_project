import streamlit as st
import os
import io
import pickle
import torch
import pandas as pd
from PIL import Image
from contextlib import redirect_stdout

# ECG prediction importing the existing script function
from predict import predict as ecg_predict_func

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
HEART_DIR = os.path.join(ROOT, "heart_disease_prediction")
MODEL_PKL = os.path.join(HEART_DIR, "model.pkl")
PREPROCESSOR_PKL = os.path.join(HEART_DIR, "preprocessor.pkl")

# Load heart disease model and preprocessor once
heart_model = None
heart_preprocessor = None

if os.path.exists(MODEL_PKL) and os.path.exists(PREPROCESSOR_PKL):
    with open(MODEL_PKL, "rb") as f:
        heart_model = pickle.load(f)
    with open(PREPROCESSOR_PKL, "rb") as f:
        heart_preprocessor = pickle.load(f)

st.set_page_config(page_title="Combined ECG + Heart Disease App", layout="wide")

st.title("Combined Heart Disease & ECG Detection")
st.markdown("This app integrates ECG image detection and tabular heart disease prediction.")

tab1, tab2 = st.tabs(["ECG Detection", "Heart Disease Tabular"])

with tab1:
    st.header("ECG Image Classification")
    uploaded = st.file_uploader("Upload ECG image", type=["png","jpg","jpeg","bmp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded ECG", use_column_width=True)
        if st.button("Run ECG Prediction"):
            temp_path = os.path.join(ROOT, "temp_ecg_for_streamlit.png")
            image.save(temp_path)
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    ecg_predict_func(temp_path)
                except Exception as e:
                    print(f"Error: {e}")
            st.text(buf.getvalue())

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
                user_vals[fld] = cols[i % 3].number_input(fld, value=0.0 if fld=="oldpeak" else 0, format="%f" if fld=="oldpeak" else "%d")
            else:
                user_vals[fld] = cols[i % 3].number_input(fld, value=0, step=1)

        if st.button("Predict Heart Disease"):
            df_in = pd.DataFrame([user_vals])
            try:
                x_enc = heart_preprocessor.transform(df_in)
                prob = heart_model.predict_proba(x_enc)[:,1][0]
                pred = (prob >= 0.5).astype(int)
                st.write(f"Probability of heart disease: {prob:.4f}")
                st.write("Prediction: ✅ Heart disease" if pred==1 else "Prediction: ❌ No heart disease")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown("---")
st.write("Built from existing ECG detection and heart_disease_prediction modules.")
