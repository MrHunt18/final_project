import gzip
import io
import os

import streamlit as st
import torch
from models.resnet_model import get_model


@st.cache_resource(ttl=24 * 3600)
def load_ecg_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Compressed model file not found: {model_path}")

    with gzip.open(model_path, "rb") as f:
        buffer = io.BytesIO(f.read())

    state_dict = torch.load(buffer, map_location="cpu")
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()
    return model
