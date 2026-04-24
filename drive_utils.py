import os

import gdown
import streamlit as st


@st.cache_data(show_spinner=False)
def download_from_gdrive(file_id: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        return dest_path

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    if not os.path.exists(dest_path):
        raise FileNotFoundError(f"Failed to download model from Google Drive: {file_id}")
    return dest_path