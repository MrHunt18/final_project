# Streamlit Deployment Guide for ECG Model

## 1. Compress the model for Streamlit

Use the helper script to save only the PyTorch state dict and compress it:

```bash
python compress_model.py --input best_model.pth --intermediate best_model_state_dict.pth --output best_model_state_dict.pth.gz --half
```

This does:
- removes optimizer/training metadata
- optionally converts float32 weights to float16
- packs the result with gzip

## 2. Upload the compressed model to Google Drive

### Manual upload
1. Open your folder:
   `https://drive.google.com/drive/u/2/folders/1jnV9OOVRxzPJTV0VSzXrxjVqJt6lnt0u`
2. Click `New` → `File upload`
3. Upload `best_model_state_dict.pth.gz`
4. Right-click the file and choose `Get link`
5. Set the link to `Anyone with the link` and `Viewer`

### Generate a direct download link
If the share link is:

```
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

Use this direct URL in your app:

```
https://drive.google.com/uc?export=download&id=FILE_ID
```

## 3. Load the model in Streamlit

The app now uses `drive_utils.py` and `model_loader.py`:
- `download_from_gdrive()` downloads the file only once per session
- `load_ecg_model()` loads the compressed model into memory once

Update `GDRIVE_FILE_ID` in `streamlit_app.py` with your file ID if you change it.

## 4. Run the app

Install required packages and then launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 5. Validate the compressed model

```bash
python validate_model.py --model best_model_state_dict.pth.gz --image path/to/sample.png
```

## 6. Git / deployment cleanup

Add this repository metadata to `.gitignore` so large files do not get pushed:

```
*.pth
*.pt
*.pth.gz
dataset/
heart_disease_prediction/myenv/
__pycache__/
```

## 7. Notes
- Keep your app under `1GB` by not committing model files or dataset folders.
- Use caching for downloads and model loading.
- Keep inference code lightweight and avoid loading the model inside button callbacks.
