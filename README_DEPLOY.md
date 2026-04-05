# Deployment and Drive Upload Guide

## 1. Upload compressed model to Google Drive

1. Open the folder link:
   https://drive.google.com/drive/u/2/folders/1jnV9OOVRxzPJTV0VSzXrxjVqJt6lnt0u
2. Upload `best_model_state_dict.pth.gz`.
3. Right-click the uploaded file and choose `Get link`.
4. Set sharing to `Anyone with the link` and `Viewer`.
5. Copy the shared link.
6. Convert it to a direct download link:
   - Shared link format:
     `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Direct download link:
     `https://drive.google.com/uc?export=download&id=FILE_ID`

## 2. Set `GDRIVE_FILE_ID` in `streamlit_app.py`

Replace the placeholder value:

```python
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"
```

with the actual `FILE_ID` from the shared link.

## 3. Programmatic upload (optional)

Install the required packages in a separate environment if needed:

```bash
pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib
```

Run the upload script:

```bash
python upload_to_drive.py --file best_model_state_dict.pth.gz --folder 1jnV9OOVRxzPJTV0VSzXrxjVqJt6lnt0u --credentials service_account.json
```

## 4. Deploy to Streamlit Cloud

1. Push your repository to GitHub.
2. Go to https://streamlit.io/cloud and log in.
3. Create a new app and connect your GitHub repo.
4. Set the app path to `streamlit_app.py`.
5. Use the repository's `requirements.txt`.
6. Deploy.

## 5. Best practice notes

- Do not commit `best_model.pth` or datasets to Git.
- Keep the repo under 1GB by storing large model files in Drive.
- Use `st.cache_resource` and `st.cache_data` for model loading and downloads.
- If the Drive download link changes, update `GDRIVE_FILE_ID`.
