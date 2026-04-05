import argparse
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def upload_file_service_account(file_path: str, folder_id: str, credentials_path: str):
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id],
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id,webViewLink").execute()
    print("Uploaded file ID:", file["id"])
    print("Web view link:", file["webViewLink"])
    print("Direct download link:", f"https://drive.google.com/uc?export=download&id={file['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Google Drive using a service account.")
    parser.add_argument("--file", required=True, help="Path to the file to upload.")
    parser.add_argument("--folder", required=True, help="Google Drive folder ID.")
    parser.add_argument("--credentials", required=True, help="Path to the service account JSON credentials file.")
    args = parser.parse_args()

    upload_file_service_account(args.file, args.folder, args.credentials)
