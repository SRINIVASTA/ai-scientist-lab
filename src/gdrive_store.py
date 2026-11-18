import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class GDriveStore:

    def __init__(self, service_account_json_str: str, folder_id: str):
        creds_dict = json.loads(service_account_json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://googleapis.com"]
        )
        self.service = build("drive", "v3", credentials=creds)
        self.folder_id = folder_id

    def upload_file(
        self, local_path: str, mime_type: str = "application/octet-stream"
    ) -> str:
        file_metadata = {
            "name": os.path.basename(local_path),
            "parents": [self.folder_id],
        }
        media = MediaFileUpload(local_path, mimetype=mime_type)
        uploaded = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id, webViewLink")
            .execute()
        )
        return uploaded.get("webViewLink", "")

    def list_artifacts(self, limit: int = 15):
        query = f"'{self.folder_id}' in parents and trashed = false"
        results = (
            self.service.files()
            .list(
                q=query,
                pageSize=limit,
                fields="files(id, name, createdTime, webViewLink)",
            )
            .execute()
        )
        return results.get("files", [])
