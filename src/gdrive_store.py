import json
import os
from typing import Union
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class GDriveStore:

    def __init__(
        self,
        service_account_info: Union[str, dict] = None,
        folder_id: str = "",
        service_account_json_str: Union[str, dict] = None,
    ):
        raw_creds = (
            service_account_info
            if service_account_info is not None
            else service_account_json_str
        )
        if raw_creds is None:
            raise ValueError("Google Service Account credentials are required.")

        if isinstance(raw_creds, str):
            creds_dict = json.loads(raw_creds, strict=False)
        else:
            creds_dict = dict(raw_creds)

        if "private_key" in creds_dict:
            pk = creds_dict["private_key"].replace("\\n", "\n").strip()
            if "..." in pk or "BEGIN PRIVATE KEY" not in pk:
                raise ValueError(
                    "Invalid Google Service Account private_key in Secrets. "
                    "Ensure you pasted the full private_key from your GCP JSON file, not '...'."
                )
            creds_dict["private_key"] = pk

        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://googleapis.com"],
        )
        self.service = build("drive", "v3", credentials=creds)
        self.folder_id = folder_id.strip()

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
            .create(
                body=file_metadata, media_body=media, fields="id, webViewLink"
            )
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
