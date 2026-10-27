"""Bounded official Drive SDK transfers anchored to the user's exact folder."""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from gstar_opd_formal.storage.files import (
    BUFFER_BYTES,
    ROOT_FOLDER_ID,
    digest_file,
    digest_value,
    file_record,
    read_json,
    verify_file,
    write_json,
)

FILE_FIELDS = (
    "id,name,mimeType,size,parents,md5Checksum,sha256Checksum,trashed,createdTime"
)
FOLDER_MIME = "application/vnd.google-apps.folder"
MAX_ATTEMPTS = 5
TRANSFER_SECONDS = 300


def http_status(error: Exception) -> int | None:
    return getattr(getattr(error, "resp", None), "status", None)


def transient_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)) or (
        isinstance(error, OSError)
        and error.errno in {54, 60, 100, 101, 104, 110, 111, 10054, 10060}
    ):
        return True
    if http_status(error) in {408, 429, 500, 502, 503, 504}:
        return True
    if http_status(error) == 403:
        try:
            reasons = json.loads(error.content).get("error", {}).get("errors", [])
        except (AttributeError, ValueError, TypeError):
            return False
        return any(
            row.get("reason") in {"rateLimitExceeded", "userRateLimitExceeded"}
            for row in reasons
        )
    return False


def retry_read(operation: object) -> object:
    deadline = time.monotonic() + TRANSFER_SECONDS
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except Exception as error:
            if (
                not transient_error(error)
                or attempt == MAX_ATTEMPTS - 1
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError("Transfer retry loop exhausted")


def authenticated_service() -> object:
    import google.auth
    import google_auth_httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import build_http

    credentials, _ = google.auth.default()
    raw = build_http()
    raw.timeout = 30
    if 308 in raw.redirect_codes:
        raise RuntimeError("Google Drive resumable-upload HTTP policy is missing")
    transport = google_auth_httplib2.AuthorizedHttp(credentials, http=raw)
    return build("drive", "v3", http=transport, cache_discovery=False)


@dataclass
class DriveStore:
    service: object
    journal: Path

    def metadata(self, file_id: str) -> dict:
        return retry_read(
            lambda: (
                self.service.files()
                .get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True)
                .execute()
            )
        )

    def require_destination(self, folder_id: str) -> dict:
        current, visited = folder_id, set()
        for _ in range(32):
            if current in visited:
                break
            visited.add(current)
            value = self.metadata(current)
            if value.get("trashed") or value.get("mimeType") != FOLDER_MIME:
                raise ValueError("Drive destination is not an accessible live folder")
            if value.get("id") == ROOT_FOLDER_ID:
                return value
            parents = value.get("parents", [])
            if len(parents) != 1:
                break
            current = parents[0]
        raise ValueError("Drive destination does not descend from the approved test ID")

    def children(self, folder_id: str) -> list[dict]:
        self.require_destination(folder_id)
        rows, token = [], None
        while True:
            page = retry_read(
                lambda token=token: (
                    self.service.files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields=f"nextPageToken,files({FILE_FIELDS})",
                        pageSize=100,
                        pageToken=token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
            )
            rows.extend(page.get("files", []))
            following = page.get("nextPageToken")
            if not following:
                return rows
            if following == token or len(rows) > 10000:
                raise ValueError("Drive listing did not advance within its bound")
            token = following

    def ensure_folder(self, name: str, parent_id: str) -> str:
        matches = [row for row in self.children(parent_id) if row["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous Drive child folder: {name}")
        if matches:
            result = matches[0]
            if result["mimeType"] != FOLDER_MIME:
                raise ValueError("A folder name collides with a non-folder file")
            return result["id"]
        record = self.reserve_id({"parent": parent_id, "folder": name})
        try:
            result = (
                self.service.files()
                .create(
                    body={
                        "id": record["id"],
                        "name": name,
                        "mimeType": FOLDER_MIME,
                        "parents": [parent_id],
                    },
                    fields=FILE_FIELDS,
                    supportsAllDrives=True,
                )
                .execute()
            )
        except Exception as error:
            if not transient_error(error) and http_status(error) != 409:
                raise
            # An ambiguous acknowledgement must be reconciled by exact ID.
            result = self.metadata(record["id"])
        if result.get("parents") != [parent_id] or result.get("name") != name:
            raise ValueError("Created folder differs from its intended parent/name")
        self.require_destination(result["id"])
        return result["id"]

    def reserve_id(self, binding: dict) -> dict:
        path = self.journal / f"{digest_value(binding)}.json"
        if path.exists():
            saved = read_json(path)
            if saved["binding"] != binding:
                raise ValueError("Drive transfer journal binding changed")
            return saved
        value = retry_read(
            lambda: self.service.files().generateIds(count=1, space="drive").execute()
        )
        result = {"binding": binding, "id": value["ids"][0]}
        write_json(path, result)
        return result

    def upload(self, path: Path, target: tuple) -> dict:
        from googleapiclient.http import MediaFileUpload

        parent, name = target
        self.require_destination(parent)
        source = file_record(path, path.parent)
        existing = [row for row in self.children(parent) if row["name"] == name]
        if len(existing) > 1:
            raise ValueError("Duplicate artifact name requires explicit reconciliation")
        if existing:
            self.verify_remote(path, (existing[0], parent, name))
            return {**source, "file_id": existing[0]["id"], "parent_id": parent}
        saved = self.reserve_id({"parent": parent, "name": name, **source})
        try:
            remote = self.metadata(saved["id"])
        except Exception as error:
            if http_status(error) != 404:
                raise
            remote = None
        if remote is None:
            media = MediaFileUpload(str(path), chunksize=BUFFER_BYTES, resumable=True)
            request = self.service.files().create(
                body={"id": saved["id"], "name": name, "parents": [parent]},
                media_body=media,
                fields=FILE_FIELDS,
                supportsAllDrives=True,
            )
            try:
                remote = finish_upload(request, source["size"])
            except Exception as error:
                if not transient_error(error) and http_status(error) != 409:
                    raise
                remote = self.metadata(saved["id"])
        verify_file(path, source)
        self.verify_remote(path, (remote, parent, name))
        return {**source, "file_id": remote["id"], "parent_id": parent}

    def read_control(self, file_id: str) -> dict:
        remote = self.metadata(file_id)
        if len(remote.get("parents", [])) != 1 or int(remote.get("size", -1)) > 2**20:
            raise ValueError("Control record lacks its parent or exceeds1MiB")
        self.require_destination(remote["parents"][0])
        raw = retry_read(
            lambda: (
                self.service.files()
                .get_media(fileId=file_id, supportsAllDrives=True)
                .execute()
            )
        )
        if len(raw) != int(remote["size"]):
            raise ValueError("Incomplete Drive control record")
        return json.loads(raw)

    def update_control(self, file_id: str, path: Path) -> None:
        from googleapiclient.http import MediaFileUpload

        remote = self.metadata(file_id)
        if len(remote.get("parents", [])) != 1:
            raise ValueError("Control file has an ambiguous destination")
        self.require_destination(remote["parents"][0])
        self.service.files().update(
            fileId=file_id,
            media_body=MediaFileUpload(str(path)),
            fields=FILE_FIELDS,
            supportsAllDrives=True,
        ).execute()

    def verify_remote(self, path: Path, context: tuple) -> None:
        remote, parent, name = context
        if (
            remote.get("parents") != [parent]
            or remote.get("name") != name
            or remote.get("trashed")
            or int(remote.get("size", -1)) != path.stat().st_size
        ):
            raise ValueError("Remote artifact metadata differs from its source")
        expected = digest_file(path)
        if remote.get("sha256Checksum"):
            valid = remote["sha256Checksum"] == expected
        elif remote.get("md5Checksum"):
            with path.open("rb") as stream:
                valid = (
                    hashlib.file_digest(stream, "md5").hexdigest()
                    == remote["md5Checksum"]
                )
        else:
            check = path.with_name(f".{path.name}.remote_check")
            try:
                self.download(
                    {
                        "file_id": remote["id"],
                        "parent_id": parent,
                        "size": path.stat().st_size,
                        "sha256": expected,
                    },
                    check,
                )
                valid = True
            finally:
                check.unlink(missing_ok=True)
        if not valid:
            raise ValueError("Server checksum differs from the local artifact")

    def download(self, record: dict, destination: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        self.require_destination(record["parent_id"])
        remote = self.metadata(record["file_id"])
        if (
            remote.get("parents") != [record["parent_id"]]
            or int(remote.get("size", -1)) != record["size"]
        ):
            raise ValueError("Download metadata differs from the published artifact")
        if destination.exists():
            verify_file(destination, record)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            with temporary.open("wb") as stream:
                request = self.service.files().get_media(
                    fileId=record["file_id"], supportsAllDrives=True
                )
                download = MediaIoBaseDownload(stream, request, chunksize=BUFFER_BYTES)
                complete = False
                for _ in range(record["size"] // BUFFER_BYTES + 3):
                    _, complete = retry_read(download.next_chunk)
                    if complete:
                        break
                if not complete:
                    raise TimeoutError(
                        "Download did not complete within its chunk bound"
                    )
            verify_file(temporary, record)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def finish_upload(request: object, size: int) -> dict:
    deadline = time.monotonic() + TRANSFER_SECONDS
    for _ in range(size // BUFFER_BYTES + 3):
        _, response = retry_read(request.next_chunk)
        if response is not None:
            return response
        if time.monotonic() >= deadline:
            break
    raise TimeoutError("Upload did not complete within its finite window")
