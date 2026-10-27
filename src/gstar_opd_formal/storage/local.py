"""Filesystem transport for independent Linux and Modal runs."""

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from gstar_opd_formal.storage.files import (
    ROOT_FOLDER_ID,
    file_record,
    relative_path,
    verify_file,
)

FOLDER_MIME = "application/vnd.google-apps.folder"
CONTROL_LIMIT_BYTES = 2**20


@dataclass
class LocalStore:
    root: Path
    journal: Path

    def __post_init__(self) -> None:
        if self.root.is_symlink():
            raise ValueError("Storage root cannot be a symlink")
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal.mkdir(parents=True, exist_ok=True)

    def path_for(self, file_id: str) -> Path:
        path = PurePosixPath(file_id)
        if not file_id or path.parts[0] != ROOT_FOLDER_ID:
            raise ValueError("Artifact escaped the configured storage root")
        relative = relative_path(file_id)
        target = self.root.joinpath(*relative.parts[1:])
        if any(candidate.is_symlink() for candidate in (target, *target.parents)):
            raise ValueError("Storage artifacts cannot use symlinks")
        return target

    def metadata(self, file_id: str) -> dict:
        path = self.path_for(file_id)
        if not path.exists():
            raise FileNotFoundError(file_id)
        parent = str(PurePosixPath(file_id).parent)
        return {
            "id": file_id,
            "name": path.name,
            "mimeType": FOLDER_MIME if path.is_dir() else "application/octet-stream",
            "size": path.stat().st_size if path.is_file() else 0,
            "parents": [] if file_id == ROOT_FOLDER_ID else [parent],
            "trashed": False,
        }

    def require_destination(self, folder_id: str) -> dict:
        path = self.path_for(folder_id)
        if not path.is_dir():
            raise ValueError("Storage destination is not a directory")
        return self.metadata(folder_id)

    def children(self, folder_id: str) -> list[dict]:
        folder = self.path_for(folder_id)
        self.require_destination(folder_id)
        return [
            self.metadata(f"{folder_id}/{child.name}")
            for child in sorted(folder.iterdir())
        ]

    def ensure_folder(self, name: str, parent_id: str) -> str:
        if len(relative_path(name).parts) != 1:
            raise ValueError("Folder name must have one path component")
        parent = self.path_for(parent_id)
        self.require_destination(parent_id)
        target = parent / name
        if target.is_symlink():
            raise ValueError("Storage folders cannot be symlinks")
        target.mkdir(exist_ok=True)
        file_id = f"{parent_id}/{name}"
        self.require_destination(file_id)
        return file_id

    def upload(self, path: Path, target: tuple[str, str]) -> dict:
        parent_id, name = target
        if len(relative_path(name).parts) != 1:
            raise ValueError("Artifact name must have one path component")
        self.require_destination(parent_id)
        record = file_record(path, path.parent)
        file_id = f"{parent_id}/{name}"
        destination = self.path_for(file_id)
        if destination.exists():
            verify_file(destination, record)
        else:
            temporary = destination.with_name(f".{name}.{uuid.uuid4().hex}.partial")
            try:
                shutil.copyfile(path, temporary)
                verify_file(temporary, record)
                if destination.exists():
                    verify_file(destination, record)
                else:
                    os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return {**record, "file_id": file_id, "parent_id": parent_id}

    def verify_remote(self, path: Path, context: tuple) -> None:
        remote, parent_id, name = context
        if remote["id"] != f"{parent_id}/{name}":
            raise ValueError("Artifact location differs from its manifest")
        stored = self.path_for(remote["id"])
        verify_file(stored, file_record(path, path.parent))

    def download(self, record: dict, destination: Path) -> None:
        source = self.path_for(record["file_id"])
        if self.metadata(record["file_id"])["parents"] != [record["parent_id"]]:
            raise ValueError("Artifact parent differs from its manifest")
        verify_file(source, record)
        if destination.exists():
            verify_file(destination, record)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            shutil.copyfile(source, temporary)
            verify_file(temporary, record)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def read_control(self, file_id: str) -> dict:
        path = self.path_for(file_id)
        if not path.is_file() or path.stat().st_size > CONTROL_LIMIT_BYTES:
            raise ValueError("Control document is missing or too large")
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("Control document must be a JSON object")
        return value

    def update_control(self, file_id: str, path: Path) -> None:
        destination = self.path_for(file_id)
        if not destination.is_file() or path.stat().st_size > CONTROL_LIMIT_BYTES:
            raise ValueError("Cannot replace this control document")
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            shutil.copyfile(path, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
