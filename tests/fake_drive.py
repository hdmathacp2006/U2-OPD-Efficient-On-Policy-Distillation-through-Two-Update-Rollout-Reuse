"""Recorded-byte Drive transport for real manifest/lease/restore integration."""

import json
import uuid
from pathlib import Path

from gstar_opd_formal.storage.files import digest_file, verify_file


class FileDrive:
    def __init__(self, root: Path):
        self.journal = root / "drive_journal"
        self.journal.mkdir(parents=True)
        self.entries = {
            "runs": {
                "id": "runs",
                "name": "runs",
                "parents": ["root"],
                "mimeType": "folder",
            }
        }
        self.data = {}

    def children(self, parent):
        return [
            dict(row) for row in self.entries.values() if row["parents"] == [parent]
        ]

    def ensure_folder(self, name, parent):
        matches = [row for row in self.children(parent) if row["name"] == name]
        if matches:
            return matches[0]["id"]
        identity = uuid.uuid4().hex
        self.entries[identity] = {
            "id": identity,
            "name": name,
            "parents": [parent],
            "mimeType": "folder",
        }
        return identity

    def upload(self, path, target):
        parent, name = target
        matches = [row for row in self.children(parent) if row["name"] == name]
        if matches:
            identity = matches[0]["id"]
            if self.data[identity] != path.read_bytes():
                raise ValueError("Immutable object changed")
        else:
            identity = uuid.uuid4().hex
        raw = path.read_bytes()
        self.data[identity] = raw
        self.entries[identity] = {
            "id": identity,
            "name": name,
            "parents": [parent],
            "mimeType": "file",
            "size": str(len(raw)),
            "sha256Checksum": digest_file(path),
        }
        return {
            "path": path.name,
            "file_id": identity,
            "parent_id": parent,
            "size": len(raw),
            "sha256": digest_file(path),
        }

    def read_control(self, identity):
        return json.loads(self.data[identity])

    def update_control(self, identity, path):
        self.data[identity] = path.read_bytes()
        self.entries[identity]["size"] = str(path.stat().st_size)
        self.entries[identity]["sha256Checksum"] = digest_file(path)

    def verify_remote(self, path, context):
        remote, parent, name = context
        if (
            remote["parents"] != [parent]
            or remote["name"] != name
            or self.data[remote["id"]] != path.read_bytes()
        ):
            raise ValueError("Remote object mismatch")

    def download(self, record, target):
        if self.entries[record["file_id"]]["parents"] != [record["parent_id"]]:
            raise ValueError("Wrong object parent")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(self.data[record["file_id"]])
        verify_file(target, record)
