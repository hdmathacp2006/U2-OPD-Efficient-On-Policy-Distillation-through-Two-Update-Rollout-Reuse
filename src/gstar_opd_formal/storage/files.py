"""Byte-preserving manifests and safe local archive transactions."""

import hashlib
import json
import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path, PurePosixPath

ROOT_FOLDER_ID = os.environ.get("GSTAR_DRIVE_ROOT_ID", "local-root")
BUFFER_BYTES = 8 * 2**20
MAX_ARCHIVE_BYTES = 20 * 2**30


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_value(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
    ):
        raise ValueError(f"Unsafe relative artifact path: {value!r}")
    return Path(*path.parts)


def require_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("A full lowercase SHA256 is required")


def file_record(path: Path, base: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Artifact must be a regular file: {path}")
    return {
        "path": path.relative_to(base).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest_file(path),
    }


def verify_file(path: Path, record: dict) -> None:
    require_digest(record["sha256"])
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["size"]
        or digest_file(path) != record["sha256"]
    ):
        raise ValueError(f"Artifact bytes differ from their manifest: {path}")


def verify_tree(base: Path, records: list[dict]) -> None:
    names = [row["path"] for row in records]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate paths")
    for record in records:
        path = base / relative_path(record["path"])
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("Artifact escapes its declared root")
        verify_file(path, record)


def create_archive(destination: Path, context: tuple) -> dict:
    base, paths = context
    records = [file_record(path, base) for path in sorted(paths)]
    verify_tree(base, records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for record in records:
                archive.add(
                    base / record["path"], arcname=record["path"], recursive=False
                )
        # A source changing while being archived must not create a trusted release.
        verify_tree(base, records)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"files": records, "archive": file_record(destination, destination.parent)}


def extract_archive(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Extract only into a fresh directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError("Archive contains duplicate paths")
            if sum(member.size for member in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds the extraction allowance")
            for member in members:
                relative_path(member.name)
                if not member.isfile():
                    raise ValueError(
                        "Only regular files are allowed in release archives"
                    )
            archive.extractall(stage, members=members, filter="data")
        os.replace(stage, destination)
    finally:
        if stage.exists():
            resolved = stage.resolve()
            if stage.is_symlink() or resolved.parent != destination.parent.resolve():
                raise ValueError("Refuse cleanup outside the archive staging parent")
            shutil.rmtree(resolved)
