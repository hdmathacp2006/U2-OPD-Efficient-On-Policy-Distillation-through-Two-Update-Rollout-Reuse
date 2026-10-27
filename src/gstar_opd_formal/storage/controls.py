"""Resolve exact study receipts within the selected storage destination."""

from pathlib import Path

from gstar_opd_formal.storage.files import digest_value, read_json, write_json


def read_named(drive: object, context: tuple) -> dict | None:
    parent, name = context
    matches = [row for row in drive.children(parent) if row["name"] == name]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous Drive control document: {name}")
    return drive.read_control(matches[0]["id"]) if matches else None


def publish_named(drive: object, path: Path, context: tuple) -> dict:
    parent, name = context
    matches = [row for row in drive.children(parent) if row["name"] == name]
    if len(matches) > 1:
        raise ValueError("Ambiguous study receipt")
    if matches:
        drive.update_control(matches[0]["id"], path)
        file_id = matches[0]["id"]
    else:
        file_id = drive.upload(path, (parent, name))["file_id"]
    if drive.read_control(file_id) != read_json(path):
        raise ValueError("Study receipt readback differs from published bytes")
    return {"file_id": file_id, "value_sha256": digest_value(read_json(path))}


def destination(drive: object, root_id: str) -> dict:
    from gstar_opd_formal.storage.files import ROOT_FOLDER_ID

    if root_id != ROOT_FOLDER_ID:
        raise ValueError("Storage root differs from the selected destination")
    formal = drive.ensure_folder("GStar_OPD_Formal", root_id)
    return {
        "root_folder_id": root_id,
        "formal_folder_id": formal,
        "runs_folder_id": drive.ensure_folder("runs", formal),
        "studies_folder_id": drive.ensure_folder("studies", formal),
    }


def write_control(path: Path, value: dict) -> None:
    write_json(path, value)
