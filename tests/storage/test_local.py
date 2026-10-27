"""Filesystem transport protects artifact identity across separate processes."""

from pathlib import Path

import pytest

from gstar_opd_formal.storage.controls import destination
from gstar_opd_formal.storage.files import digest_file, write_json
from gstar_opd_formal.storage.local import LocalStore


def test_upload_restore_and_control_replace(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "storage", tmp_path / "journal")
    selected = destination(store, "local-root")
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"checkpoint-v1")
    target = (selected["runs_folder_id"], "checkpoint.pt")
    record = store.upload(source, target)
    restored = tmp_path / "restored.pt"
    store.download(record, restored)
    assert digest_file(restored) == record["sha256"]

    control = tmp_path / "receipt.json"
    write_json(control, {"status": "active"})
    published = store.upload(control, (selected["studies_folder_id"], "receipt.json"))
    assert store.read_control(published["file_id"]) == {"status": "active"}
    write_json(control, {"status": "complete"})
    store.update_control(published["file_id"], control)
    assert store.read_control(published["file_id"]) == {"status": "complete"}


def test_rejects_changed_or_escaping_artifacts(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "storage", tmp_path / "journal")
    source = tmp_path / "source.json"
    source.write_bytes(b"first")
    store.upload(source, ("local-root", "source.json"))
    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="Artifact bytes"):
        store.upload(source, ("local-root", "source.json"))
    with pytest.raises(ValueError, match="escaped"):
        store.metadata("../outside")
    with pytest.raises(ValueError, match="Unsafe relative artifact path"):
        store.ensure_folder("../outside", "local-root")
