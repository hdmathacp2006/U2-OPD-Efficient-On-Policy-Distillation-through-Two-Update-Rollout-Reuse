"""Hash-verified V2 manifests plus durable transfer timing sidecars."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from gstar_opd_formal.storage.base import BaseRunStore, output_files
from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    file_record,
    read_json,
    write_json,
)
from gstar_opd_formal.storage.progress import progress_for


@dataclass
class RunStore(BaseRunStore):
    source: tuple = ()
    sequence: int = 0
    remote_objects: dict = field(default_factory=dict)
    object_index_loaded: bool = False

    def latest_manifest(self) -> dict | None:
        from gstar_opd_formal.storage.files import relative_path

        markers = sorted(
            self.drive.children(self.folders["commits"]),
            key=lambda row: row["name"],
            reverse=True,
        )
        if not markers:
            return None
        record = markers[0]
        parts = record["name"].removesuffix(".json").split("_")
        if len(parts) != 2 or len(parts[0]) != 20 or not parts[0].isdigit():
            raise ValueError("Invalid commit marker filename")
        target = self.drive.journal / record["name"]
        self.drive.download(
            {
                "file_id": record["id"],
                "parent_id": self.folders["commits"],
                "size": int(record["size"]),
                "sha256": parts[1],
            },
            target,
        )
        value = read_json(target)
        if (
            value.get("schema") != "formal_boundary_v1"
            or value.get("binding_sha256") != digest_value(self.binding)
            or value.get("sequence") != int(parts[0])
        ):
            raise ValueError("Commit marker binding/sequence/schema mismatch")
        paths = [record["path"] for record in value["files"]]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate paths in committed manifest")
        for record in value["files"]:
            relative_path(record["path"])
            if record["parent_id"] != self.folders["objects"]:
                raise ValueError("Committed object escaped the run folder")
        return value

    def commit(self, output: Path, boundary: int) -> dict:
        self.require_writer()
        if read_json(output / "binding.json") != self.binding:
            raise ValueError("Execution binding changed before commit")
        progress = progress_for(output, (self.binding, self.source))
        actual = progress.get("completed_count", progress.get("completed_updates", 0))
        if boundary != actual:
            raise ValueError("Commit boundary differs from verified local progress")
        started = time.perf_counter()
        records = self.upload_objects(output)
        object_seconds = time.perf_counter() - started
        self.sequence = max(self.sequence + 1, time.time_ns())
        value = {
            "schema": "formal_boundary_v1",
            "binding_sha256": digest_value(self.binding),
            "boundary": boundary,
            "sequence": self.sequence,
            "created_ns": self.sequence,
            "progress": progress,
            "files": records,
        }
        marker = output.parent / f".{output.name}_commit_v2.json"
        write_json(marker, value)
        checksum = digest_file(marker)
        marker_started = time.perf_counter()
        uploaded = self.drive.upload(
            marker, (self.folders["commits"], f"{self.sequence:020d}_{checksum}.json")
        )
        receipt = {
            "schema": "formal_transfer_v1",
            "binding_sha256": digest_value(self.binding),
            "sequence": self.sequence,
            "boundary": boundary,
            "marker_sha256": checksum,
            "marker_file_id": uploaded["file_id"],
            "marker_value_sha256": digest_value(value),
            "object_upload_seconds": object_seconds,
            "manifest_upload_seconds": time.perf_counter() - marker_started,
            "sync_seconds": time.perf_counter() - started,
            "scope": "verified_data_commit_excludes_this_small_receipt_upload",
        }
        self.save_transfer_receipt(output, receipt)
        print(
            f"drive_v2_commit_done completed={boundary} "
            f"seconds={receipt['sync_seconds']:.3f}",
            flush=True,
        )
        return receipt

    def upload_objects(self, output: Path) -> list[dict]:
        if not self.object_index_loaded:
            for row in self.drive.children(self.folders["objects"]):
                if row["name"] in self.remote_objects:
                    raise ValueError("Duplicate content-addressed Drive object")
                self.remote_objects[row["name"]] = row
            self.object_index_loaded = True
        records = []
        for path in output_files(output):
            source = file_record(path, output)
            name = f"{source['sha256']}_{path.name}"
            uploaded = self.known_objects.get(name)
            if uploaded is None:
                remote = self.remote_objects.get(name)
                if remote is not None:
                    self.drive.verify_remote(
                        path, (remote, self.folders["objects"], name)
                    )
                    uploaded = {
                        **source,
                        "file_id": remote["id"],
                        "parent_id": self.folders["objects"],
                    }
                else:
                    uploaded = self.drive.upload(path, (self.folders["objects"], name))
                self.known_objects[name] = uploaded
            records.append({**uploaded, "path": source["path"]})
        return records

    def save_transfer_receipt(self, output: Path, receipt: dict) -> None:
        # A sidecar closes the timing of the manifest that precedes it, avoiding
        # a self-referential manifest hash and an infinite final-commit chain.
        path = output / "transfers" / f"{receipt['sequence']}.json"
        write_json(path, receipt)
        self.drive.upload(
            path, (self.folders["receipts"], f"{digest_file(path)}_{path.name}")
        )

    def restore(self, output: Path) -> dict:
        markers = sorted(
            self.drive.children(self.folders["commits"]),
            key=lambda item: item["name"],
            reverse=True,
        )
        failures = []
        for record in markers:
            parts = record["name"].removesuffix(".json").split("_")
            if (
                len(parts) != 2
                or len(parts[0]) != 20
                or not parts[0].isdigit()
                or len(parts[1]) != 64
            ):
                raise ValueError("Unexpected V2 commit marker filename")
            target = output.parent / f".{output.name}_{record['name']}"
            try:
                self.drive.download(
                    {
                        "file_id": record["id"],
                        "parent_id": self.folders["commits"],
                        "size": int(record["size"]),
                        "sha256": parts[1],
                    },
                    target,
                )
                value = read_json(target)
                if value.get("binding_sha256") != digest_value(
                    self.binding
                ) or value.get("sequence") != int(parts[0]):
                    raise ValueError(
                        "Commit marker has a different binding or sequence"
                    )
                return self.restore_files(output, value)
            except (ValueError, FileNotFoundError) as error:
                failures.append(str(error))
        raise ValueError(f"No verified V2 boundary available: {failures[:3]}")

    def restore_files(self, output: Path, value: dict) -> dict:
        if value.get("schema") != "formal_boundary_v1":
            raise ValueError("Do not mix V1 and V2 run manifests")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix="colab_v2_verify_", dir=output.parent
        ) as temporary:
            stage = Path(temporary) / "verified"
            super().restore_files(stage, value)
            expected = progress_for(stage, (self.binding, self.source))
            if expected != value.get("progress"):
                raise ValueError("Manifest completed IDs differ from raw artifacts")
            self.restore_receipt(stage, value)
            self.restore_failure_receipts(stage)
            if output.exists():
                if any(output.iterdir()):
                    raise FileExistsError("Restore V2 into a fresh directory")
                output.rmdir()
            stage.replace(output)
        self.sequence = value["sequence"]
        self.known_objects = {
            f"{row['sha256']}_{Path(row['path']).name}": row for row in value["files"]
        }
        return value

    def restore_receipt(self, stage: Path, value: dict) -> None:
        suffix = f"_{value['sequence']}.json"
        records = [
            item
            for item in self.drive.children(self.folders["receipts"])
            if item["name"].endswith(suffix)
        ]
        if len(records) > 1:
            raise ValueError("Ambiguous transfer timing receipt")
        if not records:
            write_json(
                stage / "transfer_timing_warning.json",
                {
                    "sequence": value["sequence"],
                    "status": "data_committed_receipt_not_uploaded",
                },
            )
            return
        record = records[0]
        target = stage / "transfers" / f"{value['sequence']}.json"
        self.drive.download(
            {
                "file_id": record["id"],
                "parent_id": self.folders["receipts"],
                "size": int(record["size"]),
                "sha256": record["name"].split("_")[0],
            },
            target,
        )
        receipt = read_json(target)
        if receipt.get("binding_sha256") != digest_value(self.binding) or receipt.get(
            "marker_value_sha256"
        ) != digest_value(value):
            raise ValueError("Transfer timing receipt belongs to another manifest")

    def restore_failure_receipts(self, stage: Path) -> None:
        for record in self.drive.children(self.folders["receipts"]):
            if "_attempt_" not in record["name"]:
                continue
            checksum, suffix = record["name"].split("_attempt_", 1)
            key = suffix.removesuffix(".json")
            if len(key) != 32 or any(
                character not in "0123456789abcdef" for character in key
            ):
                raise ValueError("Invalid external attempt receipt name")
            target = stage / "attempts_external" / f"{key}.json"
            self.drive.download(
                {
                    "file_id": record["id"],
                    "parent_id": self.folders["receipts"],
                    "size": int(record["size"]),
                    "sha256": checksum,
                },
                target,
            )
            if read_json(target).get("binding_sha256") != digest_value(self.binding):
                raise ValueError("External attempt receipt has a different binding")


def open_store(drive: object, context: tuple) -> RunStore:
    parent, run_id, binding, source = context
    run = drive.ensure_folder(run_id, parent)
    folders = {
        name: drive.ensure_folder(name, run)
        for name in ("objects", "commits", "receipts")
    }
    folders["run"] = run
    markers = drive.children(folders["commits"])
    previous = max((int(item["name"].split("_")[0]) for item in markers), default=0)
    store = RunStore(drive, folders, binding, source=source, sequence=previous)
    return store
