"""Commit closed output files by content hash and recover verified boundaries."""

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from gstar_opd_formal.storage.drive import DriveStore
from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    file_record,
    read_json,
    relative_path,
    verify_file,
    write_json,
)


def output_files(output: Path) -> list[Path]:
    accepted = None
    if (output / "summary.json").exists():
        accepted = read_json(output / "summary.json").get("accepted_checkpoint")
    paths = []
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("Run outputs cannot contain symlinks")
        if not path.is_file() or any(
            p.startswith(".") for p in path.relative_to(output).parts
        ):
            continue
        if path.suffix in {".json", ".jsonl", ".csv"} or (
            (accepted and path.relative_to(output) == relative_path(accepted))
            or (path.parent.name == "checkpoints" and path.name == "update_0128.pt")
        ):
            paths.append(path)
    return sorted(paths)


@dataclass
class BaseRunStore:
    drive: DriveStore
    folders: dict
    binding: dict
    writer: dict | None = None
    known_objects: dict = field(default_factory=dict)

    def claim(self, operation: str) -> None:
        matches = [
            row
            for row in self.drive.children(self.folders["run"])
            if row["name"] == "writer.json"
        ]
        if len(matches) > 1:
            raise ValueError("Ambiguous writer lease; do not start another job")
        old = self.drive.read_control(matches[0]["id"]) if matches else None
        if old and old.get("status") == "active" and operation != "resume":
            raise ValueError("Remote writer may be active; confirm stopped then resume")
        path = self.drive.journal / f"writer_{uuid.uuid4().hex}.json"
        token = uuid.uuid4().hex
        value = {
            "writer_token": token,
            "status": "active",
            "claimed_at": time.time(),
            "binding_sha256": digest_value(self.binding),
        }
        write_json(path, value)
        if matches:
            file_id = matches[0]["id"]
            self.drive.update_control(file_id, path)
        else:
            uploaded = self.drive.upload(path, (self.folders["run"], "writer.json"))
            file_id = uploaded["file_id"]
        self.writer = {"file_id": file_id, "token": token, "path": path}
        self.require_writer()

    def require_writer(self) -> None:
        if self.writer is None:
            raise ValueError("Claim the writer before committing results")
        value = self.drive.read_control(self.writer["file_id"])
        if (
            value.get("writer_token") != self.writer["token"]
            or value.get("status") != "active"
        ):
            raise ValueError("Writer ownership changed; stop before further work")

    def release(self) -> None:
        self.require_writer()
        value = self.drive.read_control(self.writer["file_id"])
        write_json(
            self.writer["path"],
            {**value, "status": "released", "finished_at": time.time()},
        )
        self.drive.update_control(self.writer["file_id"], self.writer["path"])

    def commit(self, output: Path, boundary: int) -> dict:
        self.require_writer()
        if read_json(output / "binding.json") != self.binding:
            raise ValueError("Run binding changed before remote commit")
        started = time.perf_counter()
        records = []
        for path in output_files(output):
            source = file_record(path, output)
            name = f"{source['sha256']}_{path.name}"
            uploaded = self.known_objects.get(name)
            if uploaded is None:
                uploaded = self.drive.upload(path, (self.folders["objects"], name))
                self.known_objects[name] = uploaded
            records.append({**uploaded, "path": source["path"]})
        value = {
            "schema": "formal_boundary_base_v1",
            "binding_sha256": digest_value(self.binding),
            "boundary": boundary,
            "created_ns": time.time_ns(),
            "files": records,
        }
        marker = output.parent / f".{output.name}_commit.json"
        write_json(marker, value)
        checksum = digest_file(marker)
        name = f"{boundary:06d}_{checksum}.json"
        uploaded = self.drive.upload(marker, (self.folders["commits"], name))
        receipt = {
            "boundary": boundary,
            "marker": uploaded,
            "sync_seconds": time.perf_counter() - started,
            "last_remote_committed_boundary": boundary,
        }
        write_json(output / ".last_remote_commit.json", receipt)
        print(
            f"drive_commit_done boundary={boundary} files={len(records)} "
            f"seconds={receipt['sync_seconds']:.3f}",
            flush=True,
        )
        return receipt

    def restore(self, output: Path) -> dict:
        markers = self.drive.children(self.folders["commits"])
        candidates = sorted(
            markers,
            key=lambda row: (row["name"].split("_")[0], row.get("createdTime", "")),
            reverse=True,
        )
        failures = []
        for remote in candidates:
            name = remote["name"]
            parts = name.removesuffix(".json").split("_")
            if len(parts) != 2 or not parts[0].isdigit() or len(parts[1]) != 64:
                raise ValueError("Unexpected object in the run commit folder")
            target = output.parent / f".{output.name}_{name}"
            try:
                self.drive.download(
                    {
                        "file_id": remote["id"],
                        "parent_id": self.folders["commits"],
                        "size": int(remote["size"]),
                        "sha256": parts[1],
                    },
                    target,
                )
                value = read_json(target)
                if value.get("binding_sha256") != digest_value(self.binding):
                    raise ValueError(
                        "Remote run belongs to a different immutable binding"
                    )
                if value.get("boundary") != int(parts[0]):
                    raise ValueError("Boundary marker differs from its filename")
                return self.restore_files(output, value)
            except (ValueError, FileNotFoundError) as error:
                failures.append(str(error))
        raise ValueError(f"No verified remote boundary available: {failures[:3]}")

    def restore_files(self, output: Path, value: dict) -> dict:
        from tempfile import TemporaryDirectory

        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "Restore into a fresh directory, never merge stale files"
            )
        with TemporaryDirectory(
            prefix="gstar_restore_", dir=output.parent
        ) as temporary:
            stage = Path(temporary) / "output"
            stage.mkdir()
            paths = [row["path"] for row in value["files"]]
            if len(paths) != len(set(paths)):
                raise ValueError("Remote boundary contains duplicate paths")
            for record in value["files"]:
                if record["parent_id"] != self.folders["objects"]:
                    raise ValueError("Remote artifact escaped this run's object folder")
                path = stage / relative_path(record["path"])
                self.drive.download(record, path)
                verify_file(path, record)
            if read_json(stage / "binding.json") != self.binding:
                raise ValueError("Restored binding differs from its boundary marker")
            if output.exists():
                output.rmdir()
            os.replace(stage, output)
        return value


def open_run_store(drive: DriveStore, context: tuple) -> BaseRunStore:
    parent_id, run_id, binding = context
    run = drive.ensure_folder(run_id, parent_id)
    folders = {name: drive.ensure_folder(name, run) for name in ("objects", "commits")}
    folders["run"] = run
    return BaseRunStore(drive, folders, binding)
