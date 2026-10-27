"""Restore exact committed runs in worker Python; never start training."""

import uuid
from pathlib import Path

from gstar_opd_formal.storage.files import digest_file, digest_value, read_json
from gstar_opd_formal.storage.runs import open_store


def backup_local(output: Path, artifact_root: Path) -> None:
    output = output.resolve()
    artifact_root = artifact_root.resolve()
    target = output.with_name(f".{output.name}.backup_{uuid.uuid4().hex}")
    if (
        not output.is_relative_to(artifact_root)
        or not target.is_relative_to(artifact_root)
        or output == artifact_root
    ):
        raise ValueError("Backup directory escapes the artifact workspace")
    if output.is_symlink():
        raise ValueError("Do not replace a symlinked artifact directory")
    output.rename(target)


def restore_training(context: tuple, selection: dict) -> dict:
    drive, layout, destination = context
    binding = selection["binding"]
    run_id = digest_value(binding)[:32]
    if selection.get("run_id") != run_id or binding.get("action") != "train":
        raise ValueError("Training selection has the wrong binding/run ID")
    output = layout.output("training", run_id)
    source = ()
    store = open_store(drive, (destination["runs_folder_id"], run_id, binding, source))
    if output.exists():
        local = output / "binding.json"
        summary = output / "summary.json"
        if local.exists() and summary.exists() and read_json(local) == binding:
            metadata = read_json(summary)
            checkpoint = output / metadata["accepted_checkpoint"]
            if (
                metadata.get("checkpoint_sha256") == selection["checkpoint_sha256"]
                and checkpoint.exists()
                and digest_file(checkpoint) == selection["checkpoint_sha256"]
            ):
                return {"directory": output, "binding": binding}
        backup_local(output, layout.artifacts)
    store.restore(output)
    from gstar_opd_formal.training.history import completed_training

    checked = completed_training(output, binding)
    if checked["summary"]["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise ValueError("Restored final checkpoint differs from the paired receipt")
    return {"directory": output, "binding": binding}
