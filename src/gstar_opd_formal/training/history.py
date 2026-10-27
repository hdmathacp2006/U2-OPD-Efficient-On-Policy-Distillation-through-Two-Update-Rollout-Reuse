"""Validate full-update endpoints, fresh coverage and exact pending reuse state."""

import math
from pathlib import Path

from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    read_json,
    relative_path,
)


def validate_history(state: dict, binding: dict, completed: int) -> None:
    width = binding["updates_per_fresh_batch"]
    updates, fresh = state["learner_updates"], state["fresh_batches"]
    if len(updates) != completed or len(fresh) != (completed + width - 1) // width:
        raise ValueError("Accepted update/fresh batch history counts disagree")
    if (
        len(state["decisions"]) != completed
        or len(state["trajectories"]) != len(fresh) * binding["batch_size"]
    ):
        raise ValueError("Training history is incomplete")
    order = binding["definition"]["manifests"]["training"]["source_indices"]
    for index, record in enumerate(fresh):
        expected = [
            str(value)
            for value in order[
                index * binding["batch_size"] : (index + 1) * binding["batch_size"]
            ]
        ]
        if (
            record["question_ids"] != expected
            or record["behavior_version"] != index * width
        ):
            raise ValueError("Fresh questions/order or behavior version changed")
    for index, record in enumerate(updates):
        if (
            record["update"] != index + 1
            or record["reuse_index"] != index % width
            or record["fresh_batch"] != index // width
        ):
            raise ValueError("Learner/reuse update sequence changed")
        if record["question_ids"] != fresh[index // width]["question_ids"]:
            raise ValueError("Reuse lost its original questions")
        if any(
            not math.isfinite(record[key])
            for key in ("loss", "gradient_norm", "seconds")
        ):
            raise ValueError("Training metrics are not finite")
    pending = state["pending"]
    expected_pending = completed % width != 0
    if (pending is not None) != expected_pending:
        raise ValueError("Odd/even pending reuse state is inconsistent")
    if pending is not None:
        pending.validate(completed)


def accepted_checkpoint(directory: Path, binding: dict) -> dict:
    summary = read_json(directory / "summary.json")
    if summary.get("binding_sha256") != digest_value(binding):
        raise ValueError("Checkpoint summary is bound to another run")
    completed = summary.get("completed_updates")
    if type(completed) is not int or not 0 < completed <= binding["endpoint"]:
        raise ValueError("Checkpoint update count is invalid")
    path = directory / relative_path(summary["accepted_checkpoint"])
    if (
        path.parent != directory / "checkpoints"
        or path.name != f"update_{completed:04d}.pt"
        or digest_file(path) != summary["checkpoint_sha256"]
    ):
        raise ValueError("Accepted checkpoint path or bytes changed")
    if (summary.get("status") == "workload_complete") != (
        completed == binding["endpoint"]
    ):
        raise ValueError("Partial checkpoint cannot be marked complete")
    return {"path": path, "summary": summary}


def completed_training(directory: Path, binding: dict) -> dict:
    checked = accepted_checkpoint(directory, binding)
    if checked["summary"]["completed_updates"] != binding["endpoint"]:
        raise ValueError(
            "Final evaluation requires the fixed complete training endpoint"
        )
    return checked
