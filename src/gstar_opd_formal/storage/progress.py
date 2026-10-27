"""Validate persisted progress without importing models or any benchmark grader."""

from pathlib import Path

from gstar_opd_formal.storage.files import digest_value, read_json


def progress_for(output: Path, context: tuple) -> dict:
    binding, source = context
    summary = (
        read_json(output / "summary.json") if (output / "summary.json").exists() else {}
    )
    if binding["action"] == "train":
        count = summary.get("completed_updates", 0)
        if type(count) is not int or not 0 <= count <= binding["endpoint"]:
            raise ValueError("Invalid saved training progress")
        return {
            "kind": "train",
            "completed_updates": count,
            "status": summary.get("status", "pending"),
        }
    ids = {item["id"] for item in source}
    completed = []
    for path in sorted((output / "raw").glob("*.json")):
        record = read_json(path)
        row = record.get("row", {})
        key = row.get("example_id")
        if record.get("binding_sha256") != digest_value(binding) or key not in ids:
            raise ValueError("Unknown or foreign saved evaluation question")
        if path.name != digest_value(key) + ".json" or key in completed:
            raise ValueError("Duplicate or renamed evaluation question")
        completed.append(key)
    return {
        "kind": "evaluation",
        "completed_ids": sorted(completed),
        "completed_count": len(completed),
        "panel_ids_sha256": digest_value([item["id"] for item in source]),
    }
