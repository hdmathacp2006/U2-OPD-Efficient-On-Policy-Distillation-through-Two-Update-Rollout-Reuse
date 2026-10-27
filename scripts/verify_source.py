"""Verify historical preservation, actual relocations and mathematical definitions."""

import ast
import csv
import json
import re
from pathlib import Path

from gstar_opd_formal.storage.files import digest_file, write_json


def normalized_definition(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    return ast.dump(node, include_attributes=False)


def verify_sources(root: Path) -> dict:
    manifest = json.loads(
        (root / "archives/legacy_20261025/manifest.json").read_bytes()
    )
    original = Path(manifest["source_root"])
    failures = [
        row["path"]
        for row in manifest["files"]
        if digest_file(original / row["path"]) != row["sha256"]
    ]
    if failures:
        raise ValueError(f"Historical source changed: {failures}")
    if (
        digest_file(root / "archives/legacy_20261025/project.tar.gz")
        != manifest["archive_sha256"]
    ):
        raise ValueError("Historical archive bytes changed")
    rows = list(
        csv.DictReader((root / "provenance/FILE_MAP.csv").open(encoding="utf-8-sig"))
    )
    names = {
        row["source"].removeprefix("src/").removesuffix(".py").replace("/", "."): row[
            "active_destinations"
        ]
        .removeprefix("src/")
        .removesuffix(".py")
        .replace("/", ".")
        for row in rows
        if row["action"] == "PORT_MODULE"
    }
    names.update(
        {
            "gstar_opd.protocol": "gstar_opd_formal.contract_types",
            "gstar_opd.runtime.feedback": "gstar_opd_formal.models.adapter_snapshot",
            "gstar_opd.runtime.checkpoint": "gstar_opd_formal.training.checkpoint",
            "gstar_opd.evaluation": "gstar_opd_formal.evaluation.graders.math",
        }
    )
    parity_path = root / "provenance/method_parity.json"
    parity = json.loads(parity_path.read_bytes())
    for row in parity["ast_checks"]:
        source = (original / row["source"]).read_text(encoding="utf-8-sig")
        for old in sorted(names, key=len, reverse=True):
            source = re.sub(r"\b" + re.escape(old) + r"\b", names[old], source)
        target = (root / row["target"]).read_text(encoding="utf-8-sig")
        changed = [
            name
            for name in row["definitions"]
            if normalized_definition(source, name)
            != normalized_definition(target, name)
        ]
        row["all_definition_bodies_identical_after_namespace_relocation"] = not changed
        row["different_after_relocation"] = changed
        if changed:
            raise ValueError(
                f"Scientific definitions differ: {row['source']} {changed}"
            )
    write_json(parity_path, parity)
    mapping = []
    for row in rows:
        destinations = [
            name
            for name in row["active_destinations"].split(";")
            if name and (root / name).is_file()
        ]
        for target in destinations or [""]:
            mapping.append(
                {
                    "source": row["source"],
                    "source_sha256": row["source_sha256"],
                    "archive_entry": row["source"],
                    "active_destination": target,
                    "active_sha256": digest_file(root / target) if target else "",
                    "actual_action": "COPIED_BYTES"
                    if target and digest_file(root / target) == row["source_sha256"]
                    else "PORTED_OR_DERIVED"
                    if target
                    else "ARCHIVED_ONLY",
                }
            )
    path = root / "provenance/source_map.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    result = {
        "historical_files_unchanged": len(manifest["files"]),
        "historical_archive_sha256": manifest["archive_sha256"],
        "mathematical_modules_ast_verified": len(parity["ast_checks"]),
        "all_normalized_scientific_definition_checks_passed": True,
        "active_legacy_imports": [],
        "gpu_executed": False,
    }
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(
                name
                and name.startswith("gstar_opd")
                and not name.startswith("gstar_opd_formal")
                for name in modules
            ):
                raise ValueError(f"Active historical import in {path}")
    write_json(root / "provenance/source_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify_sources(Path(__file__).resolve().parents[1]), indent=2))
