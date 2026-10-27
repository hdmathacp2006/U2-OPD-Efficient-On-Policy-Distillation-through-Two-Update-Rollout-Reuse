"""Build an allowlisted standalone source release; exclude legacy/runtime output."""

from pathlib import Path
from tempfile import TemporaryDirectory

from gstar_opd_formal.contract import load_study
from gstar_opd_formal.runtime.notebooks import build_notebooks
from gstar_opd_formal.storage.files import (
    create_archive,
    digest_file,
    digest_value,
    extract_archive,
    file_record,
    verify_tree,
    write_json,
)

SOURCE_DIRECTORIES = (
    "src",
    "configs",
    "data",
    "assets",
    "requirements",
    "scripts",
    "tests",
    "docs",
)

PROVENANCE_FILES = (
    "source_verification.json",
    "method_parity.json",
    "initialization_validation.json",
    "gold_validation.json",
)


def source_paths(root: Path) -> list[Path]:
    paths = [root / "pyproject.toml", root / "README.md", root / "AGENTS.md"]
    paths.extend(root / "provenance" / name for name in PROVENANCE_FILES)
    for directory in SOURCE_DIRECTORIES:
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not any(
                part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
                for part in path.parts
            )
            and path.suffix not in {".pyc", ".pyo"}
        )
    return sorted(set(paths))


def build_release(root: Path, output: Path) -> dict:
    study = load_study(root)
    paths = source_paths(root)
    records = [file_record(path, root) for path in paths]
    release_id = digest_value(records)
    label = "20261025_" + release_id[:12]
    folder = output / label
    folder.mkdir(parents=True, exist_ok=False)
    manifest = create_archive(folder / "project.tar.gz", (root, paths))
    with TemporaryDirectory(prefix="formal_release_closure_") as temporary:
        clean = Path(temporary) / "project"
        extract_archive(folder / "project.tar.gz", clean)
        verify_tree(clean, manifest["files"])
        if load_study(clean).identity != study.identity:
            raise ValueError("Clean release has different scientific inputs/code")
    write_json(
        folder / "manifest.json",
        {
            **manifest,
            "release_id": release_id,
            "study_identity": study.identity,
            "clean_extraction_verified": True,
        },
    )
    drive_path = f"GStar_OPD_Formal/releases/{label}"
    deployment = {
        "schema": "formal_deployment_v1",
        "root_folder_id": study.design["drive_root_id"],
        "release_id": release_id,
        "study_identity": study.identity,
        "artifacts": {
            name: {
                **file_record(folder / filename, folder),
                "drive_path": drive_path + "/" + filename,
            }
            for name, filename in (
                ("source", "project.tar.gz"),
                ("manifest", "manifest.json"),
            )
        },
    }
    write_json(folder / "deployment.json", deployment)
    value = {
        "release_id": release_id,
        "study_identity": study.identity,
        "root_folder_id": study.design["drive_root_id"],
        "folder": str(folder),
        "drive_path": drive_path,
        "deployment_sha256": digest_file(folder / "deployment.json"),
        "source_files": len(paths),
        "gpu_executed": False,
    }
    notebooks = (
        build_notebooks(folder, value)
        if study.design["drive_root_id"] != "local-root"
        else []
    )
    for path in notebooks:
        import json

        notebook = json.loads(path.read_bytes())
        for index, item in enumerate(notebook["cells"]):
            if item["cell_type"] == "code":
                compile("".join(item["source"]), f"{path.name}:{index}", "exec")
    value["notebooks"] = [str(path) for path in notebooks]
    write_json(folder / "release_receipt.json", value)
    return value
