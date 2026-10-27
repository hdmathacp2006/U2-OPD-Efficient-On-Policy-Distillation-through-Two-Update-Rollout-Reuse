"""Run the formal study with user-owned filesystem storage."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from gstar_opd_formal.contract import load_study  # noqa: E402
from gstar_opd_formal.orchestration.pipeline import (  # noqa: E402
    Pipeline,
    evaluate_base,
    evaluate_methods,
    train_pair,
)
from gstar_opd_formal.storage.controls import destination  # noqa: E402
from gstar_opd_formal.storage.files import write_json  # noqa: E402
from gstar_opd_formal.storage.local import LocalStore  # noqa: E402


@dataclass(frozen=True)
class Paths:
    work: Path
    storage: Path

    @property
    def worker_python(self) -> Path:
        return self.work / "worker_env/bin/python"


def configure(paths: Paths) -> dict:
    if os.environ.get("GSTAR_DRIVE_ROOT_ID"):
        raise ValueError("Filesystem mode cannot use GSTAR_DRIVE_ROOT_ID")
    os.environ["GSTAR_STORAGE_BACKEND"] = "local"
    os.environ["GSTAR_STORAGE_ROOT"] = str(paths.storage)
    os.environ.setdefault("HF_HOME", str(paths.storage / "models"))
    paths.work.mkdir(parents=True, exist_ok=True)
    store = LocalStore(paths.storage, paths.work / ".store_setup")
    selected = destination(store, "local-root")
    write_json(paths.work / "destination.json", selected)
    study = load_study(PROJECT)
    return {
        "study_identity": study.identity,
        "storage_root": str(paths.storage),
        "training_gpu": "A100 with at least 38 GiB",
        "evaluation_gpu": "A100 or L4 with at least 22 GiB",
    }


def install_worker(paths: Paths, action: str) -> dict:
    from gstar_opd_formal.runtime.bootstrap import install_environment

    environment = paths.work / "worker_env"
    install_environment(PROJECT, environment)
    subprocess.run(
        [
            str(paths.worker_python),
            "-m",
            "gstar_opd_formal.runtime.bootstrap",
            "models",
            "--action",
            action,
        ],
        check=True,
    )
    return {"worker_python": str(paths.worker_python), "models_prepared": action}


def run_action(paths: Paths, action: str, gpu: str, resume: bool) -> dict:
    if not paths.worker_python.is_file():
        raise FileNotFoundError("Run setup before selecting a GPU action")
    if action == "train-pair":
        if gpu != "A100":
            raise ValueError("The paired study requires A100")
        settings = Pipeline(
            paths.work,
            str(paths.worker_python),
            gpu_profile="A100",
            resume_stopped=resume,
            project_directory=PROJECT,
        )
        return train_pair(settings)
    if action == "evaluate-methods":
        if gpu != "A100":
            raise ValueError("The selected 12k methods profile requires A100")
        settings = Pipeline(
            paths.work,
            str(paths.worker_python),
            gpu_profile=gpu,
            resume_stopped=resume,
            project_directory=PROJECT,
            evaluation_profile="12k_a100_b20",
        )
        return evaluate_methods(settings)
    if gpu != "L4":
        raise ValueError("The selected 12k base profile requires L4")
    settings = Pipeline(
        paths.work,
        str(paths.worker_python),
        gpu_profile=gpu,
        resume_stopped=resume,
        project_directory=PROJECT,
        evaluation_profile="12k_base_l4",
    )
    return evaluate_base(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "setup",
            "train-pair",
            "evaluate-methods",
            "evaluate-base",
        ),
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--models", choices=("train", "evaluate"), default="train")
    parser.add_argument("--gpu", choices=("A100", "L4"), default="A100")
    parser.add_argument("--resume-stopped", action="store_true")
    args = parser.parse_args()
    paths = Paths(args.work.resolve(), args.storage.resolve())
    preflight = configure(paths)
    if args.action == "preflight":
        result = preflight
    elif args.action == "setup":
        result = {**preflight, **install_worker(paths, args.models)}
    else:
        result = run_action(paths, args.action, args.gpu, args.resume_stopped)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
