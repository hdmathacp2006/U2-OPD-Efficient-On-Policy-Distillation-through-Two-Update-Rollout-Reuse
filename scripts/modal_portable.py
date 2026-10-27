"""User-owned Modal transport for the same formal training and evaluation code."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT = Path("/root/u2_opd")
VOLUME_ROOT = Path("/vol/u2_opd")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("PyYAML==6.0.3")
    .add_local_dir(
        str(PROJECT),
        remote_path="/root/u2_opd",
        ignore=[".git/**", "**/__pycache__/**", "**/*.pyc", ".runtime/**"],
    )
)
app = modal.App("u2-opd-portable")
volume = modal.Volume.from_name("u2-opd-artifacts", create_if_missing=True)


def source_key() -> str:
    digest = hashlib.sha256()
    for directory in ("src", "configs", "requirements", "data", "assets"):
        for path in sorted((REMOTE_PROJECT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(REMOTE_PROJECT)).encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def execute(command: str, *options: str) -> None:
    work = VOLUME_ROOT / "work" / source_key()
    storage = VOLUME_ROOT / "store"
    environment = {
        **os.environ,
        "GSTAR_STORAGE_BACKEND": "local",
        "GSTAR_STORAGE_ROOT": str(storage),
        "HF_HOME": str(storage / "models"),
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }
    subprocess.run(
        [
            sys.executable,
            str(REMOTE_PROJECT / "scripts/portable.py"),
            command,
            "--work",
            str(work),
            "--storage",
            str(storage),
            *options,
        ],
        env=environment,
        check=True,
    )


@app.function(image=image, volumes={"/vol": volume}, timeout=7200, memory=8192)
def setup(models: str) -> None:
    try:
        execute("setup", "--models", models)
    finally:
        volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def preflight() -> None:
    execute("preflight")


@app.function(image=image, volumes={"/vol": volume}, gpu="A100", timeout=21600)
def run_a100(action: str, resume_stopped: bool) -> None:
    options = ("--resume-stopped",) if resume_stopped else ()
    try:
        execute(action, "--gpu", "A100", *options)
    finally:
        volume.commit()


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", timeout=21600)
def run_l4(resume_stopped: bool) -> None:
    options = ("--resume-stopped",) if resume_stopped else ()
    try:
        execute("evaluate-base", "--gpu", "L4", *options)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(action: str = "preflight", resume_stopped: bool = False) -> None:
    if action == "preflight":
        preflight.remote()
    elif action in {"train-pair", "evaluate-methods"}:
        setup.remote("train" if action == "train-pair" else "evaluate")
        run_a100.remote(action, resume_stopped)
    elif action == "evaluate-base":
        setup.remote("evaluate")
        run_l4.remote(resume_stopped)
    else:
        raise ValueError(
            "Choose preflight, train-pair, evaluate-methods or evaluate-base"
        )
