"""Install one locked worker environment and pinned model cache."""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PYTHON_VERSION = "3.12.12"
STUDENT = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
TEACHER = (
    "nvidia/Nemotron-Cascade-8B-Thinking",
    "922c4fe0b33248fa02d2ed450046755f6442e638",
)


def prepare_models(action: str) -> dict:
    from huggingface_hub import snapshot_download

    selected = (STUDENT,) if action == "evaluate" else (STUDENT, TEACHER)
    paths = []
    for model, revision in selected:
        paths.append(
            snapshot_download(
                repo_id=model,
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
            )
        )
    return {"status": "pinned_models_downloaded", "paths": paths}


def install_environment(root: Path, environment: Path) -> None:
    from gstar_opd_formal.storage.files import digest_file, read_json, write_json

    expected = {
        "lock_sha256": digest_file(root / "requirements/colab-cu128.lock"),
        "package_sha256": digest_file(root / "pyproject.toml"),
        "sources": {
            path.relative_to(root).as_posix(): digest_file(path)
            for path in sorted((root / "src").rglob("*.py"))
        },
    }
    marker = environment / ".gstar_install.json"
    if marker.exists():
        if read_json(marker) != expected:
            raise ValueError("Installed environment belongs to another release")
        subprocess.run(
            [str(environment / "bin/python"), "-m", "pip", "check"], check=True
        )
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv==0.9.5"],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "venv",
            "--python",
            PYTHON_VERSION,
            "--managed-python",
            "--seed",
            "--allow-existing",
            str(environment),
        ],
        check=True,
        timeout=600,
    )
    python = environment / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "-r",
            str(root / "requirements/colab-cu128.lock"),
        ],
        check=True,
        timeout=1200,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(root)],
        check=True,
        timeout=180,
    )
    # Build/install FA2 only after the exact torch ABI is installed in this env.
    subprocess.run(
        [str(python), "-m", "gstar_opd_formal.runtime.bootstrap", "flash"],
        check=True,
        timeout=1500,
        env={**os.environ, "MAX_JOBS": "2"},
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    write_json(marker, expected)


def install_flash() -> None:
    import torch

    api = "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3"
    request = urllib.request.Request(
        api, headers={"User-Agent": "gstar-colab-bootstrap"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assets = json.load(response)["assets"]
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    suffix = f"cu12torch2.8cxx11abi{abi}-cp312-cp312-linux_x86_64.whl"
    matches = [item for item in assets if item["name"].endswith(suffix)]
    if len(matches) > 1:
        raise ValueError("Ambiguous official FlashAttention wheel")
    if matches:
        url = matches[0]["browser_download_url"]
        if not url.startswith(
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        ):
            raise ValueError("Unexpected FlashAttention release source")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", url], check=True
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "flash-attn==2.8.3",
                "--no-build-isolation",
                "--no-deps",
            ],
            check=True,
        )


def install_grader(root: Path, environment: Path) -> None:
    from gstar_opd_formal.storage.files import digest_file, read_json, write_json

    expected = {"lock_sha256": digest_file(root / "requirements/olympiad-grader.lock")}
    marker = environment / ".formal_grader.json"
    if marker.exists():
        if read_json(marker) != expected:
            raise ValueError("Official grader environment belongs to another lock")
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "venv",
            "--python",
            PYTHON_VERSION,
            "--managed-python",
            "--seed",
            "--allow-existing",
            str(environment),
        ],
        check=True,
        timeout=180,
    )
    python = environment / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(root / "requirements/olympiad-grader.lock"),
        ],
        check=True,
        timeout=300,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    write_json(marker, expected)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("install", "flash", "models", "grader"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--action", choices=("train", "evaluate"), default="evaluate")
    args = parser.parse_args()
    if args.operation == "flash":
        install_flash()
    elif args.operation == "models":
        print(json.dumps(prepare_models(args.action)))
    elif args.operation == "grader":
        install_grader(args.root, args.environment)
    else:
        install_environment(args.root, args.environment)


if __name__ == "__main__":
    main()
