"""Record actual CUDA/software evidence rather than impersonating Modal devices."""

import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

from gstar_opd_formal.storage.files import digest_value, write_json

PINNED = {
    "torch": "2.8.0",
    "vllm": "0.10.2",
    "flash-attn": "2.8.3",
    "transformers": "4.57.6",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "numpy": "2.2.6",
    "math-verify": "0.8.0",
}
MINIMUM_MEMORY = {"L4": 22 * 2**30, "A100": 38 * 2**30}


def package_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in PINNED}


def validate_packages(versions: dict) -> None:
    for name, expected in PINNED.items():
        if versions.get(name, "").split("+")[0] != expected:
            raise ValueError(f"Pinned dependency mismatch: {name} requires {expected}")


def validate_hardware(hardware: dict, profile: str) -> None:
    if profile not in MINIMUM_MEMORY:
        raise ValueError("Use an explicit L4 or A100 execution profile")
    name = hardware.get("gpu_name", "")
    capability = hardware.get("compute_capability")
    matches = name in {"NVIDIA L4", "L4"} if profile == "L4" else "A100" in name
    if (
        not matches
        or capability != ([8, 9] if profile == "L4" else [8, 0])
        or hardware.get("bf16_supported") is not True
        or hardware.get("total_memory_bytes", 0) < MINIMUM_MEMORY[profile]
    ):
        raise ValueError(f"Actual hardware does not satisfy declared {profile} profile")


def inspect_environment(profile: str) -> dict:
    import torch

    from gstar_opd_formal.runtime.bootstrap import PYTHON_VERSION

    if platform.system() != "Linux" or platform.python_version() != PYTHON_VERSION:
        raise ValueError(f"Colab workers require Linux and Python {PYTHON_VERSION}")
    if not torch.cuda.is_available():
        raise ValueError(
            "A CUDA runtime is required; CPU preflight is a separate action"
        )
    versions = package_versions()
    validate_packages(versions)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    hardware = {
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    validate_hardware(hardware, profile)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    immutable = {
        "profile": profile,
        "hardware": hardware,
        "packages": versions,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "driver": driver,
        "cudnn": torch.backends.cudnn.version(),
    }
    return {
        **immutable,
        "environment_id": digest_value(immutable),
        "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
    }


def save_environment(path: Path, value: dict) -> None:
    write_json(path, value)
