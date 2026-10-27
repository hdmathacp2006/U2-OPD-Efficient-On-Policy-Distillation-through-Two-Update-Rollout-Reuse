"""Atomically persist adapter, optimizer, policy, accounting, and random states."""

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from gstar_opd_formal.opd.costs import CostLedger
from gstar_opd_formal.opd.engine import OpdEngine

CHECKPOINT_VERSION = 1


class CheckpointPolicy(Protocol):
    """Persistence needs state methods, not a specific controller observation API."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...


@dataclass(frozen=True)
class CheckpointRequest:
    path: Path
    run_contract: dict[str, object]
    run_state: dict[str, object]


def capture_random_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def validate_random_state(state: dict[str, object]) -> None:
    """Check saved generators without advancing the live training generators."""
    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Invalid checkpoint random state fields")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (
        TypeError,
        ValueError,
        RuntimeError,
        IndexError,
        KeyError,
        OverflowError,
    ) as error:
        raise ValueError("Invalid checkpoint random generator state") from error
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        for value in cuda_states
    ):
        raise ValueError("Invalid checkpoint CUDA random state")
    if cuda_states:
        if (
            not torch.cuda.is_available()
            or len(cuda_states) != torch.cuda.device_count()
        ):
            raise ValueError("Checkpoint CUDA RNG topology differs from this machine")
        try:
            for device_index, cuda_state in enumerate(cuda_states):
                torch.Generator(device=torch.device("cuda", device_index)).set_state(
                    cuda_state
                )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "Invalid checkpoint CUDA random generator state"
            ) from error


def restore_random_state(state: dict[str, object]) -> None:
    validate_random_state(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        cuda_states = state["torch_cuda"]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    engine: OpdEngine, policy: CheckpointPolicy, request: CheckpointRequest
) -> None:
    """Save at a completed block boundary; run_state includes data cursor/context."""
    payload = build_checkpoint_payload(engine, policy, request)
    write_checkpoint(payload, request.path)


def build_checkpoint_payload(
    engine: OpdEngine, policy: CheckpointPolicy, request: CheckpointRequest
) -> dict[str, object]:
    if getattr(policy, "pending_index", None) is not None:
        raise ValueError(
            "Checkpoint only after completing and observing a controller block"
        )
    engine.ledger.synchronize()
    adapter_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_peft_model_state_dict(
            engine.bundle.student, save_embedding_layers=False
        ).items()
    }
    return {
        "version": CHECKPOINT_VERSION,
        "engine_spec": asdict(engine.spec),
        "run_contract": request.run_contract,
        "run_state": request.run_state,
        "adapter": adapter_state,
        "optimizer": engine.optimizer.state_dict(),
        "completed_steps": engine.completed_steps,
        "policy": policy.state_dict(),
        "ledger": engine.ledger.state_dict(),
        "random": capture_random_state(),
    }


def write_checkpoint(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    # A partial write must not replace the last successfully saved checkpoint.
    os.replace(temporary_path, path)


def load_checkpoint(
    engine: OpdEngine, policy: CheckpointPolicy, request: CheckpointRequest
) -> dict[str, object]:
    """Restore a project-created local checkpoint and return its run cursor/context."""
    # Python/NumPy RNG objects require loading a trusted project-created file.
    payload = torch.load(request.path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, engine, request)
    policy.load_state_dict(payload["policy"])
    set_peft_model_state_dict(engine.bundle.student, payload["adapter"])
    engine.optimizer.load_state_dict(payload["optimizer"])
    engine.completed_steps = payload["completed_steps"]
    engine.ledger.load_state_dict(payload["ledger"])
    restore_random_state(payload["random"])
    return payload["run_state"]


def validate_checkpoint_payload(
    payload: dict[str, object], engine: OpdEngine, request: CheckpointRequest
) -> None:
    if not isinstance(payload, dict) or (
        type(payload.get("version")) is not int
        or payload["version"] != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported checkpoint version")
    if payload.get("engine_spec") != asdict(engine.spec):
        raise ValueError("Checkpoint engine settings differ from this run")
    if payload.get("run_contract") != request.run_contract:
        raise ValueError("Checkpoint model revisions, split, or run contract differ")
    expected = get_peft_model_state_dict(
        engine.bundle.student, save_embedding_layers=False
    )
    adapter = payload["adapter"]
    if (
        not isinstance(adapter, dict)
        or adapter.keys() != expected.keys()
        or any(
            not isinstance(adapter[name], torch.Tensor)
            or adapter[name].shape != tensor.shape
            or adapter[name].dtype != tensor.dtype
            or not torch.isfinite(adapter[name]).all()
            for name, tensor in expected.items()
        )
    ):
        raise ValueError("Checkpoint adapter structure differs from the student")
    if type(payload["completed_steps"]) is not int or payload["completed_steps"] < 0:
        raise ValueError("Invalid completed optimizer step count")
    # Validate accounting and RNG before changing the adapter or controller.
    CostLedger(engine.ledger.device).load_state_dict(payload["ledger"])
    validate_random_state(payload["random"])
