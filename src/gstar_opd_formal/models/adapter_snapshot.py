"""Content identities for the current LoRA adapter without feedback services."""

import hashlib
import json
from dataclasses import dataclass

import torch
from peft import get_peft_model_state_dict
from torch import Tensor, nn


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class AdapterSnapshot:
    identity: str
    state: dict[str, Tensor]


def snapshot_identity(model_identity: str, state: dict[str, Tensor]) -> str:
    if not isinstance(model_identity, str) or not model_identity:
        raise ValueError("Adapter model identity must be nonempty")
    if not isinstance(state, dict) or not state:
        raise ValueError("Adapter snapshot must contain tensors")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("Adapter tensor names must be nonempty strings")
    digest = hashlib.sha256(canonical_json(model_identity).encode("utf-8"))
    for name in sorted(state):
        tensor = state[name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Adapter snapshot tensors must be finite and on CPU")
        metadata = canonical_json((name, str(tensor.dtype), tuple(tensor.shape)))
        digest.update(metadata.encode("utf-8"))
        # Byte views support BF16, which NumPy cannot represent as a native dtype.
        digest.update(
            tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def capture_adapter_snapshot(
    student: nn.Module, model_identity: str
) -> AdapterSnapshot:
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in get_peft_model_state_dict(
            student, save_embedding_layers=False
        ).items()
    }
    return AdapterSnapshot(snapshot_identity(model_identity, state), state)


def validate_adapter_snapshot(
    snapshot: AdapterSnapshot, student: nn.Module, model_identity: str
) -> None:
    if not isinstance(snapshot, AdapterSnapshot):
        raise ValueError("Expected a captured adapter snapshot")
    if snapshot.identity != snapshot_identity(model_identity, snapshot.state):
        raise ValueError("Adapter snapshot identity or model provenance differs")
    expected = get_peft_model_state_dict(student, save_embedding_layers=False)
    if snapshot.state.keys() != expected.keys() or any(
        snapshot.state[name].shape != tensor.shape
        or snapshot.state[name].dtype != tensor.dtype
        for name, tensor in expected.items()
    ):
        raise ValueError("Adapter snapshot structure differs from this student")
