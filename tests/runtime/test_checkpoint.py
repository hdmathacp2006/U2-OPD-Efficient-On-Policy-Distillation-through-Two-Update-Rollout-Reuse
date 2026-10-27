"""Restoration must reproduce the next update, not merely reload adapter weights."""

import random
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from gstar_opd_formal.controllers.adaptive import AdaptiveHorizonPolicy
from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.opd.engine import OpdEngine
from gstar_opd_formal.training.checkpoint import (
    CheckpointRequest,
    capture_random_state,
    load_checkpoint,
    restore_random_state,
    save_checkpoint,
)


def checkpoint_request(path: Path) -> CheckpointRequest:
    return CheckpointRequest(
        path,
        {"student_revision": "local", "teacher_revision": "local", "split": "fixed"},
        {"data_cursor": 2},
    )


def test_resume_reproduces_the_next_optimizer_update(
    local_engine: OpdEngine, tmp_path: Path
) -> None:
    batch = (Gsm8kExample(0, "What is 2 + 2?", "#### 4"),)
    policy = AdaptiveHorizonPolicy((2, 4))
    local_engine.step(batch, 2)
    request = checkpoint_request(tmp_path / "state.pt")
    save_checkpoint(local_engine, policy, request)
    reference = local_engine.step(batch, 2)
    expected = {
        name: value.clone()
        for name, value in local_engine.bundle.student.named_parameters()
        if value.requires_grad
    }
    load_checkpoint(local_engine, policy, request)
    resumed = local_engine.step(batch, 2)
    assert resumed.loss == reference.loss and all(
        torch.equal(expected[name], value)
        for name, value in local_engine.bundle.student.named_parameters()
        if value.requires_grad
    )


def test_resume_restores_data_cursor(local_engine: OpdEngine, tmp_path: Path) -> None:
    policy = AdaptiveHorizonPolicy((2, 4))
    request = checkpoint_request(tmp_path / "state.pt")
    save_checkpoint(local_engine, policy, request)
    assert load_checkpoint(local_engine, policy, request) == {"data_cursor": 2}


def test_resume_into_reconstructed_engine_reproduces_next_update(
    engine_factory: Callable[[], OpdEngine], tmp_path: Path
) -> None:
    original = engine_factory()
    policy = AdaptiveHorizonPolicy((2, 4))
    batch = (Gsm8kExample(0, "What is 2 + 2?", "#### 4"),)
    original.step(batch, 2)
    request = checkpoint_request(tmp_path / "state.pt")
    save_checkpoint(original, policy, request)
    original.step(batch, 2)
    reconstructed = engine_factory()
    load_checkpoint(reconstructed, AdaptiveHorizonPolicy((2, 4)), request)
    reconstructed.step(batch, 2)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original.bundle.student.parameters(),
            reconstructed.bundle.student.parameters(),
            strict=True,
        )
    )


def test_resume_rejects_changed_run_contract(
    local_engine: OpdEngine, tmp_path: Path
) -> None:
    policy = AdaptiveHorizonPolicy((2, 4))
    request = checkpoint_request(tmp_path / "state.pt")
    save_checkpoint(local_engine, policy, request)
    incompatible = CheckpointRequest(
        request.path, {"student_revision": "different"}, {}
    )
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(local_engine, policy, incompatible)


def test_invalid_accounting_is_rejected_before_changing_student(
    local_engine: OpdEngine, tmp_path: Path
) -> None:
    policy = AdaptiveHorizonPolicy((2, 4))
    request = checkpoint_request(tmp_path / "state.pt")
    save_checkpoint(local_engine, policy, request)
    local_engine.step((Gsm8kExample(0, "What is 2 + 2?", "#### 4"),), 2)
    expected = {
        name: value.detach().clone()
        for name, value in local_engine.bundle.student.named_parameters()
    }
    payload = torch.load(request.path, weights_only=False)
    payload["ledger"]["charged_seconds"] = float("nan")
    torch.save(payload, request.path)
    with pytest.raises(ValueError, match="Charged seconds"):
        load_checkpoint(local_engine, policy, request)
    assert all(
        torch.equal(expected[name], value)
        for name, value in local_engine.bundle.student.named_parameters()
    )


def test_invalid_cuda_topology_does_not_change_cpu_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = capture_random_state()
    saved = capture_random_state()
    saved["python"] = random.Random(23).getstate()
    saved["numpy"] = np.random.RandomState(23).get_state()
    saved["torch_cpu"] = torch.Generator().manual_seed(23).get_state()
    saved["torch_cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="topology"):
        restore_random_state(saved)
    actual = capture_random_state()
    assert (
        actual["python"] == expected["python"]
        and np.array_equal(actual["numpy"][1], expected["numpy"][1])
        and actual["numpy"][2:] == expected["numpy"][2:]
        and torch.equal(actual["torch_cpu"], expected["torch_cpu"])
    )


@pytest.mark.parametrize(
    "invalid_state",
    [
        ("python", ()),
        ("numpy", ()),
        ("torch_cpu", torch.zeros(1, dtype=torch.uint8)),
    ],
)
def test_invalid_random_state_does_not_change_live_torch_generator(
    invalid_state: tuple[str, object],
) -> None:
    expected = torch.get_rng_state()
    saved = capture_random_state()
    field, invalid = invalid_state
    saved[field] = invalid
    with pytest.raises(ValueError, match="random generator"):
        restore_random_state(saved)
    assert torch.equal(torch.get_rng_state(), expected)


def test_malformed_cuda_state_is_rejected_before_changing_cpu_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = torch.get_rng_state()
    saved = capture_random_state()
    saved["torch_cpu"] = torch.Generator().manual_seed(23).get_state()
    saved["torch_cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    cpu_generator = torch.Generator

    def make_generator(device: str | torch.device) -> object:
        if torch.device(device).type == "cuda":
            generator = Mock()
            generator.set_state.side_effect = RuntimeError("Invalid RNG state")
            return generator
        return cpu_generator(device=device)

    # Validate the CUDA failure path without creating a GPU context in CPU tests.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch, "Generator", make_generator)
    with pytest.raises(ValueError, match="CUDA random generator"):
        restore_random_state(saved)
    assert torch.equal(torch.get_rng_state(), expected)
