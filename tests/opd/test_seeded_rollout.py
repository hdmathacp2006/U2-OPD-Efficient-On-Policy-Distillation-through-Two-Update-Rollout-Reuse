"""Fixed question draws remain independent of training RNG and batch order."""

import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.opd.rollout import (
    RolloutRequest,
    SamplingSpec,
    generate_responses,
    generation_config,
    question_sampling_seed,
)


def seeded_request(engine) -> RolloutRequest:
    return RolloutRequest(
        tuple(Gsm8kExample(index, "2 + 2?", "#### 4") for index in range(3)),
        engine.spec.prompt,
        SamplingSpec(4, 0.6, 0.95, top_k=20, min_p=0.0),
        64,
        question_seed=20261013,
    )


def test_seeded_generation_preserves_training_rng(local_engine) -> None:
    before = torch.get_rng_state().clone()
    generate_responses(local_engine.bundle, seeded_request(local_engine))
    assert torch.equal(before, torch.get_rng_state())


def test_seeded_generation_preserves_mixed_module_modes(local_engine) -> None:
    local_engine.bundle.student.train()
    local_engine.bundle.student.base_model.eval()
    modes = {
        module: module.training for module in local_engine.bundle.student.modules()
    }
    generate_responses(local_engine.bundle, seeded_request(local_engine))
    assert modes == {
        module: module.training for module in local_engine.bundle.student.modules()
    }


def test_seeded_question_outputs_ignore_batch_order(local_engine) -> None:
    request = seeded_request(local_engine)
    first = generate_responses(local_engine.bundle, request)
    torch.rand(9)
    reversed_outputs = generate_responses(
        local_engine.bundle, replace(request, examples=request.examples[::-1])
    )
    assert all(
        torch.equal(left.input_ids, right.input_ids)
        for left, right in zip(first, reversed_outputs[::-1], strict=True)
    )


def test_seeded_question_output_ignores_neighbor_batch(local_engine) -> None:
    request = seeded_request(local_engine)
    batched = generate_responses(local_engine.bundle, request)[1]
    single = generate_responses(
        local_engine.bundle, replace(request, examples=(request.examples[1],))
    )[0]
    assert torch.equal(batched.input_ids, single.input_ids)


def test_seeded_exception_preserves_rng_and_modes(local_engine, monkeypatch) -> None:
    def fail_generation(**kwargs):
        torch.rand(3)
        raise RuntimeError("sample failed")

    before = torch.get_rng_state().clone()
    local_engine.bundle.student.train()
    monkeypatch.setattr(local_engine.bundle.student, "generate", fail_generation)
    with pytest.raises(RuntimeError, match="sample failed"):
        generate_responses(local_engine.bundle, seeded_request(local_engine))
    assert (
        torch.equal(before, torch.get_rng_state())
        and local_engine.bundle.student.training
    )


def test_feedback_processors_are_explicit(local_engine) -> None:
    config = generation_config(
        local_engine.bundle, seeded_request(local_engine).sampling
    )
    assert (config.temperature, config.top_p, config.top_k, config.min_p) == (
        0.6,
        0.95,
        20,
        0.0,
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {"top_k": -1},
        {"top_k": True},
        {"min_p": -0.1},
        {"min_p": 1.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"max_new_tokens": True},
    ],
)
def test_sampling_rejects_invalid_processors(replacement) -> None:
    with pytest.raises(ValueError, match="sampling"):
        replace(SamplingSpec(4), **replacement)


def test_question_seed_namespace_and_source_id_change_the_draw() -> None:
    assert (
        len(
            {
                question_sampling_seed(7, 1),
                question_sampling_seed(7, 2),
                question_sampling_seed(8, 1),
            }
        )
        == 3
    )


def test_seeded_request_rejects_boolean_seed(local_engine) -> None:
    with pytest.raises(ValueError, match="integer"):
        replace(seeded_request(local_engine), question_seed=True)


def test_seeded_backend_failure_restores_all_parent_random_streams(
    local_engine,
) -> None:
    def fail_backend(request):
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("backend failed")

    local_engine.bundle = replace(
        local_engine.bundle, rollout=SimpleNamespace(generate=fail_backend)
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="backend failed"):
        generate_responses(local_engine.bundle, seeded_request(local_engine))
    restored_numpy = np.random.get_state()
    assert (
        random.getstate() == python_state
        and np.array_equal(restored_numpy[1], numpy_state[1])
        and restored_numpy[0] == numpy_state[0]
        and restored_numpy[2:] == numpy_state[2:]
        and torch.equal(torch_state, torch.get_rng_state())
    )
