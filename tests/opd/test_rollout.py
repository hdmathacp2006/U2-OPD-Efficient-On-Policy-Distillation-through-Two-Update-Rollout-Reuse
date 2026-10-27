"""Greedy measurements and sampled microbatches retain distinct decoding policies."""

from dataclasses import replace

import pytest
import torch

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.opd.engine import OpdEngine
from gstar_opd_formal.opd.rollout import (
    GreedyRequest,
    GreedySpec,
    RolloutRequest,
    SamplingSpec,
    generate_greedy_responses,
    generate_response_microbatches,
    generate_responses,
    greedy_generation_config,
)


def rollout_examples() -> tuple[Gsm8kExample, ...]:
    return tuple(Gsm8kExample(index, "2 + 2?", "#### 4") for index in range(3))


def greedy_request(engine: OpdEngine) -> GreedyRequest:
    return GreedyRequest(rollout_examples(), engine.spec.prompt, GreedySpec(3), 64)


def test_greedy_decoding_preserves_training_rng(local_engine: OpdEngine) -> None:
    before = torch.get_rng_state().clone()
    generate_greedy_responses(local_engine.bundle, greedy_request(local_engine))
    assert torch.equal(before, torch.get_rng_state())


def test_greedy_decoding_preserves_student_mode(local_engine: OpdEngine) -> None:
    local_engine.bundle.student.train()
    local_engine.bundle.student.base_model.eval()
    modes = {
        module: module.training for module in local_engine.bundle.student.modules()
    }
    generate_greedy_responses(local_engine.bundle, greedy_request(local_engine))
    assert modes == {
        module: module.training for module in local_engine.bundle.student.modules()
    }


def test_greedy_decoding_is_repeatable(local_engine: OpdEngine) -> None:
    first = generate_greedy_responses(local_engine.bundle, greedy_request(local_engine))
    second = generate_greedy_responses(
        local_engine.bundle, greedy_request(local_engine)
    )
    assert all(
        torch.equal(left.input_ids, right.input_ids)
        for left, right in zip(first, second, strict=True)
    )


def test_greedy_config_ignores_inherited_processors(local_engine: OpdEngine) -> None:
    local_engine.bundle.student.generation_config.repetition_penalty = 2.0
    config = greedy_generation_config(local_engine.bundle, GreedySpec(3))
    assert (
        config.do_sample,
        config.temperature,
        config.top_p,
        config.top_k,
        config.repetition_penalty,
    ) == (False, None, None, None, 1.0)


def test_greedy_generation_ignores_model_sampling_defaults(
    local_engine: OpdEngine,
) -> None:
    request = greedy_request(local_engine)
    expected = generate_greedy_responses(local_engine.bundle, request)
    config = local_engine.bundle.student.generation_config
    config.do_sample = True
    config.num_return_sequences = 2
    config.repetition_penalty = 5.0
    observed = generate_greedy_responses(local_engine.bundle, request)
    assert all(
        torch.equal(left.input_ids, right.input_ids)
        for left, right in zip(expected, observed, strict=True)
    )


def test_sampled_generation_ignores_model_sampling_defaults(
    local_engine: OpdEngine,
) -> None:
    request = RolloutRequest(
        rollout_examples(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    rng = torch.get_rng_state()
    expected = generate_responses(local_engine.bundle, request)
    torch.set_rng_state(rng)
    config = local_engine.bundle.student.generation_config
    config.top_p = 0.05
    config.temperature = 0.1
    config.num_return_sequences = 2
    config.repetition_penalty = 5.0
    observed = generate_responses(local_engine.bundle, request)
    assert all(
        torch.equal(left.input_ids, right.input_ids)
        for left, right in zip(expected, observed, strict=True)
    )


def test_greedy_exception_restores_rng_and_modes(
    local_engine: OpdEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_generation(**kwargs: object) -> None:
        torch.rand(3)
        raise RuntimeError("decode failed")

    before = torch.get_rng_state().clone()
    local_engine.bundle.student.train()
    monkeypatch.setattr(local_engine.bundle.student, "generate", fail_generation)
    with pytest.raises(RuntimeError, match="decode failed"):
        generate_greedy_responses(local_engine.bundle, greedy_request(local_engine))
    assert (
        torch.equal(before, torch.get_rng_state())
        and local_engine.bundle.student.training
    )


def test_sampled_microbatches_preserve_input_order(local_engine: OpdEngine) -> None:
    request = RolloutRequest(
        rollout_examples(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    responses = generate_response_microbatches(local_engine.bundle, request, 2)
    assert tuple(response.example_id for response in responses) == ("0", "1", "2")


def test_sampled_microbatch_calls_do_not_update_adapters(
    local_engine: OpdEngine,
) -> None:
    before = {
        name: tensor.clone()
        for name, tensor in local_engine.bundle.student.named_parameters()
    }
    request = RolloutRequest(
        rollout_examples(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    generate_response_microbatches(local_engine.bundle, request, 1)
    assert all(
        torch.equal(before[name], tensor)
        for name, tensor in local_engine.bundle.student.named_parameters()
    )


def test_one_microbatch_matches_existing_sampled_generation(
    local_engine: OpdEngine,
) -> None:
    request = RolloutRequest(
        rollout_examples(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    before = torch.get_rng_state()
    expected = generate_responses(local_engine.bundle, request)
    torch.set_rng_state(before)
    observed = generate_response_microbatches(local_engine.bundle, request, 3)
    assert all(
        torch.equal(left.input_ids, right.input_ids)
        for left, right in zip(expected, observed, strict=True)
    )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_sampled_microbatch_size_requires_positive_integer(
    local_engine: OpdEngine, batch_size: object
) -> None:
    request = RolloutRequest(
        rollout_examples(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    with pytest.raises(ValueError, match="positive integer"):
        generate_response_microbatches(local_engine.bundle, request, batch_size)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_greedy_horizon_requires_positive_integer(horizon: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GreedySpec(horizon)


def test_greedy_rejects_empty_batch(local_engine: OpdEngine) -> None:
    request = replace(greedy_request(local_engine), examples=())
    with pytest.raises(ValueError, match="at least one"):
        generate_greedy_responses(local_engine.bundle, request)
