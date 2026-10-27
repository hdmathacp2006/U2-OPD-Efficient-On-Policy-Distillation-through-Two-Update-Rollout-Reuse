"""Unequal prompts use the original vLLM converter, without loading vLLM/CUDA."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.models.vllm_rollout import VllmRollout
from gstar_opd_formal.opd.reuse_state import PendingBatch, validate_trajectory
from gstar_opd_formal.opd.rollout import (
    RolloutRequest,
    SamplingSpec,
    tokenize_questions,
)


def converted_trajectories(engine: object) -> tuple:
    examples = (
        Gsm8kExample(1, "2 ?", "#### 2"),
        Gsm8kExample(2, "What is 2 + 2 ? Solve Final answer : 4", "#### 4"),
    )
    request = RolloutRequest(
        examples, engine.spec.prompt, SamplingSpec(8), engine.spec.max_prompt_tokens
    )
    encoded = tokenize_questions(engine.bundle, request)
    backend = SimpleNamespace(owner=engine, adapter_identity="cpu_converter_test")
    trajectories = []
    tokens = [5, 6, 2]
    for index in range(len(examples)):
        prompt = encoded["input_ids"][index][
            encoded["attention_mask"][index].bool()
        ].tolist()
        completion = SimpleNamespace(
            token_ids=tokens,
            logprobs=[{token: SimpleNamespace(logprob=-1.0)} for token in tokens],
        )
        output = SimpleNamespace(
            prompt_token_ids=prompt, finished=True, outputs=[completion]
        )
        trajectories.append(
            VllmRollout.convert_output(
                backend, output, (request, encoded, index, prompt)
            )
        )
    return tuple(trajectories)


def test_original_vllm_padded_output_passes_reuse_validation(engine):
    trajectories = converted_trajectories(engine)
    for item in trajectories:
        validate_trajectory(item)
    assert int((trajectories[0].attention_mask == 0).sum()) > 0


def test_padding_remains_exact_in_pending_snapshot(engine):
    trajectories = converted_trajectories(engine)
    pending = PendingBatch(
        trajectories,
        tuple(torch.full((t.response_length,), -1.0) for t in trajectories),
        0,
        8,
    ).cpu()
    pending.validate(0)
    assert all(
        torch.equal(a.attention_mask, b.attention_mask)
        and torch.equal(a.input_ids, b.input_ids)
        for a, b in zip(trajectories, pending.trajectories, strict=True)
    )


@pytest.mark.parametrize(
    "attention",
    [
        [1, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 2, 1, 1],
    ],
)
def test_invalid_attention_masks_are_rejected(trajectory, attention):
    with pytest.raises(ValueError, match="alignment"):
        validate_trajectory(
            replace(trajectory, attention_mask=torch.tensor([attention]))
        )


def test_padded_targets_match_dense_unpadded_reference(engine, trajectory):
    trimmed = replace(
        trajectory,
        input_ids=trajectory.input_ids[:, 2:],
        attention_mask=trajectory.attention_mask[:, 2:],
        response_mask=trajectory.response_mask[:, 2:],
    )
    padded = engine.collect_targets((trajectory,))[0].targets
    unpadded = engine.collect_targets((trimmed,))[0].targets
    torch.testing.assert_close(vars(padded), vars(unpadded), atol=1e-6, rtol=1e-6)
