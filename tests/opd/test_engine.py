"""Integration checks use random local Qwen models; no downloads or GPU jobs."""

from collections.abc import Iterable
from dataclasses import replace

import pytest
import torch

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.evaluation.graders.math import score_gsm8k
from gstar_opd_formal.models.loading import LoraSpec, ModelLoadSpec, load_model_bundle
from gstar_opd_formal.opd.engine import OpdEngine
from gstar_opd_formal.opd.objective import CandidateTargets, compute_clipped_opd_loss
from gstar_opd_formal.opd.rollout import (
    RolloutRequest,
    SamplingSpec,
    generate_responses,
    response_logits,
    split_generated_responses,
    tokenize_questions,
)


def example_batch() -> tuple[Gsm8kExample, ...]:
    return (
        Gsm8kExample(0, "What is 2 + 2?", "#### 4"),
        Gsm8kExample(1, "2 + 2?", "#### 4"),
    )


def test_update_has_finite_loss_and_nonzero_gradient(local_engine: OpdEngine) -> None:
    result = local_engine.step(example_batch(), 4)
    assert torch.isfinite(torch.tensor(result.loss)) and result.grad_norm > 0


def test_teacher_weights_are_unchanged_after_update(local_engine: OpdEngine) -> None:
    before = {
        name: value.clone()
        for name, value in local_engine.bundle.teacher.state_dict().items()
    }
    local_engine.step(example_batch(), 4)
    assert all(
        torch.equal(before[name], value)
        for name, value in local_engine.bundle.teacher.state_dict().items()
    )


def test_only_lora_parameters_change(local_engine: OpdEngine) -> None:
    before = {
        name: value.clone()
        for name, value in local_engine.bundle.student.named_parameters()
    }
    local_engine.step(example_batch(), 4)
    changed = {
        name
        for name, value in local_engine.bundle.student.named_parameters()
        if not torch.equal(before[name], value)
    }
    assert changed and all("lora_" in name for name in changed)


def test_teacher_has_no_gradients(local_engine: OpdEngine) -> None:
    local_engine.step(example_batch(), 4)
    assert all(value.grad is None for value in local_engine.bundle.teacher.parameters())


def test_stage_costs_and_total_are_recorded(local_engine: OpdEngine) -> None:
    result = local_engine.step(example_batch(), 4)
    ledger = local_engine.ledger
    assert (
        set(ledger.stage_seconds)
        == {"student_rollout", "teacher_scoring", "student_update"}
        and result.charged_seconds >= sum(ledger.stage_seconds.values()) > 0
    )


def test_generated_eos_is_kept_but_eos_padding_is_not(local_engine: OpdEngine) -> None:
    bundle = local_engine.bundle
    bundle.tokenizer.pad_token = bundle.tokenizer.eos_token
    request = RolloutRequest(
        example_batch(), local_engine.spec.prompt, SamplingSpec(4), 64
    )
    encoded = tokenize_questions(bundle, request)
    generated = torch.cat(
        (encoded["input_ids"], torch.tensor([[7, 2, 2, 2], [8, 9, 10, 2]])), dim=1
    )
    trajectories = split_generated_responses(bundle, generated, (request, encoded))
    assert tuple(
        (item.response_length, int(item.response_mask.sum())) for item in trajectories
    ) == ((2, 2), (4, 4))


def test_eos_marker_does_not_corrupt_numeric_answer(local_engine: OpdEngine) -> None:
    bundle = local_engine.bundle
    request = RolloutRequest(
        (example_batch()[0],), local_engine.spec.prompt, SamplingSpec(5), 64
    )
    encoded = tokenize_questions(bundle, request)
    response = bundle.tokenizer.encode("Final answer: 4", add_special_tokens=False)
    generated = torch.cat((encoded["input_ids"], torch.tensor([response + [2]])), dim=1)
    trajectory = split_generated_responses(bundle, generated, (request, encoded))[0]
    assert score_gsm8k(trajectory.example_id, trajectory.text, "#### 4").correct


def test_response_logits_use_last_prompt_position(local_engine: OpdEngine) -> None:
    bundle = local_engine.bundle
    request = RolloutRequest(
        (example_batch()[0],), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    encoded = tokenize_questions(bundle, request)
    generated = torch.cat((encoded["input_ids"], torch.tensor([[7, 2]])), dim=1)
    trajectory = split_generated_responses(bundle, generated, (request, encoded))[0]
    bundle.student.eval()
    logits = bundle.student(
        input_ids=generated, attention_mask=torch.ones_like(generated)
    ).logits
    width = encoded["input_ids"].shape[1]
    assert torch.allclose(
        response_logits(bundle.student, trajectory),
        logits[0, width - 1 : width + 1],
        atol=1e-6,
    )


def test_left_padding_does_not_change_valid_response_scores(
    local_engine: OpdEngine,
) -> None:
    bundle = local_engine.bundle
    request = RolloutRequest(
        example_batch(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    encoded = tokenize_questions(bundle, request)
    generated = torch.cat((encoded["input_ids"], torch.tensor([[7, 2], [7, 2]])), dim=1)
    padded = split_generated_responses(bundle, generated, (request, encoded))[1]
    valid = padded.attention_mask[0].bool()
    unpadded_ids = padded.input_ids[:, valid]
    unpadded_response_mask = torch.zeros_like(unpadded_ids[:, 1:], dtype=torch.bool)
    unpadded_response_mask[:, -padded.response_length :] = True
    unpadded = replace(
        padded,
        input_ids=unpadded_ids,
        attention_mask=torch.ones_like(unpadded_ids),
        response_mask=unpadded_response_mask,
    )
    bundle.student.eval()
    assert torch.allclose(
        response_logits(bundle.student, padded),
        response_logits(bundle.student, unpadded),
        atol=1e-6,
    )


def test_old_and_current_student_probabilities_agree_before_update(
    local_engine: OpdEngine,
) -> None:
    bundle = local_engine.bundle
    request = RolloutRequest(
        example_batch(), local_engine.spec.prompt, SamplingSpec(2), 64
    )
    trajectory = generate_responses(bundle, request)[0]
    targets = local_engine.collect_targets((trajectory,))[0].targets
    bundle.student.train()
    current = (
        response_logits(bundle.student, trajectory)
        .float()
        .log_softmax(-1)
        .gather(-1, targets.candidate_ids)
    )
    assert torch.allclose(current, targets.behavior_log_probs, atol=1e-6)


def test_prompt_limit_rejects_silent_truncation(local_engine: OpdEngine) -> None:
    request = RolloutRequest(
        example_batch(), local_engine.spec.prompt, SamplingSpec(2), 1
    )
    with pytest.raises(ValueError, match="silent truncation"):
        generate_responses(local_engine.bundle, request)


def test_nonreentrant_checkpointing_preserves_lora_gradients(
    local_engine: OpdEngine,
) -> None:
    local_engine.bundle.student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    result = local_engine.step(example_batch(), 4)
    assert result.grad_norm > 0 and torch.isfinite(torch.tensor(result.loss))


def test_unequal_responses_use_the_global_token_mean(local_engine: OpdEngine) -> None:
    bundle = local_engine.bundle
    request = RolloutRequest(
        example_batch(), local_engine.spec.prompt, SamplingSpec(4), 64
    )
    encoded = tokenize_questions(bundle, request)
    generated = torch.cat(
        (encoded["input_ids"], torch.tensor([[7, 2, 0, 0], [7, 8, 9, 2]])), dim=1
    )
    trajectories = split_generated_responses(bundle, generated, (request, encoded))
    scored = local_engine.collect_targets(trajectories)
    # A single concatenated objective is an independent reference for microbatching.
    bundle.student.train()
    log_probs = torch.cat(
        [
            response_logits(bundle.student, item.trajectory)
            .float()
            .log_softmax(-1)
            .gather(-1, item.targets.candidate_ids)
            for item in scored
        ]
    )
    targets = CandidateTargets(
        candidate_ids=torch.cat([item.targets.candidate_ids for item in scored]),
        behavior_log_probs=torch.cat(
            [item.targets.behavior_log_probs for item in scored]
        ),
        teacher_log_probs=torch.cat(
            [item.targets.teacher_log_probs for item in scored]
        ),
        teacher_top_ids=torch.cat([item.targets.teacher_top_ids for item in scored]),
    )
    expected_loss = compute_clipped_opd_loss(log_probs, targets, local_engine.spec.opd)
    expected_loss.backward()
    expected_gradients = {
        name: value.grad.clone()
        for name, value in bundle.student.named_parameters()
        if value.requires_grad
    }
    # Capture gradients before clipping; production may rescale their shared norm.
    original_clip = torch.nn.utils.clip_grad_norm_
    captured: dict[str, torch.Tensor] = {}

    def capture_gradients(
        parameters: Iterable[torch.Tensor], max_norm: float, **kwargs: object
    ) -> torch.Tensor:
        captured.update(
            {
                name: value.grad.clone()
                for name, value in bundle.student.named_parameters()
                if value.requires_grad
            }
        )
        return original_clip(parameters, max_norm, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", capture_gradients)
        local_engine.update_student(scored)
    assert all(
        torch.allclose(captured[name], gradient, atol=1e-7, rtol=1e-5)
        for name, gradient in expected_gradients.items()
    )


def test_production_loader_fails_before_download_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = ModelLoadSpec(None, LoraSpec(2, 4, 0.0, ("q_proj",)), "sdpa", False)
    with pytest.raises(RuntimeError, match="CUDA"):
        load_model_bundle(spec)
