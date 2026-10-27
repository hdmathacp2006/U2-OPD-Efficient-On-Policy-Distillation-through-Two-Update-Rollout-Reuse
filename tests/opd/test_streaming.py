"""Check actual tiny Qwen decoder/head alignment rather than mocking the math."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gstar_opd_formal.opd.engine import OpdEngine, StreamingEngineSpec
from gstar_opd_formal.opd.objective import make_candidate_targets
from gstar_opd_formal.opd.rollout import ResponseTrajectory, response_logits
from gstar_opd_formal.opd.streaming import (
    native_teacher_trajectory,
    project_targets,
    response_hidden_states,
    score_streamed_batch,
)


def trajectory_for(length: int) -> ResponseTrajectory:
    ids = torch.tensor([[3, 4, 5] + [6] * length])
    mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
    mask[:, 2:] = True
    return ResponseTrajectory(
        "1",
        ids,
        torch.ones_like(ids),
        mask,
        torch.arange(length),
        length,
        "4",
        False,
        question="2+2?",
    )


@pytest.mark.parametrize("lengths", [(1, 3), (4, 2), (3, 3)])
def test_streamed_targets_match_full_vocabulary(local_engine, lengths):
    bundle = local_engine.bundle
    bundle.student.eval()
    trajectories = tuple(trajectory_for(length) for length in lengths)
    actual = score_streamed_batch(bundle, trajectories, (3, 2))
    expected = tuple(
        make_candidate_targets(
            response_logits(bundle.student, item),
            response_logits(bundle.teacher, item),
            3,
        )
        for item in trajectories
    )
    assert all(
        torch.allclose(
            getattr(left, field), getattr(right, field), atol=1e-6, rtol=1e-6
        )
        for left, right in zip(actual, expected, strict=True)
        for field in vars(left)
    )


def test_native_teacher_preserves_response_ids(local_engine):
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **options: "native prompt",
        encode=lambda rendered, **options: [7, 8, 9, 10, 11],
    )
    bundle = replace(local_engine.bundle, teacher_tokenizer=tokenizer)
    source = trajectory_for(3)
    teacher = native_teacher_trajectory(bundle, source)
    assert torch.equal(teacher.input_ids[0, -3:], source.input_ids[0, -3:])


def test_native_teacher_first_response_is_predicted_after_its_prefix(local_engine):
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **options: "native prompt",
        encode=lambda rendered, **options: [7, 8, 9, 10, 11],
    )
    teacher = native_teacher_trajectory(
        replace(local_engine.bundle, teacher_tokenizer=tokenizer), trajectory_for(3)
    )
    assert teacher.response_mask.nonzero()[:, 1].tolist() == [4, 5, 6]


def test_native_teacher_rejects_missing_question(local_engine):
    bundle = replace(local_engine.bundle, teacher_tokenizer=SimpleNamespace())
    with pytest.raises(ValueError, match="original question"):
        native_teacher_trajectory(bundle, replace(trajectory_for(2), question=""))


def test_native_teacher_batch_has_correct_causal_alignment(local_engine):
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **options: "native prompt",
        encode=lambda rendered, **options: [7, 8, 9, 10, 11],
    )
    bundle = replace(local_engine.bundle, teacher_tokenizer=tokenizer)
    trajectories = (trajectory_for(4), trajectory_for(2))
    actual = score_streamed_batch(bundle, trajectories, (3, 2))
    expected = tuple(
        make_candidate_targets(
            response_logits(bundle.student, item),
            response_logits(bundle.teacher, native_teacher_trajectory(bundle, item)),
            3,
        )
        for item in trajectories
    )
    assert all(
        torch.allclose(
            left.teacher_log_probs, right.teacher_log_probs, atol=1e-6, rtol=1e-6
        )
        for left, right in zip(actual, expected, strict=True)
    )


def test_full_vocabulary_normalization_is_not_top_k_normalization():
    head = torch.nn.Linear(4, 17, bias=False)
    hidden = torch.randn(7, 4)
    actual = project_targets((head, head), (hidden, hidden), (3, 2))
    assert bool((actual.teacher_log_probs.exp().sum(-1) < 1).all())


def test_response_hidden_projection_matches_native_head(local_engine):
    model = local_engine.bundle.student
    model.eval()
    trajectory = trajectory_for(3)
    hidden = response_hidden_states(model, (trajectory,))[0]
    assert torch.allclose(
        model.get_base_model().lm_head(hidden),
        response_logits(model, trajectory),
        atol=1e-7,
    )


def test_streamed_and_dense_targets_give_same_gradients(local_engine):
    engine = OpdEngine(
        local_engine.bundle,
        StreamingEngineSpec(
            **vars(local_engine.spec),
            teacher_scoring_batch_size=2,
            scoring_chunk_rows=2,
        ),
    )
    trajectory = trajectory_for(3)
    engine.bundle.student.eval()
    with torch.no_grad():
        dense = make_candidate_targets(
            response_logits(engine.bundle.student, trajectory),
            response_logits(engine.bundle.teacher, trajectory),
            engine.spec.opd.candidate_top_k,
        )
    streamed = engine.collect_targets((trajectory,))[0]
    from gstar_opd_formal.opd.engine import ScoredTrajectory

    gradients = []
    for item in (ScoredTrajectory(trajectory, dense), streamed):
        engine.optimizer.zero_grad(set_to_none=True)
        engine.backward_trajectory(item, trajectory.response_length)
        gradients.append(
            torch.cat(
                [
                    value.grad.flatten()
                    for value in engine.bundle.student.parameters()
                    if value.requires_grad
                ]
            )
        )
    assert torch.allclose(*gradients, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_bounded_backward_and_adam_match_independent_dense_engine(
    engine_factory, checkpointing
):
    reference = engine_factory()
    candidate = engine_factory()
    if checkpointing:
        candidate.bundle.student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    bounded = OpdEngine(candidate.bundle, StreamingEngineSpec(**vars(candidate.spec)))
    scored = reference.collect_targets((trajectory_for(129), trajectory_for(3)))
    expected_loss = reference.calculate_gradients(scored)
    actual_loss = bounded.calculate_gradients(scored)
    expected_gradients = tuple(
        value.grad.clone()
        for value in reference.bundle.student.parameters()
        if value.requires_grad
    )
    actual_gradients = tuple(
        value.grad.clone()
        for value in bounded.bundle.student.parameters()
        if value.requires_grad
    )
    reference.commit_gradients()
    bounded.commit_gradients()
    torch.testing.assert_close(
        (
            actual_loss,
            actual_gradients,
            bounded.optimizer.state_dict(),
            tuple(bounded.bundle.student.parameters()),
        ),
        (
            expected_loss,
            expected_gradients,
            reference.optimizer.state_dict(),
            tuple(reference.bundle.student.parameters()),
        ),
        atol=1e-6,
        rtol=1e-5,
    )
