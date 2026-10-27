"""Small probability tables serve as independent numerical OPD references."""

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from gstar_opd_formal.contract_types import OpdSpec
from gstar_opd_formal.opd.objective import (
    CandidateTargets,
    MaskedCandidateBatch,
    compute_candidate_advantages,
    compute_clipped_opd_loss,
    compute_masked_clipped_opd_loss,
    make_candidate_targets,
)


def make_probability_targets(
    student: list[list[float]], teacher: list[list[float]], count: int
) -> CandidateTargets:
    return make_candidate_targets(
        torch.tensor(student, dtype=torch.float64).log(),
        torch.tensor(teacher, dtype=torch.float64).log(),
        count,
    )


def make_one_candidate_target(correction: float) -> CandidateTargets:
    return CandidateTargets(
        candidate_ids=torch.tensor([[0]]),
        behavior_log_probs=torch.tensor([[math.log(0.25)]], dtype=torch.float64),
        teacher_log_probs=torch.tensor(
            [[math.log(0.25) + correction]], dtype=torch.float64
        ),
        teacher_top_ids=torch.tensor([[0]]),
    )


def test_candidate_ids_come_from_behavior_student_not_teacher() -> None:
    targets = make_probability_targets([[0.6, 0.3, 0.1]], [[0.1, 0.3, 0.6]], 2)

    assert targets.candidate_ids.tolist() == [[0, 1]]


def test_teacher_probabilities_keep_full_vocabulary_normalization() -> None:
    targets = make_probability_targets([[0.6, 0.3, 0.1]], [[0.1, 0.3, 0.6]], 2)

    torch.testing.assert_close(
        targets.teacher_log_probs.exp(), torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    )


def test_advantage_weights_are_normalized_behavior_top_k_probabilities() -> None:
    targets = make_probability_targets([[0.7, 0.2, 0.1]], [[0.4, 0.5, 0.1]], 2)
    expected = torch.tensor(
        [[7 / 9 * math.log(4 / 7), 2 / 9 * math.log(5 / 2)]], dtype=torch.float64
    )

    torch.testing.assert_close(compute_candidate_advantages(targets), expected)


def test_loss_sums_candidates_instead_of_averaging_them() -> None:
    targets = make_probability_targets([[0.7, 0.2, 0.1]], [[0.4, 0.5, 0.1]], 2)
    loss = compute_clipped_opd_loss(
        targets.behavior_log_probs, targets, OpdSpec(2, 0.2, 1e-6)
    )
    expected = -(7 / 9 * math.log(4 / 7) + 2 / 9 * math.log(5 / 2))

    assert loss.item() == pytest.approx(expected)


def test_loss_averages_valid_response_positions_with_equal_weight() -> None:
    targets = make_probability_targets(
        [[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]],
        [[0.4, 0.5, 0.1], [0.5, 0.3, 0.2]],
        2,
    )
    loss = compute_clipped_opd_loss(
        targets.behavior_log_probs, targets, OpdSpec(2, 0.2, 1e-6)
    )
    first_correction = 7 / 9 * math.log(4 / 7) + 2 / 9 * math.log(5 / 2)
    second_correction = (math.log(5 / 4) + math.log(3 / 4)) / 2

    assert loss.item() == pytest.approx(-(first_correction + second_correction) / 2)


@pytest.mark.parametrize(
    ("ratio", "correction", "expected_multiplier"),
    [
        (2.0, math.log(2), 1.2),
        (0.5, math.log(2), 0.5),
        (0.5, -math.log(2), 0.8),
        (2.0, -math.log(2), 2.0),
    ],
)
def test_clipping_handles_both_advantage_signs(
    ratio: float, correction: float, expected_multiplier: float
) -> None:
    targets = make_one_candidate_target(correction)
    current = targets.behavior_log_probs + math.log(ratio)
    loss = compute_clipped_opd_loss(current, targets, OpdSpec(1, 0.2, 1e-6))

    assert loss.item() == pytest.approx(-expected_multiplier * correction)


def test_targets_are_detached_from_both_logit_inputs() -> None:
    student = torch.tensor([[0.2, 0.0, -0.4]], requires_grad=True)
    teacher = torch.tensor([[0.0, 0.3, -0.5]], requires_grad=True)
    targets = make_candidate_targets(student, teacher, 2)

    assert not any(
        tensor.requires_grad
        for tensor in (targets.behavior_log_probs, targets.teacher_log_probs)
    )


def test_only_current_student_receives_loss_gradients() -> None:
    targets = make_one_candidate_target(math.log(2))
    targets = replace(
        targets, teacher_log_probs=targets.teacher_log_probs.requires_grad_()
    )
    current = targets.behavior_log_probs.clone().requires_grad_()

    compute_clipped_opd_loss(current, targets, OpdSpec(1, 0.2, 1e-6)).backward()

    assert (current.grad.item(), targets.teacher_log_probs.grad) == pytest.approx(
        (-math.log(2), None)
    )


def test_clipped_improving_candidate_has_zero_gradient() -> None:
    targets = make_one_candidate_target(math.log(2))
    current = (targets.behavior_log_probs + math.log(2)).requires_grad_()

    compute_clipped_opd_loss(current, targets, OpdSpec(1, 0.2, 1e-6)).backward()

    assert current.grad.item() == pytest.approx(0.0)


def test_current_logit_gradient_matches_finite_differences() -> None:
    targets = make_probability_targets([[0.6, 0.3, 0.1]], [[0.4, 0.5, 0.1]], 2)
    current_logits = torch.tensor(
        [[math.log(0.6), math.log(0.3), math.log(0.1)]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def loss_from_logits(logits: Tensor) -> Tensor:
        log_probs = logits.log_softmax(dim=-1).gather(-1, targets.candidate_ids)
        return compute_clipped_opd_loss(log_probs, targets, OpdSpec(2, 0.2, 1e-6))

    assert torch.autograd.gradcheck(loss_from_logits, (current_logits,))


def test_nan_padding_is_removed_before_objective_math() -> None:
    targets = make_one_candidate_target(math.log(2))
    padded_targets = CandidateTargets(
        candidate_ids=torch.tensor([[0], [-1]]),
        behavior_log_probs=torch.cat(
            (
                targets.behavior_log_probs,
                torch.tensor([[math.nan]], dtype=torch.float64),
            )
        ),
        teacher_log_probs=torch.cat(
            (targets.teacher_log_probs, torch.tensor([[math.nan]], dtype=torch.float64))
        ),
        teacher_top_ids=torch.tensor([[0], [-1]]),
    )
    current = padded_targets.behavior_log_probs.clone().requires_grad_()
    batch = MaskedCandidateBatch(current, padded_targets, torch.tensor([True, False]))

    loss = compute_masked_clipped_opd_loss(batch, OpdSpec(1, 0.2, 1e-6))
    loss.backward()

    assert (loss.item(), current.grad.tolist()) == (
        pytest.approx(-math.log(2)),
        [[pytest.approx(-math.log(2))], [0.0]],
    )


def test_valid_nan_is_rejected_instead_of_silently_ignored() -> None:
    targets = make_one_candidate_target(math.log(2))

    with pytest.raises(ValueError, match="finite"):
        compute_clipped_opd_loss(
            torch.tensor([[math.nan]]), targets, OpdSpec(1, 0.2, 1e-6)
        )


def test_overflowing_ratio_is_rejected_instead_of_changing_the_objective() -> None:
    targets = make_one_candidate_target(math.log(2))
    targets = replace(
        targets,
        behavior_log_probs=torch.tensor([[-1000.0]]),
        teacher_log_probs=torch.tensor([[-999.0]]),
    )

    with pytest.raises(ValueError, match="importance ratio"):
        compute_clipped_opd_loss(torch.tensor([[0.0]]), targets, OpdSpec(1, 0.2, 1e-6))


def test_empty_response_mask_is_rejected() -> None:
    targets = make_one_candidate_target(math.log(2))
    batch = MaskedCandidateBatch(
        targets.behavior_log_probs, targets, torch.tensor([False])
    )

    with pytest.raises(ValueError, match="at least one"):
        compute_masked_clipped_opd_loss(batch, OpdSpec(1, 0.2, 1e-6))


@pytest.mark.parametrize("candidate_count", [0, 4, True])
def test_invalid_top_k_is_rejected(candidate_count: int) -> None:
    with pytest.raises(ValueError, match="top-k"):
        make_probability_targets([[0.6, 0.3, 0.1]], [[0.4, 0.5, 0.1]], candidate_count)
