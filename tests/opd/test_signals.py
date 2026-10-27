"""Check the different position/candidate denominators in boundary statistics."""

import math
from dataclasses import astuple, replace

import pytest
import torch

from gstar_opd_formal.opd.objective import CandidateTargets, make_candidate_targets
from gstar_opd_formal.opd.signals import (
    BoundaryBatch,
    BoundarySignals,
    compute_boundary_signals,
    compute_utilization,
)


def make_boundary_batch() -> BoundaryBatch:
    # At the first boundary state only token 1 is shared. At the second state,
    # every candidate is shared. Non-boundary states have perfect agreement.
    student = torch.tensor([[0.6, 0.3, 0.1]] * 4, dtype=torch.float64)
    teacher = torch.tensor(
        [[0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1]],
        dtype=torch.float64,
    )
    targets = make_candidate_targets(student.log(), teacher.log(), 2)
    return BoundaryBatch(targets, torch.arange(4), 4, 2, (4,))


def test_boundary_excludes_prefix_positions() -> None:
    signals = compute_boundary_signals(make_boundary_batch())

    assert signals.overlap == pytest.approx(0.75)


def test_shared_mass_uses_full_vocabulary_probabilities() -> None:
    signals = compute_boundary_signals(make_boundary_batch())

    assert signals.shared_mass == pytest.approx(min((0.3 + 0.9) / 2, (0.5 + 0.9) / 2))


def test_shared_greedy_uses_highest_behavior_probability_shared_candidate() -> None:
    signals = compute_boundary_signals(make_boundary_batch())
    expected = max(0, -(1 / 3 * math.log(5 / 3) + 2 / 3 * math.log(1 / 2)) / 2)

    assert signals.shared_greedy_penalty == pytest.approx(expected)


def test_unshared_penalty_divides_by_candidate_entries() -> None:
    signals = compute_boundary_signals(make_boundary_batch())

    assert signals.unshared_penalty == pytest.approx(-2 / 3 * math.log(1 / 3))


def test_empty_boundary_returns_no_progress_observation() -> None:
    batch = make_boundary_batch()
    targets = CandidateTargets(
        candidate_ids=batch.targets.candidate_ids[:2],
        behavior_log_probs=batch.targets.behavior_log_probs[:2],
        teacher_log_probs=batch.targets.teacher_log_probs[:2],
        teacher_top_ids=batch.targets.teacher_top_ids[:2],
    )

    assert (
        compute_boundary_signals(
            replace(
                batch,
                targets=targets,
                response_positions=torch.arange(2),
                response_lengths=(2,),
            )
        )
        is None
    )


def test_disjoint_top_k_returns_no_progress_observation() -> None:
    student = torch.tensor([[0.6, 0.3, 0.05, 0.05]], dtype=torch.float64)
    teacher = torch.tensor([[0.05, 0.05, 0.3, 0.6]], dtype=torch.float64)
    targets = make_candidate_targets(student.log(), teacher.log(), 2)
    batch = BoundaryBatch(targets, torch.tensor([0]), 1, 1, (1,))

    assert compute_boundary_signals(batch) is None


def test_fully_shared_candidates_have_zero_unshared_penalty() -> None:
    targets = make_candidate_targets(torch.zeros(1, 2), torch.zeros(1, 2), 2)
    batch = BoundaryBatch(targets, torch.tensor([0]), 1, 1, (1,))

    assert compute_boundary_signals(batch) == BoundarySignals(1.0, 1.0, 0.0, 0.0)


def test_mass_minimum_and_greedy_clamp_happen_after_position_averaging() -> None:
    targets = CandidateTargets(
        candidate_ids=torch.tensor([[0], [0]]),
        behavior_log_probs=torch.tensor([[0.8], [0.2]], dtype=torch.float64).log(),
        teacher_log_probs=torch.tensor([[0.2], [0.8]], dtype=torch.float64).log(),
        teacher_top_ids=torch.tensor([[0], [0]]),
    )
    batch = BoundaryBatch(targets, torch.arange(2), 2, 2, (2,))

    assert compute_boundary_signals(batch) == BoundarySignals(1.0, 0.5, 0.0, 0.0)


def test_overlap_includes_empty_intersections_but_mass_excludes_them() -> None:
    targets = CandidateTargets(
        candidate_ids=torch.tensor([[0], [0], [0]]),
        behavior_log_probs=torch.tensor(
            [[0.8], [0.2], [0.7]], dtype=torch.float64
        ).log(),
        teacher_log_probs=torch.tensor(
            [[0.2], [0.8], [0.1]], dtype=torch.float64
        ).log(),
        teacher_top_ids=torch.tensor([[0], [0], [1]]),
    )
    batch = BoundaryBatch(targets, torch.arange(3), 3, 3, (3,))
    expected = BoundarySignals(2 / 3, 0.5, 0.0, math.log(7))

    assert astuple(compute_boundary_signals(batch)) == pytest.approx(astuple(expected))


def test_utilization_uses_inclusive_paper_thresholds() -> None:
    assert compute_utilization((1, 2, 3, 4), 4, 2) == (0.25, 0.75)


def test_first_stage_reach_threshold_is_zero() -> None:
    assert compute_utilization((0, 1, 2), 2, 2) == (1 / 3, 1.0)


def test_response_positions_restart_for_each_rollout() -> None:
    targets = make_candidate_targets(torch.zeros(3, 2), torch.zeros(3, 2), 2)
    batch = BoundaryBatch(targets, torch.tensor([0, 1, 0]), 2, 1, (2, 1))

    assert compute_boundary_signals(batch) == BoundarySignals(1.0, 1.0, 0.0, 0.0)


def test_prompt_or_padding_positions_are_rejected() -> None:
    batch = replace(
        make_boundary_batch(), response_positions=torch.tensor([1, 2, 3, 4])
    )

    with pytest.raises(ValueError, match="valid positions"):
        compute_boundary_signals(batch)


@pytest.mark.parametrize("lengths", [(), (-1,), (5,), (True,)])
def test_invalid_utilization_lengths_are_rejected(lengths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        compute_utilization(lengths, 4, 2)
