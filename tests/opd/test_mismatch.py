"""Sampled-token likelihood checks cover alignment, chunking and stale policies."""

import pytest
import torch

from gstar_opd_formal.models.rollout_spec import LogprobThresholds
from gstar_opd_formal.opd.mismatch import check_rollout_log_probs
from gstar_opd_formal.opd.rollout import ResponseTrajectory


def trajectory_for(logits: torch.Tensor) -> ResponseTrajectory:
    count = logits.shape[0]
    return ResponseTrajectory(
        "0",
        torch.zeros(1, count + 1, dtype=torch.long),
        torch.ones(1, count + 1),
        torch.ones(1, count, dtype=torch.bool),
        torch.arange(count),
        count,
        "",
        False,
        logits.log_softmax(-1)[:, 0],
    )


def test_matching_likelihoods_pass_across_chunk_boundary() -> None:
    logits = torch.zeros(257, 8)
    report = check_rollout_log_probs(
        logits, trajectory_for(logits), LogprobThresholds()
    )
    assert report["mean_absolute_difference"] == 0


def test_significant_mismatch_is_rejected_before_an_update() -> None:
    logits = torch.zeros(3, 8)
    trajectory = trajectory_for(logits)
    trajectory.rollout_log_probs.add_(0.11)
    with pytest.raises(RuntimeError, match="adapter synchronization"):
        check_rollout_log_probs(logits, trajectory, LogprobThresholds())


def test_nonfinite_sampler_likelihood_is_rejected() -> None:
    logits = torch.zeros(3, 8)
    trajectory = trajectory_for(logits)
    trajectory.rollout_log_probs[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        check_rollout_log_probs(logits, trajectory, LogprobThresholds())


def test_observed_bf16_difference_emits_warning_without_stopping() -> None:
    logits = torch.zeros(1015, 8)
    trajectory = trajectory_for(logits)
    trajectory.rollout_log_probs.add_(0.0210206)
    report = check_rollout_log_probs(logits, trajectory, LogprobThresholds())
    assert report["warning"] and report["mean_absolute_difference"] == pytest.approx(
        0.0210206, abs=1e-6
    )


@pytest.mark.parametrize("length", [1024, 2048, 3072, 4096])
def test_warning_is_not_increased_by_response_length(length: int) -> None:
    logits = torch.zeros(length, 8)
    trajectory = trajectory_for(logits)
    trajectory.rollout_log_probs.add_(0.0210206)
    assert check_rollout_log_probs(logits, trajectory, LogprobThresholds())["warning"]


def test_response_and_logits_length_must_match() -> None:
    logits = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="logits must align"):
        check_rollout_log_probs(logits[:2], trajectory_for(logits), LogprobThresholds())
