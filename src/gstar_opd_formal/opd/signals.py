"""Boundary agreement statistics from Adaptive FastOPD Eqs. (4)-(10), (17)-(18)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from gstar_opd_formal.opd.objective import (
    CandidateTargets,
    compute_candidate_advantages,
    select_candidate_targets,
    validate_candidate_targets,
)


@dataclass(frozen=True)
class BoundarySignals:
    overlap: float
    shared_mass: float
    shared_greedy_penalty: float
    unshared_penalty: float


@dataclass(frozen=True)
class BoundaryBatch:
    """Valid response rows flattened in rollout order, with zero-based positions."""

    targets: CandidateTargets
    response_positions: Tensor
    horizon: int
    increment: int
    response_lengths: tuple[int, ...]


def compute_boundary_signals(batch: BoundaryBatch) -> BoundarySignals | None:
    """Skip observations with no boundary positions or no shared candidates."""
    validate_boundary_batch(batch)
    boundary_mask = (batch.response_positions >= batch.horizon - batch.increment) & (
        batch.response_positions < batch.horizon
    )
    if not bool(boundary_mask.any()):
        return None

    targets = select_candidate_targets(batch.targets, boundary_mask)
    # Membership is by vocabulary ID; student and teacher top-k rank orders differ.
    shared = (
        targets.candidate_ids.unsqueeze(-1) == targets.teacher_top_ids.unsqueeze(-2)
    ).any(dim=-1)
    shared_rows = shared.any(dim=-1)
    if not bool(shared_rows.any()):
        return None

    advantages = compute_candidate_advantages(targets)
    student_mass = (targets.behavior_log_probs.detach().exp() * shared).sum(dim=-1)
    teacher_mass = (targets.teacher_log_probs.detach().exp() * shared).sum(dim=-1)

    # P averages only nonempty intersections, then takes min of the two averages.
    # Taking a per-row min first would implement a different statistic.
    shared_mass = torch.minimum(
        student_mass[shared_rows].mean(), teacher_mass[shared_rows].mean()
    )
    greedy_indices = (
        targets.behavior_log_probs.detach()
        .masked_fill(~shared, -torch.inf)
        .argmax(dim=-1)
    )
    greedy_advantages = advantages.gather(-1, greedy_indices.unsqueeze(-1)).squeeze(-1)

    # G/U clamp after averaging signed advantages, rather than averaging penalties.
    # Their denominators differ: G uses shared positions, U uses unshared candidates.
    greedy_penalty = (-greedy_advantages[shared_rows].mean()).clamp_min(0)
    unshared = ~shared
    unshared_penalty = (
        (-advantages[unshared].mean()).clamp_min(0)
        if bool(unshared.any())
        else advantages.new_zeros(())
    )
    return BoundarySignals(
        overlap=float(shared.to(dtype=torch.float32).mean().item()),
        shared_mass=float(shared_mass.item()),
        shared_greedy_penalty=float(greedy_penalty.item()),
        unshared_penalty=float(unshared_penalty.item()),
    )


def compute_utilization(
    response_lengths: tuple[int, ...], horizon: int, increment: int
) -> tuple[float, float]:
    """Compute unsmoothed horizon-hit and boundary-reach rates for all rollouts."""
    validate_response_lengths(response_lengths, horizon, increment)
    rollout_count = len(response_lengths)
    hit_rate = sum(length >= horizon for length in response_lengths) / rollout_count
    # Preserve the paper's inclusive Eq. (18), including the H == increment case.
    reach_rate = (
        sum(length >= horizon - increment for length in response_lengths)
        / rollout_count
    )
    return hit_rate, reach_rate


def validate_boundary_batch(batch: BoundaryBatch) -> None:
    validate_candidate_targets(batch.targets)
    validate_response_lengths(batch.response_lengths, batch.horizon, batch.increment)
    positions = batch.response_positions
    if positions.dtype != torch.long or positions.ndim != 1:
        raise ValueError("Response positions must be a one-dimensional int64 tensor")
    if positions.device != batch.targets.candidate_ids.device:
        raise ValueError("Response positions and candidate targets must share a device")
    if positions.shape[0] != batch.targets.candidate_ids.shape[0]:
        raise ValueError("Every candidate row must have one response position")

    # Enforce the causal row convention shared with the engine: EOS is a valid
    # response token, but prompt and post-EOS padding are absent from these rows.
    expected_positions = torch.cat(
        [
            torch.arange(length, device=positions.device)
            for length in batch.response_lengths
        ]
    )
    if not torch.equal(positions, expected_positions):
        raise ValueError(
            "Response rows must contain each rollout's valid positions in order"
        )


def validate_response_lengths(
    response_lengths: tuple[int, ...], horizon: int, increment: int
) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Horizon must be a positive integer")
    if (
        isinstance(increment, bool)
        or not isinstance(increment, int)
        or not 0 < increment <= horizon
    ):
        raise ValueError("Increment must be positive and no greater than the horizon")
    if not response_lengths:
        raise ValueError("Utilization needs at least one rollout")
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 0 <= length <= horizon
        for length in response_lengths
    ):
        raise ValueError("Response lengths must be integers between zero and horizon")
