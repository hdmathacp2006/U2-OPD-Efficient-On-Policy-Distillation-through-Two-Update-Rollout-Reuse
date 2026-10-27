import json

import pytest

from gstar_opd_formal.controllers.adaptive import (
    AdaptiveHorizonPolicy,
    AdaptivePolicySpec,
    HorizonObservation,
    compute_utilization,
)
from gstar_opd_formal.opd.signals import BoundarySignals

CHOICES = (512, 1024, 1536, 2048)
SIGNALS = BoundarySignals(0.5, 0.4, 0.2, 0.3)


def boundary_observation(length: int) -> HorizonObservation:
    return HorizonObservation(SIGNALS, (length,))


def test_baseline_warmup_does_not_count_toward_plateau() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(3):
        policy.observe(boundary_observation(512))
    assert policy.plateau_count == 0


def test_constant_signals_expand_after_exact_patience() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    horizons = [policy.observe(boundary_observation(512)) for _ in range(8)]
    assert horizons == [512] * 7 + [1024]


def test_new_stage_resets_progress_and_utilization() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(8):
        policy.observe(boundary_observation(512))
    assert (
        policy.baseline_samples,
        policy.baseline,
        policy.smoothed_badness,
        policy.best_badness,
        policy.plateau_count,
        policy.hit_ema,
        policy.reach_ema,
    ) == ([], None, None, None, 0, None, None)


def test_missing_boundary_retains_progress_counter() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(5):
        policy.observe(boundary_observation(512))
    policy.observe(HorizonObservation(None, (100,)))
    assert policy.plateau_count == 2


def test_strict_new_minimum_resets_plateau() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(5):
        policy.observe(boundary_observation(512))
    better = BoundarySignals(0.9, 0.8, 0.01, 0.01)
    policy.observe(HorizonObservation(better, (512,)))
    assert policy.plateau_count == 0


def test_max_aggregation_keeps_a_stubborn_metric() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(3):
        policy.observe(boundary_observation(512))
    better_except_unshared = BoundarySignals(0.9, 0.8, 0.01, 0.3)
    policy.observe(HorizonObservation(better_except_unshared, (512,)))
    assert policy.plateau_count == 1


def test_plateau_without_utilization_cannot_expand_second_stage() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(8):
        policy.observe(boundary_observation(512))
    for _ in range(12):
        policy.observe(boundary_observation(100))
    assert policy.current_horizon == 1024


def test_boundary_reach_gate_is_an_or_with_hit_gate() -> None:
    policy = AdaptiveHorizonPolicy(
        CHOICES, AdaptivePolicySpec(hit_threshold=1.0, reach_threshold=0.3)
    )
    for _ in range(8):
        policy.observe(boundary_observation(512))
    for _ in range(8):
        policy.observe(boundary_observation(512))
    assert policy.current_horizon == 1536


def test_reach_equation_includes_exact_boundary_start() -> None:
    assert compute_utilization((511, 512, 1024), 1024, 512) == (1 / 3, 2 / 3)


def test_maximum_horizon_does_not_overflow() -> None:
    policy = AdaptiveHorizonPolicy((512,))
    for _ in range(20):
        policy.observe(boundary_observation(512))
    assert policy.current_horizon == 512


def test_softmax_roundoff_does_not_reject_shared_mass() -> None:
    policy = AdaptiveHorizonPolicy(CHOICES)
    rounded_signals = BoundarySignals(1.0, 1.0000001, 0, 0)
    policy.observe(HorizonObservation(rounded_signals, (512,)))
    assert policy.baseline_samples[0] == (0, 0, 0, 0)


def test_baseline_checkpoint_restores_future_transitions() -> None:
    original = AdaptiveHorizonPolicy(CHOICES)
    for _ in range(6):
        original.observe(boundary_observation(512))
    restored = AdaptiveHorizonPolicy(CHOICES)
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
    original.observe(boundary_observation(512))
    restored.observe(boundary_observation(512))
    assert restored.state_dict() == original.state_dict()


def test_baseline_rejects_checkpoint_for_other_hyperparameters() -> None:
    original = AdaptiveHorizonPolicy(CHOICES)
    restored = AdaptiveHorizonPolicy(CHOICES, AdaptivePolicySpec(ema_beta=0.6))
    with pytest.raises(ValueError, match="spec"):
        restored.load_state_dict(original.state_dict())
