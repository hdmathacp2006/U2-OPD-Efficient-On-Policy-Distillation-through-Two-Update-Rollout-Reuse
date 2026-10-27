"""Adaptive FastOPD horizon baseline implementing Eqs. 11-19.

Method reference: https://arxiv.org/html/2607.29494v1.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from gstar_opd_formal.opd.signals import BoundarySignals, compute_utilization

POLICY_STATE_VERSION = 1
SIGNAL_PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AdaptivePolicySpec:
    baseline_observations: int = 3
    ema_beta: float = 0.8
    plateau_patience: int = 5
    baseline_floor: float = 1e-6
    hit_threshold: float = 0.1
    reach_threshold: float = 0.3

    def __post_init__(self) -> None:
        validate_positive_integer(self.baseline_observations)
        validate_positive_integer(self.plateau_patience)
        validate_unit_interval(self.ema_beta)
        if self.ema_beta == 1:
            raise ValueError("EMA beta must be below one")
        validate_positive_finite(self.baseline_floor)
        validate_unit_interval(self.hit_threshold)
        validate_unit_interval(self.reach_threshold)


@dataclass(frozen=True)
class HorizonObservation:
    signals: BoundarySignals | None
    response_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            type(length) is not int or length < 0 for length in self.response_lengths
        ):
            raise ValueError("Response lengths must be nonnegative integers")


class AdaptiveHorizonPolicy:
    """Expand by one chunk after a stage-relative plateau AND utilization."""

    def __init__(
        self, choices: tuple[int, ...], spec: AdaptivePolicySpec | None = None
    ) -> None:
        self.choices = validate_horizons(choices)
        self.spec = spec or AdaptivePolicySpec()
        self.horizon_index = 0
        self.reset_stage()
        self.last_decision: dict[str, object] = {}

    @property
    def current_horizon(self) -> int:
        return self.choices[self.horizon_index]

    def select(self, context: HorizonObservation) -> int:
        return self.current_horizon

    def reset_stage(self) -> None:
        self.baseline_samples: list[tuple[float, ...]] = []
        self.baseline: tuple[float, ...] | None = None
        self.smoothed_badness: float | None = None
        self.best_badness: float | None = None
        self.plateau_count = 0
        self.hit_ema: float | None = None
        self.reach_ema: float | None = None

    def observe(self, observation: HorizonObservation) -> int:
        """Consume one optimizer-step batch, retaining progress on invalid data."""
        observed_horizon = self.current_horizon
        if observation.response_lengths:
            hit, reach = compute_utilization(
                observation.response_lengths, self.current_horizon, self.increment
            )
            self.hit_ema = update_ema(self.hit_ema, hit, self.spec.ema_beta)
            self.reach_ema = update_ema(self.reach_ema, reach, self.spec.ema_beta)
        badness = compute_badness(observation.signals)
        if badness is None:
            self.capture_decision(
                observed_horizon, "invalid_signal_retain_progress", None
            )
            return self.current_horizon

        if self.baseline is None:
            self.baseline_samples.append(badness)
            if len(self.baseline_samples) < self.spec.baseline_observations:
                self.capture_decision(observed_horizon, "baseline_warmup", badness)
                return self.current_horizon
            self.baseline = tuple(
                max(self.spec.baseline_floor, float(np.mean(values)))
                for values in zip(*self.baseline_samples, strict=True)
            )

        # The paper does not specify warmup replay. Initialize EMA on the Nth
        # valid observation, when the stage baseline first becomes available;
        # warmup steps cannot increment the plateau counter.

        # Eqs. 13-14 use the maximum: one stubborn signal can prevent a false
        # declaration of overall progress based only on the improving metrics.
        aggregate = max(
            math.log1p(value / base)
            for value, base in zip(badness, self.baseline, strict=True)
        )
        self.smoothed_badness = update_ema(
            self.smoothed_badness, aggregate, self.spec.ema_beta
        )
        if self.best_badness is None or self.smoothed_badness < self.best_badness:
            self.best_badness = self.smoothed_badness
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        utilized = (
            self.hit_ema is not None and self.hit_ema >= self.spec.hit_threshold
        ) or (
            self.reach_ema is not None and self.reach_ema >= self.spec.reach_threshold
        )
        if (
            self.plateau_count >= self.spec.plateau_patience
            and utilized
            and self.horizon_index + 1 < len(self.choices)
        ):
            self.capture_decision(
                observed_horizon, "expand_plateau_and_utilization", badness
            )
            self.last_decision["aggregate_badness"] = aggregate
            self.last_decision["next_horizon"] = self.choices[self.horizon_index + 1]
            self.horizon_index += 1
            self.reset_stage()
        else:
            self.capture_decision(observed_horizon, "hold", badness)
            self.last_decision["aggregate_badness"] = aggregate
        return self.current_horizon

    def capture_decision(
        self, horizon: int, reason: str, badness: tuple | None
    ) -> None:
        # Capture the stage state BEFORE expansion resets its baselines and EMAs.
        self.last_decision = {
            "kind": "adaptive",
            "horizon": horizon,
            "next_horizon": horizon,
            "reason": reason,
            "badness": badness,
            "baseline": self.baseline,
            "baseline_observations": len(self.baseline_samples),
            "normalized_badness": None
            if badness is None or self.baseline is None
            else [
                value / base for value, base in zip(badness, self.baseline, strict=True)
            ],
            "smoothed_badness": self.smoothed_badness,
            "historical_best_badness": self.best_badness,
            "plateau_count": self.plateau_count,
            "hit_ema": self.hit_ema,
            "reach_ema": self.reach_ema,
            "warmup_convention": "initialize_ema_on_third_valid_observation_no_replay",
        }

    @property
    def increment(self) -> int:
        return (
            self.choices[1] - self.choices[0]
            if len(self.choices) > 1
            else self.choices[0]
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": POLICY_STATE_VERSION,
            "kind": "adaptive_fastopd",
            "choices": list(self.choices),
            "spec": asdict(self.spec),
            "horizon_index": self.horizon_index,
            "baseline_samples": [list(sample) for sample in self.baseline_samples],
            "baseline": list(self.baseline) if self.baseline is not None else None,
            "smoothed_badness": self.smoothed_badness,
            "best_badness": self.best_badness,
            "plateau_count": self.plateau_count,
            "hit_ema": self.hit_ema,
            "reach_ema": self.reach_ema,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        validate_state_identity(state, self.state_dict())
        index = read_horizon_index(state, len(self.choices))
        samples = np.asarray(state["baseline_samples"], dtype=float)
        if samples.size and (samples.ndim != 2 or samples.shape[1] != 4):
            raise ValueError("Invalid baseline sample shape")
        if not np.isfinite(samples).all() or (samples < 0).any():
            raise ValueError("Baseline samples must be finite and nonnegative")
        if len(samples) > self.spec.baseline_observations:
            raise ValueError("Too many stage baseline samples")
        baseline_value = state["baseline"]
        baseline = None
        if baseline_value is not None:
            vector = read_vector(baseline_value, 4)
            if (vector < self.spec.baseline_floor).any():
                raise ValueError("Stage baseline is below its floor")
            baseline = tuple(float(value) for value in vector)
        if (baseline is not None) != (len(samples) == self.spec.baseline_observations):
            raise ValueError("Stage baseline and warmup samples disagree")
        if baseline is not None:
            expected_baseline = np.maximum(
                self.spec.baseline_floor, samples.mean(axis=0)
            )
            if not np.allclose(baseline, expected_baseline, rtol=1e-12, atol=1e-12):
                raise ValueError("Stage baseline differs from its recorded warmup")
        smoothed = read_optional_nonnegative(state["smoothed_badness"])
        best = read_optional_nonnegative(state["best_badness"])
        if (smoothed is None) != (best is None) or (baseline is None) != (best is None):
            raise ValueError("Progress state is inconsistent")
        if best is not None and smoothed is not None and best > smoothed:
            raise ValueError("Best badness cannot exceed current smoothed badness")
        plateau = state["plateau_count"]
        if type(plateau) is not int or plateau < 0:
            raise ValueError("Plateau count must be a nonnegative integer")
        hit = read_optional_nonnegative(state["hit_ema"])
        reach = read_optional_nonnegative(state["reach_ema"])
        if (hit is None) != (reach is None) or any(
            value is not None and value > 1 for value in (hit, reach)
        ):
            raise ValueError("Utilization EMAs must be paired rates")
        self.horizon_index = index
        self.baseline_samples = [tuple(row) for row in samples.tolist()]
        self.baseline = baseline
        self.smoothed_badness = smoothed
        self.best_badness = best
        self.plateau_count = plateau
        self.hit_ema = hit
        self.reach_ema = reach


def compute_badness(signals: BoundarySignals | None) -> tuple[float, ...] | None:
    if signals is None:
        return None
    values = (
        signals.overlap,
        signals.shared_mass,
        signals.shared_greedy_penalty,
        signals.unshared_penalty,
    )
    if any(value is None for value in values):
        return None
    overlap, mass, greedy, unshared = (float(value) for value in values)
    overlap = clamp_signal_probability(overlap)
    mass = clamp_signal_probability(mass)
    if any(not math.isfinite(value) or value < 0 for value in (greedy, unshared)):
        raise ValueError("Boundary penalties must be finite and nonnegative")
    return 1 - overlap, 1 - mass, greedy, unshared


def clamp_signal_probability(value: float) -> float:
    """Accept float32 softmax rounding without accepting invalid probabilities."""
    if not math.isfinite(value) or not (
        -SIGNAL_PROBABILITY_TOLERANCE <= value <= 1 + SIGNAL_PROBABILITY_TOLERANCE
    ):
        raise ValueError("Boundary probability must be within [0, 1]")
    return min(1.0, max(0.0, value))


def update_ema(previous: float | None, value: float, beta: float) -> float:
    # The algebraically equivalent increment form preserves exact equality on
    # a constant stream; weighted sums can create a one-ulp false new minimum.
    return value if previous is None else previous + (1 - beta) * (value - previous)


def validate_horizons(choices: tuple[int, ...]) -> tuple[int, ...]:
    if not choices:
        raise ValueError("At least one horizon is required")
    for choice in choices:
        validate_positive_integer(choice)
    increments = {
        right - left for left, right in zip(choices, choices[1:], strict=False)
    }
    if increments and (len(increments) != 1 or min(increments) <= 0):
        raise ValueError("Horizons must increase by one fixed positive increment")
    return tuple(choices)


def validate_positive_integer(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("Expected a positive integer")


def validate_positive_finite(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive value")


def validate_unit_interval(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite value in [0, 1]")


def validate_state_identity(
    state: dict[str, object], expected: dict[str, object]
) -> None:
    for name in ("version", "kind", "choices", "spec"):
        if state.get(name) != expected[name]:
            raise ValueError(f"Policy checkpoint differs in {name}")


def read_horizon_index(state: dict[str, object], arm_count: int) -> int:
    index = state["horizon_index"]
    if type(index) is not int or not 0 <= index < arm_count:
        raise ValueError("Invalid horizon index")
    return index


def read_vector(value: object, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("Invalid finite checkpoint vector")
    return vector.copy()


def read_optional_nonnegative(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected finite nonnegative monitoring state")
    return result
