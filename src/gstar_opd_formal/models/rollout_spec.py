"""Explicit inference-engine settings, separate from the HF training backend."""

import math
from dataclasses import dataclass

VLLM_VERSION = "0.10.2"
LOGPROB_GUARD_VERSION = 2


@dataclass(frozen=True)
class LogprobThresholds:
    warning: float = 0.02
    failure: float = 0.10

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in (self.warning, self.failure)
            )
            or self.warning >= self.failure
        ):
            raise ValueError("Log-probability thresholds require 0 < warning < failure")


@dataclass(frozen=True)
class RolloutSpec:
    backend: str = "hf"
    maximum_model_length: int = 9216
    cache_bytes: int = 2684354560
    maximum_sequences: int = 4
    logprob_tolerance: float = 0.10
    logprob_warning_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.backend not in {"hf", "vllm"}:
            raise ValueError("rollout backend must be hf or vllm")
        for value in (
            self.maximum_model_length,
            self.cache_bytes,
            self.maximum_sequences,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "vLLM dimensions and cache bytes must be positive integers"
                )
        LogprobThresholds(self.logprob_warning_threshold, self.logprob_tolerance)

    @property
    def logprob_thresholds(self) -> LogprobThresholds:
        return LogprobThresholds(self.logprob_warning_threshold, self.logprob_tolerance)


def parse_rollout_spec(config: dict[str, object]) -> RolloutSpec:
    raw = config.get("rollout", {})
    if not isinstance(raw, dict):
        raise ValueError("rollout settings must be a mapping")
    spec_type = ResidentRolloutSpec if "residency" in raw else RolloutSpec
    return spec_type(**raw)


@dataclass(frozen=True)
class ResidentRolloutSpec(RolloutSpec):
    residency: str = "resident_with_pressure_fallback"
    maximum_batched_tokens: int = 2048
    free_memory_fraction: float = 0.12
    hf_workspace_bytes: int = 6442450944

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.residency not in {
            "resident_with_pressure_fallback",
            "sleep_before_hf",
            "sleep",
        }:
            raise ValueError("Unknown vLLM residency policy")
        if (
            type(self.maximum_batched_tokens) is not int
            or self.maximum_batched_tokens <= 0
        ):
            raise ValueError("Batched token limit must be positive")
        if not 0.10 <= self.free_memory_fraction <= 0.20:
            raise ValueError("Device memory safety margin must be 10 to 20 percent")
        if type(self.hf_workspace_bytes) is not int or self.hf_workspace_bytes <= 0:
            raise ValueError("HF workspace reserve must be positive")
