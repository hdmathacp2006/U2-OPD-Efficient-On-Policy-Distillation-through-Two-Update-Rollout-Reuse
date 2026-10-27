"""Measure synchronized stage timings and charge complete blocks once."""

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from time import perf_counter

import torch


@dataclass
class CostLedger:
    device: torch.device
    charged_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    peak_memory_bytes: int = 0

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure_stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        self.synchronize()
        try:
            yield
        finally:
            # perf_counter alone would measure CUDA launches rather than completed work.
            self.synchronize()
            elapsed = perf_counter() - start
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + elapsed
            if self.device.type == "cuda":
                self.peak_memory_bytes = max(
                    self.peak_memory_bytes, torch.cuda.max_memory_allocated(self.device)
                )

    def state_dict(self) -> dict[str, object]:
        return {
            "charged_seconds": self.charged_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "peak_memory_bytes": self.peak_memory_bytes,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        charged_seconds = read_nonnegative_seconds(
            state["charged_seconds"], "Charged seconds"
        )
        stages = state["stage_seconds"]
        if not isinstance(stages, Mapping) or any(
            not isinstance(name, str) for name in stages
        ):
            raise ValueError("Stage costs must map stage names to seconds")
        stage_seconds = {
            name: read_nonnegative_seconds(value, f"Stage {name} seconds")
            for name, value in stages.items()
        }
        peak_memory_bytes = state["peak_memory_bytes"]
        if (
            isinstance(peak_memory_bytes, bool)
            or not isinstance(peak_memory_bytes, int)
            or peak_memory_bytes < 0
        ):
            raise ValueError("Peak memory must be a nonnegative integer")

        # A rejected checkpoint must leave the running budget ledger intact.
        self.charged_seconds = charged_seconds
        self.stage_seconds = stage_seconds
        self.peak_memory_bytes = peak_memory_bytes


def read_nonnegative_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return seconds
