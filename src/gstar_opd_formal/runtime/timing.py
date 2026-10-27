"""Persist aggregate host spans without writing one file per decoded token."""

import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from gstar_opd_formal.storage.files import write_json


@dataclass
class Timeline:
    output: Path
    binding_sha256: str
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    wall_started: float = field(default_factory=time.time)
    durations: dict = field(default_factory=lambda: defaultdict(float))
    counts: dict = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def measure(self, stage: str):
        began = time.perf_counter()
        try:
            yield
        finally:
            self.durations[stage] += time.perf_counter() - began
            self.counts[stage] += 1

    def flush(self) -> dict:
        value = {
            "schema": "formal_timing_v1",
            "binding_sha256": self.binding_sha256,
            "segment_id": self.segment_id,
            "wall_started_at": self.wall_started,
            "worker_observed_seconds": time.perf_counter() - self.started,
            "stage_seconds_inclusive": dict(self.durations),
            "stage_calls": dict(self.counts),
            "scope": (
                "host_wall_aggregates_nested_spans_are_not_additive; "
                "excludes_unobserved_install_and_idle"
            ),
        }
        write_json(self.output / "timing" / f"{self.segment_id}.json", value)
        return value
