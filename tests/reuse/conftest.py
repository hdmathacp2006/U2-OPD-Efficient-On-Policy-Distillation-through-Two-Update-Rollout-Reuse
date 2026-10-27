"""Deterministic small-model U2 fixtures without remote execution."""

from dataclasses import replace

import pytest
import torch

from gstar_opd_formal.opd.engine import StreamingEngineSpec
from gstar_opd_formal.opd.reuse_engine import ReuseEngine
from gstar_opd_formal.opd.rollout import ResponseTrajectory


class FixtureRollout:
    peak_memory_bytes = 0

    def __init__(self, trajectory: ResponseTrajectory):
        self.trajectory = trajectory
        self.generations = 0

    def generate(self, request):
        self.generations += 1
        return tuple(
            replace(self.trajectory, example_id=str(e.source_index))
            for e in request.examples
        )

    def prepare_hf_phase(self):
        return None

    def sample_memory(self):
        return None

    def close(self):
        return None


@pytest.fixture
def trajectory():
    ids = torch.tensor([[0, 0, 3, 4, 5, 6, 2]])
    return ResponseTrajectory(
        "1",
        ids,
        torch.tensor([[0, 0, 1, 1, 1, 1, 1]]),
        torch.tensor([[False, False, False, True, True, True]]),
        torch.arange(3),
        3,
        r"\boxed{2}",
        False,
        question="1+1?",
    )


@pytest.fixture
def engine(local_engine, trajectory):
    rollout = FixtureRollout(trajectory)
    bundle = replace(local_engine.bundle, rollout=rollout)
    spec = StreamingEngineSpec(
        **{**vars(local_engine.spec), "horizon_increment": 8},
        teacher_scoring_batch_size=1,
        scoring_chunk_rows=2,
    )
    return ReuseEngine(bundle, spec)
