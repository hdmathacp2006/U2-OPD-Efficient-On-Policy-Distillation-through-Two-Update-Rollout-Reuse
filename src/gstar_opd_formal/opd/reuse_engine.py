"""Reuse A8 models, scoring, optimizer and first-step gradient machinery."""

import random
from dataclasses import replace

import numpy as np
import torch

from gstar_opd_formal.models.loading import load_model_bundle
from gstar_opd_formal.opd.engine import OpdEngine, move_candidate_targets
from gstar_opd_formal.opd.probabilities import projected_selected_log_probs
from gstar_opd_formal.opd.reuse_objective import (
    WeightedCandidates,
    weighted_candidate_loss,
)
from gstar_opd_formal.opd.reuse_state import PendingBatch, trajectory_on
from gstar_opd_formal.opd.streaming import (
    causal_parts,
    response_hidden_states,
    sampled_log_probs,
)


class ReuseEngine(OpdEngine):
    def __init__(self, bundle: object, spec: object) -> None:
        super().__init__(bundle, spec)
        self.prefix_factors: dict[str, torch.Tensor] = {}

    def backward_trajectory(self, item: object, total_tokens: int) -> float:
        if item.trajectory.example_id not in self.prefix_factors:
            return super().backward_trajectory(item, total_tokens)
        targets = move_candidate_targets(item.targets, self.bundle.device)
        hidden = response_hidden_states(self.bundle.student, (item.trajectory,))[0]
        current = projected_selected_log_probs(
            causal_parts(self.bundle.student)[1], hidden, targets.candidate_ids
        )
        weights = self.prefix_factors[item.trajectory.example_id].to(current.device)
        loss = weighted_candidate_loss(
            WeightedCandidates(current, targets, weights), self.spec.opd
        )
        scaled = loss * (item.trajectory.response_length / total_tokens)
        scaled.backward()
        return float(scaled.detach())


def create_reuse_engine(settings: object) -> ReuseEngine:
    random.seed(settings.run_seed)
    np.random.seed(settings.run_seed)
    torch.manual_seed(settings.run_seed)
    engine = ReuseEngine(load_model_bundle(settings.model_load), settings.engine)
    if settings.rollout.backend == "vllm":
        from gstar_opd_formal.models.graph_rollout import GraphRollout as VllmRollout

        engine.bundle = replace(
            engine.bundle, rollout=VllmRollout(engine, settings, settings.run_seed)
        )
    return engine


@torch.no_grad()
def chosen_probabilities(engine: ReuseEngine, trajectories: tuple) -> tuple:
    engine.bundle.student.eval()
    values = []
    for item in trajectories:
        hidden = response_hidden_states(engine.bundle.student, (item,))[0]
        chosen = item.input_ids[0, -item.response_length :]
        values.append(
            sampled_log_probs(engine.bundle.student, hidden, chosen).detach().cpu()
        )
    return tuple(values)


def refresh_targets(engine: ReuseEngine, batch: PendingBatch) -> tuple:
    batch.validate(engine.completed_steps)
    # Old vLLM likelihoods belong to mu. Checking them against the new student
    # would confuse intentional policy drift with backend numerical mismatch.
    trajectories = tuple(
        replace(
            trajectory_on(t, engine.bundle.device),
            rollout_log_probs=None,
            rollout_adapter_identity=None,
        )
        for t in batch.trajectories
    )
    with engine.ledger.measure_stage("teacher_and_current_student_scoring"):
        scored = engine.collect_targets(trajectories)
    with engine.ledger.measure_stage("reuse_probability_observer"):
        current_chosen = chosen_probabilities(engine, trajectories)
    return scored, current_chosen
