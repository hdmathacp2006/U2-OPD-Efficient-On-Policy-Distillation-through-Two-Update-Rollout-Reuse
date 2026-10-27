"""One fresh on-policy candidate-distillation update per batch."""

from dataclasses import replace

from gstar_opd_formal.controllers.adaptive import HorizonObservation
from gstar_opd_formal.opd.engine import summarize_boundary
from gstar_opd_formal.opd.finish import validate_finish_flags
from gstar_opd_formal.opd.reuse_state import PendingBatch
from gstar_opd_formal.opd.rollout import (
    RolloutRequest,
    SamplingSpec,
    generate_response_microbatches,
)


class AdaptiveMethod:
    name = "adaptive"
    updates_per_fresh_batch = 1

    def fresh(self, engine: object, context: tuple) -> tuple:
        policy, examples = context
        horizon = policy.horizon
        with engine.ledger.measure_stage("student_rollout"):
            trajectories = generate_response_microbatches(
                engine.bundle,
                RolloutRequest(
                    examples,
                    engine.spec.prompt,
                    SamplingSpec(horizon),
                    engine.spec.max_prompt_tokens,
                ),
                engine.spec.generation_batch_size,
            )
            validate_finish_flags(trajectories, (engine.bundle.student.config, horizon))
        with engine.ledger.measure_stage("teacher_and_current_student_scoring"):
            if engine.bundle.rollout is not None:
                engine.bundle.rollout.prepare_hf_phase()
            scored = engine.collect_targets(trajectories)
        pending = PendingBatch(trajectories, None, engine.completed_steps, horizon)
        pending.validate(engine.completed_steps)
        return pending, scored

    def update(self, engine: object, context: tuple) -> tuple:
        policy, pending, scored = context
        pending.validate(engine.completed_steps)
        if pending.accepted_updates != 0 or scored is None:
            raise ValueError("Adaptive requires a fresh pre-update scored batch")
        engine.prefix_factors = {}
        with engine.ledger.measure_stage("student_update"):
            loss, norm = engine.update_student(scored)
        with engine.ledger.measure_stage("horizon_policy"):
            policy.observe(
                HorizonObservation(
                    summarize_boundary(
                        scored, (pending.horizon, engine.spec.horizon_increment)
                    ),
                    tuple(item.response_length for item in pending.trajectories),
                )
            )
        return replace(pending, accepted_updates=1), {
            "loss": loss,
            "gradient_norm": norm,
            "drift": [],
        }
