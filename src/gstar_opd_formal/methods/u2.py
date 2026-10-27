"""Original fixed-two U2: fresh first update and one refreshed reuse update."""

from dataclasses import replace

from gstar_opd_formal.methods.adaptive import AdaptiveMethod
from gstar_opd_formal.opd.reuse_engine import chosen_probabilities, refresh_targets
from gstar_opd_formal.opd.reuse_objective import prefix_weights


class U2Method(AdaptiveMethod):
    name = "u2"
    updates_per_fresh_batch = 2

    def fresh(self, engine: object, context: tuple) -> tuple:
        pending, scored = super().fresh(engine, context)
        with engine.ledger.measure_stage("behavior_probability_snapshot"):
            behavior = chosen_probabilities(engine, pending.trajectories)
        pending = replace(pending, behavior_chosen=behavior).cpu()
        pending.validate(engine.completed_steps)
        return pending, scored

    def update(self, engine: object, context: tuple) -> tuple:
        policy, pending, scored = context
        if pending.accepted_updates == 0:
            return super().update(engine, context)
        pending.validate(engine.completed_steps)
        if scored is not None:
            raise ValueError(
                "Reuse targets must be refreshed after the accepted first update"
            )
        refreshed, current = refresh_targets(engine, pending)
        engine.prefix_factors = {}
        drift = []
        with engine.ledger.measure_stage("reuse_observer"):
            for item, now, behavior in zip(
                refreshed, current, pending.behavior_chosen, strict=True
            ):
                weights, diagnostic = prefix_weights((now, behavior), engine.reuse_cap)
                engine.prefix_factors[item.trajectory.example_id] = weights
                drift.append({"question_id": item.trajectory.example_id, **diagnostic})
        with engine.ledger.measure_stage("student_update"):
            loss, norm = engine.update_student(refreshed)
        return replace(pending, accepted_updates=2), {
            "loss": loss,
            "gradient_norm": norm,
            "drift": drift,
        }
