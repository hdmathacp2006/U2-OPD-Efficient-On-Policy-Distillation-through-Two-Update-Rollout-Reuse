"""Shared finite training loop for matched Adaptive and original fixed-two U2."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from gstar_opd_formal.controllers.reuse import FreshPolicy
from gstar_opd_formal.methods.adaptive import AdaptiveMethod
from gstar_opd_formal.methods.u2 import U2Method
from gstar_opd_formal.opd.reuse_engine import create_reuse_engine
from gstar_opd_formal.runtime.events import report_progress
from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    read_json,
    write_json,
)
from gstar_opd_formal.training.checkpoint import (
    CheckpointRequest,
    load_checkpoint,
    save_checkpoint,
)
from gstar_opd_formal.training.history import accepted_checkpoint, validate_history
from gstar_opd_formal.training.initialization import apply_initial_adapter
from gstar_opd_formal.training.settings import build_settings, training_examples


def new_state() -> dict:
    return {
        "learner_updates": [],
        "fresh_batches": [],
        "decisions": [],
        "trajectories": [],
        "pending": None,
    }


def export_history(output: Path, state: dict) -> None:
    for name in ("learner_updates", "fresh_batches", "decisions", "trajectories"):
        path = output / f"{name}.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(
            "".join(
                json.dumps(row, allow_nan=False) + "\n" for row in state[name]
            ).encode()
        )
        temporary.replace(path)


@dataclass
class TrainingRun:
    session: object
    engine: object
    method: object
    policy: FreshPolicy
    state: dict
    examples: tuple
    last_update_seconds: float = 0.0

    def save(self, reason: str) -> dict:
        completed = self.engine.completed_steps
        if completed == 0:
            raise ValueError("No accepted optimizer update to checkpoint")
        validate_history(self.state, self.session.binding, completed)
        if self.policy.fresh_observations != len(self.state["fresh_batches"]):
            raise ValueError(
                "Adaptive observation count differs from fresh batch count"
            )
        output = self.session.output
        relative = f"checkpoints/update_{completed:04d}.pt"
        with self.engine.ledger.measure_stage("checkpoint"):
            save_checkpoint(
                self.engine,
                self.policy,
                CheckpointRequest(output / relative, self.session.binding, self.state),
            )
        export_history(output, self.state)
        endpoint = self.session.binding["endpoint"]
        value = {
            "schema": "formal_training_v1",
            "binding_sha256": digest_value(self.session.binding),
            "status": "workload_complete"
            if completed == endpoint
            else "checkpoint_available",
            "arm": self.method.name,
            "requested_updates": endpoint,
            "completed_updates": completed,
            "fresh_batches": len(self.state["fresh_batches"]),
            "fresh_question_count": len(self.state["fresh_batches"])
            * self.session.binding["batch_size"],
            "horizon_observations": self.policy.fresh_observations,
            "pending_reuse": self.state["pending"] is not None,
            "accepted_checkpoint": relative,
            "checkpoint_sha256": digest_file(output / relative),
            "accepted_learning_seconds": self.engine.ledger.charged_seconds,
            "stage_seconds": dict(self.engine.ledger.stage_seconds),
            "ledger_after_checkpoint": self.engine.ledger.state_dict(),
            "stop_reason": reason,
            "horizon": self.policy.horizon,
            "cache_policy": "full_forward_rescore_no_hidden_cache",
            "timing_scope": (
                "learning_includes_rollout_scoring_backward_observers_controller;"
                " checkpoint_and_Drive_reported_separately; "
                "nested_stages_not_additive"
            ),
        }
        write_json(output / "summary.json", value)
        self.session.timeline.flush()
        self.session.store.commit(output, completed)
        return value

    def accept_update(self) -> dict:
        began = time.perf_counter()
        pending, scored = self.state["pending"], None
        if pending is None:
            index = len(self.state["fresh_batches"])
            width = self.session.binding["batch_size"]
            examples = self.examples[index * width : (index + 1) * width]
            if len(examples) != width:
                raise ValueError("Training exhausted the frozen paired manifest")
            pending, scored = self.method.fresh(self.engine, (self.policy, examples))
            self.state["fresh_batches"].append(
                {
                    "fresh_batch": index,
                    "behavior_version": pending.behavior_version,
                    "horizon": pending.horizon,
                    "question_ids": [item.example_id for item in pending.trajectories],
                }
            )
            self.state["trajectories"].extend(
                {
                    "fresh_batch": index,
                    "behavior_version": pending.behavior_version,
                    "example_id": item.example_id,
                    "output": item.text,
                    "response_length": item.response_length,
                    "truncated": item.truncated,
                    "response_token_ids": item.input_ids[
                        0, -item.response_length :
                    ].tolist(),
                }
                for item in pending.trajectories
            )
        reuse_index = pending.accepted_updates
        pending, metrics = self.method.update(
            self.engine, (self.policy, pending, scored)
        )
        backend = self.engine.bundle.rollout
        if backend is not None:
            backend.sample_memory()
        self.engine.ledger.synchronize()
        seconds = time.perf_counter() - began
        self.engine.ledger.charged_seconds += seconds
        self.last_update_seconds = max(self.last_update_seconds, seconds)
        record = {
            "update": self.engine.completed_steps,
            "fresh_batch": len(self.state["fresh_batches"]) - 1,
            "reuse_index": reuse_index,
            "behavior_version": pending.behavior_version,
            "horizon": pending.horizon,
            "next_horizon": self.policy.horizon,
            "question_ids": [item.example_id for item in pending.trajectories],
            "response_lengths": [item.response_length for item in pending.trajectories],
            "seconds": seconds,
            **metrics,
        }
        self.state["learner_updates"].append(record)
        self.state["pending"] = (
            pending.cpu()
            if pending.accepted_updates < self.method.updates_per_fresh_batch
            else None
        )
        self.state["decisions"].append(
            {
                "update": self.engine.completed_steps,
                "action": "reuse" if self.state["pending"] is not None else "refresh",
                "adaptive": dict(self.policy.adaptive.last_decision),
            }
        )
        report_progress(
            "formal_training_progress",
            completed=self.engine.completed_steps,
            total=self.session.binding["endpoint"],
            fresh_batches=len(self.state["fresh_batches"]),
            horizon=pending.horizon,
            reuse_index=reuse_index,
            accepted_learning_seconds=self.engine.ledger.charged_seconds,
        )
        return record

    def loop(self) -> dict:
        endpoint = self.session.binding["endpoint"]
        reason = "requested_endpoint"
        while self.engine.completed_steps < endpoint:
            margin = max(
                self.session.study.design["checkpoint_margin_seconds"],
                self.last_update_seconds * 1.5,
            )
            if (
                self.session.stop_requested()
                or time.perf_counter() + margin >= self.session.deadline
            ):
                reason = "stopped_or_deadline_boundary"
                break
            self.session.store.require_writer()
            self.accept_update()
            if (
                self.engine.completed_steps
                % self.session.study.design["checkpoint_interval"]
                == 0
            ):
                self.save("periodic")
        return self.save(reason)


def run_training(session: object) -> dict:
    settings = build_settings(session.study)
    with session.timeline.measure("model_loading"):
        engine = create_reuse_engine(settings)
    engine.reuse_cap = session.study.methods["u2"]["prefix_cap"]
    method = U2Method() if session.binding["arm"] == "u2" else AdaptiveMethod()
    policy = FreshPolicy(settings)
    state = new_state()
    try:
        previous = session.output / "summary.json"
        saved = read_json(previous) if previous.exists() else {}
        if saved.get("accepted_checkpoint"):
            checked = accepted_checkpoint(session.output, session.binding)
            state = load_checkpoint(
                engine, policy, CheckpointRequest(checked["path"], session.binding, {})
            )
            validate_history(state, session.binding, engine.completed_steps)
            engine.ledger.load_state_dict(checked["summary"]["ledger_after_checkpoint"])
        else:
            apply_initial_adapter(engine, session.study)
        run = TrainingRun(
            session, engine, method, policy, state, training_examples(session.study)
        )
        return run.loop()
    finally:
        engine.optimizer.zero_grad(set_to_none=True)
        if engine.bundle.rollout is not None:
            engine.bundle.rollout.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
