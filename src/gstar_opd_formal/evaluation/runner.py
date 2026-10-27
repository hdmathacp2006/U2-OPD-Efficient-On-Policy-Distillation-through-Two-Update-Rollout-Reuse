"""Restore selected student checkpoints, evaluate missing IDs and close the engine."""

import json

from gstar_opd_formal.evaluation.backend import RollingBackend
from gstar_opd_formal.evaluation.rows import QuestionStore
from gstar_opd_formal.evaluation.scheduler import RollingScheduler
from gstar_opd_formal.models.inference import load_student
from gstar_opd_formal.reporting.metrics import evaluation_metrics
from gstar_opd_formal.storage.files import digest_value, write_json


def export_metrics(session: object, source: tuple) -> dict:
    store = QuestionStore(session, source)
    rows = store.load()
    ordered = [rows[item["id"]] for item in source if item["id"] in rows]
    metrics = evaluation_metrics(ordered, len(source))
    if session.binding["benchmark"] == "math500":
        historical = set(session.study.manifests["math500"].get("historical_ids", []))
        unseen = [row for row in ordered if row["example_id"] not in historical]
        metrics["outside_historical_math103"] = evaluation_metrics(unseen, 397)
    value = {
        "schema": "formal_evaluation_v1",
        "binding_sha256": digest_value(session.binding),
        "arm": session.binding["arm"],
        "benchmark": session.binding["benchmark"],
        "checkpoint_update": session.binding["checkpoint_update"],
        **metrics,
    }
    write_json(session.output / "summary.json", value)
    (session.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    return value


def run_evaluation(session: object) -> dict:
    source = session.source
    questions = QuestionStore(session, source)
    completed = questions.load()
    if len(completed) == len(source):
        return export_metrics(session, source)
    with session.timeline.measure("model_loading_and_adapter_restore"):
        bundle, settings, capacity = load_student(session)
    backend = RollingBackend(
        session, bundle, settings, session.binding["evaluation_config"]
    )
    backend.timeline = session.timeline
    write_json(
        session.output / "model_evidence.json",
        {
            "capacity": capacity,
            "hardware": session.environment["hardware"],
            "checkpoint_sha256": session.binding.get("checkpoint_sha256"),
        },
    )
    try:
        with session.timeline.measure("engine_startup_and_adapter_sync"):
            backend.start()
        RollingScheduler(session, backend, questions, completed).run(source)
        write_json(
            session.output / "generation_provenance.json", bundle.rollout.provenance()
        )
        value = export_metrics(session, source)
        session.timeline.flush()
        session.store.commit(session.output, value["count"])
        return value
    finally:
        with session.timeline.measure("engine_close"):
            backend.close()
