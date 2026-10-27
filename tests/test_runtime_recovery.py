"""Checkpoint publication and worker recovery failures remain visible."""

from types import SimpleNamespace

import pytest

from gstar_opd_formal.runtime import worker
from gstar_opd_formal.storage.files import write_json


def test_valid_local_advance_survives_remote_zero_boundary(tmp_path, monkeypatch):
    output = tmp_path / "run"
    write_json(
        output / "summary.json",
        {"status": "checkpoint_available", "completed_updates": 1},
    )
    store = SimpleNamespace(
        drive=SimpleNamespace(children=lambda _: [1]),
        folders={"commits": "commits"},
        latest_manifest=lambda: {"boundary": 0, "files": []},
        restore=lambda _: pytest.fail("Must keep the newer verified local checkpoint"),
    )

    def missing(*_):
        raise ValueError("Local checkpoint is ahead of last acknowledged commit")

    from gstar_opd_formal.storage import local_recovery

    monkeypatch.setattr(worker, "verify_tree", missing)
    monkeypatch.setattr(local_recovery, "local_training_boundary", lambda *_: 1)
    worker.prepare_output(
        SimpleNamespace(resume_stopped=True, allow_new=False),
        (store, SimpleNamespace(artifacts=tmp_path), {"action": "train"}, output),
    )
    assert output.exists()


def test_completed_worker_reuses_summary_without_importing_training(
    tmp_path, monkeypatch
):
    summary = {"status": "workload_complete", "completed_updates": 128}
    write_json(tmp_path / "summary.json", summary)
    released = []
    session = SimpleNamespace(
        output=tmp_path, store=SimpleNamespace(release=lambda: released.append(True))
    )
    monkeypatch.setattr(worker, "open_session", lambda *_: session)
    monkeypatch.setattr(worker, "finish_attempt", lambda *_: None)
    monkeypatch.setattr(
        worker, "result_receipt", lambda _, result, reused: {**result, "reused": reused}
    )
    result = worker.run_selected(SimpleNamespace(action="train"), ())
    assert result["reused"] and released == [True]


def test_final_pair_cannot_publish_without_remote_boundaries(tmp_path, monkeypatch):
    from gstar_opd_formal.contract import load_study
    from gstar_opd_formal.orchestration.receipts import study_receipt
    from tests.test_formal_contract import ROOT, training_result

    study = load_study(ROOT)
    value = study_receipt(
        {arm: training_result(study, arm) for arm in ("adaptive", "u2")}, study
    )
    write_json(tmp_path / "pair.json", value)
    monkeypatch.setattr(
        worker, "open_store", lambda *_: SimpleNamespace(latest_manifest=lambda: None)
    )
    with pytest.raises(ValueError, match="committed"):
        worker.publish_study(
            SimpleNamespace(input=tmp_path / "pair.json"),
            (study, None, {"runs_folder_id": "runs"}),
        )


def test_eval_stop_preserves_completed_questions_and_aborts_inflight(tmp_path):
    import time

    from gstar_opd_formal.evaluation.scheduler import RollingScheduler

    commits = []
    session = SimpleNamespace(
        output=tmp_path,
        deadline=time.perf_counter() + 100,
        stop_requested=lambda: True,
        binding={
            "evaluation_config": {
                "concurrency": 2,
                "drain_margin_seconds": 1,
                "flush_margin_seconds": 1,
            }
        },
        timeline=SimpleNamespace(flush=lambda: None),
        store=SimpleNamespace(commit=lambda *values: commits.append(values[1])),
    )
    backend = SimpleNamespace(contexts={"running": True}, abort=lambda: ["running"])
    result = RollingScheduler(session, backend, None, {"done": {}}).run(
        ({"id": "next"},)
    )
    assert result == {"done": {}} and commits == [1]
