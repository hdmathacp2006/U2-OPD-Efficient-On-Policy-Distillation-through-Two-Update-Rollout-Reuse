"""Real raw-output persistence and grading recover without another generation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gstar_opd_formal.evaluation.rows import QuestionStore
from gstar_opd_formal.opd.rollout import question_sampling_seed
from gstar_opd_formal.reporting.metrics import evaluation_metrics
from gstar_opd_formal.storage.files import digest_value, write_json


@pytest.fixture
def store(tmp_path):
    source = {
        "id": "math:1",
        "question": "1+1?",
        "answer": "2",
        "sampling_index": 1,
        "metadata": {},
    }
    session = SimpleNamespace(
        output=tmp_path,
        binding={
            "benchmark": "math500",
            "evaluation_config": {"seed": 17, "max_new_tokens": 32},
        },
        grader_python="unused",
        study=SimpleNamespace(root=Path(__file__).resolve().parents[1]),
    )
    return QuestionStore(session, (source,))


def raw_row(store):
    source = store.source[0]
    return {
        "example_id": source["id"],
        "question_sha256": digest_value(source["question"]),
        "sampling_index": 1,
        "question_seed": question_sampling_seed(17, 1),
        "output": r"\boxed{2}",
        "response_length": 4,
        "truncated": False,
        "finish_reason": "stop",
    }


def test_ungraded_answer_recovers_without_generation(store):
    path = store.output / "ungraded" / f"{digest_value('math:1')}.json"
    write_json(
        path,
        {"binding_sha256": digest_value(store.session.binding), "row": raw_row(store)},
    )
    recovered = store.load()
    assert recovered["math:1"]["correct"] and not path.exists()


def test_complete_question_cannot_be_overwritten(store):
    store.save(raw_row(store))
    with pytest.raises(ValueError, match="overwritten"):
        store.save(raw_row(store))


@pytest.mark.parametrize(
    "field,value",
    [
        ("question_seed", 0),
        ("response_length", 33),
        ("finish_reason", "length"),
        ("sampling_index", 2),
    ],
)
def test_changed_generation_identity_is_rejected(store, field, value):
    row = {**raw_row(store), field: value}
    with pytest.raises(ValueError):
        store.save(row)


def test_infrastructure_failure_preserves_raw_output(store, monkeypatch):
    row = raw_row(store)
    path = store.output / "ungraded" / f"{digest_value('math:1')}.json"
    write_json(
        path, {"binding_sha256": digest_value(store.session.binding), "row": row}
    )

    def fail(*_):
        raise RuntimeError("grader process unavailable")

    monkeypatch.setattr(QuestionStore, "grade", fail)
    with pytest.raises(RuntimeError):
        store.load()
    assert path.exists() and not list((store.output / "raw").glob("*.json"))


def test_partial_metrics_never_claim_full_accuracy(store):
    row = store.grade(raw_row(store), store.source[0])
    summary = evaluation_metrics([row], 2)
    assert summary["accuracy"] is None and summary["partial_accuracy"] == 1.0
