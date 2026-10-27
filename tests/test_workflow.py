"""Verify phase order and failure stops without launching any GPU subprocess."""

from pathlib import Path

import pytest

from gstar_opd_formal.contract import load_study
from gstar_opd_formal.orchestration import pipeline
from gstar_opd_formal.orchestration.pipeline import Pipeline
from tests.test_formal_contract import training_result


@pytest.fixture
def settings(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    study = load_study(root)
    monkeypatch.setattr(pipeline, "load_study", lambda _: study)
    return Pipeline(tmp_path, "worker_python"), study


def test_train_pair_runs_adaptive_then_u2_and_stops(settings, monkeypatch):
    config, study = settings
    calls = []

    def phase(_, context):
        action, arm, _, _ = context
        calls.append((action, arm))
        return training_result(study, arm), {"seconds": 1.0}

    monkeypatch.setattr(pipeline, "run_phase", phase)
    monkeypatch.setattr(
        pipeline, "invoke", lambda *_: ({"drive": {"verified": True}}, 0.0)
    )
    result = pipeline.train_pair(config)
    assert (
        calls == [("train", "adaptive"), ("train", "u2")]
        and result["evaluation_started"] is False
    )


def test_first_arm_failure_never_starts_the_second(settings, monkeypatch):
    config, _ = settings
    calls = []

    def fail(_, context):
        calls.append(context[1])
        raise RuntimeError("checkpoint publication failed")

    monkeypatch.setattr(pipeline, "run_phase", fail)
    with pytest.raises(RuntimeError):
        pipeline.train_pair(config)
    assert calls == ["adaptive"]


@pytest.mark.parametrize(
    "entry,arms",
    [
        (pipeline.evaluate_methods, ("adaptive", "u2")),
        (pipeline.evaluate_base, ("base",)),
    ],
)
def test_eval_notebooks_dispatch_only_their_models(settings, monkeypatch, entry, arms):
    config, _ = settings
    calls = []

    def phase(_, context):
        calls.append(context)
        return {
            "run_id": "a" * 32,
            "accuracy": 0.5,
            "count": 500,
            "reused_complete": False,
        }, {"seconds": 1.0}

    monkeypatch.setattr(pipeline, "run_phase", phase)
    result = entry(config)
    assert (
        len(calls) == len(arms)
        and all(row[0] == "evaluate" for row in calls)
        and {row[1] for row in calls} == set(arms)
        and {row[2] for row in calls} == {"math500"}
        and all(row[3] == 128 for row in calls)
        and result["training_started"] is False
    )


def test_eval_rejects_unknown_benchmark_before_worker(settings, monkeypatch):
    config, _ = settings
    monkeypatch.setattr(
        pipeline,
        "run_phase",
        lambda *_: pytest.fail("Invalid benchmark must not start a worker"),
    )
    invalid = Pipeline(
        config.base,
        config.python,
        selected_benchmarks=("math500", "unknown"),
    )
    with pytest.raises(ValueError, match="benchmark selection"):
        pipeline.evaluate_methods(invalid)


def test_base_selection_never_reads_a_training_receipt(settings, monkeypatch):
    from types import SimpleNamespace

    from gstar_opd_formal.runtime import worker

    _, study = settings
    monkeypatch.setattr(
        worker,
        "read_named",
        lambda *_: pytest.fail("Base inference must not depend on training"),
    )
    source, selection = worker.select_evaluation(
        SimpleNamespace(arm="base", benchmark="math500"), (study, None, {})
    )
    assert len(source) == 500 and selection["checkpoint_update"] == 0


def test_method_selection_requires_the_saved_pair(settings, monkeypatch):
    from types import SimpleNamespace

    from gstar_opd_formal.runtime import worker

    _, study = settings
    monkeypatch.setattr(worker, "read_named", lambda *_: None)
    with pytest.raises(ValueError, match="paired train-only"):
        worker.select_evaluation(
            SimpleNamespace(arm="u2", benchmark="math500"),
            (study, None, {"studies_folder_id": "fake"}),
        )
