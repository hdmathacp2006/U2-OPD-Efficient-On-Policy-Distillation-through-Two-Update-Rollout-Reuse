"""Catch input and runtime identity failures before any user GPU launch."""

from pathlib import Path

import pytest

from gstar_opd_formal.contract import (
    load_study,
    read_yaml,
    stable_environment,
    validate_configuration,
)
from gstar_opd_formal.orchestration.receipts import study_receipt
from gstar_opd_formal.storage.files import digest_value

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study():
    return load_study(ROOT)


def training_result(study, arm):
    binding = study.training_binding(arm, {"environment_id": "cpu_fixture"})
    return {
        "binding": binding,
        "run_id": digest_value(binding)[:32],
        "status": "workload_complete",
        "completed_updates": study.design["updates"],
        "checkpoint_sha256": "a" * 64,
        "accepted_learning_seconds": 1.0,
        "reused_complete": False,
        "retained_checkpoints": {
            str(update): {
                "path": f"checkpoints/update_{update:04d}.pt",
                "sha256": "a" * 64,
            }
            for update in study.design["retained_updates"]
        },
    }


def test_disk_free_space_cannot_change_training_run_identity(study):
    first = study.training_binding(
        "u2", {"environment_id": "same", "free_disk_bytes": 500}
    )
    second = study.training_binding(
        "u2", {"environment_id": "same", "free_disk_bytes": 100}
    )
    assert first == second


def test_actual_hardware_identity_is_retained():
    assert stable_environment(
        {"hardware": {"gpu_name": "L4"}, "free_disk_bytes": 1}
    ) == {"hardware": {"gpu_name": "L4"}}


def test_training_pool_and_reserved_dev_are_disjoint(study):
    assert not set(study.manifests["training"]["source_indices"]) & set(
        study.manifests["dev"]["source_indices"]
    )


def test_all_benchmark_ids_and_counts_are_frozen(study):
    assert {
        name: len(study.manifests[name]["ids"]) for name in study.design["benchmarks"]
    } == {"math500": 500, "minerva_math": 272}


def test_math397_secondary_has_exactly103_exposed_ids(study):
    assert len(set(study.manifests["math500"]["historical_ids"])) == 103


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 8),
        ("teacher_scoring_batch_size", 4),
        ("attention_backend", "eager"),
    ],
)
def test_unsupported_batch_or_backend_is_rejected(study, field, value):
    training = {**study.training, field: value}
    with pytest.raises(ValueError):
        validate_configuration(
            (study.design, training, study.evaluation, study.methods)
        )


def test_context_must_cover_prompt_and_output(study):
    evaluation = {**study.evaluation, "maximum_model_length": 32000}
    with pytest.raises(ValueError):
        validate_configuration(
            (study.design, study.training, evaluation, study.methods)
        )


def test_16k_profile_preserves_training_study_identity(study):
    original = study.identity
    profile = read_yaml(ROOT / "configs/evaluation_16k.yaml")
    assert profile == {
        **study.evaluation,
        "max_new_tokens": 16384,
        "maximum_model_length": 17664,
    }
    assert load_study(ROOT).identity == original


def test_partial_training_cannot_be_published_as_a_pair(study):
    results = {arm: training_result(study, arm) for arm in ("adaptive", "u2")}
    results["u2"]["completed_updates"] = study.design["updates"] - 1
    with pytest.raises(ValueError):
        study_receipt(results, study)


def test_foreign_checkpoint_run_binding_is_rejected(study):
    results = {arm: training_result(study, arm) for arm in ("adaptive", "u2")}
    results["u2"]["run_id"] = "0" * 32
    with pytest.raises(ValueError):
        study_receipt(results, study)


def test_fresh_coverage_is_half_only_for_original_u2(study):
    assert {
        arm: study.design["updates"]
        // study.methods[arm]["updates_per_fresh_batch"]
        * study.training["batch_size"]
        for arm in ("adaptive", "u2")
    } == {"adaptive": 512, "u2": 256}
