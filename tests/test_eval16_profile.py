"""A shorter evaluation can reuse the verified historical training pair."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gstar_opd_formal.contract import load_study
from gstar_opd_formal.orchestration.receipts import study_receipt
from gstar_opd_formal.runtime import worker
from gstar_opd_formal.storage.files import digest_file, digest_value
from tests.test_formal_contract import training_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "profile,cap",
    [
        ("16k", 16384),
        ("12k_a100", 12288),
        ("12k_a100_b20", 12288),
        ("12k_a100_b10", 12288),
    ],
)
def test_shorter_evaluation_accepts_prior_training_runtime(
    tmp_path, monkeypatch, profile, cap
):
    study = load_study(ROOT)
    checkpoint = tmp_path / "checkpoints/update_0128.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"previous verified checkpoint")
    checksum = digest_file(checkpoint)
    results = {}
    for arm in ("adaptive", "u2"):
        result = training_result(study, arm)
        result["binding"]["execution_code"] = "a" * 64
        result["run_id"] = digest_value(result["binding"])[:32]
        result["checkpoint_sha256"] = checksum
        result["retained_checkpoints"]["128"]["sha256"] = checksum
        results[arm] = result
    prior_study = replace(study, code={**study.code, "execution": "a" * 64})
    pair = study_receipt(results, prior_study)
    monkeypatch.setattr(worker, "read_named", lambda *_: pair)
    monkeypatch.setattr(worker, "restore_training", lambda *_: {"directory": tmp_path})
    args = SimpleNamespace(
        action="evaluate",
        arm="u2",
        benchmark="math500",
        checkpoint_update=128,
        artifacts=tmp_path,
        evaluation_profile=profile,
        gpu_profile="A100",
    )
    context = (study, None, {"studies_folder_id": "test"})
    source, selection = worker.select_evaluation(args, context)
    assert len(source) == 500 and selection["checkpoint_sha256"] == checksum

    binding, _, _ = worker.execution_binding(
        args, (*context, {"environment_id": "cpu_fixture"})
    )
    assert binding["study_identity"] == study.identity
    assert binding["evaluation_config"]["max_new_tokens"] == cap
    assert binding["evaluation_profile"] == profile
    if profile.startswith("12k_a100"):
        from gstar_opd_formal.evaluation.capacity import validate_capacity

        capacity = validate_capacity(
            SimpleNamespace(
                max_position_embeddings=40960,
                num_hidden_layers=28,
                num_key_value_heads=8,
                head_dim=128,
            ),
            binding["evaluation_config"],
        )
        batch, cache = {
            "12k_a100": (8, 16),
            "12k_a100_b20": (20, 35),
            "12k_a100_b10": (10, 18),
        }[profile]
        assert binding["evaluation_config"]["concurrency"] == batch
        assert capacity["cache_bytes"] == cache * 2**30
        assert capacity["worst_case_batch_kv_bytes"] < capacity["cache_bytes"] * 0.85

    args.evaluation_profile = "full"
    with pytest.raises(ValueError, match="another runtime"):
        worker.select_evaluation(args, context)


@pytest.mark.parametrize("arm,gpu", [("base", "A100"), ("u2", "L4")])
def test_12k_rejects_base_and_l4_before_restore(arm, gpu, monkeypatch):
    study = load_study(ROOT)
    monkeypatch.setattr(
        worker, "select_evaluation", lambda *_: pytest.fail("Reject before restore")
    )
    args = SimpleNamespace(
        action="evaluate", arm=arm, gpu_profile=gpu, evaluation_profile="12k_a100"
    )
    with pytest.raises(ValueError, match="reserved for Adaptive/U2"):
        worker.execution_binding(args, (study, None, {}, {}))


def test_base_l4_profile_uses_12k_four_requests_without_checkpoint(monkeypatch):
    study = load_study(ROOT)
    monkeypatch.setattr(
        worker, "read_named", lambda *_: pytest.fail("Base needs no training receipt")
    )
    args = SimpleNamespace(
        action="evaluate",
        arm="base",
        gpu_profile="L4",
        evaluation_profile="12k_base_l4",
        benchmark="math500",
    )
    binding, source, selection = worker.execution_binding(args, (study, None, {}, {}))
    config = binding["evaluation_config"]
    assert len(source) == 500 and selection["checkpoint_sha256"] is None
    assert (config["max_new_tokens"], config["concurrency"]) == (12288, 4)
    assert config["cache_bytes"] == 12 * 2**30
    assert config["temperature"] == study.evaluation["temperature"]
    assert config["top_p"] == study.evaluation["top_p"]


@pytest.mark.parametrize("arm,gpu", [("u2", "L4"), ("base", "A100")])
def test_base_l4_profile_rejects_wrong_model_or_gpu(arm, gpu):
    study = load_study(ROOT)
    args = SimpleNamespace(
        action="evaluate",
        arm=arm,
        gpu_profile=gpu,
        evaluation_profile="12k_base_l4",
    )
    with pytest.raises(ValueError, match="base model on L4"):
        worker.execution_binding(args, (study, None, {}, {}))
