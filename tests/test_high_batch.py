"""High concurrency keeps native sampling and captures the actual decode sizes."""

from types import SimpleNamespace

import pytest

from gstar_opd_formal.runtime import high_batch


@pytest.mark.parametrize("batch,cache", [(20, 35), (10, 18)])
def test_engine_configuration_covers_requested_batch(batch, cache, monkeypatch):
    calls = []
    monkeypatch.setattr(
        high_batch, "installed_version", lambda _: high_batch.VLLM_VERSION
    )
    monkeypatch.setattr(high_batch, "capture_random_state", lambda: "saved_rng")
    restored = []
    monkeypatch.setattr(high_batch, "restore_random_state", restored.append)
    monkeypatch.setattr(
        high_batch.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            LLM=lambda **kwargs: calls.append(kwargs) or "engine"
        ),
    )
    rollout = object.__new__(high_batch.EvaluationRollout)
    rollout.settings = SimpleNamespace(
        protocol=SimpleNamespace(student_model="pinned", student_revision="revision"),
        model_load=SimpleNamespace(lora=SimpleNamespace(rank=16)),
    )
    rollout.seed = 17
    rollout.spec = SimpleNamespace(
        maximum_model_length=13568,
        maximum_sequences=batch,
        maximum_batched_tokens=4096,
        cache_bytes=cache * 2**30,
    )
    assert rollout.create_llm() == "engine"
    options = calls[0]
    assert options["max_num_seqs"] == batch
    assert options["kv_cache_memory_bytes"] == cache * 2**30
    assert options["compilation_config"]["cudagraph_capture_sizes"][-1] == batch
    assert options["compilation_config"]["use_inductor"] is False
    assert options["dtype"] == "bfloat16" and options["max_model_len"] == 13568
    assert options["generation_config"] == "vllm" and options["enable_lora"] is True
    assert restored == ["saved_rng"]
