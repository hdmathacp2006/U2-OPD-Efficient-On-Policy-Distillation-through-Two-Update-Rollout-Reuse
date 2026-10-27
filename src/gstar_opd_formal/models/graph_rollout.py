"""Use the A100 graph configuration without TorchInductor compilation."""

import importlib
from contextlib import nullcontext
from dataclasses import replace

from gstar_opd_formal.models.attention import installed_version
from gstar_opd_formal.models.rollout_spec import VLLM_VERSION
from gstar_opd_formal.models.vllm_rollout import VllmRollout
from gstar_opd_formal.training.checkpoint import (
    capture_random_state,
    restore_random_state,
)


class GraphRollout(VllmRollout):
    def create_llm(self) -> object:
        if installed_version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"Formal runtime requires vllm=={VLLM_VERSION}")
        module = importlib.import_module("vllm")
        protocol = self.settings.protocol
        state = capture_random_state()
        try:
            return module.LLM(
                model=protocol.student_model,
                revision=protocol.student_revision,
                tokenizer_revision=protocol.student_revision,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                tensor_parallel_size=1,
                seed=self.seed,
                max_model_len=self.spec.maximum_model_length,
                max_num_seqs=self.spec.maximum_sequences,
                max_num_batched_tokens=getattr(
                    self.spec, "maximum_batched_tokens", 1024
                ),
                gpu_memory_utilization=0.25,
                kv_cache_memory_bytes=self.spec.cache_bytes,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                enable_sleep_mode=True,
                enable_lora=True,
                max_lora_rank=self.settings.model_load.lora.rank,
                max_loras=1,
                max_cpu_loras=2,
                generation_config="vllm",
                enforce_eager=False,
                compilation_config={
                    "level": 3,
                    "use_inductor": False,
                    "cudagraph_mode": 1,
                    "cudagraph_capture_sizes": [1, 2, 4],
                },
                disable_log_stats=True,
            )
        finally:
            restore_random_state(state)

    def wake(self) -> None:
        with self.measure("rollout_startup_or_wake"):
            super().wake()

    def sync_adapter(self) -> None:
        with self.measure("rollout_adapter_export_and_sync"):
            super().sync_adapter()

    def sleep(self) -> None:
        with self.measure("rollout_sleep"):
            super().sleep()

    def measure(self, stage: str) -> object:
        ledger = getattr(self.owner, "ledger", None)
        return ledger.measure_stage(stage) if ledger is not None else nullcontext()

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "enforce_eager": False,
            "use_inductor": False,
            "training_caps": "shared_formal_8192",
            "prefill_decode_timing": "combined_generation_not_separately_measured",
        }


def replace_rollout(bundle: object) -> object:
    original = bundle.rollout
    if not isinstance(original, VllmRollout) or original.llm is not None:
        raise RuntimeError("Replace only an unstarted rollout engine")
    updated = replace(
        bundle,
        rollout=GraphRollout(original.owner, original.settings, original.seed),
    )
    original.owner.bundle = updated
    return updated
