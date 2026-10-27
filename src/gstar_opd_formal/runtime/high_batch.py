"""Evaluation-only graph sizes; preserve the frozen training implementation."""

import importlib
from dataclasses import replace

from gstar_opd_formal.evaluation.backend import RollingBackend
from gstar_opd_formal.evaluation.rows import QuestionStore
from gstar_opd_formal.evaluation.runner import export_metrics
from gstar_opd_formal.evaluation.scheduler import RollingScheduler
from gstar_opd_formal.models.attention import installed_version
from gstar_opd_formal.models.graph_rollout import GraphRollout
from gstar_opd_formal.models.inference import load_student
from gstar_opd_formal.models.rollout_spec import VLLM_VERSION
from gstar_opd_formal.storage.files import write_json
from gstar_opd_formal.training.checkpoint import (
    capture_random_state,
    restore_random_state,
)


def capture_sizes(maximum: int) -> list[int]:
    return sorted({size for size in (1, 2, 4, 8, 10, 12, 16, 20) if size <= maximum})


class EvaluationRollout(GraphRollout):
    def create_llm(self) -> object:
        if installed_version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"Formal runtime requires vllm=={VLLM_VERSION}")
        protocol = self.settings.protocol
        state = capture_random_state()
        try:
            return importlib.import_module("vllm").LLM(
                model=protocol.student_model,
                revision=protocol.student_revision,
                tokenizer_revision=protocol.student_revision,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                tensor_parallel_size=1,
                seed=self.seed,
                max_model_len=self.spec.maximum_model_length,
                max_num_seqs=self.spec.maximum_sequences,
                max_num_batched_tokens=self.spec.maximum_batched_tokens,
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
                    "cudagraph_capture_sizes": capture_sizes(
                        self.spec.maximum_sequences
                    ),
                },
                disable_log_stats=True,
            )
        finally:
            restore_random_state(state)

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "evaluation_capture_sizes": capture_sizes(self.spec.maximum_sequences),
            "evaluation_concurrency": self.spec.maximum_sequences,
        }


def evaluation_bundle(session: object) -> tuple:
    bundle, settings, capacity = load_student(session)
    previous = bundle.rollout
    rollout = EvaluationRollout(previous.owner, settings, previous.seed)
    bundle = replace(bundle, rollout=rollout)
    rollout.owner.bundle = bundle
    return bundle, settings, capacity


def run_evaluation(session: object) -> dict:
    questions = QuestionStore(session, session.source)
    completed = questions.load()
    if len(completed) == len(session.source):
        return export_metrics(session, session.source)
    with session.timeline.measure("model_loading_and_adapter_restore"):
        bundle, settings, capacity = evaluation_bundle(session)
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
        RollingScheduler(session, backend, questions, completed).run(session.source)
        write_json(
            session.output / "generation_provenance.json", bundle.rollout.provenance()
        )
        value = export_metrics(session, session.source)
        session.timeline.flush()
        session.store.commit(session.output, value["count"])
        return value
    finally:
        with session.timeline.measure("engine_close"):
            backend.close()
