"""Pinned, synchronous vLLM LoRA inference for fresh student trajectories.

Sources: vllm-project/vllm v0.10.2 entrypoints/llm.py and lora/worker_manager.py.
"""

import atexit
import hashlib
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from gstar_opd_formal.models.adapter_snapshot import capture_adapter_snapshot
from gstar_opd_formal.models.attention import installed_version
from gstar_opd_formal.models.rollout_spec import (
    LOGPROB_GUARD_VERSION,
    VLLM_VERSION,
    ResidentRolloutSpec,
)
from gstar_opd_formal.opd.rollout import (
    GreedyRequest,
    ResponseTrajectory,
    RolloutRequest,
    eos_token_ids,
    question_sampling_seed,
    split_generated_responses,
    tokenize_questions,
)
from gstar_opd_formal.runtime.events import report_progress
from gstar_opd_formal.training.checkpoint import (
    capture_random_state,
    restore_random_state,
)


def sampling_seed(run_seed: int, step: int, source_id: int) -> int:
    encoded = json.dumps((run_seed, step, source_id)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


class VllmRollout:
    """One engine per process; immutable adapter IDs prevent stale LoRA reuse."""

    def __init__(self, owner: object, settings: object, seed: int) -> None:
        self.owner = owner
        self.settings = settings
        self.spec = settings.rollout
        self.seed = seed
        self.llm = None
        self.sleeping = False
        self.adapter_directory = None
        self.adapter_identity = None
        self.adapter_request = None
        self.adapter_counter = 0
        self.peak_memory_bytes = 0
        self.last_mismatch = None
        self.diagnostic_path = None
        self.diagnostic_count = 0
        self.warning_count = 0
        self.model_identity = (
            settings.protocol.student_model + "@" + settings.protocol.student_revision
        )

    def create_llm(self) -> object:
        if installed_version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"The rollout image requires vllm=={VLLM_VERSION}")
        module = importlib.import_module("vllm")
        protocol = self.settings.protocol
        state = capture_random_state()
        try:
            # The worker is spawned by the image environment. It does not inherit
            # the trainer's initialized CUDA context or mutate its saved RNG.
            return module.LLM(
                model=protocol.student_model,
                revision=protocol.student_revision,
                tokenizer_revision=protocol.student_revision,
                # Prompts are already tokenized by the verified HF tokenizer and
                # outputs are decoded there. Avoid per-adapter tokenizer lookup.
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
                disable_log_stats=True,
            )
        finally:
            restore_random_state(state)

    def wake(self) -> None:
        if self.owner.bundle.device.type == "cuda" and (
            self.llm is None or self.sleeping
        ):
            # HF's caching allocator can retain a large scoring/backward peak.
            # Release unused blocks before the worker restores its KV cache.
            torch.cuda.empty_cache()
        if self.llm is None:
            report_progress("vllm_loading_start", version=VLLM_VERSION)
            self.llm = self.create_llm()
            atexit.register(self.close)
            report_progress("vllm_ready", cache_gib=self.spec.cache_bytes / 2**30)
        elif self.sleeping:
            self.llm.wake_up()
            self.sleeping = False
        self.record_event("vllm_wake", {"sleeping": self.sleeping})

    def record_event(self, event: str, details: dict) -> None:
        journal = getattr(self.owner, "journal", None)
        if journal is not None:
            journal.write(
                "events",
                {"event": event, "step": self.owner.completed_steps, **details},
            )

    def prepare_hf_phase(self) -> None:
        if (
            not isinstance(self.spec, ResidentRolloutSpec)
            or self.sleeping
            or self.llm is None
        ):
            return
        if self.spec.residency in {"sleep_before_hf", "sleep"}:
            # Retain the worker across feedback generations, then release its
            # KV cache before teacher scoring and student backward.
            self.sleep()
            return
        free, total = torch.cuda.mem_get_info(self.owner.bundle.device)
        required = total * self.spec.free_memory_fraction + self.spec.hf_workspace_bytes
        if free < required:
            self.record_event(
                "memory_pressure_fallback", {"free_bytes": free, "total_bytes": total}
            )
            self.sleep()

    def sample_memory(self) -> None:
        if self.owner.bundle.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.owner.bundle.device)
            self.peak_memory_bytes = max(self.peak_memory_bytes, total - free)
            ledger = getattr(self.owner, "ledger", None)
            if ledger is not None:
                ledger.peak_memory_bytes = max(
                    ledger.peak_memory_bytes, self.peak_memory_bytes
                )

    def sleep(self) -> None:
        if self.llm is None or self.sleeping:
            return
        self.llm.sleep(level=1)
        self.sleeping = True
        report_progress("vllm_sleep", adapter_id=self.adapter_counter)
        self.record_event("vllm_sleep", {"adapter_id": self.adapter_counter})

    def sync_adapter(self) -> None:
        snapshot = capture_adapter_snapshot(
            self.owner.bundle.student, self.model_identity
        )
        if snapshot.identity == self.adapter_identity:
            return
        if self.adapter_request is not None:
            self.llm.llm_engine.remove_lora(self.adapter_counter)
        old_directory = self.adapter_directory
        directory = TemporaryDirectory(prefix="gstar_vllm_lora_")
        try:
            save_file(snapshot.state, directory.name + "/adapter_model.safetensors")
            self.owner.bundle.student.peft_config["default"].save_pretrained(
                directory.name
            )
            request_type = importlib.import_module("vllm.lora.request").LoRARequest
            self.adapter_counter += 1
            self.adapter_request = request_type(
                snapshot.identity, self.adapter_counter, directory.name
            )
            self.adapter_directory = directory
            self.adapter_identity = snapshot.identity
        except Exception:
            directory.cleanup()
            raise
        finally:
            if old_directory is not None:
                old_directory.cleanup()
        report_progress(
            "vllm_adapter_synced",
            adapter_id=self.adapter_counter,
            step=self.owner.completed_steps,
        )
        self.record_event(
            "adapter_synced",
            {"adapter_id": self.adapter_counter, "adapter_identity": snapshot.identity},
        )

    def make_sampling_params(self, request: RolloutRequest | GreedyRequest) -> list:
        params_type = importlib.import_module("vllm").SamplingParams
        greedy = isinstance(request, GreedyRequest)
        temperature = 0.0 if greedy else request.sampling.temperature
        top_p = 1.0 if greedy else request.sampling.top_p
        return [
            params_type(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=-1
                if greedy or request.sampling.top_k == 0
                else request.sampling.top_k,
                min_p=0.0 if greedy else request.sampling.min_p,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                # Explicit stops preserve HF's EOS contract without relying on
                # a skipped tokenizer or inherited generation defaults.
                stop_token_ids=sorted(eos_token_ids(self.owner.bundle.student.config)),
                max_tokens=request.sampling.max_new_tokens,
                seed=self.seed_for(request, item.source_index),
                logprobs=0,
                detokenize=False,
            )
            for item in request.examples
        ]

    def seed_for(self, request: RolloutRequest | GreedyRequest, source_id: int) -> int:
        if isinstance(request, RolloutRequest) and request.question_seed is not None:
            return question_sampling_seed(request.question_seed, source_id)
        return sampling_seed(self.seed, self.owner.completed_steps, source_id)

    def verify_trajectories(self, trajectories: tuple[ResponseTrajectory, ...]) -> None:
        snapshot = capture_adapter_snapshot(
            self.owner.bundle.student, self.model_identity
        )
        if self.adapter_identity != snapshot.identity or any(
            item.rollout_adapter_identity != snapshot.identity for item in trajectories
        ):
            # Numerical thresholds never relax the requirement for fresh weights.
            raise RuntimeError("Student adapter changed between rollout and scoring")

    def record_mismatch(self, diagnostic: dict, trajectory: ResponseTrajectory) -> None:
        self.last_mismatch = diagnostic
        self.diagnostic_count += 1
        self.warning_count += int(diagnostic["warning"])
        if self.diagnostic_path is not None:
            path = Path(self.diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                **diagnostic,
                "step": self.owner.completed_steps + 1,
                "question_id": trajectory.example_id,
                "response_length": trajectory.response_length,
                "adapter_identity": trajectory.rollout_adapter_identity,
                "guard_version": LOGPROB_GUARD_VERSION,
                "scope": "physical_scoring_checks_including_partial_blocks",
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def convert_output(self, output: object, context: tuple) -> object:
        request, encoded, row, prompt_ids = context
        if list(output.prompt_token_ids) != prompt_ids:
            raise ValueError("vLLM changed the rendered prompt token IDs")
        if not output.finished or len(output.outputs) != 1:
            raise ValueError("vLLM returned an incomplete or multi-response request")
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        if not tokens or len(tokens) > request.sampling.max_new_tokens:
            raise ValueError("vLLM response violates the token cap")
        if completion.logprobs is None or len(completion.logprobs) != len(tokens):
            raise ValueError(
                "vLLM must return one sampled-token log-probability per token"
            )
        probabilities = [
            float(record[token].logprob)
            for token, record in zip(tokens, completion.logprobs, strict=True)
        ]
        if any(not math.isfinite(value) or value > 0.000001 for value in probabilities):
            raise ValueError("vLLM returned invalid sampled-token log-probabilities")
        bundle = self.owner.bundle
        sequence = torch.cat(
            (
                encoded["input_ids"][row : row + 1],
                torch.tensor([tokens], device=bundle.device, dtype=torch.long),
            ),
            dim=1,
        )
        single_request = replace(request, examples=(request.examples[row],))
        single_encoded = {name: value[row : row + 1] for name, value in encoded.items()}
        trajectory = split_generated_responses(
            bundle, sequence, (single_request, single_encoded)
        )[0]
        if trajectory.response_length != len(tokens):
            raise ValueError("vLLM decoded past the first declared EOS token")
        return replace(
            trajectory,
            rollout_log_probs=torch.tensor(
                probabilities[: trajectory.response_length], dtype=torch.float32
            ),
            rollout_adapter_identity=self.adapter_identity,
        )

    def infer(self, request: RolloutRequest | GreedyRequest) -> tuple:
        bundle = self.owner.bundle
        encoded = tokenize_questions(bundle, request)
        prompts = [
            encoded["input_ids"][row][encoded["attention_mask"][row].bool()].tolist()
            for row in range(len(request.examples))
        ]
        if any(
            len(prompt) + request.sampling.max_new_tokens
            > self.spec.maximum_model_length
            for prompt in prompts
        ):
            raise ValueError("vLLM request exceeds the declared context length")
        self.wake()
        try:
            self.sync_adapter()
            self.sample_memory()
            outputs = self.llm.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                self.make_sampling_params(request),
                lora_request=self.adapter_request,
                use_tqdm=True,
            )
            if len(outputs) != len(prompts):
                raise ValueError("vLLM returned the wrong number of requests")
            trajectories = tuple(
                self.convert_output(output, (request, encoded, row, prompts[row]))
                for row, output in enumerate(outputs)
            )
            self.sample_memory()
            return trajectories
        finally:
            # Even failed inference must release KV/weight memory before HF work.
            if (
                not isinstance(self.spec, ResidentRolloutSpec)
                or self.spec.residency == "sleep"
            ):
                self.sleep()

    def generate(self, request: RolloutRequest) -> tuple:
        return self.infer(request)

    def close(self) -> None:
        """Close the pinned V1 client before Python tears down worker resources."""
        if self.llm is not None:
            # v0.10.2 exposes shutdown on EngineCoreClient, not on offline LLM.
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None
        if self.adapter_directory is not None:
            self.adapter_directory.cleanup()
            self.adapter_directory = None
        atexit.unregister(self.close)

    def generate_greedy(self, request: GreedyRequest) -> tuple:
        return self.infer(request)

    def provenance(self) -> dict:
        return {
            "backend": "vllm",
            "version": VLLM_VERSION,
            "adapter_identity": self.adapter_identity,
            "adapter_revision_count": self.adapter_counter,
            "sleeping": self.sleeping,
            "peak_device_memory_bytes": self.peak_memory_bytes,
            "last_training_logprob_difference": self.last_mismatch,
            "logprob_guard_version": LOGPROB_GUARD_VERSION,
            "logprob_thresholds": {
                "warning": self.spec.logprob_warning_threshold,
                "failure": self.spec.logprob_tolerance,
            },
            "diagnostics_in_current_process": self.diagnostic_count,
            "warnings_in_current_process": self.warning_count,
        }
