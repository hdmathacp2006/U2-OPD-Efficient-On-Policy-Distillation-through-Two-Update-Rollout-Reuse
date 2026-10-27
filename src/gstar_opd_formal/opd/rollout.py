"""Generate trajectories and identify only their causal response targets."""

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import GenerationConfig

from gstar_opd_formal.contract_types import PromptSpec
from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.data.prompts import render_question
from gstar_opd_formal.models.loading import ModelBundle
from gstar_opd_formal.runtime.events import (
    GenerationProgress,
    GenerationProgressSpec,
    report_progress,
)


@dataclass(frozen=True)
class SamplingSpec:
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.max_new_tokens) is not int
            or self.max_new_tokens <= 0
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
            or isinstance(self.top_p, bool)
            or not 0 < self.top_p <= 1
            or type(self.top_k) is not int
            or self.top_k < 0
            or isinstance(self.min_p, bool)
            or not 0 <= self.min_p <= 1
        ):
            raise ValueError("Invalid response sampling settings")


@dataclass(frozen=True)
class RolloutRequest:
    examples: tuple[Gsm8kExample, ...]
    prompt: PromptSpec
    sampling: SamplingSpec
    max_prompt_tokens: int
    question_seed: int | None = None

    def __post_init__(self) -> None:
        if self.question_seed is not None and type(self.question_seed) is not int:
            raise ValueError("Question seed must be an integer")


def question_sampling_seed(base_seed: int, source_index: int) -> int:
    """Keep each realized answer independent of checkpoint and sampling order."""
    if type(base_seed) is not int or type(source_index) is not int or source_index < 0:
        raise ValueError("Question seeds require an integer seed and source ID")
    encoded = json.dumps(("fixed_question_v1", base_seed, source_index)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


@dataclass(frozen=True)
class GreedySpec:
    max_new_tokens: int

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("Greedy response horizon must be a positive integer")


@dataclass(frozen=True)
class GreedyRequest:
    examples: tuple[Gsm8kExample, ...]
    prompt: PromptSpec
    spec: GreedySpec
    max_prompt_tokens: int

    @property
    def sampling(self) -> GreedySpec:
        """Share causal response slicing without changing the sampled policy."""
        return self.spec


@dataclass(frozen=True)
class ResponseTrajectory:
    example_id: str
    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    response_positions: Tensor
    response_length: int
    text: str
    truncated: bool
    rollout_log_probs: Tensor | None = None
    rollout_adapter_identity: str | None = None
    question: str = ""


def generation_config(bundle: ModelBundle, spec: SamplingSpec) -> GenerationConfig:
    """Avoid inherited top-k/repetition processors that change the behavior policy."""
    return GenerationConfig(
        max_new_tokens=spec.max_new_tokens,
        do_sample=True,
        temperature=spec.temperature,
        top_p=spec.top_p,
        top_k=spec.top_k,
        min_p=spec.min_p,
        repetition_penalty=1.0,
        eos_token_id=bundle.student.config.eos_token_id,
        pad_token_id=bundle.tokenizer.pad_token_id,
        bos_token_id=bundle.tokenizer.bos_token_id,
        use_cache=True,
        return_dict_in_generate=False,
        output_scores=False,
    )


def tokenize_questions(
    bundle: ModelBundle, request: RolloutRequest | GreedyRequest
) -> dict[str, Tensor]:
    if not request.examples:
        raise ValueError("A rollout batch must contain at least one question")
    tokenizer = bundle.tokenizer
    if tokenizer.padding_side != "left":
        raise ValueError("Batched autoregressive generation requires left padding")
    prompts = [
        render_question(tokenizer, item.question, request.prompt)
        for item in request.examples
    ]
    # The template inserts special tokens; adding them twice changes prefixes.
    encoded = tokenizer(
        prompts, padding=True, add_special_tokens=False, return_tensors="pt"
    )
    if int(encoded["attention_mask"].sum(dim=1).max()) > request.max_prompt_tokens:
        raise ValueError(
            "Prompt exceeds configured limit; silent truncation is prohibited"
        )
    return {
        name: value.to(bundle.device)
        for name, value in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }


def eos_token_ids(model_config: Any) -> tuple[int, ...]:
    eos = model_config.eos_token_id
    if isinstance(eos, int):
        return (eos,)
    if not eos:
        raise ValueError("A causal model must define at least one EOS token")
    return tuple(eos)


def split_generated_responses(
    bundle: ModelBundle,
    sequences: Tensor,
    batch: tuple[RolloutRequest | GreedyRequest, dict[str, Tensor]],
) -> tuple[ResponseTrajectory, ...]:
    request, encoded = batch
    prompt_width = encoded["input_ids"].shape[1]
    if sequences.ndim != 2 or sequences.shape[0] != len(request.examples):
        raise ValueError("Generated sequences must match the question batch")
    response_width = sequences.shape[1] - prompt_width
    if response_width <= 0:
        raise ValueError("Generation produced no response tokens")
    if response_width > request.sampling.max_new_tokens:
        raise ValueError("Generation exceeded the requested response horizon")
    if not torch.equal(sequences[:, :prompt_width], encoded["input_ids"]):
        raise ValueError("Generated sequences changed their prompt prefix")
    endings = eos_token_ids(bundle.student.config)
    trajectories = []
    for row, example in enumerate(request.examples):
        response = sequences[row, prompt_width:]
        eos_position = next(
            (
                index
                for index, value in enumerate(response.tolist())
                if value in endings
            ),
            None,
        )
        length = eos_position + 1 if eos_position is not None else response_width
        input_ids = sequences[row : row + 1, : prompt_width + length].detach()
        attention_mask = torch.cat(
            (
                encoded["attention_mask"][row : row + 1],
                torch.ones((1, length), device=bundle.device, dtype=torch.long),
            ),
            dim=1,
        )
        # logits[t] predicts input_ids[t+1], so the first response is predicted at P-1.
        response_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
        response_mask[:, prompt_width - 1 :] = True
        # Remove EOS/chat markers for grading; Qwen's thinking tags remain textual.
        trajectories.append(
            ResponseTrajectory(
                example_id=str(example.source_index),
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                response_positions=torch.arange(length, device=bundle.device),
                response_length=length,
                text=bundle.tokenizer.decode(
                    response[:length], skip_special_tokens=True
                ),
                truncated=eos_position is None
                and length == request.sampling.max_new_tokens,
                question=example.question,
            )
        )
    return tuple(trajectories)


def generate_responses(
    bundle: ModelBundle, request: RolloutRequest
) -> tuple[ResponseTrajectory, ...]:
    if request.question_seed is not None:
        return generate_seeded_responses(bundle, request)
    backend = getattr(bundle, "rollout", None)
    if backend is not None:
        return backend.generate(request)
    encoded = tokenize_questions(bundle, request)
    bundle.student.eval()
    # Rescore later to avoid retaining full-vocabulary scores for every decode step.
    progress_spec = GenerationProgressSpec(
        f"sampled_ids={','.join(str(item.source_index) for item in request.examples)}",
        encoded["input_ids"].shape[1],
        request.sampling.max_new_tokens,
    )
    with torch.no_grad(), GenerationProgress(progress_spec) as progress:
        sequences = bundle.student.generate(
            **encoded,
            generation_config=generation_config(bundle, request.sampling),
            use_model_defaults=False,
            logits_processor=[progress],
        )
        progress.update_tokens(sequences.shape[1] - progress_spec.prompt_tokens)
    return split_generated_responses(bundle, sequences, (request, encoded))


def generate_seeded_responses(
    bundle: ModelBundle, request: RolloutRequest
) -> tuple[ResponseTrajectory, ...]:
    """Isolate feedback/evaluation draws from the OPD training random stream."""
    if not request.examples:
        raise ValueError("A rollout batch must contain at least one question")
    if request.question_seed is None:
        raise ValueError("Seeded sampling requires a question seed")
    modes = {module: module.training for module in bundle.student.modules()}
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count()))
    try:
        bundle.student.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            backend = getattr(bundle, "rollout", None)
            if backend is not None:
                # vLLM accepts a distinct SamplingParams seed for each prompt.
                return backend.generate(request)
            trajectories = []
            for example in request.examples:
                torch.manual_seed(
                    question_sampling_seed(request.question_seed, example.source_index)
                )
                # HF exposes one global generator, so fixed outcomes require one
                # question per call rather than a seed for the changing batch.
                single_request = replace(
                    request, examples=(example,), question_seed=None
                )
                trajectories.extend(generate_responses(bundle, single_request))
            return tuple(trajectories)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes.items():
            module.training = training


def generate_response_microbatches(
    bundle: ModelBundle, request: RolloutRequest, batch_size: int
) -> tuple[ResponseTrajectory, ...]:
    """Sample all effective-batch responses from the same unchanged student."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Generation microbatch size must be a positive integer")
    if not request.examples:
        raise ValueError("A rollout batch must contain at least one question")
    trajectories = []
    for start in range(0, len(request.examples), batch_size):
        microbatch = replace(
            request, examples=request.examples[start : start + batch_size]
        )
        report_progress(
            "rollout_microbatch_start",
            question=start + 1,
            total=len(request.examples),
            horizon=request.sampling.max_new_tokens,
        )
        trajectories.extend(generate_responses(bundle, microbatch))
    return tuple(trajectories)


def greedy_generation_config(bundle: ModelBundle, spec: GreedySpec) -> GenerationConfig:
    """Use pure argmax decoding with no inherited sampling or penalty settings."""
    return GenerationConfig(
        max_new_tokens=spec.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        eos_token_id=bundle.student.config.eos_token_id,
        pad_token_id=bundle.tokenizer.pad_token_id,
        bos_token_id=bundle.tokenizer.bos_token_id,
        use_cache=True,
        return_dict_in_generate=False,
        output_scores=False,
    )


def generate_greedy_responses(
    bundle: ModelBundle, request: GreedyRequest
) -> tuple[ResponseTrajectory, ...]:
    """Measure a frozen student without perturbing the training random stream."""
    backend = getattr(bundle, "rollout", None)
    if backend is not None:
        return backend.generate_greedy(request)
    encoded = tokenize_questions(bundle, request)
    modes = {module: module.training for module in bundle.student.modules()}
    cuda_devices = [bundle.device.index or 0] if bundle.device.type == "cuda" else []
    try:
        bundle.student.eval()
        source_ids = ",".join(str(item.source_index) for item in request.examples)
        progress_spec = GenerationProgressSpec(
            f"feedback_ids={source_ids}",
            encoded["input_ids"].shape[1],
            request.spec.max_new_tokens,
        )
        with (
            torch.random.fork_rng(devices=cuda_devices),
            torch.no_grad(),
            GenerationProgress(progress_spec) as progress,
        ):
            sequences = bundle.student.generate(
                **encoded,
                generation_config=greedy_generation_config(bundle, request.spec),
                use_model_defaults=False,
                logits_processor=[progress],
            )
            progress.update_tokens(sequences.shape[1] - progress_spec.prompt_tokens)
        return split_generated_responses(bundle, sequences, (request, encoded))
    finally:
        # Restore mixed submodule modes as well as the top-level train/eval flag.
        for module, training in modes.items():
            module.training = training


def response_logits(model: torch.nn.Module, trajectory: ResponseTrajectory) -> Tensor:
    """Apply identical left-padding-aware positions during rescoring and updating."""
    positions = trajectory.attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(trajectory.attention_mask == 0, 0)
    prediction_positions = trajectory.response_mask[0].nonzero().flatten()
    outputs = model(
        input_ids=trajectory.input_ids,
        attention_mask=trajectory.attention_mask,
        position_ids=positions,
        use_cache=False,
        # Qwen's native head projects only valid prediction positions to vocabulary.
        logits_to_keep=prediction_positions,
    )
    return outputs.logits[0]
