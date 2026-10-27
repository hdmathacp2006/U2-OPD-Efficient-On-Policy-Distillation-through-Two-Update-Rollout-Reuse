"""Continuous vLLM requests with the same native prompt and stable question seeds."""

import importlib
from dataclasses import dataclass, field

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.data.prompts import render_question
from gstar_opd_formal.opd.rollout import (
    RolloutRequest,
    SamplingSpec,
    tokenize_questions,
)
from gstar_opd_formal.storage.files import digest_value, write_json


@dataclass
class RollingBackend:
    session: object
    bundle: object
    settings: object
    config: dict
    contexts: dict = field(default_factory=dict)
    timeline: object = None

    def start(self) -> None:
        self.bundle.rollout.wake()
        if self.session.binding["arm"] != "base":
            self.bundle.rollout.sync_adapter()

    def add(self, source: dict) -> None:
        key = source["id"]
        if key in self.contexts:
            raise ValueError("Duplicate active question ID")
        prompt_text = render_question(
            self.bundle.tokenizer, source["question"], self.settings.protocol.prompt
        )
        actual = len(
            self.bundle.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        if actual > self.config["max_prompt_tokens"]:
            raise ValueError(
                "Benchmark question would be truncated by the declared prompt budget"
            )
        request = RolloutRequest(
            (Gsm8kExample(source["sampling_index"], source["question"], "unused"),),
            self.settings.protocol.prompt,
            SamplingSpec(
                self.config["max_new_tokens"],
                self.config["temperature"],
                self.config["top_p"],
            ),
            self.config["max_prompt_tokens"],
            question_seed=self.config["seed"],
        )
        encoded = tokenize_questions(self.bundle, request)
        prompt = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
        if (
            len(prompt) != actual
            or len(prompt) + self.config["max_new_tokens"]
            > self.config["maximum_model_length"]
        ):
            raise ValueError(
                "Rendered prompt or context differs from the frozen capacity evidence"
            )
        params = self.bundle.rollout.make_sampling_params(request)[0]
        params.output_kind = importlib.import_module(
            "vllm.sampling_params"
        ).RequestOutputKind.FINAL_ONLY
        self.bundle.rollout.llm.llm_engine.add_request(
            key,
            {"prompt_token_ids": prompt},
            params,
            lora_request=self.bundle.rollout.adapter_request,
        )
        self.contexts[key] = source, request, encoded, prompt

    def step(self) -> list[dict]:
        rows = []
        for output in self.bundle.rollout.llm.llm_engine.step():
            if not output.finished:
                continue
            if output.request_id not in self.contexts:
                raise ValueError("Unknown or duplicate engine completion")
            source, request, encoded, prompt = self.contexts.pop(output.request_id)
            trajectory = self.bundle.rollout.convert_output(
                output, (request, encoded, 0, prompt)
            )
            row = {
                "example_id": source["id"],
                "question_sha256": digest_value(source["question"]),
                "output": trajectory.text,
                "response_length": trajectory.response_length,
                "truncated": trajectory.truncated,
                "finish_reason": output.outputs[0].finish_reason,
                "sampling_index": source["sampling_index"],
                "question_seed": self.bundle.rollout.seed_for(
                    request, request.examples[0].source_index
                ),
            }
            write_json(
                self.session.output / "ungraded" / f"{digest_value(source['id'])}.json",
                {"binding_sha256": digest_value(self.session.binding), "row": row},
            )
            rows.append(row)
        self.bundle.rollout.sample_memory()
        return rows

    def abort(self) -> list[str]:
        keys = list(self.contexts)
        for key in keys:
            self.bundle.rollout.llm.llm_engine.abort_request(key)
        self.contexts.clear()
        return keys

    def close(self) -> None:
        if self.contexts:
            self.abort()
        self.bundle.rollout.close()
