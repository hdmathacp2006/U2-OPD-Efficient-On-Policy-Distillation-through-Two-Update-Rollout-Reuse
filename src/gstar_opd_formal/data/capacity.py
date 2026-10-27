"""Freeze the common native context after rendering all actual benchmark prompts."""

import math
from pathlib import Path

import yaml

from gstar_opd_formal.contract import load_study
from gstar_opd_formal.data.manifests import read_jsonl
from gstar_opd_formal.data.prompts import render_question
from gstar_opd_formal.evaluation.capacity import validate_capacity
from gstar_opd_formal.storage.files import digest_value, write_json


def freeze_prompt_capacity(root: Path) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    from gstar_opd_formal.contract_types import PromptSpec

    study = load_study(root)
    models = study.models["models"]
    tokenizer = AutoTokenizer.from_pretrained(
        models["student"], revision=models["student_revision"]
    )
    native = AutoConfig.from_pretrained(
        models["student"], revision=models["student_revision"]
    )
    spec = PromptSpec(study.models["prompt"]["instruction"], models["thinking_mode"])
    maxima, token_records = {}, []
    for name in study.design["benchmarks"]:
        lengths = []
        for row in study.manifests[name]["records"]:
            text = render_question(tokenizer, row["question"], spec)
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            token_records.append({"id": row["id"], "prompt_tokens": length})
            lengths.append(length)
        maxima[name] = max(lengths)
    raw = read_jsonl(root / "data/raw/gsm8k_train.jsonl")
    for role in ("training", "dev"):
        lengths = [
            len(
                tokenizer(
                    render_question(tokenizer, raw[index]["question"], spec),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for index in study.manifests[role]["source_indices"]
        ]
        maxima[role] = max(lengths)
    if maxima["training"] > study.training["max_prompt_tokens"]:
        raise ValueError(
            "Selected training questions exceed the preserved prompt budget"
        )
    required = max(maxima[name] for name in [*study.design["benchmarks"], "dev"])
    budget = max(1024, math.ceil(required / 128) * 128)
    config = {
        **study.evaluation,
        "max_prompt_tokens": budget,
        "maximum_model_length": budget + study.evaluation["max_new_tokens"],
    }
    capacity = validate_capacity(native, config)
    (root / "configs/evaluation.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    result = {
        "schema": "formal_prompt_capacity_v1",
        "student_revision": models["student_revision"],
        "maxima": maxima,
        "token_records_sha256": digest_value(token_records),
        "config": config,
        "capacity": capacity,
        "benchmark_questions_truncated": 0,
    }
    write_json(root / "data/manifests/prompt_capacity.json", result)
    return result
