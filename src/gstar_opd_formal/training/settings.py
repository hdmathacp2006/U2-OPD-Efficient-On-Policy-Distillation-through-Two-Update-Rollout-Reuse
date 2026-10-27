"""Build the original engine value types from the standalone formal contract."""

from dataclasses import dataclass

from gstar_opd_formal.contract import Study
from gstar_opd_formal.contract_types import (
    DualClipOpdSpec,
    EvaluationSpec,
    Gsm8kSplitSpec,
    HorizonSpec,
    PromptSpec,
    Protocol,
)
from gstar_opd_formal.controllers.adaptive import AdaptivePolicySpec
from gstar_opd_formal.models.loading import LoraSpec, NativeTeacherLoadSpec
from gstar_opd_formal.models.rollout_spec import ResidentRolloutSpec
from gstar_opd_formal.opd.engine import StreamingEngineSpec


@dataclass(frozen=True)
class RuntimeSettings:
    protocol: Protocol
    model_load: NativeTeacherLoadSpec
    engine: StreamingEngineSpec
    adaptive: AdaptivePolicySpec
    batch_size: int
    run_seed: int
    rollout: ResidentRolloutSpec


def build_settings(study: Study) -> RuntimeSettings:
    models, training = study.models["models"], study.training
    source = study.manifests["training"]["source_files"][0]
    horizons = training["horizons"]
    opd = training["opd"]
    evaluation = study.evaluation
    protocol = Protocol(
        1,
        Gsm8kSplitSpec(
            "https://github.com/openai/grade-school-math",
            study.root / source["path"],
            source["sha256"],
            7473,
            20261007,
            128,
            372,
        ),
        PromptSpec(study.models["prompt"]["instruction"], models["thinking_mode"]),
        models["student"],
        models["teacher"],
        models["student_revision"],
        models["teacher_revision"],
        models["precision"],
        DualClipOpdSpec(
            opd["candidate_top_k"],
            opd["clip_ratio"],
            opd["baseline_floor_epsilon"],
            opd["dual_clip_ratio"],
            opd["log_ratio_limit"],
        ),
        HorizonSpec(
            tuple(horizons["choices"]),
            horizons["initial"],
            horizons["increment"],
            horizons["maximum"],
        ),
        EvaluationSpec(
            "math500",
            "minerva_math",
            1,
            evaluation["temperature"],
            evaluation["top_p"],
            evaluation["max_new_tokens"],
            "fixed128",
        ),
    )
    lora = LoraSpec(**models["lora"], target_modules=tuple(models["lora_targets"]))
    load = NativeTeacherLoadSpec(
        protocol,
        lora,
        training["attention_backend"],
        training["gradient_checkpointing"],
    )
    engine = StreamingEngineSpec(
        protocol.opd,
        protocol.prompt,
        protocol.horizons.increment,
        training["learning_rate"],
        training["weight_decay"],
        training["max_grad_norm"],
        training["max_prompt_tokens"],
        training["generation_batch_size"],
        training["teacher_scoring_batch_size"],
        training["scoring_chunk_rows"],
    )
    return RuntimeSettings(
        protocol,
        load,
        engine,
        AdaptivePolicySpec(**training["adaptive_policy"]),
        training["batch_size"],
        study.seed,
        ResidentRolloutSpec(**training["rollout"]),
    )


def training_examples(study: Study) -> tuple:
    from gstar_opd_formal.data.dataset import Gsm8kExample
    from gstar_opd_formal.data.manifests import read_jsonl

    rows = read_jsonl(study.root / "data/raw/gsm8k_train.jsonl")
    return tuple(
        Gsm8kExample(index, rows[index]["question"], rows[index]["answer"])
        for index in study.manifests["training"]["source_indices"]
    )
