"""Load only a pinned student, with a verified trained adapter when requested."""

import random
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoConfig, AutoTokenizer

from gstar_opd_formal.evaluation.capacity import validate_capacity
from gstar_opd_formal.models.attention import validate_attention_runtime
from gstar_opd_formal.models.graph_rollout import GraphRollout
from gstar_opd_formal.models.loading import attach_lora, load_model_weights
from gstar_opd_formal.models.rollout_spec import ResidentRolloutSpec
from gstar_opd_formal.storage.files import digest_file
from gstar_opd_formal.training.settings import build_settings


@dataclass(frozen=True)
class StudentBundle:
    student: object
    tokenizer: object
    device: object
    rollout: object = None
    attention_environment: dict = None


def validate_adapter(adapter: dict, student: object) -> None:
    expected = get_peft_model_state_dict(student, save_embedding_layers=False)
    if not isinstance(adapter, dict) or adapter.keys() != expected.keys():
        raise ValueError("Evaluation adapter keys differ from the declared LoRA")
    if any(
        not isinstance(adapter[name], torch.Tensor)
        or adapter[name].shape != tensor.shape
        or adapter[name].dtype != tensor.dtype
        or not torch.isfinite(adapter[name]).all()
        for name, tensor in expected.items()
    ):
        raise ValueError("Evaluation adapter shape, dtype or finite values changed")


def load_student(session: object) -> tuple:
    settings = build_settings(session.study)
    config = session.binding["evaluation_config"]
    protocol = settings.protocol
    native = AutoConfig.from_pretrained(
        protocol.student_model,
        revision=protocol.student_revision,
        local_files_only=True,
    )
    capacity = validate_capacity(native, config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    environment = validate_attention_runtime(settings.model_load.attention_backend)
    tokenizer = AutoTokenizer.from_pretrained(
        protocol.student_model, revision=protocol.student_revision
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda", torch.cuda.current_device())
    student = load_model_weights(
        protocol.student_model,
        protocol.student_revision,
        settings.model_load.attention_backend,
    ).to(device)
    checkpoint = session.selection.get("checkpoint")
    if checkpoint is not None:
        from gstar_opd_formal.training.history import validate_history

        if digest_file(checkpoint) != session.binding["checkpoint_sha256"]:
            raise ValueError("Evaluation checkpoint bytes changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        training = session.selection["training_binding"]
        expected_update = session.binding["checkpoint_update"]
        if (
            payload.get("run_contract") != training
            or payload.get("completed_steps") != expected_update
        ):
            raise ValueError(
                "Evaluation checkpoint belongs to a different training run/update"
            )
        validate_history(payload["run_state"], training, expected_update)
        student = attach_lora(student, settings.model_load.lora)
        expected = student.peft_config
        if not expected:
            raise ValueError("Trained evaluation requires the declared LoRA config")
        validate_adapter(payload["adapter"], student)
        set_peft_model_state_dict(student, payload["adapter"])
        del payload
    student.requires_grad_(False)
    student.eval()
    rollout = ResidentRolloutSpec(
        backend="vllm",
        maximum_model_length=config["maximum_model_length"],
        cache_bytes=config["cache_bytes"],
        maximum_sequences=config["concurrency"],
        maximum_batched_tokens=config["maximum_batched_tokens"],
        residency="sleep",
    )
    settings = replace(settings, rollout=rollout)
    bundle = StudentBundle(
        student, tokenizer, device, attention_environment=environment
    )
    owner = SimpleNamespace(
        bundle=bundle, completed_steps=session.binding["checkpoint_update"]
    )
    backend = GraphRollout(owner, settings, config["seed"])
    bundle = replace(bundle, rollout=backend)
    owner.bundle = bundle
    return bundle, settings, capacity
