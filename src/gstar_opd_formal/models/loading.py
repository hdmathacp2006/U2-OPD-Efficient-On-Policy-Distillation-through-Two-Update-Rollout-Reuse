"""Load the single-device BF16 teacher and LoRA student used by the OPD engine."""

from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from gstar_opd_formal.contract_types import Protocol
from gstar_opd_formal.models.attention import (
    validate_attention_runtime,
    verify_model_attention,
)
from gstar_opd_formal.models.contract import (
    check_model_contract,
    check_native_teacher_contract,
)
from gstar_opd_formal.runtime.events import report_progress


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0 or not self.target_modules:
            raise ValueError("LoRA rank, alpha, and target modules must be nonempty")
        if self.dropout != 0:
            raise ValueError("On-policy likelihood comparison requires zero dropout")


@dataclass(frozen=True)
class ModelBundle:
    student: nn.Module
    teacher: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    rollout: object | None = None
    teacher_tokenizer: PreTrainedTokenizerBase | None = None
    student_instruction: str = ""


@dataclass(frozen=True)
class ModelLoadSpec:
    protocol: Protocol
    lora: LoraSpec
    attention_backend: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class NativeTeacherLoadSpec(ModelLoadSpec):
    teacher_conditioning: str = "native_prompt_student_prefix"

    def __post_init__(self) -> None:
        if self.teacher_conditioning != "native_prompt_student_prefix":
            raise ValueError("Unsupported teacher conditioning")


def attach_lora(student: nn.Module, spec: LoraSpec) -> nn.Module:
    """Freeze the base model and expose only the requested LoRA adapters."""
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
    )
    adapted_student = get_peft_model(student, adapter)
    verify_student_parameters(adapted_student)
    return adapted_student


def verify_student_parameters(student: nn.Module) -> None:
    trainable_names = [
        name for name, value in student.named_parameters() if value.requires_grad
    ]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise ValueError("Only LoRA parameters may be trainable")
    # Train-mode checkpointing must not introduce a second stochastic policy.
    if any(
        module.p != 0 for module in student.modules() if isinstance(module, nn.Dropout)
    ):
        raise ValueError("Student dropout must be zero during rollout and updates")
    if getattr(student.config, "attention_dropout", 0) != 0:
        raise ValueError("Student attention dropout must be zero")


def freeze_teacher(teacher: nn.Module) -> None:
    teacher.requires_grad_(False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.grad = None


def load_model_weights(
    model_name: str, revision: str, attention_backend: str
) -> nn.Module:
    """Use identical precision and attention settings for both pinned models."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    )
    verify_model_attention(model, attention_backend)
    return model


def load_model_bundle(spec: ModelLoadSpec) -> ModelBundle:
    """Fail before downloads when the production BF16 CUDA requirement is unmet."""
    report_progress("attention_preflight_start", backend=spec.attention_backend)
    validate_attention_runtime(spec.attention_backend)
    report_progress("attention_preflight_done", backend=spec.attention_backend)
    device = torch.device("cuda", torch.cuda.current_device())
    protocol = spec.protocol
    if isinstance(spec, NativeTeacherLoadSpec):
        check_native_teacher_contract(protocol)
    else:
        check_model_contract(protocol)
    report_progress(
        "model_contract_verified",
        student=protocol.student_model,
        teacher=protocol.teacher_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        protocol.student_model, revision=protocol.student_revision
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    report_progress("student_loading_start", model=protocol.student_model)
    student = load_model_weights(
        protocol.student_model,
        protocol.student_revision,
        spec.attention_backend,
    ).to(device)
    student = attach_lora(student, spec.lora)
    if spec.gradient_checkpointing:
        # Non-reentrant checkpointing supports frozen embedding/base parameters.
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    report_progress("student_ready", device=device)
    report_progress("teacher_loading_start", model=protocol.teacher_model)
    teacher = load_model_weights(
        protocol.teacher_model,
        protocol.teacher_revision,
        spec.attention_backend,
    ).to(device)
    freeze_teacher(teacher)
    report_progress("teacher_ready", device=device)
    teacher_tokenizer = None
    if isinstance(spec, NativeTeacherLoadSpec):
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            protocol.teacher_model, revision=protocol.teacher_revision
        )
        teacher_tokenizer.padding_side = "left"
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    return ModelBundle(
        student,
        teacher,
        tokenizer,
        device,
        teacher_tokenizer=teacher_tokenizer,
        student_instruction=protocol.prompt.instruction,
    )
