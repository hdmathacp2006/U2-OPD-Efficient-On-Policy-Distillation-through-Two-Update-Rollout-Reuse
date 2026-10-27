"""Check Qwen model metadata and token-id compatibility without loading weights."""

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from gstar_opd_formal.contract_types import PromptSpec, Protocol
from gstar_opd_formal.data.prompts import (
    render_native_teacher_question,
    render_question,
)


def check_model_contract(protocol: Protocol) -> dict[str, str | int | bool]:
    """Ensure student token IDs can be scored by the frozen teacher."""
    student_config = AutoConfig.from_pretrained(
        protocol.student_model, revision=protocol.student_revision
    )
    teacher_config = AutoConfig.from_pretrained(
        protocol.teacher_model, revision=protocol.teacher_revision
    )
    check_model_metadata(student_config, teacher_config)

    student_tokenizer = AutoTokenizer.from_pretrained(
        protocol.student_model, revision=protocol.student_revision
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        protocol.teacher_model, revision=protocol.teacher_revision
    )
    check_tokenizer_contract(student_tokenizer, teacher_tokenizer, protocol.prompt)
    return {
        "student_revision": protocol.student_revision,
        "teacher_revision": protocol.teacher_revision,
        "model_type": student_config.model_type,
        "vocab_size": student_config.vocab_size,
        "eos_token_id": student_config.eos_token_id,
        "tokenizer_mapping_equal": True,
        "tokenizer_backend_equal": True,
        "chat_template_equal": True,
        "thinking_mode": protocol.prompt.thinking_mode,
        "weight_dtype": "bfloat16",
    }


def check_model_metadata(
    student_config: PretrainedConfig, teacher_config: PretrainedConfig
) -> None:
    """Reject incompatible next-token spaces before loading weight files."""
    if student_config.model_type != teacher_config.model_type:
        raise ValueError("Student and teacher architectures differ")
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("Student and teacher vocabulary sizes differ")
    if student_config.eos_token_id != teacher_config.eos_token_id:
        raise ValueError("Student and teacher EOS token IDs differ")
    if student_config.to_dict().get("dtype") != "bfloat16":
        raise ValueError("Student metadata does not specify BF16 weights")
    if teacher_config.to_dict().get("dtype") != "bfloat16":
        raise ValueError("Teacher metadata does not specify BF16 weights")


def check_token_space(
    student: PreTrainedTokenizerBase, teacher: PreTrainedTokenizerBase
) -> None:
    """A native teacher prompt still requires exactly the same next-token IDs."""
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Student and teacher token-to-ID mappings differ")
    if student.backend_tokenizer.to_str() != teacher.backend_tokenizer.to_str():
        raise ValueError("Student and teacher tokenizer rules differ")
    if student.eos_token_id != teacher.eos_token_id:
        raise ValueError("Student and teacher EOS IDs differ")


def check_native_teacher_contract(protocol: Protocol) -> dict[str, object]:
    if protocol.teacher_model != "nvidia/Nemotron-Cascade-8B-Thinking":
        raise ValueError("Native teacher mode is validated only for pinned Nemotron")
    student = AutoConfig.from_pretrained(
        protocol.student_model, revision=protocol.student_revision
    )
    teacher = AutoConfig.from_pretrained(
        protocol.teacher_model, revision=protocol.teacher_revision
    )
    check_model_metadata(student, teacher)
    tokenizers = tuple(
        AutoTokenizer.from_pretrained(name, revision=revision)
        for name, revision in (
            (protocol.student_model, protocol.student_revision),
            (protocol.teacher_model, protocol.teacher_revision),
        )
    )
    check_token_space(*tokenizers)
    if not tokenizers[1].chat_template:
        raise ValueError("Native teacher conditioning requires a chat template")
    sample = render_native_teacher_question(
        tokenizers[1], "What is 2 + 2?", protocol.prompt
    )
    if " /think<|im_end|>" not in sample or not sample.endswith(
        "<|im_start|>assistant\n"
    ):
        raise ValueError("Pinned Nemotron thinking template changed")
    return {
        "student_revision": protocol.student_revision,
        "teacher_revision": protocol.teacher_revision,
        "tokenizer_mapping_equal": True,
        "tokenizer_backend_equal": True,
        "chat_template_equal": tokenizers[0].chat_template
        == tokenizers[1].chat_template,
        "teacher_conditioning": "native_prompt_student_prefix",
    }


def check_tokenizer_contract(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    prompt: PromptSpec,
) -> None:
    """Identical vocabularies alone do not ensure identical response token IDs."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher token-to-ID mappings differ")
    if (
        student_tokenizer.backend_tokenizer.to_str()
        != teacher_tokenizer.backend_tokenizer.to_str()
    ):
        raise ValueError("Student and teacher tokenizer rules differ")
    if student_tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise ValueError("Student and teacher chat templates differ")

    sample_question = "What is 2 + 2?"
    student_prompt = render_question(student_tokenizer, sample_question, prompt)
    teacher_prompt = render_question(teacher_tokenizer, sample_question, prompt)
    if student_prompt != teacher_prompt:
        raise ValueError("Rendered student and teacher prompts differ")
    if student_tokenizer.encode(student_prompt) != teacher_tokenizer.encode(
        teacher_prompt
    ):
        raise ValueError("Rendered prompts produce different token IDs")
