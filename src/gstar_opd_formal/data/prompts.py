"""Use the native Qwen chat template for all project question sets."""

from typing import Protocol

from gstar_opd_formal.contract_types import PromptSpec


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str: ...


def render_question(tokenizer: ChatTokenizer, question: str, spec: PromptSpec) -> str:
    """Apply one fixed user instruction with explicit Qwen thinking mode."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    messages = [{"role": "user", "content": f"{question}\n\n{spec.instruction}"}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=spec.thinking_mode,
    )


def render_native_teacher_question(
    tokenizer: ChatTokenizer, question: str, spec: PromptSpec
) -> str:
    """Use Nemotron's native system and thinking tag with the same task text."""
    if not question.strip() or not spec.thinking_mode:
        raise ValueError("Nemotron teacher requires a nonempty thinking-mode question")
    messages = [{"role": "user", "content": f"{question}\n\n{spec.instruction}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
