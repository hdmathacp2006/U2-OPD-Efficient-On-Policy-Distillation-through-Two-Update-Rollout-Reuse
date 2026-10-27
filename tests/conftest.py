"""Small local causal models exercise the real Transformers/PEFT integration."""

from collections.abc import Callable, Iterator

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from gstar_opd_formal.contract_types import OpdSpec, PromptSpec
from gstar_opd_formal.models.loading import (
    LoraSpec,
    ModelBundle,
    attach_lora,
    freeze_teacher,
)
from gstar_opd_formal.opd.engine import EngineSpec, OpdEngine


@pytest.fixture(scope="session", autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_local_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    words = (
        "user",
        "assistant",
        "What",
        "is",
        "2",
        "+",
        "?",
        "Solve",
        "Final",
        "answer",
        ":",
        "4",
    )
    vocab.update({word: index + len(vocab) for index, word in enumerate(words)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        padding_side="left",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tokenizer


def make_local_engine() -> OpdEngine:
    torch.manual_seed(73)
    tokenizer = make_local_tokenizer()
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    student = attach_lora(
        Qwen3ForCausalLM(config), LoraSpec(2, 4, 0.0, ("q_proj", "v_proj"))
    )
    teacher = Qwen3ForCausalLM(config)
    freeze_teacher(teacher)
    bundle = ModelBundle(student, teacher, tokenizer, torch.device("cpu"))
    spec = EngineSpec(
        OpdSpec(4, 0.2, 1e-6), PromptSpec("Final answer:", False), 2, 1e-3, 0.0, 1.0, 64
    )
    return OpdEngine(bundle, spec)


@pytest.fixture
def local_engine() -> OpdEngine:
    return make_local_engine()


@pytest.fixture
def engine_factory() -> Callable[[], OpdEngine]:
    return make_local_engine
