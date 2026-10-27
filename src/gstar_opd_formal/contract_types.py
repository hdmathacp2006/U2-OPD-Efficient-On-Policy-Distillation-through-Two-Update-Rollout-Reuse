"""Pure protocol value types shared by formal training and evaluation."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Gsm8kSplitSpec:
    source_url: str
    source_path: Path
    source_sha256: str
    expected_count: int
    seed: int
    feedback_count: int
    tuning_count: int


@dataclass(frozen=True)
class PromptSpec:
    instruction: str
    thinking_mode: bool


@dataclass(frozen=True)
class OpdSpec:
    candidate_top_k: int
    clip_ratio: float
    baseline_floor_epsilon: float


@dataclass(frozen=True)
class DualClipOpdSpec(OpdSpec):
    dual_clip_ratio: float = 3.0
    log_ratio_limit: float = 20.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.dual_clip_ratio) or self.dual_clip_ratio <= 1:
            raise ValueError("Dual clipping requires a finite coefficient above one")
        if not math.isfinite(self.log_ratio_limit) or self.log_ratio_limit <= 0:
            raise ValueError("Log-ratio limit must be positive and finite")


@dataclass(frozen=True)
class HorizonSpec:
    choices: tuple[int, ...]
    initial: int
    increment: int
    maximum: int


@dataclass(frozen=True)
class EvaluationSpec:
    primary: str
    secondary: str
    samples_per_question: int
    temperature: float
    top_p: float
    max_new_tokens: int
    final_checkpoint: str


@dataclass(frozen=True)
class Protocol:
    version: int
    gsm8k: Gsm8kSplitSpec
    prompt: PromptSpec
    student_model: str
    teacher_model: str
    student_revision: str
    teacher_revision: str
    precision: str
    opd: OpdSpec
    horizons: HorizonSpec
    evaluation: EvaluationSpec


def read_mapping(settings: dict[str, Any], key: str) -> dict[str, Any]:
    value = settings.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def read_text(settings: dict[str, Any], key: str) -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def read_positive_int(settings: dict[str, Any], key: str) -> int:
    value = settings.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def read_positive_float(settings: dict[str, Any], key: str) -> float:
    value = settings.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{key} must be positive")
    return float(value)


def read_revision(settings: dict[str, Any], key: str) -> str:
    revision = read_text(settings, key)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{key} must be a pinned commit SHA")
    return revision
