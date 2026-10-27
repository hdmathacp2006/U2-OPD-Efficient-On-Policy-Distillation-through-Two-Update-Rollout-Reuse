"""Strict final-answer extraction and held-out math grading."""

import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

BOX_PREFIX = r"\boxed{"
FINAL_LINE = re.compile(
    r"(?im)^\s*(?:####|final\s+answer(?:\s+is)?\s*[:=]?|answer\s*[:=])\s*(.+?)\s*$"
)
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?%?")
MAX_ANSWER_CHARACTERS = 512
MATH_VERIFY_TIMEOUT = 5 if os.name == "posix" else None


@dataclass(frozen=True)
class ScoredAnswer:
    example_id: str
    extracted_answer: str | None
    correct: bool


def extract_final_answer(output: str) -> str | None:
    """Use an explicit final-answer marker after any completed thinking block."""
    visible_output = extract_visible_answer_text(output)
    if visible_output is None:
        return None

    candidates = find_boxed_candidates(visible_output)
    candidates.extend(
        (match.start(), match.group(1).strip())
        for match in FINAL_LINE.finditer(visible_output)
    )
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1] or None

    standalone_answer = visible_output.strip().rstrip(".")
    if NUMBER.fullmatch(standalone_answer):
        return standalone_answer
    return None


def score_gsm8k(example_id: str, output: str, gold_solution: str) -> ScoredAnswer:
    """Compare explicit GSM8K final numbers exactly after formatting cleanup."""
    gold_matches = list(FINAL_LINE.finditer(gold_solution))
    if not gold_matches:
        raise ValueError(f"Missing GSM8K gold delimiter for {example_id}")
    gold_number = parse_numeric_fraction(gold_matches[-1].group(1))
    if gold_number is None:
        raise ValueError(f"Non-numeric GSM8K gold answer for {example_id}")

    extracted_answer = extract_final_answer(output)
    predicted_number = parse_numeric_fraction(extracted_answer)
    return ScoredAnswer(
        example_id=example_id,
        extracted_answer=extracted_answer,
        correct=predicted_number is not None and predicted_number == gold_number,
    )


def score_math500(example_id: str, output: str, gold_answer: str) -> ScoredAnswer:
    """Check mathematical equivalence with a pinned Math-Verify version."""
    extracted_answer = extract_final_answer(output)
    if extracted_answer is None:
        return ScoredAnswer(example_id, None, False)
    if len(extracted_answer) > MAX_ANSWER_CHARACTERS:
        return ScoredAnswer(example_id, extracted_answer, False)

    gold_expressions = parse_math_answer(gold_answer)
    if not gold_expressions:
        raise ValueError(f"Cannot parse MATH-500 gold answer for {example_id}")
    predicted_expressions = parse_math_answer(extracted_answer)
    correct = bool(predicted_expressions) and verify(
        gold_expressions,
        predicted_expressions,
        timeout_seconds=MATH_VERIFY_TIMEOUT,
    )
    return ScoredAnswer(example_id, extracted_answer, correct)


def exact_answer_accuracy(scores: list[ScoredAnswer]) -> float:
    """Return the fraction correct; an empty evaluation is invalid."""
    if not scores:
        raise ValueError("Cannot evaluate an empty score list")
    return sum(score.correct for score in scores) / len(scores)


def extract_visible_answer_text(output: str) -> str | None:
    if "<think>" not in output:
        return output
    closing_tag = output.rfind("</think>")
    if closing_tag < output.rfind("<think>"):
        return None
    return output[closing_tag + len("</think>") :]


def find_boxed_candidates(output: str) -> list[tuple[int, str]]:
    candidates = []
    search_start = 0
    while (box_start := output.find(BOX_PREFIX, search_start)) >= 0:
        content_start = box_start + len(BOX_PREFIX)
        content_end = find_matching_brace(output, content_start)
        if content_end is None:
            return []
        candidates.append((box_start, output[content_start:content_end].strip()))
        search_start = content_end + 1
    return candidates


def find_matching_brace(output: str, content_start: int) -> int | None:
    depth = 1
    for position in range(content_start, len(output)):
        if output[position] == "{":
            depth += 1
        elif output[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def parse_numeric_fraction(answer: str | None) -> Fraction | None:
    if answer is None:
        return None
    normalized = answer.strip().rstrip(". ").replace(",", "")
    normalized = normalized.removeprefix("$").removesuffix("$").strip()
    normalized = normalized.removesuffix("%").strip()
    if not NUMBER.fullmatch(normalized):
        return None
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", maxsplit=1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(normalized))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def parse_math_answer(answer: str) -> list[object]:
    latex_answer = rf"\boxed{{{answer}}}"
    return parse(
        latex_answer,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
        fallback_mode="no_fallback",
        parsing_timeout=MATH_VERIFY_TIMEOUT,
    )
