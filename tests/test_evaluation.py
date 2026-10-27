import pytest

from gstar_opd_formal.evaluation.graders.math import (
    ScoredAnswer,
    exact_answer_accuracy,
    extract_final_answer,
    score_gsm8k,
    score_math500,
)


def test_final_answer_is_taken_after_thinking() -> None:
    output = "<think>Incorrect \\boxed{7}</think>Final answer: \\boxed{9}"
    assert extract_final_answer(output) == "9"


def test_nested_boxed_expression_is_preserved() -> None:
    assert extract_final_answer(r"Therefore \boxed{\frac{14}{3}}") == r"\frac{14}{3}"


def test_unfinished_thinking_and_unmarked_reasoning_are_not_answers() -> None:
    assert extract_final_answer("<think>42") is None
    assert extract_final_answer("First calculate 3, then 42.") is None


def test_gsm8k_scores_normalized_numbers() -> None:
    result = score_gsm8k("item", r"Final answer: \boxed{1,200}", "Work\n#### 1200")
    assert result.correct


def test_gsm8k_rejects_extraction_failure() -> None:
    result = score_gsm8k("item", "The calculation reached 72 and stopped", "#### 72")
    assert not result.correct


def test_gsm8k_rejects_missing_gold_marker() -> None:
    with pytest.raises(ValueError, match="gold delimiter"):
        score_gsm8k("item", "\\boxed{4}", "4")


def test_math500_accepts_equivalent_fractions() -> None:
    result = score_math500("item", r"Final answer: \boxed{14/3}", r"\frac{14}{3}")
    assert result.correct


def test_math500_accepts_equivalent_symbolic_expressions() -> None:
    result = score_math500("symbolic", r"\boxed{x+x}", "2x")
    assert result.correct


def test_math500_rejects_wrong_or_missing_answer() -> None:
    assert not score_math500("item", r"\boxed{3}", "4").correct
    assert not score_math500("item", "I calculated 4", "4").correct


def test_accuracy_requires_nonempty_scores() -> None:
    assert (
        exact_answer_accuracy(
            [ScoredAnswer("a", "1", True), ScoredAnswer("b", "2", False)]
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="empty"):
        exact_answer_accuracy([])
