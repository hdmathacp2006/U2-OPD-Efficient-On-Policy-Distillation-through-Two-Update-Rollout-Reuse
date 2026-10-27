"""Report every selected item, including truncation and extraction failures."""


def evaluation_metrics(rows: list[dict], expected: int) -> dict:
    count = len(rows)
    if count > expected or len({row["example_id"] for row in rows}) != count:
        raise ValueError("Evaluation has duplicate or excessive results")
    correct = sum(row["correct"] for row in rows)
    return {
        "status": "evaluation_complete" if count == expected else "evaluation_partial",
        "count": count,
        "expected_count": expected,
        "correct": correct,
        "accuracy": correct / count if count == expected else None,
        "partial_accuracy": correct / count if 0 < count < expected else None,
        "truncated_count": sum(row["truncated"] for row in rows),
        "extraction_failures": sum(row["extracted_answer"] is None for row in rows),
        "generated_tokens": sum(row["response_length"] for row in rows),
    }
