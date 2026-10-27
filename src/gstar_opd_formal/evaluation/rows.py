"""Durable per-question raw answers and deterministic benchmark grading."""

from dataclasses import dataclass
from pathlib import Path

from gstar_opd_formal.evaluation.graders.math import (
    extract_final_answer,
    parse_math_answer,
    score_gsm8k,
    score_math500,
)
from gstar_opd_formal.opd.rollout import question_sampling_seed
from gstar_opd_formal.storage.files import digest_value, read_json, write_json


def validate_row(row: dict, source: dict, config: dict) -> None:
    if (
        row.get("example_id") != source["id"]
        or row.get("question_sha256") != digest_value(source["question"])
        or row.get("sampling_index") != source["sampling_index"]
        or row.get("question_seed")
        != question_sampling_seed(config["seed"], source["sampling_index"])
        or not isinstance(row.get("output"), str)
        or type(row.get("response_length")) is not int
        or not 0 < row["response_length"] <= config["max_new_tokens"]
        or type(row.get("truncated")) is not bool
    ):
        raise ValueError("Saved question identity/seed/output dimensions changed")
    if row["truncated"] != (row.get("finish_reason") == "length"):
        raise ValueError("Truncation and engine finish reason disagree")


@dataclass
class QuestionStore:
    session: object
    source: tuple

    @property
    def output(self) -> Path:
        return self.session.output

    def grade(self, row: dict, source: dict) -> dict:
        benchmark = self.session.binding["benchmark"]
        if benchmark == "olympiadbench" or (
            benchmark == "minerva_math" and not parse_math_answer(source["answer"])
        ):
            from gstar_opd_formal.evaluation.graders.olympiad import grade_official

            answer = extract_final_answer(row["output"])
            correct = False
            if answer is not None and len(answer) <= 512:
                result = grade_official(
                    [
                        {
                            "gold": source["answer"].replace("np.arcsin", r"\arcsin"),
                            "prediction": answer.replace("np.arcsin", r"\arcsin"),
                            "metadata": source["metadata"],
                        }
                    ],
                    (self.session.grader_python, self.session.study.root),
                )
                correct = result[0]["correct"]
        else:
            scorer = score_gsm8k if benchmark == "dev128" else score_math500
            score = scorer(source["id"], row["output"], source["answer"])
            answer, correct = score.extracted_answer, score.correct
        return {**row, "extracted_answer": answer, "correct": correct}

    def save(self, row: dict) -> None:
        source = next(item for item in self.source if item["id"] == row["example_id"])
        validate_row(row, source, self.session.binding["evaluation_config"])
        value = self.grade(row, source)
        path = self.output / "raw" / f"{digest_value(source['id'])}.json"
        if path.exists():
            raise ValueError("A completed evaluation question must not be overwritten")
        write_json(
            path, {"binding_sha256": digest_value(self.session.binding), "row": value}
        )
        (self.output / "ungraded" / path.name).unlink(missing_ok=True)

    def load(self) -> dict:
        by_id, rows = {item["id"]: item for item in self.source}, {}
        for path in sorted((self.output / "raw").glob("*.json")):
            value = read_json(path)
            row = value["row"]
            key = row["example_id"]
            if (
                key not in by_id
                or key in rows
                or value["binding_sha256"] != digest_value(self.session.binding)
                or path.name != digest_value(key) + ".json"
            ):
                raise ValueError("Duplicate/foreign saved evaluation row")
            validate_row(row, by_id[key], self.session.binding["evaluation_config"])
            if type(row.get("correct")) is not bool or "extracted_answer" not in row:
                raise ValueError("Saved grade is incomplete")
            rows[key] = row
        for path in sorted((self.output / "ungraded").glob("*.json")):
            value = read_json(path)
            if value["binding_sha256"] != digest_value(self.session.binding):
                raise ValueError("Foreign ungraded raw output")
            row = value["row"]
            if row["example_id"] not in rows:
                self.save(row)
        if len(rows) != len(list((self.output / "raw").glob("*.json"))):
            return self.load()
        return rows
