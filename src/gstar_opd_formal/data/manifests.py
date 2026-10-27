"""Prepare frozen source-index manifests and benchmark-specific gold records."""

import json
import random
from pathlib import Path

from gstar_opd_formal.contract import COUNTS, read_yaml
from gstar_opd_formal.storage.files import (
    digest_value,
    file_record,
    read_json,
    write_json,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def normalize_question(question: str) -> str:
    return " ".join(question.split())


def minerva_gold(row: dict) -> str:
    from gstar_opd_formal.evaluation.graders.math import find_boxed_candidates

    candidates = find_boxed_candidates(row["solution"])
    if not candidates:
        raise ValueError(f"Minerva solution has no complete gold box: {row['idx']}")
    return candidates[-1][1]


def benchmark_records(benchmark: str, rows: list[dict]) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        if benchmark == "math500":
            identity, question, gold = (
                str(row.get("unique_id", index)),
                row["problem"],
                row["answer"],
            )
        elif benchmark == "minerva_math":
            identity, question, gold = (
                str(row["idx"]),
                row["problem"],
                minerva_gold(row),
            )
        elif benchmark == "olympiadbench":
            identity = str(row["id"])
            question = "\n\n".join(
                value for value in (row.get("context"), row["question"]) if value
            )
            if "<img_" in question:
                raise ValueError("Formal Olympiad evaluation is English text-only math")
            answers = row["final_answer"]
            if (
                not isinstance(answers, list)
                or len(answers) != 1
                or not answers[0].strip()
            ):
                raise ValueError(
                    (
                        "Olympiad gold needs one final-answer field, possibly "
                        "containing several answers"
                    )
                )
            gold = answers[0]
        else:
            raise ValueError("Unknown benchmark")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(gold, str)
            or not gold.strip()
        ):
            raise ValueError("Empty benchmark question or gold")
        records.append(
            {
                "id": f"{benchmark}:{identity}",
                "sampling_index": index,
                "question": question,
                "answer": gold,
                "metadata": {
                    key: row.get(key)
                    for key in (
                        "type",
                        "level",
                        "subfield",
                        "unit",
                        "error",
                        "answer_type",
                        "is_multiple_answer",
                    )
                },
            }
        )
    return records


def prepare_data(root: Path) -> dict:
    design = read_yaml(root / "configs/studies/formal128.yaml")
    manifests = root / "data/manifests"
    forbidden = set()
    for benchmark, count in COUNTS.items():
        source = root / f"data/raw/{benchmark}.jsonl"
        rows = benchmark_records(benchmark, read_jsonl(source))
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise ValueError("Wrong benchmark count or duplicate IDs")
        forbidden.update(normalize_question(row["question"]) for row in rows)
        value = {
            "schema": "formal_benchmark_manifest_v1",
            "benchmark": benchmark,
            "source_files": [file_record(source, root)],
            "ids": [row["id"] for row in rows],
            "records": rows,
            "records_sha256": digest_value(rows),
            "selection": "full_declared_benchmark",
        }
        if benchmark == "math500":
            exposed = read_json(manifests / "historical_math103.json")
            if (
                exposed["source_sha256"] != value["source_files"][0]["sha256"]
                or len(exposed["source_indices"]) != 103
            ):
                raise ValueError("Historical MATH103 exposure manifest changed")
            value["historical_ids"] = [
                rows[index]["id"] for index in exposed["source_indices"]
            ]
        write_json(manifests / f"{benchmark}.json", value)
    source = root / "data/raw/gsm8k_train.jsonl"
    examples = read_jsonl(source)
    split = read_json(manifests / "gsm8k_split.v1.json")
    if (
        len(examples) != split["total"]
        or file_record(source, root)["sha256"] != split["source_sha256"]
    ):
        raise ValueError("GSM8K train source differs from the original frozen split")
    dev = split["feedback_indices"][: design["dev_count"]]
    reserved = set(split["feedback_indices"]) | set(split["tuning_indices"])
    forbidden.update(
        normalize_question(examples[index]["question"]) for index in reserved
    )
    eligible = [
        index
        for index in split["training_indices"]
        if index not in reserved
        and normalize_question(examples[index]["question"]) not in forbidden
    ]
    random.Random(design["seed"]).shuffle(eligible)
    unique, order = set(), []
    for index in eligible:
        question = normalize_question(examples[index]["question"])
        if question not in unique:
            unique.add(question)
            order.append(index)
        if len(order) == design["training_pool"]:
            break
    if len(order) != 1024 or set(order) & set(dev):
        raise ValueError("Insufficient disjoint, deduplicated training questions")
    files = [file_record(source, root)]
    write_json(
        manifests / f"train1024_seed{design['seed']}.json",
        {
            "schema": "formal_train_manifest_v1",
            "source_files": files,
            "seed": design["seed"],
            "source_indices": order,
            "reserved_indices_sha256": digest_value(sorted(reserved)),
            "benchmark_overlap_excluded_before_selection": True,
            "order_policy": "paired_seed_shared_order_U2_consumes_first256",
        },
    )
    write_json(
        manifests / "dev128.json",
        {
            "schema": "formal_dev_manifest_v1",
            "source_files": files,
            "source_indices": dev,
            "role": "checkpoint_reporting_only_not_final_test_selection",
        },
    )
    return {
        "training_pool": len(order),
        "dev": len(dev),
        "benchmarks": design["benchmarks"],
        "prepared_snapshots": COUNTS,
    }
