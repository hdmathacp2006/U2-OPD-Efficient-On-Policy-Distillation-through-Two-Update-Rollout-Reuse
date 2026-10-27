"""Aggregate complete formal runs with explicit hardware and timing comparability."""

from pathlib import Path

from gstar_opd_formal.reporting.timing import timing_report
from gstar_opd_formal.storage.files import read_json, write_json


def paired_counts(left: list[dict], right: list[dict]) -> dict:
    before, after = (
        {row["example_id"]: row for row in left},
        {row["example_id"]: row for row in right},
    )
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation requires the identical question IDs")
    return {
        "count": len(before),
        "gains": sum(
            not before[key]["correct"] and after[key]["correct"] for key in before
        ),
        "losses": sum(
            before[key]["correct"] and not after[key]["correct"] for key in before
        ),
    }


def build_report(artifacts: Path) -> dict:
    evaluations, training = [], []
    for directory, target in (("evaluation", evaluations), ("training", training)):
        for path in sorted((artifacts / directory).glob("*/summary.json")):
            binding = read_json(path.parent / "binding.json")
            target.append(
                {
                    "run_id": path.parent.name,
                    "binding": binding,
                    "summary": read_json(path),
                    "timing": timing_report(path.parent),
                }
            )
    studies = sorted(
        {row["binding"]["study_identity"] for row in evaluations + training}
    )
    value = {
        "schema": "formal_report_v1",
        "studies": studies,
        "training": training,
        "evaluations": evaluations,
        "status": "no_results_yet" if not studies else "results_available",
        "interpretation": (
            "fixed128 compares equal optimizer updates, not equal fresh "
            "coverage or compute; historical mixed-hardware results are "
            "exploratory"
        ),
    }
    pairs = {}
    for study in studies:
        selected = {
            row["binding"]["arm"]: row
            for row in training
            if row["binding"]["study_identity"] == study
            and row["summary"]["status"] == "workload_complete"
        }
        if set(selected) == {"adaptive", "u2"}:
            a, u = selected["adaptive"], selected["u2"]
            comparable = a["binding"]["environment"] == u["binding"]["environment"]
            pairs[study] = {
                "matched_runtime": comparable,
                "learning_speedup": a["summary"]["accepted_learning_seconds"]
                / u["summary"]["accepted_learning_seconds"]
                if comparable
                else None,
                "process_speedup": a["timing"]["closed_work_process_seconds"]
                / u["timing"]["closed_work_process_seconds"]
                if comparable and u["timing"]["closed_work_process_seconds"]
                else None,
            }
    value["training_pairs"] = pairs
    write_json(artifacts / "reports/comparison.json", value)
    return value
