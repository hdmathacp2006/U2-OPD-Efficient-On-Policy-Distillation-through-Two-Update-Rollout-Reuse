"""Lightweight notebook orchestration; every ML/restore operation runs in workers."""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from gstar_opd_formal.contract import ARMS, load_study
from gstar_opd_formal.orchestration.receipts import study_receipt
from gstar_opd_formal.runtime.progress import CommandView, run_command
from gstar_opd_formal.storage.files import digest_value, read_json, write_json


@dataclass(frozen=True)
class Pipeline:
    base: Path
    python: str
    gpu_profile: str = "A100"
    resume_stopped: bool = False
    project_directory: Path | None = None
    evaluation_profile: str = "full"
    selected_benchmarks: tuple[str, ...] = ("math500",)

    @property
    def project(self) -> Path:
        return self.project_directory or self.base / "project"

    @property
    def artifacts(self) -> Path:
        return self.base / "artifacts"

    @property
    def grader(self) -> Path:
        return self.base / "grader_env/bin/python"


def command(settings: Pipeline) -> list[str]:
    return [
        settings.python,
        "-u",
        "-m",
        "gstar_opd_formal.runtime.worker",
        "--project",
        str(settings.project),
        "--artifacts",
        str(settings.artifacts),
        "--destination",
        str(settings.base / "destination.json"),
        "--gpu-profile",
        settings.gpu_profile,
        "--grader-python",
        str(settings.grader),
    ]


def invoke(settings: Pipeline, context: tuple) -> tuple:
    options, label, endpoint, timeout = context
    receipt = settings.base / "worker_receipts" / f"{uuid.uuid4().hex}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.unlink(missing_ok=True)
    started = time.perf_counter()
    view = CommandView(
        settings.project,
        settings.base / "ui_logs",
        label,
        action="train"
        if "train" in options
        else "evaluate"
        if "evaluate" in options
        else "setup",
        endpoint=endpoint,
        timeout_seconds=timeout,
    )
    try:
        run_command(command(settings) + options + ["--receipt", str(receipt)], view)
    except BaseException:
        seconds = time.perf_counter() - started
        if receipt.exists() and ("train" in options or "evaluate" in options):
            preserve_failed_timing(settings, (receipt, seconds))
        raise
    seconds = time.perf_counter() - started
    if not receipt.exists():
        raise ValueError("Worker did not publish a fresh completion receipt")
    return read_json(receipt), seconds


def preserve_failed_timing(settings: Pipeline, context: tuple) -> None:
    receipt, seconds = context
    result = read_json(receipt)
    if (
        not isinstance(result.get("binding"), dict)
        or result.get("run_id") != digest_value(result["binding"])[:32]
    ):
        return
    timing = {
        "id": uuid.uuid4().hex,
        "run_id": result["run_id"],
        "binding_sha256": digest_value(result["binding"]),
        "phase": result["binding"]["action"],
        "status": "failed",
        "seconds": seconds,
        "reused_complete": False,
        "scope": "measured_full_failed_child_lifetime",
    }
    path = settings.base / "client_receipts" / f"{timing['id']}_failed.json"
    write_json(path, timing)
    try:
        invoke(
            settings,
            (
                [
                    "--action",
                    "record-timing",
                    "--input",
                    str(receipt),
                    "--timing",
                    str(path),
                ],
                "Save failed process time to Drive",
                0,
                1800,
            ),
        )
    except Exception as error:
        print(f"failed_process_timing_local_only={path}; publication_error={error}")


def run_phase(settings: Pipeline, context: tuple) -> tuple:
    action, arm, benchmark, update = context
    study = load_study(settings.project)
    endpoint = (
        study.design["updates"]
        if action == "train"
        else (128 if benchmark == "dev128" else study.design["benchmarks"][benchmark])
    )
    options = ["--action", action, "--arm", arm]
    if action == "evaluate":
        options += [
            "--benchmark",
            benchmark,
            "--checkpoint-update",
            str(update),
            "--evaluation-profile",
            settings.evaluation_profile,
            "--allow-new",
        ]
    elif arm == "u2":
        options.append("--allow-new")
    if settings.resume_stopped:
        options.append("--resume-stopped")
    result, seconds = invoke(
        settings,
        (options, f"{arm.upper()} {settings.gpu_profile} {action}", endpoint, 15300),
    )
    if (
        result.get("run_id") != digest_value(result.get("binding"))[:32]
        or result["binding"].get("study_identity") != study.identity
        or result["binding"].get("arm") != arm
        or result["binding"].get("action") != action
    ):
        raise ValueError("Worker result is bound to a foreign run/arm")
    timing = {
        "id": uuid.uuid4().hex,
        "run_id": result["run_id"],
        "binding_sha256": digest_value(result["binding"]),
        "phase": action,
        "seconds": seconds,
        "reused_complete": result["reused_complete"],
        "scope": (
            "full_child_lifetime_includes_models_commits_shutdown_"
            "excludes_setup_restore_before_worker_and_timing_publication"
        ),
    }
    result_path = settings.base / "client_receipts" / f"{timing['id']}_result.json"
    timing_path = result_path.with_name(f"{timing['id']}_timing.json")
    write_json(result_path, result)
    write_json(timing_path, timing)
    invoke(
        settings,
        (
            [
                "--action",
                "record-timing",
                "--input",
                str(result_path),
                "--timing",
                str(timing_path),
            ],
            "Save measured process time to Drive",
            0,
            1800,
        ),
    )
    expected = "workload_complete" if action == "train" else "evaluation_complete"
    if result["status"] != expected:
        raise RuntimeError(
            (
                "Partial work is saved. Confirm the old process stopped, then "
                "resume; the next phase was not started."
            )
        )
    return result, timing


def train_pair(settings: Pipeline) -> dict:
    if settings.gpu_profile != "A100":
        raise ValueError("Train the paired study on A10040GB")
    study = load_study(settings.project)
    results, timings = {}, {}
    for arm in ARMS:
        results[arm], timings[arm] = run_phase(
            settings, ("train", arm, None, study.design["updates"])
        )
    receipt = study_receipt(results, study)
    path = settings.base / "paired_study.json"
    write_json(path, receipt)
    saved, _ = invoke(
        settings,
        (
            ["--action", "publish-study", "--input", str(path)],
            "Save verified paired study to Drive",
            0,
            1800,
        ),
    )
    result = {
        "action": "train_only",
        "evaluation_started": False,
        "study_identity": study.identity,
        "drive_receipt": saved["drive"],
        "arms": {
            arm: {
                "run_id": results[arm]["run_id"],
                "completed_updates": results[arm]["completed_updates"],
                "learning_seconds": results[arm]["accepted_learning_seconds"],
                "process_seconds_this_invocation": timings[arm]["seconds"],
                "reused_complete": results[arm]["reused_complete"],
                "checkpoint_sha256": results[arm]["checkpoint_sha256"],
            }
            for arm in ARMS
        },
    }
    write_json(settings.base / "train_pair_result.json", result)
    return result


def evaluate_arms(settings: Pipeline, arms: tuple[str, ...]) -> dict:
    study = load_study(settings.project)
    if not arms or any(arm not in study.design["eval_arms"] for arm in arms):
        raise ValueError("Unknown evaluation arm selection")
    benchmarks = settings.selected_benchmarks
    if (
        not benchmarks
        or len(set(benchmarks)) != len(benchmarks)
        or any(benchmark not in study.design["benchmarks"] for benchmark in benchmarks)
    ):
        raise ValueError("Unknown or duplicate evaluation benchmark selection")
    results = []
    for arm in arms:
        for benchmark in benchmarks:
            result, timing = run_phase(
                settings, ("evaluate", arm, benchmark, study.design["updates"])
            )
            results.append(
                {
                    "arm": arm,
                    "benchmark": benchmark,
                    "run_id": result["run_id"],
                    "accuracy": result["accuracy"],
                    "count": result["count"],
                    "process_seconds_this_invocation": timing["seconds"],
                    "reused_complete": result["reused_complete"],
                }
            )
    value = {
        "action": (
            "evaluate_base_only" if arms == ("base",) else "evaluate_methods_only"
        ),
        "training_started": False,
        "study_identity": study.identity,
        "results": results,
    }
    write_json(settings.base / f"{value['action']}_result.json", value)
    return value


def evaluate_methods(settings: Pipeline) -> dict:
    return evaluate_arms(settings, ARMS)


def evaluate_base(settings: Pipeline) -> dict:
    return evaluate_arms(settings, ("base",))
