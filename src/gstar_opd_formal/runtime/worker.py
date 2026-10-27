"""Worker-only training/evaluation, verified recovery and durable receipts."""

import argparse
import json
import os
import re
import signal
import time
import uuid
from dataclasses import replace
from pathlib import Path

from gstar_opd_formal.contract import ARMS, load_study, read_yaml, stable_environment
from gstar_opd_formal.orchestration.receipts import validate_pair
from gstar_opd_formal.paths import Layout
from gstar_opd_formal.runtime.session import Session
from gstar_opd_formal.runtime.timing import Timeline
from gstar_opd_formal.storage.controls import publish_named, read_named
from gstar_opd_formal.storage.drive import DriveStore, authenticated_service
from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    read_json,
    verify_tree,
    write_json,
)
from gstar_opd_formal.storage.local import LocalStore
from gstar_opd_formal.storage.restore import backup_local, restore_training
from gstar_opd_formal.storage.runs import open_store


def open_storage(args: argparse.Namespace) -> tuple:
    destination = read_json(args.destination)
    study = load_study(args.project)
    if destination["root_folder_id"] != study.design["drive_root_id"]:
        raise ValueError("Execution destination changed")
    backend = os.environ.get("GSTAR_STORAGE_BACKEND", "local")
    if backend == "local":
        storage_root = os.environ.get("GSTAR_STORAGE_ROOT")
        if not storage_root:
            raise ValueError("Set GSTAR_STORAGE_ROOT for local or Modal storage")
        drive = LocalStore(Path(storage_root), args.artifacts / ".store_transfers")
    elif backend == "drive":
        if not os.environ.get("GSTAR_DRIVE_ROOT_ID"):
            raise ValueError("Set GSTAR_DRIVE_ROOT_ID for Drive storage")
        drive = DriveStore(authenticated_service(), args.artifacts / ".drive_transfers")
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
    drive.require_destination(destination["runs_folder_id"])
    drive.require_destination(destination["studies_folder_id"])
    return study, drive, destination


def select_evaluation(args: argparse.Namespace, context: tuple) -> tuple:
    study, drive, destination = context
    if args.benchmark == "dev128":
        from gstar_opd_formal.data.manifests import read_jsonl

        raw = read_jsonl(study.root / "data/raw/gsm8k_train.jsonl")
        source = tuple(
            {
                "id": f"dev128:{index}",
                "sampling_index": index,
                "question": raw[index]["question"],
                "answer": raw[index]["answer"],
                "metadata": {},
            }
            for index in study.manifests["dev"]["source_indices"]
        )
    elif args.benchmark in study.design["benchmarks"]:
        source = tuple(study.manifests[args.benchmark]["records"])
    else:
        raise ValueError("Unknown evaluation benchmark/panel")
    if args.arm == "base":
        return source, {
            "checkpoint": None,
            "checkpoint_sha256": None,
            "checkpoint_update": 0,
            "model_revision": study.models["models"]["student_revision"],
        }
    pair = read_named(
        drive, (destination["studies_folder_id"], study.identity + ".json")
    )
    if pair is None:
        raise ValueError("Complete the paired train-only study before final evaluation")
    paired_study = study
    if args.evaluation_profile in {"16k", "12k_a100", "12k_a100_b20", "12k_a100_b10"}:
        try:
            training_code = pair["arms"]["adaptive"]["binding"]["execution_code"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Paired receipt lacks its training runtime identity"
            ) from error
        if not isinstance(training_code, str) or not re.fullmatch(
            r"[a-f0-9]{64}", training_code
        ):
            raise ValueError("Paired receipt has an invalid training runtime identity")
        paired_study = replace(study, code={**study.code, "execution": training_code})
    validate_pair(pair, paired_study)
    result = pair["arms"][args.arm]
    layout = Layout(study.root, args.artifacts)
    restored = restore_training((drive, layout, destination), result)
    update = args.checkpoint_update
    if update not in study.design["retained_updates"] or (
        args.benchmark != "dev128" and update != study.design["updates"]
    ):
        raise ValueError("Final benchmarks require the declared endpoint checkpoint")
    selected = result["retained_checkpoints"].get(str(update))
    if selected is None:
        raise ValueError("Paired receipt lacks the requested retained checkpoint")
    checkpoint = restored["directory"] / selected["path"]
    if digest_file(checkpoint) != selected["sha256"]:
        raise ValueError("Retained checkpoint hash differs from paired receipt")
    return source, {
        "checkpoint": checkpoint,
        "checkpoint_sha256": selected["sha256"],
        "checkpoint_update": update,
        "training_binding": result["binding"],
    }


def execution_binding(args: argparse.Namespace, context: tuple) -> tuple:
    study, drive, destination, environment = context
    if args.action == "train":
        if (
            args.arm not in ARMS
            or args.gpu_profile != "A100"
            or args.evaluation_profile != "full"
        ):
            raise ValueError("Train Adaptive/U2 on A100")
        return study.training_binding(args.arm, environment), (), {}
    if args.arm not in {"base", *ARMS}:
        raise ValueError("Unknown evaluation model")
    if args.evaluation_profile in {"12k_a100", "12k_a100_b20", "12k_a100_b10"} and (
        args.arm not in ARMS or args.gpu_profile != "A100"
    ):
        raise ValueError("12k A100 evaluation is reserved for Adaptive/U2 on A100")
    if args.evaluation_profile == "12k_base_l4" and (
        args.arm != "base" or args.gpu_profile != "L4"
    ):
        raise ValueError("12k base evaluation requires the base model on L4")
    source, selection = select_evaluation(args, (study, drive, destination))
    evaluation = study.evaluation
    if args.evaluation_profile == "16k":
        evaluation = read_yaml(study.root / "configs/evaluation_16k.yaml")
        if evaluation != {
            **study.evaluation,
            "max_new_tokens": 16384,
            "maximum_model_length": 17664,
        }:
            raise ValueError("16k profile changed fields outside the output budget")
    elif args.evaluation_profile == "12k_a100":
        evaluation = {
            **study.evaluation,
            "max_new_tokens": 12288,
            "maximum_model_length": 13568,
            "concurrency": 8,
            "cache_bytes": 16 * 2**30,
            "maximum_batched_tokens": 4096,
        }
    elif args.evaluation_profile == "12k_base_l4":
        evaluation = {
            **study.evaluation,
            "max_new_tokens": 12288,
            "maximum_model_length": 13568,
            "concurrency": 4,
        }
    elif args.evaluation_profile in {"12k_a100_b20", "12k_a100_b10"}:
        batch, cache_gib = (
            (20, 35) if args.evaluation_profile == "12k_a100_b20" else (10, 18)
        )
        evaluation = {
            **study.evaluation,
            "max_new_tokens": 12288,
            "maximum_model_length": 13568,
            "concurrency": batch,
            "cache_bytes": cache_gib * 2**30,
            "maximum_batched_tokens": 4096,
        }
    binding = {
        "schema": "opd_formal_v1",
        "action": "evaluate",
        "study_identity": study.identity,
        "execution_code": study.code["execution"],
        "environment": stable_environment(environment),
        "arm": args.arm,
        "benchmark": args.benchmark,
        "evaluation_config": evaluation,
        "source_ids_sha256": digest_value([row["id"] for row in source]),
        "checkpoint_update": selection["checkpoint_update"],
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "model_revision": study.models["models"]["student_revision"],
    }
    if args.evaluation_profile != "full":
        binding["evaluation_profile"] = args.evaluation_profile
    return binding, source, selection


def prepare_output(args: argparse.Namespace, context: tuple) -> None:
    store, layout, binding, output = context
    markers = store.drive.children(store.folders["commits"])
    if markers:
        manifest = store.latest_manifest()
        keep_local = False
        if output.exists():
            try:
                verify_tree(output, manifest["files"])
                keep_local = True
            except (ValueError, FileNotFoundError):
                if args.resume_stopped and binding["action"] == "train":
                    from gstar_opd_formal.storage.local_recovery import (
                        local_training_boundary,
                    )

                    local_count = local_training_boundary(output, binding)
                    keep_local = local_count > 0 and local_count >= manifest["boundary"]
            if not keep_local:
                backup_local(output, layout.artifacts)
        if not keep_local:
            store.restore(output)
        summary = (
            read_json(output / "summary.json")
            if (output / "summary.json").exists()
            else {}
        )
        complete = summary.get("status") in {"workload_complete", "evaluation_complete"}
        if not complete and not args.resume_stopped:
            raise ValueError(
                (
                    "An incomplete run exists; resume only after its previous "
                    "worker stopped"
                )
            )
    elif args.resume_stopped and not args.allow_new:
        raise ValueError("No committed run to resume; do not silently start fresh")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Uncommitted local output exists; preserve it before selecting recovery"
        )


def open_session(args: argparse.Namespace, context: tuple) -> Session:
    from gstar_opd_formal.runtime.environment import inspect_environment

    study, drive, destination = context
    environment = inspect_environment(args.gpu_profile)
    binding, source, selection = execution_binding(args, (*context, environment))
    layout = Layout(study.root, args.artifacts)
    run_id = digest_value(binding)[:32]
    role = (
        args.arm
        if args.action == "train"
        else f"{args.arm}_{args.benchmark}_{binding['checkpoint_update']}"
    )
    if args.action == "evaluate" and args.evaluation_profile != "full":
        role += f"_{args.evaluation_profile}"
    registry_name = f"allocation_{study.identity}_{args.action}_{role}.json"
    allocated = read_named(drive, (destination["studies_folder_id"], registry_name))
    if allocated is not None and (
        allocated.get("binding") != binding or allocated.get("run_id") != run_id
    ):
        raise ValueError(
            (
                "This study role already belongs to another runtime binding; do"
                " not silently duplicate it"
            )
        )
    if allocated is None:
        allocation = args.artifacts / "allocations" / registry_name
        write_json(allocation, {"run_id": run_id, "binding": binding})
        publish_named(
            drive, allocation, (destination["studies_folder_id"], registry_name)
        )
    output = layout.output(
        "training" if args.action == "train" else "evaluation", run_id
    )
    store = open_store(drive, (destination["runs_folder_id"], run_id, binding, source))
    store.claim("resume" if args.resume_stopped else "run")
    try:
        prepare_output(args, (store, layout, binding, output))
        output.mkdir(parents=True, exist_ok=True)
        if (output / "binding.json").exists() and read_json(
            output / "binding.json"
        ) != binding:
            raise ValueError("Local binding differs from selected execution")
        write_json(output / "binding.json", binding)
        write_json(output / "environment.json", environment)
    except BaseException:
        store.release()
        raise
    session = Session(
        study,
        output,
        binding,
        environment,
        store,
        Timeline(output, digest_value(binding)),
        time.perf_counter() + study.design["segment_seconds"],
        source,
        selection,
        str(args.grader_python or ""),
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: session.stop.set())
    print(f"formal_run_id={run_id} action={args.action} output={output}", flush=True)
    return session


def result_receipt(session: Session, summary: dict, reused: bool) -> dict:
    value = {
        **summary,
        "run_id": digest_value(session.binding)[:32],
        "output_dir": str(session.output),
        "binding": session.binding,
        "reused_complete": reused,
    }
    if session.binding["action"] == "train":
        value["retained_checkpoints"] = {
            str(update): {
                "path": f"checkpoints/update_{update:04d}.pt",
                "sha256": digest_file(
                    session.output / f"checkpoints/update_{update:04d}.pt"
                ),
            }
            for update in session.study.design["retained_updates"]
            if (session.output / f"checkpoints/update_{update:04d}.pt").exists()
        }
    return value


def finish_attempt(
    session: Session, attempt: dict, error: BaseException | None
) -> None:
    attempt = {
        **attempt,
        "finished_at": time.time(),
        "error": repr(error) if error else None,
        "binding_sha256": digest_value(session.binding),
    }
    path = session.output / "attempts" / f"{attempt['attempt_id']}.json"
    write_json(path, attempt)
    session.timeline.flush()
    summary_path = session.output / "summary.json"
    if not summary_path.exists():
        if session.binding["action"] == "evaluate":
            from gstar_opd_formal.reporting.metrics import evaluation_metrics

            rows = [
                read_json(path)["row"]
                for path in (session.output / "raw").glob("*.json")
            ]
            summary = {
                "schema": "formal_evaluation_v1",
                "binding_sha256": digest_value(session.binding),
                **evaluation_metrics(rows, len(session.source)),
            }
        else:
            summary = {
                "schema": "formal_training_v1",
                "binding_sha256": digest_value(session.binding),
                "status": "checkpoint_unavailable",
                "completed_updates": 0,
            }
        write_json(summary_path, summary)
    if summary_path.exists():
        summary = read_json(summary_path)
        if error is not None and session.binding["action"] == "evaluate":
            from gstar_opd_formal.reporting.metrics import evaluation_metrics

            rows = [
                read_json(path)["row"]
                for path in (session.output / "raw").glob("*.json")
            ]
            summary.update(evaluation_metrics(rows, len(session.source)))
            write_json(summary_path, summary)
        boundary = summary.get("completed_updates", summary.get("count", 0))
        session.store.commit(session.output, boundary)
    else:
        session.store.drive.upload(
            path,
            (
                session.store.folders["receipts"],
                f"{digest_file(path)}_attempt_{attempt['attempt_id']}.json",
            ),
        )


def run_selected(args: argparse.Namespace, context: tuple) -> dict:
    from gstar_opd_formal.runtime.events import configure_progress_logging

    configure_progress_logging()
    session = open_session(args, context)
    attempt = {
        "attempt_id": uuid.uuid4().hex,
        "action": args.action,
        "started_at": time.time(),
    }
    error = None
    try:
        path = session.output / "summary.json"
        summary = read_json(path) if path.exists() else {}
        reused = summary.get("status") in {"workload_complete", "evaluation_complete"}
        if not reused:
            if args.action == "train":
                from gstar_opd_formal.training.runner import run_training

                summary = run_training(session)
            else:
                if args.evaluation_profile in {"12k_a100_b20", "12k_a100_b10"}:
                    from gstar_opd_formal.runtime.high_batch import run_evaluation
                else:
                    from gstar_opd_formal.evaluation.runner import run_evaluation

                summary = run_evaluation(session)
        attempt.update(status="completed", measurement_status=summary["status"])
        return result_receipt(session, summary, reused)
    except BaseException as caught:
        error = caught
        attempt["status"] = "failed"
        saved = session.output / "summary.json"
        previous = read_json(saved) if saved.exists() else {}
        write_json(
            args.receipt,
            {
                "status": "failed",
                "binding": session.binding,
                "run_id": digest_value(session.binding)[:32],
                "output_dir": str(session.output),
                "reused_complete": False,
                "completed_updates": previous.get("completed_updates", 0),
                "count": previous.get("count", 0),
            },
        )
        raise
    finally:
        try:
            finish_attempt(session, attempt, error)
        finally:
            session.store.release()


def record_timing(args: argparse.Namespace, context: tuple) -> dict:
    study, drive, destination = context
    result = read_json(args.input)
    output = Path(result["output_dir"])
    binding = read_json(output / "binding.json")
    if binding != result["binding"] or binding["study_identity"] != study.identity:
        raise ValueError("Timing output belongs to another run/study")
    source = (
        tuple(study.manifests[binding["benchmark"]]["records"])
        if binding["action"] == "evaluate" and binding["benchmark"] != "dev128"
        else ()
    )
    if binding["action"] == "evaluate" and binding["benchmark"] == "dev128":
        from gstar_opd_formal.data.manifests import read_jsonl

        rows = read_jsonl(study.root / "data/raw/gsm8k_train.jsonl")
        source = tuple(
            {"id": f"dev128:{i}", "question": rows[i]["question"]}
            for i in study.manifests["dev"]["source_indices"]
        )
    store = open_store(
        drive, (destination["runs_folder_id"], result["run_id"], binding, source)
    )
    store.claim("record_timing")
    try:
        timing = read_json(args.timing)
        if timing.get("run_id") != result["run_id"] or timing.get(
            "binding_sha256"
        ) != digest_value(binding):
            raise ValueError("Client timing receipt does not match the result")
        write_json(output / "client_timing" / f"{timing['id']}.json", timing)
        saved = output / "summary.json"
        summary = read_json(saved) if saved.exists() else {}
        store.commit(output, summary.get("completed_updates", summary.get("count", 0)))
        return {"status": "timing_saved", "run_id": result["run_id"]}
    finally:
        store.release()


def publish_study(args: argparse.Namespace, context: tuple) -> dict:
    study, drive, destination = context
    receipt = validate_pair(read_json(args.input), study)
    for arm in ARMS:
        result = receipt["arms"][arm]
        store = open_store(
            drive,
            (destination["runs_folder_id"], result["run_id"], result["binding"], ()),
        )
        manifest = store.latest_manifest()
        if (
            manifest is None
            or manifest["boundary"] != study.design["updates"]
            or manifest["progress"].get("status") != "workload_complete"
        ):
            raise ValueError(
                "Both final training boundaries must be committed to Drive"
            )
        files = {row["path"]: row for row in manifest["files"]}
        for selected in result["retained_checkpoints"].values():
            if files.get(selected["path"], {}).get("sha256") != selected["sha256"]:
                raise ValueError(
                    "Retained checkpoint is absent from the final Drive commit"
                )
    record = publish_named(
        drive, args.input, (destination["studies_folder_id"], study.identity + ".json")
    )
    return {
        "status": "paired_study_saved",
        "study_identity": study.identity,
        "drive": record,
        "arms": {arm: receipt["arms"][arm]["run_id"] for arm in ARMS},
        "evaluation_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=("train", "evaluate", "record-timing", "publish-study"),
        required=True,
    )
    parser.add_argument("--arm", default="adaptive")
    parser.add_argument("--benchmark", default="math500")
    parser.add_argument("--checkpoint-update", type=int, default=128)
    parser.add_argument(
        "--evaluation-profile",
        choices=(
            "full",
            "16k",
            "12k_a100",
            "12k_base_l4",
            "12k_a100_b20",
            "12k_a100_b10",
        ),
        default="full",
    )
    parser.add_argument("--gpu-profile", choices=("A100", "L4"), default="A100")
    parser.add_argument("--resume-stopped", action="store_true")
    parser.add_argument("--allow-new", action="store_true")
    parser.add_argument("--grader-python", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    args.receipt.unlink(missing_ok=True)
    context = open_storage(args)
    if args.action == "record-timing":
        value = record_timing(args, context)
    elif args.action == "publish-study":
        value = publish_study(args, context)
    else:
        value = run_selected(args, context)
    write_json(args.receipt, value)
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "status",
                    "run_id",
                    "completed_updates",
                    "count",
                    "accuracy",
                )
                if key in value
            }
        )
    )


if __name__ == "__main__":
    main()
