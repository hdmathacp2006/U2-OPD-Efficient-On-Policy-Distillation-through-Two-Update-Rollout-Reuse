"""Immutable formal study inputs without importing torch, models or graders."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from gstar_opd_formal.storage.files import (
    digest_file,
    digest_value,
    read_json,
    relative_path,
    verify_file,
)

FAMILY = "opd_formal_v1"
ARMS = ("adaptive", "u2")
COUNTS = {"math500": 500, "minerva_math": 272, "olympiadbench": 675}
FINAL_BENCHMARKS = {"math500": 500, "minerva_math": 272}


def stable_environment(environment: dict) -> dict:
    keys = (
        "profile",
        "hardware",
        "packages",
        "python",
        "cuda",
        "driver",
        "cudnn",
        "environment_id",
    )
    return {key: environment[key] for key in keys if key in environment}


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def source_identity(root: Path) -> dict:
    scientific, execution, interface = {}, {}, {}
    for path in sorted((root / "src/gstar_opd_formal").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        if name.endswith(("runtime/progress.py", "orchestration/pipeline.py")):
            selected = interface
        elif "/runtime/" in name or "/storage/" in name or name.endswith("cli.py"):
            selected = execution
        else:
            selected = scientific
        selected[name] = digest_file(path)
    execution.update(
        {
            path.relative_to(root).as_posix(): digest_file(path)
            for path in sorted((root / "requirements").glob("*"))
            if path.is_file()
        }
    )
    return {
        "scientific": digest_value(scientific),
        "execution": digest_value(execution),
        "interface": digest_value(interface),
    }


@dataclass(frozen=True)
class Study:
    root: Path
    design: dict
    models: dict
    training: dict
    evaluation: dict
    methods: dict
    initialization: dict
    manifests: dict
    code: dict

    @property
    def seed(self) -> int:
        return self.design["seed"]

    @property
    def identity(self) -> str:
        return digest_value(self.definition())

    def definition(self) -> dict:
        return {
            "family": FAMILY,
            "design": self.design,
            "models": self.models,
            "training": self.training,
            "evaluation": self.evaluation,
            "methods": self.methods,
            "initialization": self.initialization,
            "manifests": {
                name: {
                    key: value for key, value in manifest.items() if key != "records"
                }
                for name, manifest in self.manifests.items()
            },
            "scientific_code": self.code["scientific"],
        }

    def training_binding(self, arm: str, environment: dict) -> dict:
        if arm not in ARMS:
            raise ValueError("Unknown formal training arm")
        return {
            "schema": FAMILY,
            "action": "train",
            "study_identity": self.identity,
            "definition": self.definition(),
            "execution_code": self.code["execution"],
            "environment": stable_environment(environment),
            "arm": arm,
            "seed": self.seed,
            "endpoint": self.design["updates"],
            "batch_size": self.training["batch_size"],
            "updates_per_fresh_batch": self.methods[arm]["updates_per_fresh_batch"],
        }


def validate_configuration(config: tuple) -> None:
    design, training, evaluation, methods = config
    if (
        design.get("schema") != FAMILY
        or design.get("arms") != list(ARMS)
        or design.get("eval_arms") != ["base", *ARMS]
        or design.get("updates") != 128
        or design.get("training_pool") != 1024
        or design.get("benchmarks") != FINAL_BENCHMARKS
        or design.get("dev_count") != 128
        or design.get("checkpoint_interval") != 32
        or design.get("retained_updates") != [128]
        or type(design.get("seed")) is not int
    ):
        raise ValueError("Formal128 design differs from the declared paired study")
    required = {
        "batch_size": 4,
        "generation_batch_size": 4,
        "teacher_scoring_batch_size": 1,
        "scoring_chunk_rows": 128,
        "attention_backend": "flash_attention_2",
        "gradient_checkpointing": True,
    }
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Shared A100 training batch/backend settings changed")
    if methods["adaptive"]["updates_per_fresh_batch"] != 1 or (
        methods["u2"]["updates_per_fresh_batch"] != 2
        or methods["u2"].get("prefix_cap") != 4.0
        or methods["u2"].get("teacher_cache") != "full_forward_rescore"
    ):
        raise ValueError("Use original Adaptive/U2, without an auxiliary objective")
    if (
        evaluation["maximum_model_length"]
        < (evaluation["max_prompt_tokens"] + evaluation["max_new_tokens"])
        or evaluation["samples_per_question"] != 1
    ):
        raise ValueError("Evaluation context or sample count is inconsistent")


def load_study(root: Path) -> Study:
    root = root.resolve()
    design = read_yaml(root / "configs/studies/formal128.yaml")
    drive_root_id = os.environ.get("GSTAR_DRIVE_ROOT_ID")
    if drive_root_id:
        design["drive_root_id"] = drive_root_id
    training = read_yaml(root / "configs/training.yaml")
    evaluation = read_yaml(root / "configs/evaluation.yaml")
    models = read_yaml(root / "configs/models.yaml")
    methods = {arm: read_yaml(root / f"configs/methods/{arm}.yaml") for arm in ARMS}
    validate_configuration((design, training, evaluation, methods))
    initialization = read_json(root / "assets/initialization/manifest.json")
    if initialization.get("completed_updates") != 0:
        raise ValueError("Initialization must be the preserved zero-update adapter")
    if (
        digest_file(root / relative_path(initialization["path"]))
        != initialization["sha256"]
    ):
        raise ValueError("Initialization adapter bytes changed")
    manifests = {
        name: read_json(root / "data/manifests" / filename)
        for name, filename in {
            "training": f"train1024_seed{design['seed']}.json",
            "dev": "dev128.json",
            **{key: key + ".json" for key in design["benchmarks"]},
        }.items()
    }
    for manifest in manifests.values():
        for record in manifest.get("source_files", []):
            verify_file(root / relative_path(record["path"]), record)
    order, dev = (
        manifests["training"]["source_indices"],
        manifests["dev"]["source_indices"],
    )
    split = read_json(root / "data/manifests/gsm8k_split.v1.json")
    if any(type(index) is not int for index in order + dev) or not set(order).issubset(
        split["training_indices"]
    ):
        raise ValueError("Training IDs escaped the reserved source split")
    if dev != split["feedback_indices"][: design["dev_count"]]:
        raise ValueError("Reserved dev membership changed")
    if (
        len(order) != 1024
        or len(set(order)) != 1024
        or len(dev) != 128
        or set(order) & set(dev)
    ):
        raise ValueError(
            "Training/dev manifests have wrong count, duplicates or overlap"
        )
    for benchmark, count in FINAL_BENCHMARKS.items():
        ids = manifests[benchmark]["ids"]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Benchmark identity manifest changed")
        records = manifests[benchmark]["records"]
        if [row["id"] for row in records] != ids or digest_value(records) != manifests[
            benchmark
        ]["records_sha256"]:
            raise ValueError("Prepared benchmark records differ from their IDs/hash")
    return Study(
        root,
        design,
        models,
        training,
        evaluation,
        methods,
        initialization,
        manifests,
        source_identity(root),
    )
