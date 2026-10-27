"""Deterministic GSM8K training partitions with source-integrity checks."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from gstar_opd_formal.contract_types import Gsm8kSplitSpec


@dataclass(frozen=True)
class Gsm8kExample:
    source_index: int
    question: str
    answer: str


@dataclass(frozen=True)
class SplitManifest:
    source_url: str
    source_sha256: str
    seed: int
    total: int
    feedback_indices: tuple[int, ...]
    tuning_indices: tuple[int, ...]
    training_indices: tuple[int, ...]

    def validate(self) -> None:
        groups = (
            self.feedback_indices,
            self.tuning_indices,
            self.training_indices,
        )
        all_indices = [index for group in groups for index in group]
        if len(all_indices) != self.total or set(all_indices) != set(range(self.total)):
            raise ValueError("Split indices must partition the source exactly once")


@dataclass(frozen=True)
class Gsm8kPartitions:
    training: tuple[Gsm8kExample, ...]
    feedback: tuple[Gsm8kExample, ...]
    tuning: tuple[Gsm8kExample, ...]


def load_gsm8k_train(spec: Gsm8kSplitSpec) -> tuple[Gsm8kExample, ...]:
    """Reject a modified source file or malformed official GSM8K rows."""
    observed_hash = hashlib.sha256(spec.source_path.read_bytes()).hexdigest()
    if observed_hash != spec.source_sha256:
        raise ValueError("GSM8K train SHA256 differs from the frozen protocol")

    examples = []
    with spec.source_path.open("r", encoding="utf-8") as source_file:
        for source_index, line in enumerate(source_file):
            examples.append(parse_gsm8k_row(json.loads(line), source_index))

    if len(examples) != spec.expected_count:
        raise ValueError("GSM8K train row count differs from the frozen protocol")
    return tuple(examples)


def parse_gsm8k_row(row: object, source_index: int) -> Gsm8kExample:
    """Validate the official row schema independently of source-file integrity."""
    if not isinstance(row, dict):
        raise ValueError(f"Invalid GSM8K row at row {source_index}: expected an object")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Invalid GSM8K question at row {source_index}")
    if not isinstance(answer, str) or "####" not in answer:
        raise ValueError(f"Invalid GSM8K answer at row {source_index}")
    return Gsm8kExample(source_index, question, answer)


def make_split_manifest(spec: Gsm8kSplitSpec) -> SplitManifest:
    """Shuffle source row indices once and preserve source order within each split."""
    indices = list(range(spec.expected_count))
    random.Random(spec.seed).shuffle(indices)
    feedback_end = spec.feedback_count
    tuning_end = feedback_end + spec.tuning_count
    manifest = SplitManifest(
        source_url=spec.source_url,
        source_sha256=spec.source_sha256,
        seed=spec.seed,
        total=spec.expected_count,
        feedback_indices=tuple(sorted(indices[:feedback_end])),
        tuning_indices=tuple(sorted(indices[feedback_end:tuning_end])),
        training_indices=tuple(sorted(indices[tuning_end:])),
    )
    manifest.validate()
    return manifest


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    """Persist the exact membership used by all later runs."""
    manifest.validate()
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def load_split_manifest(path: Path, spec: Gsm8kSplitSpec) -> SplitManifest:
    """Reject manifests that differ from the frozen source and split settings."""
    content = json.loads(path.read_text(encoding="utf-8"))
    manifest = SplitManifest(
        source_url=content["source_url"],
        source_sha256=content["source_sha256"],
        seed=content["seed"],
        total=content["total"],
        feedback_indices=tuple(content["feedback_indices"]),
        tuning_indices=tuple(content["tuning_indices"]),
        training_indices=tuple(content["training_indices"]),
    )
    manifest.validate()
    if manifest != make_split_manifest(spec):
        raise ValueError("Manifest differs from the frozen split specification")
    return manifest


def partition_examples(
    examples: tuple[Gsm8kExample, ...], manifest: SplitManifest
) -> Gsm8kPartitions:
    """Return named roles so the training loop cannot mistake feedback for training."""
    manifest.validate()
    if len(examples) != manifest.total:
        raise ValueError("Example count and manifest total differ")
    if any(example.source_index != index for index, example in enumerate(examples)):
        raise ValueError("Examples must retain original GSM8K source order")
    return Gsm8kPartitions(
        training=tuple(examples[index] for index in manifest.training_indices),
        feedback=tuple(examples[index] for index in manifest.feedback_indices),
        tuning=tuple(examples[index] for index in manifest.tuning_indices),
    )
