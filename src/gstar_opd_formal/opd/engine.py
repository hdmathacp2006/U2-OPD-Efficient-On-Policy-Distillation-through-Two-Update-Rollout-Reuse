"""One fresh-rollout, frozen-teacher, clipped-OPD update shared by both policies."""

import math
from dataclasses import asdict, dataclass
from time import perf_counter

import torch

from gstar_opd_formal.contract_types import OpdSpec, PromptSpec
from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.models.loading import (
    ModelBundle,
    freeze_teacher,
    verify_student_parameters,
)
from gstar_opd_formal.opd.costs import CostLedger
from gstar_opd_formal.opd.mismatch import check_rollout_log_probs
from gstar_opd_formal.opd.objective import (
    CandidateTargets,
    compute_clipped_opd_loss,
    make_candidate_targets,
)
from gstar_opd_formal.opd.probabilities import (
    projected_selected_log_probs,
    selected_log_probs,
)
from gstar_opd_formal.opd.rollout import (
    ResponseTrajectory,
    RolloutRequest,
    SamplingSpec,
    generate_response_microbatches,
    response_logits,
)
from gstar_opd_formal.opd.signals import (
    BoundaryBatch,
    BoundarySignals,
    compute_boundary_signals,
)
from gstar_opd_formal.runtime.events import report_progress


@dataclass(frozen=True)
class EngineSpec:
    opd: OpdSpec
    prompt: PromptSpec
    horizon_increment: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    max_prompt_tokens: int
    generation_batch_size: int = 1

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value)
            for value in (self.learning_rate, self.weight_decay, self.max_grad_norm)
        ):
            raise ValueError("Optimizer settings must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.horizon_increment <= 0 or self.max_prompt_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if (
            type(self.generation_batch_size) is not int
            or self.generation_batch_size <= 0
        ):
            raise ValueError("generation_batch_size must be a positive integer")


@dataclass(frozen=True)
class ScoredTrajectory:
    trajectory: ResponseTrajectory
    targets: CandidateTargets


@dataclass(frozen=True)
class StreamingEngineSpec(EngineSpec):
    teacher_scoring_batch_size: int = 2
    scoring_chunk_rows: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            type(value) is not int or value <= 0
            for value in (self.teacher_scoring_batch_size, self.scoring_chunk_rows)
        ):
            raise ValueError("Scoring batch and chunk sizes must be positive integers")


@dataclass(frozen=True)
class UpdateResult:
    step: int
    horizon: int
    loss: float
    grad_norm: float
    response_lengths: tuple[int, ...]
    truncated_count: int
    signals: BoundarySignals | None
    charged_seconds: float


def move_candidate_targets(
    targets: CandidateTargets, device: torch.device
) -> CandidateTargets:
    return CandidateTargets(
        candidate_ids=targets.candidate_ids.to(device),
        behavior_log_probs=targets.behavior_log_probs.to(device),
        teacher_log_probs=targets.teacher_log_probs.to(device),
        teacher_top_ids=targets.teacher_top_ids.to(device),
    )


def score_trajectory(
    bundle: ModelBundle, trajectory: ResponseTrajectory, candidate_top_k: int
) -> ScoredTrajectory:
    bundle.student.eval()
    with torch.no_grad():
        student_logits = response_logits(bundle.student, trajectory)
        if trajectory.rollout_log_probs is not None:
            backend = bundle.rollout
            diagnostic = check_rollout_log_probs(
                student_logits, trajectory, backend.spec.logprob_thresholds
            )
            backend.record_mismatch(diagnostic, trajectory)
            report_progress("rollout_likelihood_check", **diagnostic)
            if diagnostic["warning"]:
                report_progress(
                    "rollout_likelihood_warning",
                    question_id=trajectory.example_id,
                    step=backend.owner.completed_steps + 1,
                    mean_absolute_difference=diagnostic["mean_absolute_difference"],
                )
        teacher_logits = response_logits(bundle.teacher, trajectory)
        targets = make_candidate_targets(
            student_logits, teacher_logits, candidate_top_k
        )
    # Only O(response_length * k) detached targets persist until the update phase.
    return ScoredTrajectory(
        trajectory, move_candidate_targets(targets, torch.device("cpu"))
    )


def summarize_boundary(
    scored: tuple[ScoredTrajectory, ...], spec: tuple[int, int]
) -> BoundarySignals | None:
    horizon, increment = spec
    targets = CandidateTargets(
        candidate_ids=torch.cat([item.targets.candidate_ids for item in scored]),
        behavior_log_probs=torch.cat(
            [item.targets.behavior_log_probs for item in scored]
        ),
        teacher_log_probs=torch.cat(
            [item.targets.teacher_log_probs for item in scored]
        ),
        teacher_top_ids=torch.cat([item.targets.teacher_top_ids for item in scored]),
    )
    return compute_boundary_signals(
        BoundaryBatch(
            targets=targets,
            response_positions=torch.cat(
                [item.trajectory.response_positions.cpu() for item in scored]
            ),
            horizon=horizon,
            increment=increment,
            response_lengths=tuple(item.trajectory.response_length for item in scored),
        )
    )


class OpdEngine:
    def __init__(self, bundle: ModelBundle, spec: EngineSpec) -> None:
        verify_student_parameters(bundle.student)
        freeze_teacher(bundle.teacher)
        self.bundle = bundle
        self.spec = spec
        self.optimizer = torch.optim.AdamW(
            [value for value in bundle.student.parameters() if value.requires_grad],
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
        self.ledger = CostLedger(bundle.device)
        self.completed_steps = 0
        self.journal = None
        self.current_block: int | None = None
        self.ratio_diagnostics: list[dict[str, float]] = []

    def collect_targets(
        self, trajectories: tuple[ResponseTrajectory, ...]
    ) -> tuple[ScoredTrajectory, ...]:
        # All targets must refer to one frozen behavior policy before any update.
        if self.bundle.rollout is not None and any(
            item.rollout_log_probs is not None for item in trajectories
        ):
            self.bundle.rollout.verify_trajectories(trajectories)
        scored = []
        if isinstance(self.spec, StreamingEngineSpec):
            from gstar_opd_formal.opd.streaming import score_streamed_batch

            for start in range(
                0, len(trajectories), self.spec.teacher_scoring_batch_size
            ):
                batch = trajectories[
                    start : start + self.spec.teacher_scoring_batch_size
                ]
                targets = score_streamed_batch(
                    self.bundle,
                    batch,
                    (self.spec.opd.candidate_top_k, self.spec.scoring_chunk_rows),
                )
                scored.extend(
                    ScoredTrajectory(item, target)
                    for item, target in zip(batch, targets, strict=True)
                )
            return tuple(scored)
        for index, item in enumerate(trajectories, start=1):
            report_progress(
                "teacher_scoring_start",
                question=index,
                total=len(trajectories),
                response_tokens=item.response_length,
            )
            scored.append(
                score_trajectory(self.bundle, item, self.spec.opd.candidate_top_k)
            )
            report_progress(
                "teacher_scoring_done", question=index, total=len(trajectories)
            )
        return tuple(scored)

    def update_student(
        self, scored: tuple[ScoredTrajectory, ...]
    ) -> tuple[float, float]:
        loss = self.calculate_gradients(scored)
        return loss, self.commit_gradients()

    def calculate_gradients(self, scored: tuple[ScoredTrajectory, ...]) -> float:
        """Accumulate the global response-token mean without advancing Adam."""
        total_tokens = sum(item.trajectory.response_length for item in scored)
        if total_tokens == 0:
            raise ValueError("Cannot update on an empty response batch")
        self.bundle.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.ratio_diagnostics = []
        self.release_unused_memory()
        total_loss = 0.0
        for index, item in enumerate(scored, start=1):
            report_progress("student_forward_start", question=index, total=len(scored))
            total_loss += self.backward_trajectory(item, total_tokens)
            # The helper's graph/logit references are gone before the next forward.
            # Returning unused allocator blocks limits ragged-length fragmentation.
            self.release_unused_memory()
            report_progress("backward_done", question=index, total=len(scored))
        return total_loss

    def commit_gradients(self) -> float:
        """Clip exactly once and commit one real update."""
        parameters = [
            value for value in self.bundle.student.parameters() if value.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.spec.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.completed_steps += 1
        return float(grad_norm)

    def backward_trajectory(self, item: ScoredTrajectory, total_tokens: int) -> float:
        targets = move_candidate_targets(item.targets, self.bundle.device)
        if isinstance(self.spec, StreamingEngineSpec):
            from gstar_opd_formal.opd.streaming import (
                causal_parts,
                response_hidden_states,
            )

            hidden = response_hidden_states(self.bundle.student, (item.trajectory,))[0]
            head = causal_parts(self.bundle.student)[1]
            log_probs = projected_selected_log_probs(
                head, hidden, targets.candidate_ids
            )
        else:
            logits = response_logits(self.bundle.student, item.trajectory)
            log_probs = selected_log_probs(logits, targets.candidate_ids)
        loss = compute_clipped_opd_loss(log_probs, targets, self.spec.opd)
        if self.journal is not None:
            with torch.no_grad():
                ratio = (
                    (log_probs.detach() - targets.behavior_log_probs)
                    .clamp(-20, 20)
                    .exp()
                )
                self.ratio_diagnostics.append(
                    {
                        "question_id": item.trajectory.example_id,
                        "minimum": float(ratio.min()),
                        "maximum": float(ratio.max()),
                        "mean": float(ratio.mean()),
                        "clip_fraction": float(
                            (
                                (ratio < 1 - self.spec.opd.clip_ratio)
                                | (ratio > 1 + self.spec.opd.clip_ratio)
                            )
                            .float()
                            .mean()
                        ),
                    }
                )
        # Sequence microbatches still recover the same global valid-token mean.
        scaled_loss = loss * (item.trajectory.response_length / total_tokens)
        report_progress(
            "backward_start",
            question_id=item.trajectory.example_id,
            response_tokens=item.trajectory.response_length,
        )
        scaled_loss.backward()
        return float(scaled_loss.detach())

    def release_unused_memory(self) -> None:
        if self.bundle.device.type == "cuda" and not isinstance(
            self.spec, StreamingEngineSpec
        ):
            torch.cuda.empty_cache()

    def step(self, examples: tuple[Gsm8kExample, ...], horizon: int) -> UpdateResult:
        start = perf_counter()
        previous_stages = dict(self.ledger.stage_seconds)
        if self.journal is not None:
            self.journal.sample_hardware(
                {"phase": "update_start", "step": self.completed_steps + 1}
            )
        if (
            horizon < self.spec.horizon_increment
            or horizon % self.spec.horizon_increment
        ):
            raise ValueError("Horizon must follow the configured positive increment")
        report_progress(
            "update_start",
            step=self.completed_steps + 1,
            horizon=horizon,
            questions=len(examples),
        )
        request = RolloutRequest(
            examples,
            self.spec.prompt,
            SamplingSpec(horizon),
            self.spec.max_prompt_tokens,
        )
        with self.ledger.measure_stage("student_rollout"):
            trajectories = generate_response_microbatches(
                self.bundle, request, self.spec.generation_batch_size
            )
        with self.ledger.measure_stage("teacher_scoring"):
            backend = self.bundle.rollout
            if backend is not None and hasattr(backend, "prepare_hf_phase"):
                backend.prepare_hf_phase()
            scored = self.collect_targets(trajectories)
            signals = summarize_boundary(scored, (horizon, self.spec.horizon_increment))
        with self.ledger.measure_stage("student_update"):
            loss, grad_norm = self.update_student(scored)
        elapsed = perf_counter() - start
        self.ledger.charged_seconds += elapsed
        report_progress(
            "update_done",
            step=self.completed_steps,
            horizon=horizon,
            loss=round(loss, 6),
            seconds=round(elapsed, 2),
            response_tokens=sum(item.response_length for item in trajectories),
        )
        result = UpdateResult(
            step=self.completed_steps,
            horizon=horizon,
            loss=loss,
            grad_norm=grad_norm,
            response_lengths=tuple(item.response_length for item in trajectories),
            truncated_count=sum(item.truncated for item in trajectories),
            signals=signals,
            charged_seconds=elapsed,
        )
        if self.journal is not None:
            boundary_rows = sum(
                int(
                    (
                        (
                            item.trajectory.response_positions
                            >= horizon - self.spec.horizon_increment
                        )
                        & (item.trajectory.response_positions < horizon)
                    ).sum()
                )
                for item in scored
            )
            self.journal.write(
                "updates",
                {
                    **asdict(result),
                    "block": self.current_block,
                    "question_ids": [item.example_id for item in trajectories],
                    "prompt_lengths": [
                        int(item.attention_mask.sum()) - item.response_length
                        for item in trajectories
                    ],
                    "valid_tokens": sum(result.response_lengths),
                    "boundary_rows": boundary_rows,
                    "eos_count": len(trajectories) - result.truncated_count,
                    "gradient_clipped": grad_norm > self.spec.max_grad_norm,
                    "candidate_ratios": self.ratio_diagnostics,
                    "phase_seconds": {
                        name: seconds - previous_stages.get(name, 0)
                        for name, seconds in self.ledger.stage_seconds.items()
                    },
                    "timing_scope": "parent_update_inclusive_children_stage_exclusive",
                },
            )
            self.journal.sample_hardware(
                {"phase": "update_end", "step": self.completed_steps}
            )
        return result
