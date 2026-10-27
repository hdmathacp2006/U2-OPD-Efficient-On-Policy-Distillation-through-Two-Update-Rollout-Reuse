"""Exact full-vocabulary targets with bounded head projection memory.

Only response rows are projected. Neither attention context nor the teacher's
softmax denominator is truncated. Native teacher prefixes are an explicit study
adaptation, rather than an assumption that two chat templates are identical.
"""

from dataclasses import replace

import torch
from torch import Tensor, nn

from gstar_opd_formal.data.prompts import render_native_teacher_question
from gstar_opd_formal.models.loading import ModelBundle
from gstar_opd_formal.opd.objective import CandidateTargets, make_candidate_targets
from gstar_opd_formal.opd.rollout import ResponseTrajectory
from gstar_opd_formal.runtime.events import report_progress


def native_teacher_trajectory(
    bundle: ModelBundle, trajectory: ResponseTrajectory
) -> ResponseTrajectory:
    if bundle.teacher_tokenizer is None:
        return trajectory
    if not trajectory.question:
        raise ValueError("Native teacher conditioning needs the original question")
    tokenizer = bundle.teacher_tokenizer
    # The pinned teacher template supplies its /think suffix and default system
    # message. Student response IDs are appended verbatim, including thinking tags.
    from gstar_opd_formal.contract_types import PromptSpec

    rendered = render_native_teacher_question(
        tokenizer, trajectory.question, PromptSpec(bundle.student_instruction, True)
    )
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    response = trajectory.input_ids[0, -trajectory.response_length :]
    ids = torch.cat(
        (torch.tensor(prefix, device=response.device, dtype=torch.long), response)
    )[None]
    native_limit = getattr(bundle.teacher.config, "max_position_embeddings", None)
    if native_limit is not None and ids.shape[1] > native_limit:
        raise ValueError("Native teacher prompt plus full response exceeds its context")
    mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
    mask[:, len(prefix) - 1 :] = True
    return replace(
        trajectory,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=mask,
    )


def causal_parts(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise TypeError("Streaming scoring requires a Qwen causal decoder and lm_head")
    return base.model, base.lm_head


def response_hidden_states(
    model: nn.Module, trajectories: tuple[ResponseTrajectory, ...]
) -> tuple[Tensor, ...]:
    decoder, _ = causal_parts(model)
    width = max(item.input_ids.shape[1] for item in trajectories)
    device = trajectories[0].input_ids.device
    ids = torch.zeros((len(trajectories), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, item in enumerate(trajectories):
        length = item.input_ids.shape[1]
        ids[row, -length:] = item.input_ids[0]
        mask[row, -length:] = item.attention_mask[0]
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    outputs = decoder(
        input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
    )
    # R response tokens need the R states preceding them, including the prompt's
    # final state; the last response state predicts a token outside this rollout.
    return tuple(
        outputs.last_hidden_state[row, -item.response_length - 1 : -1].contiguous()
        for row, item in enumerate(trajectories)
    )


def project_targets(
    heads: tuple[nn.Module, nn.Module],
    hidden: tuple[Tensor, Tensor],
    spec: tuple[int, int],
) -> CandidateTargets:
    student_head, teacher_head = heads
    student_hidden, teacher_hidden = hidden
    top_k, chunk_rows = spec
    parts = []
    for start in range(0, student_hidden.shape[0], chunk_rows):
        student_logits = student_head(student_hidden[start : start + chunk_rows])
        teacher_logits = teacher_head(teacher_hidden[start : start + chunk_rows])
        targets = make_candidate_targets(student_logits, teacher_logits, top_k)
        parts.append(
            CandidateTargets(*(value.cpu() for value in vars(targets).values()))
        )
    result = CandidateTargets(
        *(
            torch.cat([getattr(item, name) for item in parts])
            for name in (
                "candidate_ids",
                "behavior_log_probs",
                "teacher_log_probs",
                "teacher_top_ids",
            )
        )
    )
    return result


def sampled_log_probs(model: nn.Module, hidden: Tensor, chosen: Tensor) -> Tensor:
    _, head = causal_parts(model)
    values = []
    for start in range(0, hidden.shape[0], 128):
        log_probs = head(hidden[start : start + 128]).float().log_softmax(-1)
        values.append(
            log_probs.gather(-1, chosen[start : start + 128, None]).squeeze(-1)
        )
    return torch.cat(values)


def check_streamed_likelihood(
    bundle: ModelBundle, trajectory: ResponseTrajectory, hidden: Tensor
) -> None:
    if trajectory.rollout_log_probs is None:
        return
    chosen = trajectory.input_ids[0, -trajectory.response_length :]
    expected = sampled_log_probs(bundle.student, hidden, chosen)
    difference = (expected - trajectory.rollout_log_probs.to(expected.device)).abs()
    thresholds = bundle.rollout.spec.logprob_thresholds
    if (
        not torch.isfinite(difference).all()
        or float(difference.mean()) > thresholds.failure
    ):
        raise RuntimeError(
            "Streamed vLLM/HF likelihood check exceeded the declared threshold"
        )
    diagnostic = {
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
        "warning": float(difference.mean()) > thresholds.warning,
        "warning_threshold": thresholds.warning,
        "failure_threshold": thresholds.failure,
    }
    bundle.rollout.record_mismatch(diagnostic, trajectory)
    report_progress("rollout_likelihood_check", **diagnostic)


@torch.no_grad()
def score_streamed_batch(
    bundle: ModelBundle,
    trajectories: tuple[ResponseTrajectory, ...],
    spec: tuple[int, int],
) -> tuple[CandidateTargets, ...]:
    bundle.student.eval()
    teacher_inputs = tuple(
        native_teacher_trajectory(bundle, item) for item in trajectories
    )
    teacher_hidden = response_hidden_states(bundle.teacher, teacher_inputs)
    heads = (causal_parts(bundle.student)[1], causal_parts(bundle.teacher)[1])
    targets = []
    for item, teacher_states in zip(trajectories, teacher_hidden, strict=True):
        student_states = response_hidden_states(bundle.student, (item,))[0]
        check_streamed_likelihood(bundle, item, student_states)
        result = project_targets(heads, (student_states, teacher_states), spec)
        targets.append(result)
    return tuple(targets)
