"""Fail on significant inference/training likelihood disagreement before updating."""

import torch

from gstar_opd_formal.models.rollout_spec import LogprobThresholds
from gstar_opd_formal.opd.rollout import ResponseTrajectory

LOGPROB_CHUNK_SIZE = 128


def check_rollout_log_probs(
    logits: torch.Tensor, trajectory: ResponseTrajectory, thresholds: LogprobThresholds
) -> dict[str, object]:
    if logits.ndim != 2 or logits.shape[0] != trajectory.response_length:
        raise ValueError("Training logits must align with response tokens")
    observed = trajectory.rollout_log_probs
    if observed is None or observed.shape != (trajectory.response_length,):
        raise ValueError("Rollout log-probabilities must align with response tokens")
    chosen = trajectory.input_ids[0, -trajectory.response_length :]
    probabilities = []
    # Avoid allocating another full R x vocabulary FP32 tensor at a 4K horizon.
    for start in range(0, logits.shape[0], LOGPROB_CHUNK_SIZE):
        stop = start + LOGPROB_CHUNK_SIZE
        probabilities.append(
            logits[start:stop]
            .float()
            .log_softmax(-1)
            .gather(-1, chosen[start:stop, None])
            .squeeze(-1)
        )
    expected = torch.cat(probabilities)
    difference = (expected - observed.to(expected.device)).abs()
    if not torch.isfinite(difference).all():
        raise ValueError("Nonfinite rollout/training likelihood difference")
    mean, maximum = float(difference.mean()), float(difference.max())
    if mean > thresholds.failure:
        raise RuntimeError(
            f"vLLM/HF mean log-probability difference {mean:.6g} exceeds "
            f"declared failure threshold {thresholds.failure}; "
            "check adapter synchronization and precision"
        )
    # The warning threshold is a diagnostic. BF16 kernels with identical weights
    # need not produce identical probabilities; neither threshold proves parity.
    return {
        "mean_absolute_difference": mean,
        "maximum_absolute_difference": maximum,
        "warning": mean > thresholds.warning,
        "warning_threshold": thresholds.warning,
        "failure_threshold": thresholds.failure,
    }
