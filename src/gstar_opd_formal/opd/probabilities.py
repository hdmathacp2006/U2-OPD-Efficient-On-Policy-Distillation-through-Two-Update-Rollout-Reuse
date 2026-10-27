"""Full-vocabulary normalization with bounded FP32 intermediates during backward."""

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from gstar_opd_formal.opd.objective import stable_log_softmax

LOGPROB_CHUNK_ROWS = 128
MEMORY_POLICY = "checkpoint_head_and_selected_logsoftmax_v2"


def project_log_probs_chunk(
    head: nn.Module, hidden: Tensor, candidate_ids: Tensor
) -> Tensor:
    return select_log_probs_chunk(head(hidden), candidate_ids)


def projected_selected_log_probs(
    head: nn.Module, hidden: Tensor, candidate_ids: Tensor
) -> Tensor:
    """Bound head logits AND their gradients while retaining full-vocab softmax."""
    if hidden.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("Hidden states and candidate IDs must be two-dimensional")
    if hidden.shape[0] == 0 or hidden.shape[0] != candidate_ids.shape[0]:
        raise ValueError("Hidden states and candidate IDs need matching nonempty rows")
    if candidate_ids.dtype != torch.long or candidate_ids.device != hidden.device:
        raise ValueError("Candidate IDs must be int64 on the hidden-state device")
    chunks = []
    for values, indices in zip(
        hidden.split(LOGPROB_CHUNK_ROWS),
        candidate_ids.split(LOGPROB_CHUNK_ROWS),
        strict=True,
    ):
        # Checkpoint BEFORE the head projection. Checkpointing normalization of
        # an already-materialized R x V tensor still needs its full dense gradient.
        if torch.is_grad_enabled():
            result = checkpoint(
                project_log_probs_chunk, head, values, indices, use_reentrant=False
            )
        else:
            result = project_log_probs_chunk(head, values, indices)
        chunks.append(result)
    return torch.cat(chunks)


def select_log_probs_chunk(logits: Tensor, candidate_ids: Tensor) -> Tensor:
    # Every chunk still normalizes over the entire vocabulary. Chunking the
    # response axis neither truncates context nor renormalizes the top-k set.
    return stable_log_softmax(logits).gather(-1, candidate_ids)


def selected_log_probs(logits: Tensor, candidate_ids: Tensor) -> Tensor:
    """Match dense log_softmax/gather without retaining an R x V FP32 output."""
    if logits.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("Logits and candidate IDs must be two-dimensional")
    if logits.shape[0] == 0 or logits.shape[0] != candidate_ids.shape[0]:
        raise ValueError("Logits and candidate IDs need matching nonempty rows")
    if candidate_ids.dtype != torch.long or candidate_ids.device != logits.device:
        raise ValueError("Candidate IDs must be int64 on the logits device")
    chunks = []
    pairs = zip(
        logits.split(LOGPROB_CHUNK_ROWS),
        candidate_ids.split(LOGPROB_CHUNK_ROWS),
        strict=True,
    )
    for values, indices in pairs:
        if torch.is_grad_enabled() and logits.requires_grad:
            # Native non-reentrant checkpointing recomputes only this deterministic
            # normalization. It keeps native-precision input views, not FP32 R x V
            # probabilities. SplitBackward joins native-precision gradient chunks.
            result = checkpoint(
                select_log_probs_chunk, values, indices, use_reentrant=False
            )
        else:
            result = select_log_probs_chunk(values, indices)
        chunks.append(result)
    return torch.cat(chunks)
