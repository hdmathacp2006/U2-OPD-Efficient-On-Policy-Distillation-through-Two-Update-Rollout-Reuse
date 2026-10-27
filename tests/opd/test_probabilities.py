"""Dense autograd is the independent reference for memory-saving normalization."""

import pytest
import torch

from gstar_opd_formal.opd.probabilities import (
    LOGPROB_CHUNK_ROWS,
    projected_selected_log_probs,
    selected_log_probs,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_chunked_values_match_full_vocabulary_dense_normalization(dtype) -> None:
    generator = torch.Generator().manual_seed(83)
    logits = torch.randn(257, 37, generator=generator, dtype=dtype, requires_grad=True)
    ids = torch.randint(37, (257, 16), generator=generator)
    stable = logits if dtype == torch.float64 else logits.float()
    torch.testing.assert_close(
        selected_log_probs(logits, ids), stable.log_softmax(-1).gather(-1, ids)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_chunked_gradients_match_dense_autograd_including_unselected_tokens(
    dtype,
) -> None:
    generator = torch.Generator().manual_seed(84)
    logits = torch.randn(257, 37, generator=generator, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ids = torch.randint(37, (257, 16), generator=generator)
    weights = torch.randn(257, 16, generator=generator)
    selected_log_probs(logits, ids).mul(weights).sum().backward()
    stable = reference if dtype == torch.float64 else reference.float()
    stable.log_softmax(-1).gather(-1, ids).mul(weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_no_dense_fp32_vocabulary_matrix_is_saved_for_backward() -> None:
    logits = torch.zeros(257, 37, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.zeros(257, 16, dtype=torch.long)
    saved = []

    def save_tensor(tensor: torch.Tensor) -> torch.Tensor:
        saved.append((tensor.dtype, tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        result = selected_log_probs(logits, ids)
    result.sum().backward()
    assert not any(
        dtype == torch.float32 and len(shape) == 2 and shape[-1] == 37
        for dtype, shape in saved
    )


def test_recomputation_does_not_advance_random_generators() -> None:
    logits = torch.zeros(257, 8, requires_grad=True)
    ids = torch.zeros(257, 2, dtype=torch.long)
    before = torch.get_rng_state().clone()
    selected_log_probs(logits, ids).sum().backward()
    assert torch.equal(before, torch.get_rng_state())


def test_chunked_normalization_passes_double_precision_gradcheck() -> None:
    logits = (
        torch.linspace(-1, 1, 129 * 3, dtype=torch.float64)
        .reshape(129, 3)
        .requires_grad_()
    )
    ids = torch.ones(129, 2, dtype=torch.long)
    assert torch.autograd.gradcheck(
        lambda values: selected_log_probs(values, ids), (logits,), fast_mode=True
    )


def test_normalization_remains_over_full_vocabulary() -> None:
    logits = torch.zeros(129, 32)
    ids = torch.zeros(129, 16, dtype=torch.long)
    torch.testing.assert_close(
        selected_log_probs(logits, ids),
        torch.full((129, 16), -torch.tensor(32.0).log()),
    )


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 257])
def test_projected_head_values_and_gradients_match_dense(rows):
    generator = torch.Generator().manual_seed(93)
    head = torch.nn.Linear(7, 37, bias=False, dtype=torch.float64)
    hidden = torch.randn(
        rows, 7, generator=generator, dtype=torch.float64, requires_grad=True
    )
    reference = hidden.detach().clone().requires_grad_()
    ids = torch.randint(37, (rows, 16), generator=generator)
    weights = torch.randn(rows, 16, generator=generator, dtype=torch.float64)
    actual = projected_selected_log_probs(head, hidden, ids)
    actual.mul(weights).sum().backward()
    actual_weight_gradient = head.weight.grad.clone()
    head.zero_grad(set_to_none=True)
    expected = head(reference).log_softmax(-1).gather(-1, ids)
    expected.mul(weights).sum().backward()
    torch.testing.assert_close(
        (actual, hidden.grad, actual_weight_gradient),
        (expected, reference.grad, head.weight.grad),
    )


@pytest.mark.parametrize("rows", [257, 8192])
def test_projection_never_constructs_a_full_response_vocabulary_matrix(rows):
    head = torch.nn.Linear(7, 37, bias=False)
    hidden = torch.randn(rows, 7, requires_grad=True)
    ids = torch.zeros(rows, 16, dtype=torch.long)
    projected_rows = []
    hook = head.register_forward_pre_hook(
        lambda module, inputs: projected_rows.append(inputs[0].shape[0])
    )
    try:
        projected_selected_log_probs(head, hidden, ids).sum().backward()
    finally:
        hook.remove()
    assert projected_rows and max(projected_rows) <= LOGPROB_CHUNK_ROWS


def test_projection_does_not_save_dense_logits_views_for_backward():
    head = torch.nn.Linear(7, 37, bias=False)
    hidden = torch.randn(257, 7, requires_grad=True)
    ids = torch.zeros(257, 16, dtype=torch.long)
    saved = []

    def retain_shape(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(retain_shape, lambda value: value):
        output = projected_selected_log_probs(head, hidden, ids)
    output.sum().backward()
    assert not any(len(shape) == 2 and shape[-1] == 37 for shape in saved)


def test_projection_tracks_head_gradients_with_frozen_hidden_states():
    head = torch.nn.Linear(4, 9, bias=False)
    output = projected_selected_log_probs(
        head, torch.zeros(129, 4), torch.zeros(129, 2, dtype=torch.long)
    )
    output.sum().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
