"""Independent A8 row references check loss values, gradients and Adam steps."""

import pytest
import torch

from gstar_opd_formal.contract_types import DualClipOpdSpec, OpdSpec
from gstar_opd_formal.opd.objective import (
    compute_clipped_opd_loss,
    make_candidate_targets,
    select_candidate_targets,
)
from gstar_opd_formal.opd.reuse_objective import (
    WeightedCandidates,
    prefix_weights,
    weighted_candidate_loss,
)


@pytest.mark.parametrize(
    "spec", [OpdSpec(3, 0.2, 1e-6), DualClipOpdSpec(3, 0.2, 1e-6, 3.0, 20.0)]
)
def test_weighted_gradient_and_adam_match_independent_a8_rows(spec):
    torch.manual_seed(43)
    behavior, teacher = (
        torch.randn(4, 6, dtype=torch.double),
        torch.randn(4, 6, dtype=torch.double),
    )
    targets = make_candidate_targets(behavior, teacher, 3)
    candidate = torch.nn.Parameter(
        behavior + torch.tensor([2.0, -2.0, 0.1, 1.0, -1.0, 0.0])
    )
    reference = torch.nn.Parameter(candidate.detach().clone())
    factors = torch.tensor([1.0, 0.3, 2.0, 4.0], dtype=torch.double)
    actual = weighted_candidate_loss(
        WeightedCandidates(
            candidate.log_softmax(-1).gather(-1, targets.candidate_ids),
            targets,
            factors,
        ),
        spec,
    )
    expected = (
        sum(
            compute_clipped_opd_loss(
                reference.log_softmax(-1).gather(-1, targets.candidate_ids)[i : i + 1],
                select_candidate_targets(targets, torch.tensor([i])),
                spec,
            )
            * factors[i]
            for i in range(4)
        )
        / 4
    )
    actual.backward()
    expected.backward()
    actual_optimizer, reference_optimizer = (
        torch.optim.AdamW([candidate], lr=1e-4),
        torch.optim.AdamW([reference], lr=1e-4),
    )
    torch.nn.utils.clip_grad_norm_([candidate], 0.1)
    torch.nn.utils.clip_grad_norm_([reference], 0.1)
    actual_optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(
        (actual.detach(), candidate.grad, candidate),
        (expected.detach(), reference.grad, reference),
        atol=1e-12,
        rtol=1e-12,
    )


def test_prefix_excludes_current_token_and_caps_geometric_mean():
    behavior = torch.tensor([-4.0, -4.0, -4.0, -4.0], dtype=torch.double)
    current = torch.tensor([-1.0, -3.0, -4.0, -4.0], dtype=torch.double)
    factors, _ = prefix_weights((current, behavior), 4.0)
    torch.testing.assert_close(
        factors,
        torch.tensor(
            [1.0, 4.0, 4.0, torch.exp(torch.tensor(4.0 / 3.0)).item()],
            dtype=torch.double,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_prefix_factor_is_detached():
    current = torch.tensor([-1.0, -2.0], requires_grad=True)
    assert prefix_weights((current, current.detach()), 4)[0].requires_grad is False


@pytest.mark.parametrize(
    "values", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([1.0])]
)
def test_invalid_prefix_probabilities_are_rejected(values):
    with pytest.raises(ValueError):
        prefix_weights((values, values), 4)


def test_long_prefix_diagnostic_retains_cumulative_shift():
    current, behavior = torch.full((8192,), -1.0), torch.full((8192,), -1.01)
    _, diagnostics = prefix_weights((current, behavior), 4.0)
    assert diagnostics["cumulative_log_ratio_max"] > 80
