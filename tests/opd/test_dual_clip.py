"""Dual clipping differs only on the declared negative-advantage tail."""

import pytest
import torch

from gstar_opd_formal.contract_types import DualClipOpdSpec, OpdSpec
from gstar_opd_formal.opd.objective import CandidateTargets, compute_clipped_opd_loss


def negative_targets() -> CandidateTargets:
    return CandidateTargets(
        torch.tensor([[1]]),
        torch.tensor([[-3.0]]),
        torch.tensor([[-4.0]]),
        torch.tensor([[2]]),
    )


def test_ratio_one_dual_clip_preserves_loss():
    targets = negative_targets()
    standard = compute_clipped_opd_loss(
        targets.behavior_log_probs, targets, OpdSpec(1, 0.2, 1e-6)
    )
    dual = compute_clipped_opd_loss(
        targets.behavior_log_probs, targets, DualClipOpdSpec(1, 0.2, 1e-6)
    )
    assert torch.equal(standard, dual)


def test_negative_advantage_ratio_four_is_capped_at_three():
    current = torch.tensor([[-3.0 + torch.log(torch.tensor(4.0)).item()]])
    loss = compute_clipped_opd_loss(
        current, negative_targets(), DualClipOpdSpec(1, 0.2, 1e-6)
    )
    assert float(loss) == pytest.approx(3.0)


def test_dual_clip_remains_finite_for_extreme_log_ratios():
    targets = CandidateTargets(
        torch.tensor([[1]]),
        torch.tensor([[-1000.0]]),
        torch.tensor([[-1001.0]]),
        torch.tensor([[2]]),
    )
    loss = compute_clipped_opd_loss(
        torch.tensor([[-0.1]]), targets, DualClipOpdSpec(1, 0.2, 1e-6)
    )
    assert float(loss) == pytest.approx(3.0)


@pytest.mark.parametrize("coefficient", [0.0, 1.0, float("inf")])
def test_dual_clip_rejects_invalid_tail_coefficient(coefficient):
    with pytest.raises(ValueError):
        DualClipOpdSpec(1, 0.2, 1e-6, coefficient)
