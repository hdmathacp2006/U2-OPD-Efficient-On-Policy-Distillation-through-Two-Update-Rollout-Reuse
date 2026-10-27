"""Cost checkpoints must not corrupt a running budget on validation failure."""

import pytest
import torch

from gstar_opd_formal.opd.costs import CostLedger


@pytest.fixture
def ledger() -> CostLedger:
    return CostLedger(torch.device("cpu"), 3.5, {"student_rollout": 1.5}, 64)


def test_valid_cost_state_round_trips(ledger: CostLedger) -> None:
    restored = CostLedger(torch.device("cpu"))
    restored.load_state_dict(ledger.state_dict())
    assert restored.state_dict() == ledger.state_dict()


def test_zero_cost_state_is_valid(ledger: CostLedger) -> None:
    state = {"charged_seconds": 0, "stage_seconds": {}, "peak_memory_bytes": 0}
    ledger.load_state_dict(state)
    assert ledger.state_dict() == state


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_invalid_charged_seconds_preserve_ledger(
    ledger: CostLedger, invalid: object
) -> None:
    before = ledger.state_dict()
    state = {**before, "charged_seconds": invalid}
    with pytest.raises(ValueError, match="Charged seconds"):
        ledger.load_state_dict(state)
    assert ledger.state_dict() == before


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_invalid_stage_seconds_preserve_ledger(
    ledger: CostLedger, invalid: object
) -> None:
    before = ledger.state_dict()
    state = {
        "charged_seconds": 8.0,
        "stage_seconds": {"teacher_scoring": invalid},
        "peak_memory_bytes": 128,
    }
    with pytest.raises(ValueError, match="Stage teacher_scoring seconds"):
        ledger.load_state_dict(state)
    assert ledger.state_dict() == before


@pytest.mark.parametrize("invalid", [-1, 1.5, float("nan"), float("inf"), True, "64"])
def test_invalid_peak_memory_preserves_ledger(
    ledger: CostLedger, invalid: object
) -> None:
    before = ledger.state_dict()
    state = {
        "charged_seconds": 8.0,
        "stage_seconds": {"teacher_scoring": 4.0},
        "peak_memory_bytes": invalid,
    }
    with pytest.raises(ValueError, match="Peak memory"):
        ledger.load_state_dict(state)
    assert ledger.state_dict() == before


@pytest.mark.parametrize("stages", [[], {1: 2.0}])
def test_invalid_stage_mapping_preserves_ledger(
    ledger: CostLedger, stages: object
) -> None:
    before = ledger.state_dict()
    state = {**before, "charged_seconds": 8.0, "stage_seconds": stages}
    with pytest.raises(ValueError, match="Stage costs"):
        ledger.load_state_dict(state)
    assert ledger.state_dict() == before


def test_restored_stage_costs_do_not_alias_checkpoint(ledger: CostLedger) -> None:
    state = {
        "charged_seconds": 8.0,
        "stage_seconds": {"teacher_scoring": 4.0},
        "peak_memory_bytes": 128,
    }
    ledger.load_state_dict(state)
    state["stage_seconds"]["teacher_scoring"] = 99.0
    assert ledger.stage_seconds == {"teacher_scoring": 4.0}
