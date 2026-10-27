"""Evaluation rejects incomplete or malformed LoRA states before assignment."""

import pytest
from peft import get_peft_model_state_dict

from gstar_opd_formal.models.inference import validate_adapter


def test_real_adapter_matches_expected_model(local_engine):
    state = get_peft_model_state_dict(local_engine.bundle.student)
    validate_adapter(state, local_engine.bundle.student)
    assert state


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype", "nan"])
def test_malformed_adapter_is_rejected(local_engine, mutation):
    state = {
        name: value.detach().clone()
        for name, value in get_peft_model_state_dict(
            local_engine.bundle.student
        ).items()
    }
    name = next(iter(state))
    if mutation == "missing":
        state.pop(name)
    elif mutation == "shape":
        state[name] = state[name][:1]
    elif mutation == "dtype":
        state[name] = state[name].double()
    else:
        state[name].fill_(float("nan"))
    with pytest.raises(ValueError):
        validate_adapter(state, local_engine.bundle.student)
