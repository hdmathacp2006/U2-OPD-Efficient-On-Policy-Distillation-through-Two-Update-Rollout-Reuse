"""Run both formal methods through real tiny Qwen/LoRA forward and backward."""

from types import SimpleNamespace

import torch

from gstar_opd_formal.controllers.adaptive import AdaptivePolicySpec
from gstar_opd_formal.controllers.reuse import FreshPolicy
from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.methods.adaptive import AdaptiveMethod
from gstar_opd_formal.methods.u2 import U2Method


def policy_for(engine):
    engine.reuse_cap = 4.0
    return FreshPolicy(
        SimpleNamespace(
            protocol=SimpleNamespace(horizons=SimpleNamespace(choices=(8, 16, 24, 32))),
            adaptive=AdaptivePolicySpec(),
        )
    )


def examples():
    return tuple(Gsm8kExample(index, "1+1?", "#### 2") for index in range(4))


def test_adaptive_observes_its_fresh_update(engine):
    policy, method = policy_for(engine), AdaptiveMethod()
    pending, scored = method.fresh(engine, (policy, examples()))
    method.update(engine, (policy, pending, scored))
    assert (engine.completed_steps, policy.fresh_observations) == (1, 1)


def test_u2_reuses_once_without_another_rollout(engine):
    policy, method = policy_for(engine), U2Method()
    pending, scored = method.fresh(engine, (policy, examples()))
    generations = engine.bundle.rollout.generations
    pending, _ = method.update(engine, (policy, pending, scored))
    method.update(engine, (policy, pending.cpu(), None))
    assert (
        engine.completed_steps,
        policy.fresh_observations,
        engine.bundle.rollout.generations,
    ) == (2, 1, generations)


def test_original_behavior_snapshot_is_fixed_across_second_update(engine):
    policy, method = policy_for(engine), U2Method()
    pending, scored = method.fresh(engine, (policy, examples()))
    before = tuple(value.clone() for value in pending.behavior_chosen)
    pending, _ = method.update(engine, (policy, pending, scored))
    pending, _ = method.update(engine, (policy, pending.cpu(), None))
    assert all(
        torch.equal(left, right)
        for left, right in zip(before, pending.behavior_chosen, strict=True)
    )


def test_refresh_runs_the_teacher_again_instead_of_using_a_hidden_cache(engine):
    policy, method = policy_for(engine), U2Method()
    calls = []
    hook = (
        engine.bundle.teacher.get_base_model().model.register_forward_hook(
            lambda *_: calls.append(1)
        )
        if hasattr(engine.bundle.teacher, "get_base_model")
        else engine.bundle.teacher.model.register_forward_hook(
            lambda *_: calls.append(1)
        )
    )
    try:
        pending, scored = method.fresh(engine, (policy, examples()))
        first = len(calls)
        pending, _ = method.update(engine, (policy, pending, scored))
        method.update(engine, (policy, pending.cpu(), None))
        assert len(calls) == 2 * first > 0
    finally:
        hook.remove()


def test_first_u2_step_matches_adaptive_loss_and_adam(engine):
    import copy

    other = copy.deepcopy(engine)
    left, right = policy_for(engine), policy_for(other)
    adaptive, u2 = AdaptiveMethod(), U2Method()
    pa, sa = adaptive.fresh(engine, (left, examples()))
    pu, su = u2.fresh(other, (right, examples()))
    _, a = adaptive.update(engine, (left, pa, sa))
    _, b = u2.update(other, (right, pu, su))
    torch.testing.assert_close(
        torch.tensor([a["loss"], a["gradient_norm"]]),
        torch.tensor([b["loss"], b["gradient_norm"]]),
    )
    assert all(
        torch.equal(value, dict(other.bundle.student.named_parameters())[name])
        for name, value in engine.bundle.student.named_parameters()
    )
