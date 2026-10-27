"""Checkpointable full rollout inputs; never serialize a live autograd graph."""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from gstar_opd_formal.opd.rollout import ResponseTrajectory


def trajectory_on(
    trajectory: ResponseTrajectory, device: torch.device
) -> ResponseTrajectory:
    return replace(
        trajectory,
        **{
            field.name: value.detach().to(device).clone()
            for field in fields(trajectory)
            if isinstance(value := getattr(trajectory, field.name), Tensor)
        },
    )


def validate_trajectory(trajectory: ResponseTrajectory) -> None:
    length = trajectory.response_length
    ids, mask, attention = (
        trajectory.input_ids,
        trajectory.response_mask,
        trajectory.attention_mask,
    )
    if (
        type(length) is not int
        or length <= 0
        or not isinstance(ids, Tensor)
        or ids.ndim != 2
        or ids.shape[0] != 1
        or ids.dtype != torch.long
        or ids.shape[1] <= length
        or not isinstance(mask, Tensor)
        or mask.shape != (1, ids.shape[1] - 1)
        or mask.dtype != torch.bool
        or not isinstance(attention, Tensor)
        or attention.shape != ids.shape
        or attention.dtype not in {torch.long, torch.int32, torch.bool}
        or not isinstance(trajectory.response_positions, Tensor)
    ):
        raise ValueError(
            "Stored rollout has invalid causal response alignment: "
            f"question_id={trajectory.example_id}; malformed token/mask shape or dtype"
        )
    expected_response = torch.zeros_like(mask)
    expected_response[:, -length:] = True
    # vLLM removes left padding for decoding, then reconstructs the original
    # padded prompt for HF scoring. Keep those bytes and require a 0*1* mask.
    checks = {
        "nonnegative_token_ids": bool((ids >= 0).all()),
        "binary_attention": bool(((attention == 0) | (attention == 1)).all()),
        "left_padding_only": bool(
            (attention[:, 1:].long() >= attention[:, :-1].long()).all()
        ),
        "visible_prompt": int(attention.sum()) > length,
        "visible_response": bool((attention[:, -length:] == 1).all()),
        "causal_response_mask": torch.equal(mask, expected_response),
        "response_positions": trajectory.response_positions.dtype == torch.long
        and torch.equal(trajectory.response_positions.cpu(), torch.arange(length)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "Stored rollout has invalid causal response alignment: "
            f"question_id={trajectory.example_id}; failed_checks={failures}"
        )


@dataclass(frozen=True)
class PendingBatch:
    trajectories: tuple[ResponseTrajectory, ...]
    behavior_chosen: tuple[Tensor, ...] | None
    behavior_version: int
    horizon: int
    accepted_updates: int = 0

    def validate(self, learner_version: int) -> None:
        if (
            self.accepted_updates not in {0, 1}
            or self.behavior_version < 0
            or self.behavior_version + self.accepted_updates != learner_version
            or (self.behavior_chosen is None and self.accepted_updates > 0)
            or (
                self.behavior_chosen is not None
                and len(self.trajectories) != len(self.behavior_chosen)
            )
            or not self.trajectories
            or len({t.example_id for t in self.trajectories}) != len(self.trajectories)
        ):
            raise ValueError("Stored behavior and learner versions disagree")
        for item in self.trajectories:
            validate_trajectory(item)
            if item.response_length > self.horizon:
                raise ValueError("Stored response exceeds its behavior horizon")
        if self.behavior_chosen is None:
            return
        for item, chosen in zip(self.trajectories, self.behavior_chosen, strict=True):
            if (
                chosen.shape != (item.response_length,)
                or chosen.requires_grad
                or not torch.isfinite(chosen).all()
                or bool((chosen > 0).any())
                or item.response_length > self.horizon
            ):
                raise ValueError("Invalid frozen behavior probabilities or horizon")

    def cpu(self) -> "PendingBatch":
        return replace(
            self,
            trajectories=tuple(
                trajectory_on(t, torch.device("cpu")) for t in self.trajectories
            ),
            behavior_chosen=tuple(
                t.detach().cpu().clone() for t in self.behavior_chosen
            )
            if self.behavior_chosen is not None
            else None,
        )
