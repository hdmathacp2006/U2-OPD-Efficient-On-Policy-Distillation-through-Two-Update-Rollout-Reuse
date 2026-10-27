"""Check EOS and cap evidence without importing a grader."""

from gstar_opd_formal.opd.rollout import eos_token_ids


def validate_finish_flags(trajectories: tuple, context: tuple) -> None:
    config, horizon = context
    endings = eos_token_ids(config)
    for item in trajectories:
        tokens = item.input_ids[0, -item.response_length :].tolist()
        first_eos = next(
            (i for i, token in enumerate(tokens) if token in endings), None
        )
        if item.truncated:
            valid = len(tokens) == horizon and first_eos is None
        else:
            valid = first_eos == len(tokens) - 1 and first_eos is not None
        if not valid:
            raise ValueError(
                "Rollout finish flags disagree with EOS/horizon token evidence"
            )
