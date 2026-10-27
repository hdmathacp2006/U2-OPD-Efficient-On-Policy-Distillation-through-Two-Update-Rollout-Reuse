"""Keep a verified local checkpoint when its last Drive acknowledgement failed."""

from pathlib import Path

from gstar_opd_formal.storage.files import read_json


def local_training_boundary(output: Path, binding: dict) -> int:
    if binding["action"] != "train" or not (output / "summary.json").exists():
        return 0
    if read_json(output / "binding.json") != binding:
        raise ValueError("Local recovery belongs to another execution")
    from gstar_opd_formal.training.history import accepted_checkpoint, validate_history

    checked = accepted_checkpoint(output, binding)
    import torch

    payload = torch.load(checked["path"], map_location="cpu", weights_only=False)
    count = checked["summary"]["completed_updates"]
    if (
        payload.get("run_contract") != binding
        or payload.get("completed_steps") != count
    ):
        raise ValueError("Local checkpoint payload differs from its summary")
    validate_history(payload["run_state"], binding, count)
    if (
        payload["policy"].get("fresh_observations")
        != checked["summary"]["fresh_batches"]
    ):
        raise ValueError("Local controller observations differ from history")
    return count
