"""Apply the preserved zero-update adapter and verify its content identity."""

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from gstar_opd_formal.models.adapter_snapshot import snapshot_identity
from gstar_opd_formal.storage.files import digest_file, relative_path


def apply_initial_adapter(engine: object, study: object) -> str:
    path = study.root / relative_path(study.initialization["path"])
    if digest_file(path) != study.initialization["sha256"]:
        raise ValueError("Initialization adapter bytes changed")
    adapter = load_file(str(path))
    expected = get_peft_model_state_dict(
        engine.bundle.student, save_embedding_layers=False
    )
    if adapter.keys() != expected.keys() or any(
        value.shape != expected[name].shape
        or value.dtype != expected[name].dtype
        or not torch.isfinite(value).all()
        for name, value in adapter.items()
    ):
        raise ValueError("Zero-update adapter is incompatible with the current student")
    identity = snapshot_identity(
        study.models["models"]["student"]
        + "@"
        + study.models["models"]["student_revision"],
        adapter,
    )
    if identity != study.initialization["adapter_identity"]:
        raise ValueError("Zero-update adapter identity changed")
    set_peft_model_state_dict(engine.bundle.student, adapter)
    return identity
