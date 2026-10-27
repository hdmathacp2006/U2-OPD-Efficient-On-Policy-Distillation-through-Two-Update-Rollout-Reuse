"""Validate paired-run selection without importing a training or grading stack."""

import re

from gstar_opd_formal.contract import ARMS
from gstar_opd_formal.storage.files import digest_value


def validate_pair(receipt: dict, study: object) -> dict:
    if (
        receipt.get("schema") != "formal_paired_study_v1"
        or receipt.get("study_identity") != study.identity
    ):
        raise ValueError("Paired receipt belongs to another formal study")
    arms = receipt.get("arms", {})
    if set(arms) != set(ARMS):
        raise ValueError("Paired receipt must contain both completed arms")
    for arm in ARMS:
        result = arms[arm]
        binding = result["binding"]
        if (
            result.get("status") != "workload_complete"
            or result.get("completed_updates") != study.design["updates"]
            or binding.get("arm") != arm
            or binding.get("action") != "train"
            or binding.get("study_identity") != study.identity
            or binding.get("definition") != study.definition()
            or result.get("run_id") != digest_value(binding)[:32]
            or not re.fullmatch(r"[a-f0-9]{64}", result.get("checkpoint_sha256", ""))
        ):
            raise ValueError(
                "Paired receipt contains an incomplete/foreign training arm"
            )
        if binding.get("execution_code") != study.code["execution"]:
            raise ValueError("Paired training arms use another runtime implementation")
        checkpoints = result.get("retained_checkpoints", {})
        if set(checkpoints) != {
            str(update) for update in study.design["retained_updates"]
        }:
            raise ValueError("Paired receipt has the wrong retained checkpoints")
        for update, selected in checkpoints.items():
            if selected.get(
                "path"
            ) != f"checkpoints/update_{int(update):04d}.pt" or not re.fullmatch(
                r"[a-f0-9]{64}", selected.get("sha256", "")
            ):
                raise ValueError("Invalid retained checkpoint identity")
        if (
            checkpoints[str(study.design["updates"])]["sha256"]
            != result["checkpoint_sha256"]
        ):
            raise ValueError(
                "Final retained checkpoint differs from the selected endpoint"
            )
    if (
        arms["adaptive"]["binding"]["environment"]
        != arms["u2"]["binding"]["environment"]
    ):
        raise ValueError("Matched formal training requires the same hardware/runtime")
    return receipt


def study_receipt(results: dict, study: object) -> dict:
    value = {
        "schema": "formal_paired_study_v1",
        "study_identity": study.identity,
        "seed": study.seed,
        "arms": results,
        "evaluation_started": False,
    }
    return validate_pair(value, study)
