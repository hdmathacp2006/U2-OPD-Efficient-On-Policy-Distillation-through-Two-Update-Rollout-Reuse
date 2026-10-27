"""Keep learner work, closed process spans and transfer overhead distinct."""

from pathlib import Path

from gstar_opd_formal.storage.files import read_json


def timing_report(output: Path) -> dict:
    records = [read_json(path) for path in (output / "client_timing").glob("*.json")]
    transfers = [read_json(path) for path in (output / "transfers").glob("*.json")]
    observed = [read_json(path) for path in (output / "timing").glob("*.json")]
    return {
        "closed_work_process_seconds": sum(
            row["seconds"] for row in records if not row["reused_complete"]
        ),
        "completed_result_restore_seconds": sum(
            row["seconds"] for row in records if row["reused_complete"]
        ),
        "observed_worker_segment_seconds": sum(
            row["worker_observed_seconds"] for row in observed
        ),
        "transfer_seconds_inclusive_in_process": sum(
            row["sync_seconds"] for row in transfers
        ),
        "closed_process_count": len(records),
        "failed_closed_process_count": sum(
            row.get("status") == "failed" for row in records
        ),
        "scope": (
            "nested spans are not additive; missing interrupted client time"
            " is unknown, never inferred as zero"
        ),
    }
