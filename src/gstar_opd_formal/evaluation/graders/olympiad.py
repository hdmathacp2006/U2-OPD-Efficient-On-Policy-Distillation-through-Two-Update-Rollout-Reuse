"""Bound the official judge subprocess without changing its equality tolerances."""

import json
import os
import subprocess
from pathlib import Path


def grade_official(cases: list[dict], context: tuple) -> list[dict]:
    python, root = context
    result = subprocess.run(
        [str(python), "-m", "gstar_opd_formal.evaluation.graders.olympiad_worker"],
        input=json.dumps(cases),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=root,
        timeout=max(30, 15 * len(cases)),
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(root) / "src")},
    )
    values = json.loads(result.stdout)
    if len(values) != len(cases) or any(
        type(row.get("correct")) is not bool for row in values
    ):
        raise ValueError("Official grader returned incomplete or invalid records")
    return values
