"""Use the pinned official judge in its compatible, isolated ANTLR environment."""

import json
import sys

from gstar_opd_formal.evaluation.graders.vendor.olympiad import AutoScoringJudge


def precision_for(metadata: dict) -> object:
    value = metadata.get("error")
    if not value or "Tuple" in (metadata.get("answer_type") or ""):
        return 1e-8
    if isinstance(value, str) and "," in value:
        return [float(item) if item else 1e-8 for item in value.split(",")]
    return float(value)


def main() -> None:
    from importlib.metadata import version

    required = {
        "sympy": "1.13.3",
        "mpmath": "1.3.0",
        "antlr4-python3-runtime": "4.11.1",
    }
    if any(version(name) != pin for name, pin in required.items()):
        raise RuntimeError("Official symbolic grader dependencies changed")
    cases = json.load(sys.stdin)
    judge = AutoScoringJudge()
    results = []
    for case in cases:
        correct = judge.judge(
            case["gold"], case["prediction"], precision_for(case["metadata"])
        )
        results.append({"correct": bool(correct)})
    sys.stdout.write(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
