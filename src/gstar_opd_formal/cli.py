"""Thin local commands; GPU actions run only when the user selects them."""

import argparse
import json
from pathlib import Path

from gstar_opd_formal.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "prepare-data",
            "show-study",
            "build-release",
            "train-pair",
            "evaluate-methods",
            "evaluate-base",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--gpu-profile", choices=("A100", "L4"), default="A100")
    parser.add_argument("--resume-stopped", action="store_true")
    args = parser.parse_args()
    args.root = args.root or project_root()
    if args.operation == "prepare-data":
        from gstar_opd_formal.data.manifests import prepare_data

        value = prepare_data(args.root)
    elif args.operation == "show-study":
        from gstar_opd_formal.contract import load_study

        study = load_study(args.root)
        value = {
            "study_identity": study.identity,
            "design": study.design,
            "code": study.code,
        }
    elif args.operation == "build-release":
        from gstar_opd_formal.runtime.release import build_release

        value = build_release(args.root, args.output or args.root / "releases")
    elif args.operation == "report":
        from gstar_opd_formal.reporting.comparison import build_report

        value = build_report(args.output or args.root / "artifacts")
    else:
        from gstar_opd_formal.orchestration.pipeline import (
            Pipeline,
            evaluate_base,
            evaluate_methods,
            train_pair,
        )

        if args.base is None or not args.python:
            parser.error(
                "Explicit --base and --python are required for selected GPU actions"
            )
        settings = Pipeline(
            args.base,
            args.python,
            gpu_profile=args.gpu_profile,
            resume_stopped=args.resume_stopped,
            project_directory=args.root,
        )
        actions = {
            "train-pair": train_pair,
            "evaluate-methods": evaluate_methods,
            "evaluate-base": evaluate_base,
        }
        value = actions[args.operation](settings)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
