"""Resolve project files and runtime output independently of the historical tree."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    project: Path
    artifacts: Path

    @property
    def config(self) -> Path:
        return self.project / "configs"

    def source(self, relative: str) -> Path:
        path = (self.project / relative).resolve()
        if not path.is_relative_to(self.project.resolve()):
            raise ValueError("Source path escapes this formal project")
        return path

    def output(self, kind: str, run_id: str) -> Path:
        import re

        if kind not in {"training", "evaluation", "studies"} or not re.fullmatch(
            r"[a-f0-9]{32}", run_id
        ):
            raise ValueError("Unknown artifact kind or invalid run ID")
        return self.artifacts / kind / run_id


def project_root() -> Path:
    for path in (Path.cwd().resolve(), Path(__file__).resolve().parents[2]):
        if (path / "configs/studies/formal128.yaml").is_file():
            return path
    raise ValueError("Pass the explicit formal project root outside its source tree")
