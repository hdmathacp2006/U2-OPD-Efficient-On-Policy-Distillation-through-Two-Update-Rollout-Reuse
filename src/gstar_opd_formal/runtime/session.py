"""Execution session shared by worker-only train and evaluation runners."""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event


@dataclass
class Session:
    study: object
    output: Path
    binding: dict
    environment: dict
    store: object
    timeline: object
    deadline: float
    source: tuple = ()
    selection: dict = field(default_factory=dict)
    grader_python: str = ""
    stop: Event = field(default_factory=Event)

    def stop_requested(self) -> bool:
        return self.stop.is_set()
