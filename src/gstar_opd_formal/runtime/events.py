"""Immediate stage logs and token counts without changing generation scores."""

import logging
import sys
from dataclasses import dataclass
from time import perf_counter
from types import TracebackType

import torch
from tqdm import tqdm
from transformers import LogitsProcessor

LOGGER = logging.getLogger("gstar_opd.progress")
PROGRESS_REFRESH_SECONDS = 10


def configure_progress_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def report_progress(event: str, **fields: object) -> None:
    if LOGGER.isEnabledFor(logging.INFO):
        details = " ".join(f"{name}={value}" for name, value in fields.items())
        LOGGER.info("%s %s", event, details)


@dataclass(frozen=True)
class GenerationProgressSpec:
    label: str
    prompt_tokens: int
    maximum_tokens: int


class GenerationProgress(LogitsProcessor):
    """Observe prefix lengths; never copy GPU tokens or modify model scores."""

    def __init__(self, spec: GenerationProgressSpec) -> None:
        self.spec = spec
        self.completed_tokens = 0
        self.started = perf_counter()
        self.bar = tqdm(
            total=spec.maximum_tokens,
            desc=spec.label,
            unit="tok",
            file=sys.stdout,
            mininterval=PROGRESS_REFRESH_SECONDS,
            maxinterval=PROGRESS_REFRESH_SECONDS,
            miniters=1,
            ascii=True,
            dynamic_ncols=False,
            disable=not LOGGER.isEnabledFor(logging.INFO),
            bar_format="{desc}: {percentage:3.0f}%|{bar:16}| "
            "{n_fmt}/{total_fmt} tokens [{elapsed}, {rate_fmt}]",
        )

    def __enter__(self) -> "GenerationProgress":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.bar.close()
        report_progress(
            "generation_failed" if exception_type else "generation_done",
            stage=self.spec.label,
            tokens=self.completed_tokens,
            cap=self.spec.maximum_tokens,
            seconds=round(perf_counter() - self.started, 2),
        )

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Shape metadata is available on the CPU. A streamer would instead make
        # Transformers copy generated GPU tokens to the CPU at every decode step.
        self.update_tokens(input_ids.shape[-1] - self.spec.prompt_tokens)
        return scores

    def update_tokens(self, generated: int) -> None:
        generated = min(self.spec.maximum_tokens, max(0, generated))
        if generated > self.completed_tokens:
            self.bar.update(generated - self.completed_tokens)
            self.completed_tokens = generated
