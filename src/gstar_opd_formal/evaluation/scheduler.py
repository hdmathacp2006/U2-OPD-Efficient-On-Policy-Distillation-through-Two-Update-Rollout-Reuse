"""Refill bounded active requests; commit out-of-order completions by identity."""

import time
from collections import deque
from dataclasses import dataclass

from gstar_opd_formal.storage.files import write_json


@dataclass
class RollingScheduler:
    session: object
    backend: object
    questions: object
    completed: dict

    def commit(self) -> None:
        self.session.timeline.flush()
        self.session.store.commit(self.session.output, len(self.completed))

    def admit(self, waiting: deque) -> None:
        config = self.session.binding["evaluation_config"]
        if (
            not waiting
            or self.session.stop_requested()
            or len(self.backend.contexts) >= config["concurrency"]
            or time.perf_counter() + config["drain_margin_seconds"]
            >= self.session.deadline
        ):
            return
        self.session.store.require_writer()
        with self.session.timeline.measure("request_preparation"):
            while waiting and len(self.backend.contexts) < config["concurrency"]:
                self.backend.add(waiting.popleft())

    def run(self, source: tuple) -> dict:
        config = self.session.binding["evaluation_config"]
        waiting = deque(item for item in source if item["id"] not in self.completed)
        uncommitted = 0
        aborted, last_progress = [], time.perf_counter()
        self.admit(waiting)
        while self.backend.contexts:
            if (
                self.session.stop_requested()
                or time.perf_counter() + config["flush_margin_seconds"]
                >= self.session.deadline
            ):
                aborted = self.backend.abort()
                break
            with self.session.timeline.measure("engine_generation"):
                rows = self.backend.step()
            for row in rows:
                self.questions.save(row)
                self.completed[row["example_id"]] = row
                uncommitted += 1
                if uncommitted >= config["commit_questions"]:
                    self.commit()
                    uncommitted = 0
            now = time.perf_counter()
            if rows or now - last_progress >= 30:
                print(
                    f"formal_eval_progress completed={len(self.completed)} "
                    f"total={len(source)} "
                    f"active={len(self.backend.contexts)} waiting={len(waiting)}",
                    flush=True,
                )
                last_progress = now
            self.admit(waiting)
        write_json(
            self.session.output / "scheduler.json",
            {
                "completed": len(self.completed),
                "total": len(source),
                "aborted_inflight_ids": aborted,
                "unsubmitted_ids": [item["id"] for item in waiting],
                "inflight_scope": (
                    "not_scored_or_marked_truncated; "
                    "regeneration_may_differ_after_batch_change"
                ),
            },
        )
        self.commit()
        return self.completed
