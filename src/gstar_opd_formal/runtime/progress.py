"""Display real worker counters while forwarding output through a bounded pipe."""

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

HEARTBEAT_SECONDS = 5.0
POLL_SECONDS = 0.25
COUNT_PATTERN = re.compile(
    r"(?:formal_training_progress|formal_eval_progress) "
    r"completed=(\d+) total=(\d+)"
)
OUTPUT_PATTERN = re.compile(r"formal_run_id=[a-f0-9]{32} action=\w+ output=(.+)")


@dataclass
class CommandView:
    cwd: Path
    log_directory: Path
    label: str
    action: str = "setup"
    endpoint: int = 0
    timeout_seconds: float = 7560.0


@dataclass
class ProgressState:
    action: str
    endpoint: int = 0
    completed: int = 0
    baseline: int = 0
    total: int | None = None
    phase: str = "Starting process"
    output: Path | None = None
    observed_counter: bool = False
    measurements: deque = field(default_factory=lambda: deque(maxlen=12))

    def __post_init__(self) -> None:
        self.total = self.endpoint or None
        if self.action == "train":
            self.total = self.endpoint

    def observe(self, line: str) -> None:
        output = OUTPUT_PATTERN.search(line)
        if output:
            self.output = Path(output[1].strip())
            self.read_saved_progress()
            self.baseline = self.completed
        count = COUNT_PATTERN.search(line)
        if count:
            self.read_saved_progress()
            completed, total = map(int, count.groups())
            if 0 <= completed <= total:
                self.total = total
                if completed > self.completed:
                    self.measurements.append((time.monotonic(), completed))
                self.completed = max(self.completed, completed)
                self.observed_counter = True
        self.phase = phase_for(line, self.phase)

    def read_saved_progress(self) -> None:
        if self.output is None:
            return
        try:
            value = json.loads((self.output / "summary.json").read_bytes())
        except (OSError, ValueError):
            return
        if self.action == "train":
            count = value.get("completed_updates", 0)
        elif self.action == "evaluate":
            count = value.get("count", 0)
        elif self.action == "diagnostic":
            count = len(value.get("records", []))
        else:
            return
        if type(count) is int and 0 <= count <= (self.total or 0):
            if self.observed_counter and count > self.completed:
                self.measurements.append((time.monotonic(), count))
            self.completed = max(self.completed, count)
            if not self.observed_counter:
                self.baseline = self.completed


def phase_for(line: str, previous: str) -> str:
    names = (
        ("drive_v2_commit_done", "Drive commit saved"),
        ("rolling_eval", "Generating answers"),
        ("colab_v2_training_progress", "Optimizer update accepted"),
        ("backward_start", "Student backward"),
        ("backward_done", "Backward finished"),
        ("student_forward_start", "Student forward"),
        ("vllm_sleep", "Teacher/student scoring"),
        ("vllm_ready", "vLLM ready"),
        ("Capturing CUDA graphs", "Capturing CUDA graphs"),
        ("torch.compile", "Compiling kernels"),
        ("Loading checkpoint shards", "Loading model shards"),
        ("Fetching", "Downloading model files"),
        ("Installing", "Installing dependencies"),
        ("Downloading", "Downloading dependencies"),
        ("colab_v2_run_id", "Worker checks and model loading"),
    )
    return next((name for token, name in names if token in line), previous)


def progress_eta(state: ProgressState, now: float) -> float | None:
    if state.total is None or len(state.measurements) < 2:
        return None
    first_time, first_count = state.measurements[0]
    last_time, last_count = state.measurements[-1]
    if last_count <= first_count or last_time <= first_time:
        return None
    seconds_per_item = (last_time - first_time) / (last_count - first_count)
    remaining = max(0, state.total - state.completed)
    # Initial model loading has no count interval, so it cannot inflate ETA.
    return seconds_per_item * remaining + max(0.0, now - last_time - seconds_per_item)


@dataclass
class ProgressDisplay:
    label: str
    state: ProgressState
    bar: object = None
    started: float = field(default_factory=time.monotonic)

    def refresh(self) -> None:
        from tqdm.auto import tqdm

        if self.bar is None or (self.bar.total, self.bar.initial) != (
            self.state.total,
            self.state.baseline,
        ):
            if self.bar is not None:
                self.bar.close()
            self.bar = tqdm(
                total=self.state.total,
                initial=self.state.baseline,
                desc=self.label,
                unit="update" if self.state.action == "train" else "item",
                dynamic_ncols=True,
                mininterval=1.0,
                smoothing=0.0,
                leave=True,
            )
        self.bar.update(max(0, self.state.completed - self.bar.n))
        eta = progress_eta(self.state, time.monotonic())
        estimated = "?" if eta is None else tqdm.format_interval(eta)
        elapsed = tqdm.format_interval(time.monotonic() - self.started)
        self.bar.bar_format = (
            "{desc}: [" + elapsed + " elapsed] {postfix}"
            if self.state.total is None
            else "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            + "["
            + elapsed
            + " elapsed, work ETA~"
            + estimated
            + "] {postfix}"
        )
        self.bar.set_postfix_str(self.state.phase, refresh=False)
        self.bar.refresh()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


@dataclass
class OutputReader:
    stream: object
    log: object
    events: queue.Queue

    def read(self) -> None:
        logging_failed = False
        try:
            for line in self.stream:
                if not logging_failed:
                    try:
                        self.log.write(line)
                    except OSError:
                        logging_failed = True
                self.events.put(line)
        finally:
            self.events.put(None)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
        process.wait(timeout=15)
        return
    for stop_signal, seconds in (
        (signal.SIGINT, 45),
        (signal.SIGTERM, 15),
        (signal.SIGKILL, 15),
    ):
        try:
            os.killpg(process.pid, stop_signal)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def refresh_display(display: ProgressDisplay) -> None:
    try:
        display.refresh()
    except Exception as error:
        # A widget failure must not terminate an otherwise healthy GPU worker.
        print(f"progress_display_unavailable={error}", flush=True)


def monitor(process: subprocess.Popen, context: tuple) -> int:
    view, reader, display = context
    started = time.monotonic()
    refreshed = started
    terminal_at, ended = None, False
    tail = deque(maxlen=80)
    refresh_display(display)
    while not ended:
        if time.monotonic() - started >= view.timeout_seconds:
            raise subprocess.TimeoutExpired(process.args, view.timeout_seconds)
        try:
            line = reader.events.get(timeout=POLL_SECONDS)
            if line is None:
                ended = True
            else:
                tail.append(line)
                before = display.state.completed
                display.state.observe(line)
                if display.state.completed != before:
                    refresh_display(display)
        except queue.Empty:
            pass
        now = time.monotonic()
        if now - refreshed >= HEARTBEAT_SECONDS:
            display.state.read_saved_progress()
            refresh_display(display)
            refreshed = now
        if process.poll() is not None:
            terminal_at = terminal_at or now
            if now - terminal_at >= 3 and reader.events.empty():
                break
    code = process.wait(
        timeout=max(0.1, view.timeout_seconds - (time.monotonic() - started))
    )
    if code:
        print("".join(tail), flush=True)
    return code


def run_command(command: list[str], view: CommandView) -> Path:
    view.log_directory.mkdir(parents=True, exist_ok=True)
    log_path = view.log_directory / f"{uuid.uuid4().hex}.log"
    state = ProgressState(view.action, view.endpoint)
    display = ProgressDisplay(view.label, state)
    print(f"Full worker log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=view.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(view.cwd / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
            start_new_session=os.name != "nt",
        )
        reader = OutputReader(process.stdout, log, queue.Queue(maxsize=2048))
        thread = threading.Thread(target=reader.read, daemon=True)
        thread.start()
        try:
            code = monitor(process, (view, reader, display))
            state.phase = "Segment finished" if code == 0 else f"Failed: exit {code}"
            refresh_display(display)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            stop_process(process)
            raise
        finally:
            thread.join(timeout=1)
            display.close()
    print(f"{view.label}: process finished; log={log_path}", flush=True)
    return log_path
