"""Generate three thin Colab entrypoints against one immutable formal release."""

import hashlib
import textwrap

from gstar_opd_formal.storage.files import write_json


def cell(kind: str, source: str) -> dict:
    value = {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    value["id"] = hashlib.sha256("".join(value["source"]).encode()).hexdigest()[:16]
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def source_cell(release: dict) -> str:
    return f'''
import hashlib, json, os, subprocess, sys, tarfile, tempfile, time
from pathlib import Path, PurePosixPath
from google.colab import auth
import google.auth
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import build_http, MediaIoBaseDownload
from tqdm.auto import tqdm

setup_started = time.perf_counter()
auth.authenticate_user()
credentials, _ = google.auth.default()
http = build_http()
http.timeout = 30
service = build("drive", "v3", http=AuthorizedHttp(credentials, http=http), cache_discovery=False)
root_id = "{release["root_folder_id"]}"
access = service.files().get(fileId=root_id, fields="id,capabilities(canAddChildren)", supportsAllDrives=True).execute(num_retries=3)
if not access.get("capabilities", {{}}).get("canAddChildren"):
    raise PermissionError("Authenticate the account with write access to the approved Drive folder")

def find_path(relative):
    current = {{"id": root_id}}
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\\\" in relative:
        raise ValueError("Invalid published artifact path")
    for name in path.parts:
        escaped = name.replace("'", "\\\\'")
        page = service.files().list(q=f"'{{current['id']}}' in parents and name='{{escaped}}' and trashed=false", fields="files(id,name,parents,size),nextPageToken", supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=3)
        if len(page.get("files", [])) != 1 or page.get("nextPageToken"):
            raise ValueError(f"Expected one immutable Drive artifact: {{name}}")
        current = page["files"][0]
    return current

deployment_record = find_path("{release["drive_path"]}/deployment.json")
deployment_bytes = service.files().get_media(fileId=deployment_record["id"], supportsAllDrives=True).execute(num_retries=3)
if hashlib.sha256(deployment_bytes).hexdigest() != "{release["deployment_sha256"]}":
    raise ValueError("Deployment bytes differ from this notebook release")
deployment = json.loads(deployment_bytes)
base = Path(LOCAL_ROOT)
base.mkdir(parents=True, exist_ok=True)
records = deployment["artifacts"]
for name, record in records.items():
    metadata = find_path(record["drive_path"])
    if int(metadata["size"]) != record["size"]:
        raise ValueError("Published artifact size changed")
    target = base / "staging" / record["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.with_suffix(".partial").open("wb") as stream, tqdm(total=record["size"], desc=f"Download {{name}}", unit="B", unit_scale=True) as bar:
            download = MediaIoBaseDownload(stream, service.files().get_media(fileId=metadata["id"], supportsAllDrives=True), chunksize=8 * 2**20)
            done = False
            while not done:
                status, done = download.next_chunk(num_retries=3)
                bar.update(int(status.progress() * record["size"]) - bar.n)
        target.with_suffix(".partial").replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Downloaded source artifact hash mismatch")
manifest = json.loads((base / "staging" / records["manifest"]["path"]).read_bytes())
project = base / "project"
if not project.exists():
    with tempfile.TemporaryDirectory(prefix="formal_stage_", dir=base) as temporary:
        stage = Path(temporary) / "project"
        stage.mkdir()
        with tarfile.open(base / "staging" / records["source"]["path"]) as archive:
            members = archive.getmembers()
            if len({{member.name for member in members}}) != len(members) or sum(member.size for member in members) > 2**30:
                raise ValueError("Invalid formal archive size or duplicate paths")
            for member in members:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts or "\\\\" in member.name:
                    raise ValueError("Unsafe formal archive member")
            archive.extractall(stage, members=members, filter="data")
        stage.rename(project)
for record in manifest["files"]:
    path = project / record["path"]
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Staged source differs from release: {{record['path']}}")
sys.path.insert(0, str(project / "src"))
os.environ.update(PYTHONUNBUFFERED="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn", HF_HOME="/content/gstar_formal_model_cache")
from gstar_opd_formal.storage.drive import DriveStore
from gstar_opd_formal.storage.controls import destination
from gstar_opd_formal.storage.files import write_json
destination_value = destination(DriveStore(service, base / ".setup_drive"), root_id)
write_json(base / "destination.json", destination_value)
print("Formal source verified. The historical project is not required.")
'''


def setup_cell(action: str) -> str:
    return f'''
from gstar_opd_formal.runtime.progress import CommandView, run_command
worker_python = str(base / "env/bin/python")
setup_commands = [
    ("Install locked worker", [sys.executable, "-u", "-m", "gstar_opd_formal.runtime.bootstrap", "install", "--root", str(project), "--environment", str(base / "env")], 4200),
    ("Download pinned {"student and teacher" if action == "train" else "student only"}", [worker_python, "-u", "-m", "gstar_opd_formal.runtime.bootstrap", "models", "--action", "{action}"], 2400),
]
if "{action}" == "evaluate":
    setup_commands.append(("Install official Olympiad grader", [sys.executable, "-u", "-m", "gstar_opd_formal.runtime.bootstrap", "grader", "--root", str(project), "--environment", str(base / "grader_env")], 600))
with tqdm(total=len(setup_commands), desc="Setup", unit="stage") as bar:
    for label, command, timeout in setup_commands:
        run_command(command, CommandView(project, base / "ui_logs", label, timeout_seconds=timeout))
        bar.update(1)
write_json(base / "setup_timing.json", {{"seconds": time.perf_counter() - setup_started, "scope": "setup_download_install_separate_from_training", "action": "{action}"}})
'''


def build_notebooks(folder: object, release: dict) -> list:
    paths = []
    for action, title, entry in [
        ("train", "Train_Only", "train_pair"),
        ("evaluate", "Eval_Methods_Only", "evaluate_methods"),
        ("evaluate", "Eval_Base_Only", "evaluate_base"),
    ]:
        headings = {
            "Train_Only": (
                "# Formal128: Adaptive + original U2 training only\n"
                "A10040GB, Run all: Adaptive128 → verified Drive checkpoint → "
                "close worker → U2128 → verified Drive checkpoint and paired "
                "receipt → STOP. No evaluation is started."
            ),
            "Eval_Methods_Only": (
                "# Formal128: Adaptive + U2 MATH500 evaluation, 12,288 output tokens\n"
                "Choose A10040GB, then Run all. Restore both checkpoint128 "
                "files from Drive and evaluate MATH500 only. Default concurrency "
                "20; set METHODS_CONCURRENCY=10 for the lower-memory option. "
                "Training never runs. Start after the train receipt is saved."
            ),
            "Eval_Base_Only": (
                "# Formal128: base evaluation, 12,288 output tokens on L4\n"
                "Choose L4, then Run all. Run pure inference on MATH500 "
                "only. No teacher, checkpoint or training receipt is "
                "required. This can run alongside training on a separate GPU runtime."
            ),
        }
        heading = headings[title]
        settings = f'RUN_SELECTED_ACTION = True\nRESUME_STOPPED_RUN = False\nLOCAL_ROOT = "/content/gstar_formal_{release["release_id"][:10]}"\n'
        if title == "Eval_Methods_Only":
            settings += "METHODS_CONCURRENCY = 20\n"
        hardware = f'''
import shutil, subprocess
observed_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
gpu_profile = "A100" if "A100" in observed_gpu else "L4" if "L4" in observed_gpu else None
if gpu_profile is None or ("{action}" == "train" and gpu_profile != "A100"):
    raise RuntimeError(f"Select A10040GB for training, L4/A100 for evaluation; observed {{observed_gpu}}")
if shutil.disk_usage("/content").free < {35 if action == "train" else 20} * 2**30:
    raise RuntimeError("Insufficient local disk for the locked environment and models")
print("GPU:", observed_gpu, "| formal {action} only")
'''
        profile = (
            ', evaluation_profile=f"12k_a100_b{METHODS_CONCURRENCY}"'
            if title == "Eval_Methods_Only"
            else ', evaluation_profile="12k_base_l4"'
            if action == "evaluate"
            else ""
        )
        if title == "Eval_Methods_Only":
            hardware += '\nif gpu_profile != "A100":\n    raise RuntimeError("Select A10040GB for Adaptive/U2 12k evaluation")\n'
            hardware += '\nif METHODS_CONCURRENCY not in (20, 10):\n    raise ValueError("METHODS_CONCURRENCY must be 20 or 10")\n'
        elif title == "Eval_Base_Only":
            hardware += '\nif gpu_profile != "L4":\n    raise RuntimeError("Select L4 for base 12k evaluation")\n'
        run = f"""
from gstar_opd_formal.orchestration.pipeline import Pipeline, {entry}
if RUN_SELECTED_ACTION:
    result = {entry}(Pipeline(base, worker_python, gpu_profile=gpu_profile, resume_stopped=RESUME_STOPPED_RUN{profile}))
    print(json.dumps(result, indent=2))
"""
        value = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "accelerator": "GPU",
                "colab": {"provenance": []},
            },
            "cells": [
                cell("markdown", heading),
                cell("code", settings),
                cell("code", hardware),
                cell("code", source_cell(release)),
                cell("code", setup_cell(action)),
                cell(
                    "markdown",
                    "If interrupted: confirm the old worker stopped, set RESUME_STOPPED_RUN=True and rerun. Completed verified phases are reused. No runtime reset is required by this workflow.",
                ),
                cell("code", run),
            ],
        }
        path = folder / f"GStar_OPD_Formal128_{title}.ipynb"
        write_json(path, value)
        paths.append(path)
    return paths
