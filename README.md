# U2-OPD: two updates per student rollout

This repository contains the standalone formal comparison between an adapted Adaptive FastOPD baseline and **original U2-OPD**. Both methods distill from a fixed Nemotron-Cascade-8B-Thinking teacher into a Qwen3-0.6B student. U2 uses one fresh on-policy update followed by at most one refreshed off-policy update on the same student response. No teacher-generated solutions or correctness reward enter the training objective. See [methods](docs/METHODS.md).

The research code, pinned input manifests, zero-update LoRA adapter, dependency locks, and tests are included. Model weights are downloaded from their pinned upstream revisions. Published **trained** checkpoints are not included: a new user can run both arms from initialization, but cannot evaluate the original checkpoints without obtaining those checkpoint bytes separately. The original reported aggregates and their evidence limitations are in [results](results/formal128_20261026/RESULTS.md).

## Requirements

- Linux x86-64, or Windows with WSL2 and a working NVIDIA CUDA driver. The GPU worker uses Python **3.12.12**, CUDA 12.8 wheels, FlashAttention 2.8.3 and the hash-locked [worker requirements](requirements/colab-cu128.lock).
- Training: an NVIDIA **A100 with at least 38 GiB** available; evaluation: A100 or L4 with at least 22 GiB. The runtime validates the GPU, BF16 support, package versions and actual environment before work starts.
- Sufficient persistent disk for the pinned student and teacher models, worker environment, outputs and checkpoints. Model downloads require network access.
- A separate user-owned directory or Modal Volume for durable artifacts. The portable path needs no access to the author's Google Drive.

The CPU preflight can run without a GPU. It does not establish that a later CUDA run will fit or succeed on untested hardware.

## Run on your Linux machine

From a fresh clone, install the lightweight controller environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Choose two persistent paths, then run the CPU preflight:

```bash
export U2_WORK="$PWD/.runtime/work"
export U2_STORE="$PWD/.runtime/store"
python scripts/portable.py preflight --work "$U2_WORK" --storage "$U2_STORE"
```

The setup command installs the locked worker Python and downloads both pinned models. It can take substantial time and disk space:

```bash
python scripts/portable.py setup --work "$U2_WORK" --storage "$U2_STORE" --models train
python scripts/portable.py train-pair --work "$U2_WORK" --storage "$U2_STORE" --gpu A100
python scripts/portable.py evaluate-methods --work "$U2_WORK" --storage "$U2_STORE" --gpu A100
```

Base evaluation is independent of training. On a separate L4 machine, mount or copy the same source and use its own work directory and persistent store:

```bash
python scripts/portable.py setup --work "$U2_WORK" --storage "$U2_STORE" --models evaluate
python scripts/portable.py evaluate-base --work "$U2_WORK" --storage "$U2_STORE" --gpu L4
```

Methods evaluation needs the paired receipt and checkpoints in the **same storage root** as training. A failed phase may resume only after the previous worker stopped; pass `--resume-stopped` to the corresponding action. Verified completed phases are reused. Do not run two workers for the same study and role at once. See the [runbook](docs/RUNBOOK.md).

## Run in your Modal account

Install and authenticate the Modal client locally:

```bash
python -m pip install -r requirements/modal-client.txt
modal setup
modal run scripts/modal_portable.py --action preflight
modal run scripts/modal_portable.py --action train-pair
modal run scripts/modal_portable.py --action evaluate-methods
modal run scripts/modal_portable.py --action evaluate-base
```

These commands create a `u2-opd-artifacts` Volume **in your Modal account**. Setup runs in a CPU container; training and methods evaluation request A100, while base evaluation requests L4. GPU actions can incur charges. The Volume stores the pinned worker environment, model cache, checkpoint objects and receipts. To resume a stopped action, run it again with `--resume-stopped` only after confirming the prior function is no longer active. The Modal entrypoint has not been GPU-executed as part of this repository publication; its CPU preflight and local storage tests do not replace an end-to-end Modal run.

## Reproduction scope

Training uses the frozen GSM8K pool and 128 accepted optimizer updates per arm. The two arms have **different fresh-example coverage**: 512 question slots for Adaptive and 256 for U2. Evaluation defaults to MATH500, one completion per question. The portable release has a new source and storage identity; do not substitute its run IDs for the original Colab receipts. Results from different GPU profiles or token caps should not be presented as a controlled speed test.

Run CPU checks with `python -m pytest` and `python -m ruff check .`. First-party code is [MIT licensed](LICENSE). Upstream datasets, model artifacts and the vendored symbolic grader retain their own terms and attribution; see [data sources](docs/DATA.md) and [third-party notices](licenses). No credentials, trained checkpoints or user-specific Drive objects are committed to this repository.
