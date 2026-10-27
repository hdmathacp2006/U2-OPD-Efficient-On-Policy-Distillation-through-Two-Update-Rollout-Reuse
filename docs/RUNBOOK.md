# Portable runbook

The portable entrypoint is `scripts/portable.py`. It uses a user-owned filesystem store rather than the original author's Google Drive. On Modal, `scripts/modal_portable.py` calls this same entrypoint and persists the store in the `u2-opd-artifacts` Volume in the caller's account.

## Order and dependencies

1. `preflight` verifies the release's pinned source, model and data contract on CPU. It does **not** test GPU kernels or memory.
2. `setup --models train` installs the locked worker and downloads the pinned student and teacher. This step requires network access and ample local or Volume disk space.
3. `train-pair --gpu A100` trains Adaptive to 128 accepted updates, then U2 to 128, saves each retained checkpoint and publishes a paired receipt. The methods evaluation does not start here.
4. `evaluate-methods --gpu A100` restores the paired checkpoints and evaluates each method on MATH500. It cannot run until step 3 completes.
5. `evaluate-base --gpu L4` evaluates the pinned base student. It requires `setup --models evaluate` and can run independently on another GPU machine or Modal call.

For exact commands see the [README](../README.md). Each action writes worker logs, content-hashed artifacts and timing receipts under the selected work and store directories. A restart reuses verified completed phases. A partial phase is not a final benchmark result. Set `--resume-stopped` only after confirming the prior worker has stopped; concurrent workers for the same run can corrupt the interpretation of timing and recovery.

## Evaluation profile

The portable methods profile is `12k_a100_b20` and the base profile is `12k_base_l4`. They use a 12,288 generated-token cap. The historic formal release also defined a full 31,744-token profile; those runs are a different evaluation binding. Do not compare results across profiles as if only the training method changed. Batch limits are scheduling caps, not guaranteed concurrent throughput.

## Timing and claims

Accepted learning time is accumulated inside successful updates. Process time includes worker lifetime; setup, checkpoint transfer, timing publication and evaluation have their own scopes. The historical aggregates were recorded from completion messages and training receipts, not from raw prediction files in this repository. The portable source has a new study identity and must be measured again before reporting a portable speedup or accuracy number.
