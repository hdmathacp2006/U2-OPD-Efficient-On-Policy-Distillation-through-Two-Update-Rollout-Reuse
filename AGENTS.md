# U2-OPD repository

- Preserve the original Adaptive and U2 scientific objectives. U2 performs at most two updates with refreshed targets and a fixed behavior snapshot.
- Never launch paid GPU jobs automatically. CPU verification is not CUDA proof.
- Keep secrets, model caches, generated environments, runtime artifacts and private checkpoints out of Git.
- Keep source in `src/`; run Ruff and the CPU preflight after Python changes.
- Keep scientific study identity separate from execution and storage receipts.
- Report accepted learning, full process, setup and evaluation times separately.
