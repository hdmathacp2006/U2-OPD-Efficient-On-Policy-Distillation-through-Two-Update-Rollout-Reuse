# Implementation map

The public repository contains the formal Adaptive FastOPD and original U2-OPD study. The mathematical objectives live in `src/gstar_opd_formal/opd/` and `src/gstar_opd_formal/methods/`; the adaptive horizon controller lives in `controllers/`. The worker in `runtime/worker.py` runs training and evaluation. `orchestration/pipeline.py` keeps paired training and later evaluation as separate actions.

`configs/` defines both arms and the 128-update endpoint. `data/` contains frozen inputs and manifests. `assets/initialization/` contains the zero-update LoRA adapter; trained checkpoints are produced in a user-owned store and are not in Git. `requirements/` pins the CUDA worker and isolated symbolic grader. `storage/local.py` provides filesystem and Modal Volume artifact transport; the original Drive transport remains available only when explicitly configured. `scripts/portable.py` and `scripts/modal_portable.py` expose the two supported portable entrypoints.

The portable source has its own content-derived study identity. Historical MATH500 aggregates and their evidence limits are recorded in `results/formal128_20261026/`. See [run order](RUNBOOK.md), [methods](METHODS.md), [data](DATA.md) and [timing](TIMING.md) before comparing new runs with those historical numbers.
