# Implementation and verification

The standalone root is GStar_OPD_Formal. It does not import or link to the
historical sibling. Existing source, runs and checkpoints remain there unchanged.

```text
GStar_OPD_Formal/
  src/gstar_opd_formal/
    methods/          original Adaptive and fixed-two U2
    opd/              preserved loss, scoring, backward and streaming core
    controllers/      original adaptive horizon policy
    training/         initialization, 128-update loop, checkpoint/history
    models/           pinned weights, FlashAttention2, no-Inductor vLLM graphs
    evaluation/       dataset grading, continuous requests, per-item recovery
    storage/          exact Drive destination, SHA objects, leases and commits
    runtime/          locked install, worker, timing, notebooks and release
    orchestration/    lightweight sequential train / separate eval
    reporting/        primary/partial metrics and distinct timing scopes
  configs/            shared settings plus explicit method/study definitions
  data/raw/           pinned GSM8K and all three benchmark snapshots
  data/manifests/     paired train1024, reserved dev128, complete test IDs
  assets/initialization/  preserved zero-update LoRA, verified zero B matrices
  requirements/       hash-locked CUDA runtime and isolated symbolic grader
  notebooks/          generated train, method-eval and base-eval entrypoints
  tests/              ported core regressions and new integration/recovery tests
  provenance/         actual file map, mathematical parity and source evidence
  archives/legacy_20261025/  all879 original files, exact bytes
  releases/           immutable source bundle, deployment and thin notebooks
```

The historical archive contains all19 source packages, configs, scripts, tests,
notebooks and inventoried documentation. Only the new formal namespace is
installed in Colab. Historical result/checkpoint files are referenced, not moved.
The preserved planning FILE_MAP.csv is a proposal; source_map.csv records actual
destinations. Planned tests not ported remain byte-preserved in the archive;
formal method/workflow/recovery tests replace obsolete family-specific routes.

CPU evidence covers actual small Transformers/PEFT models, first-update
loss/gradient/Adam parity, odd-boundary pending reuse, checkpoint/RNG recovery,
Drive committed-byte restore, partial metrics, raw grading recovery, paired
dispatch order, interpreter boundaries and rejected malformed adapter states.
Independent archive extraction reproduces frozen historical scientific IDs.

All1447 available benchmark golds passed the recorded CPU grading sanity audit;
the formal128 run selects only MATH500 and Minerva (772 items).
The longest actual rendered prompt is1277 tokens, so eval uses a common1280
prompt budget and33024 context, with31744 output tokens and no prompt truncation.
Five Minerva gold formats use the explicitly declared official symbolic fallback;
see DATA.md. These are adapted evaluation semantics, not a claim of an exact
paper reproduction.

No Colab/Modal GPU work was launched during implementation. CPU checks cannot
confirm future CUDA kernels, Colab availability or GPU peak memory. Full train
and evaluation remain user-selected notebook actions. No accuracy or speedup
for the new128-update study has been measured yet.

Verification sources: ../provenance/source_verification.json,
../provenance/method_parity.json, ../provenance/legacy_closure.json,
../provenance/initialization_validation.json, ../provenance/gold_validation.json,
../data/manifests/prompt_capacity.json, and ../artifacts/reports/all_tests.xml.
