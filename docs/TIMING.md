# Timing interpretation

Report separately:

1. accepted_learning_seconds: generation, scoring, backward/Adam, observers,
   controller and synchronization of accepted updates;
2. full child-process seconds: model loading, accepted work, blocking checkpoint
   and Drive commits, close/shutdown; includes resumed closed segments;
3. setup/model download/install, restore-only completed runs and publication
   overhead, separately identified;
4. evaluation generation/grading/commit/process time, outside training totals.

Nested stage spans are not additive. Transfer sidecars close the preceding
commit without self-referential hashes. Unknown time after an abrupt host/VM
loss remains unknown; do not infer zero or present it as measured GPU billing.

The primary comparison is accuracy at fixed128updates plus matched-runtime
training time. Same updates is not equal fresh coverage or GPU budget.
Offline dev checkpoint accuracy-versus-recorded-time curves are not an online
MLPerf time-to-quality measurement. Historical A8/U2 hardware/timing scopes
cannot establish a causal formal speedup. No >80% accuracy is guaranteed.
