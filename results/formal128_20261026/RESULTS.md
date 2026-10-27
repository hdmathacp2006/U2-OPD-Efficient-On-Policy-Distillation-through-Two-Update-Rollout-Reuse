# MATH-500 results received on 2026-10-26

Source: completion JSON pasted by the user in this conversation. Raw predictions,
hardware, token caps, batch settings and checkpoint hashes have not been fetched
or independently checked in this recording step. Configuration history alone
does not establish the exact execution binding of these runs.

Study: `af52b1e39c0617fcd9227da4cb224b7812bc052166038d922100f33e173cb915`.

The paired training receipt in this folder reports 128 accepted updates for
each arm. Its stage clock gives the following breakdown (seconds, rounded):

| Training stage | Adaptive | U2 |
|---|---:|---:|
| Accepted learning total | 3580.5 | 1651.4 |
| Student rollout | 3097.5 | 1165.2 |
| Teacher/current-student scoring | 226.8 | 179.8 |
| Student update | 255.0 | 252.2 |

The accepted-learning total is 53.9% lower for U2 (2.17× ratio), driven mainly
by rollout time. This is not a full process or end-to-end cost comparison; see
the receipt's `timing_scope` and [timing definitions](../../docs/TIMING.md).

| Model | Correct / questions | Accuracy | Evaluation process time |
|---|---:|---:|---:|
| Base Qwen3-0.6B | 342 / 500 | 68.4% | 3h 20m 51.10s |
| Adaptive | 350 / 500 | 70.0% | 2h 17m 08.96s |
| Original U2 | 345 / 500 | 69.0% | 2h 12m 17.43s |

Correct counts are derived from the reported accuracy and question count.
Times are `process_seconds_this_invocation`, not training time or isolated GPU
generation time. Do not infer a training speedup from this table. Paired
significance and extraction/truncation diagnostics require the actual predictions.

Original completion fields are retained in `methods_user_result.json` and
`base_user_result.json`. No training or evaluation was launched to save this report.
