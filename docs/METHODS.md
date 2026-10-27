# Preserved method

Let mu generate a fresh response, q_j be the detached current student before
update j, p the frozen teacher, and S_t=Top16(q_j at the student prefix).
The candidate advantage is

    A_t(v) = q_j(v)/sum_{u in S_t}q_j(u) * (log p(v)-log q_j(v)).

The candidate ratio is exp(clamp(log pi_theta(v)-log q_j(v), -20,20)).
Use PPO clip0.2 and the original dual-clip3 rule for negative advantages.
Sum candidates, then take the mean across **all valid response tokens** in
the effective batch. Accumulate one trajectory at a time; clip globally1 and
advance AdamW once (lr1e-6, weight decay0). Teacher and base student are frozen;
only rank16/alpha32/dropout0 LoRA parameters learn. Full-vocabulary
normalization and native teacher prompt + verbatim student response tokens
are preserved. Microbatching does not change the objective denominator.

Adaptive generates fresh responses every update and observes its boundary
statistics after that accepted update. U2 generates one batch, snapshots
chosen-token behavior likelihoods, learns once, and makes one adaptive
observation. For its second update, recompute teacher/current-student targets
and candidate support on the same response. The fixed original behavior
snapshot is retained. Prefix position t uses

    w_1=1
    w_t=exp(min(log4, sum_{k<t}(log pi_current(y_k)-log mu(y_k))/(t-1))).

Detach weights; multiply the unchanged candidate row loss before the global
token mean. This is a biased capped geometric stabilizer, not exact importance
sampling. The first update is on-policy; the one reuse update is off-policy.
No teacher-generated trajectories, SFT objective, cache or auxiliary loss.

The local baseline is adapted: student/teacher/LoRA/data/precision and batch
differ from the original paper. Do not call it an exact paper reproduction.
Source ancestry and mathematical parity evidence are in provenance/.

Source: [Adaptive FastOPD](https://arxiv.org/html/2607.29494v1), preserved legacy
gstar_opd/opd/{objective,engine,signals}.py and
gstar_opd_reuse/opd/{objective,engine,state}.py in the historical snapshot.
