# Frozen data

GSM8K7473 training rows retain their original bytes/source indices. Reserve the
historical feedback/tuning pools; dev128 uses the reserved feedback IDs.
The training pool is1024 distinct source questions, selected before model
results and excluding benchmark-overlapping normalized questions. Paired arms
share the same seed/order; Adaptive consumes the first512 and U2 the first256
fresh questions at the fixed128 endpoint. The frozen1024 pool remains unchanged.

Final evaluation uses full MATH500 (500) and Minerva Math (272). The Olympiad675
snapshot remains available for historical provenance and training-overlap
exclusion, but it is not an active formal128 evaluation panel. MATH500 also
reports the397 questions outside historical MATH103 development exposure.
Minerva gold is extracted from the solution's final complete boxed answer.
Olympiad uses final_answer plus upstream error/answer-type handling; unit and
multiple-answer metadata are retained. The official equality implementation
and its license are vendored with pinned provenance. Gold never enters OPD loss.

Every actual prompt is rendered with the pinned native tokenizer before the
release context is frozen. Benchmark questions are not truncated to fit an
arbitrary1024 prompt budget. The historical full profile caps output at31744;
the portable MATH500 profiles cap it at12288. They are different evaluation
bindings. Both use T0.7/top-p0.95, one completion, stable source-index
question seeds and a shared context configuration.
This is not the paper's Avg16 metric. Context/KV analytical bounds do not prove
total CUDA peak memory; actual environment evidence is recorded per run.

MATH retains Math-Verify0.8.0 final-answer semantics. Minerva uses it when its
gold is parseable; five known source gold expressions (NumPy arcsin syntax,
a differential equation, two trailing-decimal fractions and a symbolic list)
use the pinned official symbolic judge instead. NumPy arcsin is rendered as
the mathematically identical LaTeX arcsin on both sides. Source answers are
not edited or dropped, and the same declared policy applies to every arm.
This hybrid grader is an adapted protocol, not Qwen's exact Minerva evaluator.
The symbolic fallback retains the official AutoScoringJudge implementation
and tolerances in a separate pinned SymPy1.13.3/ANTLR4.11.1 environment; the ML
worker keeps Math-Verify's ANTLR4.13.2.
An infrastructure grading error leaves raw answers saved for recovery rather
than silently treating every affected question as an incorrect model answer.

Sources and immutable revisions:
- https://github.com/openai/grade-school-math
- https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- https://github.com/QwenLM/Qwen2.5-Math/tree/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/data
- https://github.com/OpenBMB/OlympiadBench/tree/ba5b26a7e2849940b598a9159c1190daa2b9175f

The MIT notices supplied by GSM8K, the Qwen2.5-Math evaluation repository and
OlympiadBench are copied in `licenses/`. The HuggingFaceH4 MATH-500 dataset card
is the source for the frozen MATH500 snapshot; its terms should be checked at
the upstream dataset page before republishing that snapshot elsewhere. The
initial LoRA adapter and any downloaded base model weights remain subject to
their respective model repository terms, not this repository's code license.
