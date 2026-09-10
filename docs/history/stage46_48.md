# Stage 46–48: Learned recurrent models and decision fidelity

**Evidence level:** Historical report; not re-executed during this publication.

Five 81-parameter recurrent graph models were trained on output loss, with four
hidden channels and an explicitly enforced contraction bound. They were tested
on 1,200 synthetic inputs, not a language-model benchmark.

Fixed-radius numerical recovery failed locally on all 320 Stage 46 queries and
was 36.75% slower in its primary population. An adaptive rule that only needed
to certify the output sign saved 83.50% in its local/large-graph primary subset.
The all-condition reduction was 18.06%, still below 20%.

All Stage 47 decisions matched the high-precision learned model, but 5/720 were
wrong relative to the generating teacher. This is model error, not an interval
containment failure. Near a decision boundary, larger regions or fallback were
needed; remote-edge cases could be slower.

**Central distinction:** reproducing a model's answer and improving the true
answer are different tasks. Stage 49 deliberately returned to true output risk.

[Imported key results](../../reference/history/stage46_48_key_results.csv)

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
