# Stage 31–34: Cyclic states and query-specific guarantees

**Evidence level:** Historical report; not re-executed during this publication.

For contractive continuous-state systems, query-weighted residual scheduling
reduced work by only 4.27% in the broad primary comparison (20% threshold FAIL).
A new-seed local/single-query stream achieved 44.07% fewer local evaluations
than ordinary maximum-residual scheduling.

A correct requested output could coexist with large error elsewhere. On a query
switch, 129/160 states produced by the weighted policy violated tolerance when
read without renewed validation. Query-specific correction recovered tolerance.
Sparse LU implementation and preprocessing reuse changed the runtime outcome.
Linear cached responses, and nonlinear small-change bounds, were strong controls.

**Audit correction:** a heap-initialization bug was found, fixed, and all
reported comparisons were rerun in the source study. Pre-fix results are not
accepted evidence. Small-change response caches used additional memory; this
was not a storage-compression demonstration.

[Imported key results](../../reference/history/stage31_34_key_results.csv)

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
