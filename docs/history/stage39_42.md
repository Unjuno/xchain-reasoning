# Stage 39–42: Selection overhead and failed-check short-circuiting

**Evidence level:** Historical report; not re-executed during this publication.

An inexpensive two-hop selector improved over the best fixed comparator by
6.43%, below 20%. A finite-series query-error check was slower overall, and
transfer to different graphs/query-near changes also failed the primary goal.

A later short-circuit rule avoided creating check arrays when the first positive
term already prevented that certificate from succeeding. On new seeds it saved
12.07% against the earlier finite-check method and 15.38% against global update.
Against the best of its three simultaneous controls the reduction was 10.90%.

**Limits:** source changes were fixed at four nodes, so size and change density
were confounded. The main state updates were still global synchronous sweeps.
A residual check that fails to certify an answer does not prove that the answer
itself is wrong; signed cancellations can matter.

[Imported key results](../../reference/history/stage39_42_key_results.csv)

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
