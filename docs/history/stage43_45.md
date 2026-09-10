# Stage 43–45: Fixed-density changes and local intervals

**Evidence level:** Historical report; not re-executed during this publication.

Fixing change density and forcing the query itself to change defeated the older
fast-check policy: it was 3.81% slower than the best simultaneous control.

A new local interval solver kept exterior values uncertain rather than freezing
them as old points. In the prespecified 4,096-node local-grid population it saved
80.81%; new-seed full-input-change/moving-query confirmation saved 83.28% against
that trial's control set. Small graphs, remote edges and some strong-coupling
conditions remained negative.

Erasing saved state left all 946 successful local answers unchanged. Thus this
benefit did not require memory retention. In a boundary stress test, treating
old exterior values as exact accepted 160/160 incorrect local answers, whereas
interval boundaries rejected them and safely fell back.

**Scope:** known values in a contractive tanh system. The source implementation
used float64 audited against a reference, not directed-rounding verification.
Original inputs/graphs were still stored; no compression ratio was measured.

[Imported key results](../../reference/history/stage43_45_key_results.csv)

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
