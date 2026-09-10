# Stage 30: Relational state retention

**Evidence level:** Historical report; not re-executed during this publication.

A new four-valued relation-composition task removed the trivial majority-vote
solution. Across 20,480 inputs, continuation preserving intermediate state
reached 100% on deep paths/merges while shallow restarts and intermediate-state
erasure remained near 25%.

The local operations were supervised exhaustively and the composition structure
was manually supplied. A standard topological pass using the same learned local
operations produced the identical final probability vectors with as few as
one-sixteenth the nominal local updates at depth 16.

**Conclusion:** intermediate state can transport task-relevant information;
this does not establish an advantage over ordinary dependency-ordered execution.
The experiment was not end-to-end discovery of a general reasoning algorithm.

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
