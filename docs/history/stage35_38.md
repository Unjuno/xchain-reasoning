# Stage 35–38: Changing operators and repairing certificates

**Evidence level:** Historical report; not re-executed during this publication.

With changing coupling, an old query-influence vector was not automatically a
valid stopping certificate. A stress control violated tolerance on 1,222/2,560
requests, including 316/1,280 with an unchanged query.

Adding the positive residual defect of an approximate influence vector to the
output-error bound allowed incomplete certificate repair. Safe-side policies
had no observed tolerance failures in 30,720 outputs in the source study.

The Stage 37 512-node primary population saved 19.06%, below its fixed 20%
criterion. A prespecified 2,048-node local/fixed-query control saved 59.35%.
Later same-input timings did not overwrite the primary FAIL. On moving queries,
resetting the influence estimate could be cheaper than preserving it.

**Scope:** known contractive operators, float64 numerical audits rather than
rigorous directed-rounding interval certification. No LLM or compression test.

[Imported key results](../../reference/history/stage35_38_key_results.csv)

[History index](../RESEARCH_HISTORY.md) · [Source provenance](../../provenance/history_sources.json)
