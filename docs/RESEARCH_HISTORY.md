# Research history and corrections

These are experiment identifiers, not 51 independent validated discoveries.
The public history preserves failed hypotheses and superseded interpretations.
Only Stage 49–51 is a maintained runnable export and was fully re-executed for
this publication. Earlier entries are curated from the supplied reports; they
are not all independently reproduced here. See [source hashes](../provenance/history_sources.json).

| Stages | Question and result | Evidence level |
|---|---|---|
| [01–08](history/stage01_08.md) | Early reach and conductance studies | Historical report |
| [09](history/stage09.md) | Independent-headroom reconstruction | Historical report |
| [10–15](history/stage10_15.md) | Compute allocation and transfer | Historical report |
| [16–28](history/stage16_28.md) | Belief–evidence mismatch and expected value | Historical report |
| [29](history/stage29.md) | Full-depth controls and cheap correction | Historical report |
| [30](history/stage30.md) | Relational state retention | Historical report |
| [31–34](history/stage31_34.md) | Cyclic states and query-specific guarantees | Historical report |
| [35–38](history/stage35_38.md) | Changing operators and repairing certificates | Historical report |
| [39–42](history/stage39_42.md) | Selection overhead and failed-check short-circuiting | Historical report |
| [43–45](history/stage43_45.md) | Fixed-density changes and local intervals | Historical report |
| [46–48](history/stage46_48.md) | Learned recurrent models and decision fidelity | Historical report |
| [49–51](history/stage49_51.md) | True output accuracy and relation corruption | Publication rerun |

## Corrections that control interpretation

- A full sampler trajectory is not one neural-function evaluation. Reach also
  depends on update order and coordinate-wise dilation classes.
- The Stage 4 dimensional transport explanation was partially withdrawn when
  contact-level reachability was corrected in Stage 5.
- Shared R1 measurement noise can create apparent headroom/gain correlation;
  Stage 9 used independent calibration and evaluation samples.
- Conductance explanation is not cost-effective scheduling. A binary R1/R8
  compute frontier is not the full R1/R2/R4/R8 frontier.
- Some early mismatch features used observations the model had not received;
  those results were excluded and accessible-only features retested.
- Local convergence does not establish correctness. Topological passes and
  direct readout correction are essential cheap controls.
- The Stage 31–34 priority-queue bug was corrected and results rerun in that
  source study. An old certificate can also become invalid after query/operator
  changes even when the numerical code is bug-free.
- Model-fidelity speedups are not true task-accuracy gains. Cold local interval
  computation can work without saved state, so it is not evidence for memory
  retention or compression.
- Stage 49–51 returned to true output accuracy. Correct exterior relations help;
  wrong relations can harm despite convergence. Ordinary Bayes/linear solvers
  remain strong baselines.

## Reading numeric files

CSV units differ by experiment: some reductions are fractions, some are percent,
and accuracy differences use percentage points. Keep the source headers and
population definitions. Timing studies changed hardware, workload and control
sets. They cannot be combined into one universal speedup or accuracy curve.
The historical `PASS` field means that experiment's stated criterion, often
within a chosen subgroup; it is not a general validation certificate.
