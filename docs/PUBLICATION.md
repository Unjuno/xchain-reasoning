# Public export policy and scope

This is a **curated initial publication**, not an unfiltered upload of the working
container. The repository was initialized with an Apache-2.0 license; that file
is kept unchanged.

## Included

Portable Stage 49–51 source, exact reference summaries, protocols, a mathematical
specification, tests, and a curated history spanning Stage 1–51. Historical key
result CSVs retain their original units and verdicts. Source-package hashes
record provenance without publishing the packages or confidential storage paths.

## Not included in this source tree

Conversation transcripts, personal profile/context, tool-response IDs,
authentication material, private connector data, raw terminal logs, system
metadata beyond a minimal numerical environment record, third-party papers or
font files. Bulk synthetic input arrays and generated row-level runs are also
excluded from Git; the supported runner deterministically generates them.

Legacy code and weights for Stage 1–48 are **not yet a supported portable export**.
Their research conclusions are documented as historical reports, not silently
claimed to be independently verified here. Source integrity checks are not the
same as rerunning an experiment. Earlier large archives are kept out of this
initial publication rather than dumped with broken container-specific paths.

## Release checklist

1. Identify source files and the right to distribute them; preserve third-party
   notices and do not relicense external material by accident.
2. Inspect executable code, exclude secrets and personal records, and remove
   runtime-specific paths and broken sandbox links from published documents.
3. Distinguish reported history, reproduced current results and proposed work.
4. Run numerical tests and the full supported protocol; compare saved results.
5. Preserve counterexamples, corrections, uncertainty and compute limitations.
6. Record file hashes and review the public diff before merging.

The automated scan detects selected credential formats and forbidden local
artifacts; it is not a guarantee that all sensitive content is detectable.
No confidential material is needed to reproduce the supported experiments.

## Attribution and scientific status

The materials were created with AI assistance and are maintained by Unjuno.
Stage numbering is internal experimental chronology. This repository is not a
peer-reviewed publication, not a novelty certificate, and not a claim of
patent clearance. No DOI or archival release identifier is fabricated.
