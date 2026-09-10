# Claims that are not established

This repository does not establish that X-shaped layouts are inherently better,
that more iteration always increases accuracy, or that explicit workspace use
is a new inference principle. Classical Bayesian inference, iterative linear
solvers, residual scheduling and interval methods are strong related baselines.

It does not establish benefits on real LLMs, unknown graphs, natural-language
reasoning or production retrieval. Explicit workspace variables here are not
private chain-of-thought transcripts. It does not demonstrate model compression,
lossless semantic compaction, storage savings or recovery of discarded facts.

For the current Gaussian study, the generating covariance, graph and noise law
are known. Query truth is withheld, but the structure is oracle knowledge in the
model specification. Graph-sign corruption changes the inference assumption,
not the true data. A fixed point can therefore be stable and wrong.

The gate calibrated on 50% corrupted relations is not an online correctness
detector. It damages intact relations. The implemented calibration objective is
mean **normalized** squared error (for the MSE gate), not unnormalized raw MSE.
Some early narrative descriptions used 'raw MSE' loosely; the code and this
public statement control that ambiguity.

Historical timing improvements are confined to their recorded hardware,
locality, query and tolerance settings. Different stages changed processors,
baselines, tasks and endpoints. Do not combine percentages into a universal
speedup curve. Source CSVs can use fractions, percentage points or percent;
read their headers and the accompanying historical note.

All conclusions are limited by synthetic task design and repeated research
exploration. Local protocol hashes are not external preregistration, theoretical
convergence is not truth, and model fidelity is not task correctness.
