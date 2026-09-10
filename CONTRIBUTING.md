# Contributing

Keep empirical findings, mathematical claims and hypotheses separate. A new
result should identify the task, available observations, latent truth, update
schedule, reference solver, seeds, compute accounting and stopping rule.

For experiment changes, commit the protocol before evaluating held-out inputs.
Local hashes are not an external preregistration. Preserve failed hypotheses;
never rename a post-hoc subgroup into the original primary population. Count
graph/training seeds, not repeated trajectories, as the independent blocks when
that is the design. Fit calibration on separate data and keep the test labels
out of inference and policy selection.

Run `python -m unittest discover -s tests -v` and the smoke command in README.
Changes to the current study should also pass a full run followed by
`python tools/compare_reference.py runs/full`, or explain an intentional change
in results in a new result directory. Do not overwrite historical reference
numbers to make a test pass.

Do not contribute conversations, private datasets, API tokens, personal contact
information, unsafe serialized objects, unlicensed third-party code, or bulk
arrays through a normal source PR. Use the publication checklist in
[docs/PUBLICATION.md](docs/PUBLICATION.md). Reports should include uncertainties,
negative controls, provenance and the exact code commit.

This is a research prototype. Filing an issue is not evidence that a claim has
been reproduced. Report a minimal synthetic reproduction and state the outcome.
