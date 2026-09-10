# Security and data handling

The supported experiments generate synthetic numerical data and require no
credentials or external services. Do not add an API key to run them.

Do not publish secrets or private data in issues, logs, pull requests or test
fixtures. If GitHub private vulnerability reporting is enabled for this
repository, use that channel for a sensitive report. Otherwise open a minimal
issue asking the maintainer for a private channel, without disclosing the
sensitive payload. No private reporting service is asserted to be configured.

Never load untrusted pickle or PyTorch checkpoint files. No such files are part
of this initial public source tree. Numerical NPZ files, when generated locally,
should be read with object deserialization disabled.

Secret-pattern scans are heuristic, not a privacy or security proof. Repository
code has not undergone an independent security assessment. Research tests are
not production safety certification.
