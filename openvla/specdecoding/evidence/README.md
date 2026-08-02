# Paper evidence tooling

This directory turns diagnostic LIBERO runs into auditable paper evidence. It
is deliberately separate from the online decoding implementation.

`build_p0_evidence.py` consumes three summaries produced by the P0 launcher:

1. synchronized stage profiling for a plain one-layer DFlash action;
2. the paper-compatible wrapped-AR baseline with compact action, state, and
   visual traces;
3. strict VTPF fused-prefill versus serial-AR parity.

It writes raw-input hashes, CSV tables, vector PDF figures, 300-dpi PNG review
copies, and explicit claim boundaries. Generated plots use the official ICLR
2026 `5.5 in` text width, embedded TrueType fonts, a colorblind-safe palette,
and labels readable at final paper size.

Run the suite-aware launcher rather than assembling these inputs by hand:

```bash
SPEC_CKPT=/absolute/path/to/epoch_100_step_xxxxxx \
P0_TRIALS=6 \
bash openvla/specdecoding/decode-scripts/run_dflash_p0_evidence.sh goal
```

These short runs are mechanism diagnostics. They do not replace the 50-trial
main-table protocol or the still-separate same-state counterfactual recovery
experiment.

The counterfactual experiment is a separate target-only run:

```bash
P0_REFERENCE_EPISODES=1 P0_FORKS_PER_EPISODE=2 \
bash openvla/specdecoding/decode-scripts/run_dflash_p0_counterfactual.sh goal
```

It reconstructs an identical simulator and controller history by deterministic
replay, checks the fork-state difference against the reference, commits bounded
historical or control actions, resumes the frozen target policy, and flushes
each branch to JSONL.  A target-continuation positive control must pass before
the result is considered valid.  The paired inverse-age branch uses the same
historical target action and hold depth as the unscaled branch, changing only
the continuous control scale at ages two and three.
