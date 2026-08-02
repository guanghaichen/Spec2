# P0 evidence pack: libero_spatial

This directory is generated from immutable evaluation summaries. Raw inputs are
stored as deterministic gzip files with SHA-256 hashes in `manifest.json`.

## Current evidence

- Profiler actions: 315
- Temporal transitions: 674
- VTPF causally comparable logit positions: 824
- VTPF top-1 mismatches: 9
- Fused-verifier accepted tokens that differ from serial AR: 0 / 555

## Interpretation boundary

Figure 2 diagnoses the model-call cost structure. Figure 3 establishes temporal
action persistence after outcome/phase stratification but does not establish
closed-loop recoverability. Figure 4 compares a fused verifier with an
independent token-by-token AR run only where both share the same causal prefix.
It explicitly reports finite-precision top-1 and accepted-token divergences;
positions after the first candidate mismatch are excluded by construction.
Counterfactual state-fork evidence remains a separate P0 experiment.
