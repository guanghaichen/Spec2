# Final four-suite results, 2026-08-09

This directory freezes the compact manifest for the current Spec² method:

```text
Minimal DFlash + VTPF + PacedHarmonic (shared Target anchor)
```

Both H1 and H2 are gated against the latest Target image. The unit-depth visual
budget is `0.075`, giving cumulative depth bounds `0.075/0.15`. The protocol is
one RTX 4090, BF16, seed 7, 50 trials for each of 10 LIBERO tasks,
`TIMING_SCOPE=last_task`, and `SYNC_CUDA_TIMING=False`.

`paced_harmonic_runs.json` records the four canonical 500-episode runs,
including checkpoint paths, success rate, latency, AR-relative speedup,
speculative length, Target-prefill rate, H1/H2 counts, VTPF match rate, and the
SHA-256 digest of each source summary.

The Object run uses the actual epoch-60 checkpoint. Goal, Spatial, and
LIBERO-10 use epoch 100. Large per-episode logs remain in the server-side
`specvla-data/paper_results/` archive and are not duplicated in Git.
