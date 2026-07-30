# Minimal Draft e100 Goal evidence

This directory freezes the four 500-episode LIBERO Goal evaluations completed on
2026-07-29/30 with seed 7 and the same Minimal Draft checkpoint:

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/
minimal-100epoch/epoch_100_step_044800
```

The checkpoint weights are intentionally not committed. `checkpoint_dflash_config.json`
records the model configuration; the evaluation text, timing JSON and summary JSON are
preserved verbatim. The paper-wrapped AR denominator is `0.182718 s`.

| Route | SR | Last-task mean | Speedup | Length |
| --- | ---: | ---: | ---: | ---: |
| Linear strict | 0.792 | 0.160411 s | 1.139x | 2.172 |
| VTPF strict | 0.776 | 0.141044 s | 1.295x | 2.432 |
| VTPF-TD-Fast | 0.754 | 0.072119 s | 2.534x | 3.573 |
| VTPF-TD-Adaptive | 0.746 | 0.070320 s | 2.599x | 3.544 |

These results show that e100 preserves the practical online performance of the more
complex Golden e200 route. They do not establish that fewer than 100 epochs are enough.
