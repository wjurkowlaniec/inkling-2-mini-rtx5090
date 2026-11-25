# SGLang benchmark results

All results in this document describe SGLang and the uncensored checkpoint; see the root README for Basalt. Run commands from the repository root.

## Measured results

Measurements on one RTX 5090 on 2026-11-21, with the pinned image and the **uncensored checkpoint**. Results depend on prompt, context, sampling, batching and hardware. Output token counts include reasoning tokens when applicable.

| Measurement | Result | Method |
| --- | --- | --- |
| Earlier DSpark baseline | ~232 output tok/s | 1,024-token fixed-output test, three warm runs, 4.42 s each, concurrency 1 |
| No speculative decoding | ~162 output tok/s | Same 1,024-token test, warm runs 6.30–6.34 s, concurrency 1 |
| Two short requests | 2.93 s parallel vs 5.83 s sequential | Two identical 512-token outputs; synthetic, not an agent workload |
| Long-context retrieval | 130,759 input tokens and 208 output tokens | Synthetic repeated filler and one embedded fact, correct result |
| Long-context + vision | 130,225 input tokens including image tokens | Synthetic image and text fact recognized correctly |

In one later real-use log sample, prompt-processing batches of at least 512 tokens had a median of ~11,558 tok/s, and single-request generation samples a median of ~213 tok/s. These are sampled server throughput values, not complete-request benchmarks or guaranteed speeds. In another earlier observation, two active requests yielded ~351 output tok/s combined at the median; workloads differed. Full numerical test artifacts are in [`benchmarks/sglang/`](.).

### DFlash2 context sweep (2026-11-25)

Measured on the uncensored checkpoint, one RTX 5090, vision enabled, 128k limit, two server slots. The client submits fresh random prefixes with repetitive synthetic filler, temperature 0 and 512 forced output tokens. The first token time and batch wall time include prefill, network and queueing; output includes reasoning. Three batches per point. The server remained live during the runs; results are not quality evaluations.

| Prompt per request | 1 request: batch time | 2 requests: batch time | 1 request: estimated TG | 2 requests: aggregate output / batch time |
| ---: | ---: | ---: | ---: | ---: |
| ~1k | 1.42 s | 1.91 s | 385 tok/s | 537 tok/s |
| ~16k | 2.59 s | 4.48 s | 356 tok/s | 229 tok/s |
| ~30k | 4.11 s | 7.13 s | 280 tok/s | 144 tok/s |
| ~60k | 6.71 s | 12.79 s | 336 tok/s | 80 tok/s |

Estimated TG is output tokens divided by first-visible-chunk-to-end-of-stream time. Aggregate throughput is output tokens divided by wall time of the whole batch **including prefill**; these are different measurements. High TG on repetitive synthetic prompts should not be generalized to agent tasks. Full per-request measurements: [one request](results/dflash2-rtx5090-c1-2026-11-25.json) and [two requests](results/dflash2-rtx5090-c2-2026-11-25.json).

In an isolated three-slot experiment, three ~1k requests completed in 2.19 s and three ~30k requests in 9.38 s (medians of two batches). The provided SGLang configuration retains two slots. See [agent guidance](../../docs/AGENT_CONTEXT.md) for limits and the [experimental results](results/dflash2-rtx5090-c3-experimental-2026-11-25.json).

Reproduce with one or two simultaneous requests (creates load on the running server):

```bash
python benchmarks/sglang/benchmark_generation.py --concurrency 1 --iterations 3 --output-tokens 512
python benchmarks/sglang/benchmark_generation.py --concurrency 2 --iterations 3 --output-tokens 512
```

Reproduce the fixed-output and synthetic retrieval checks against a running server:

```bash
python benchmarks/sglang/measure_speed.py dspark
python benchmarks/sglang/test_long_context.py 30000 256
```

The speed script forces 1,024 output tokens and writes a local, ignored results file. The context script uses repetitive synthetic filler and writes its result to a separate ignored file. These tests create inference load; run them when the server is idle.

See [Agent context and compression recommendations](../../docs/AGENT_CONTEXT.md) for suggested budgets and their limitations.

### DFlash2 verification width experiment (2026-11-25)

On the same RTX 5090 and uncensored checkpoint, the pinned DFlash2 build was tested with verification widths 6, 8 and 12. Vision, two server slots and the 131,072-token limit remained configured. Each short-task figure below is the median of three greedy `/generate` requests, each forced to emit 512 tokens. Rates divide output tokens by complete request time, including short-prompt prefill and HTTP overhead; they are not pure GPU decode rates. The two-request figure is one simultaneous batch. The 16k case is highly repetitive synthetic text, with a warm cached prefix for the median, and is not representative agent performance.

| Width | Prose | Code | Two short requests combined | Repetitive 16k, warm |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 232 tok/s | 267 tok/s | 433 tok/s | 442 tok/s |
| 8 (default) | 262 tok/s | 292 tok/s | 453 tok/s | 640 tok/s |
| 12 | 240 tok/s | 265 tok/s | 390 tok/s | 706 tok/s |

Width 8 remains the default: width 12 improved this artificial repetitive case but slowed the realistic short examples. These few prompts do not establish an overall optimum. The greedy serial output hashes match between widths on these three fixtures; concurrent hashes varied, so this is not a comprehensive correctness comparison. Raw results: [6](results/dflash2-width6-2026-11-25.json), [8](results/dflash2-width8-2026-11-25.json), [12](results/dflash2-width12-2026-11-25.json), plus context sweeps for [6](results/dflash2-width6-context-2026-11-25.json) and [12](results/dflash2-width12-context-2026-11-25.json). Context-sweep prompts have randomized prefixes, so their output and decode rates are not matched across runs. Reproduce short tasks with `python benchmarks/sglang/measure_dflash_width.py http://127.0.0.1:6020 width8` against each separately configured server. Changing width requires a server restart and can interrupt requests.

A separate [sampling check](results/dflash2-sampling-2026-11-25.json) on three short raw prompts produced median rates of 261/380/252 tok/s at temperature 0 and 232/249/244 tok/s at temperature 0.7 (prose/code/analysis, respectively). Three seeds per task and temperature; the 512-token cap was reached in these runs. Output tokens divided by complete request time, so the figures include prefill and are not a quality benchmark. Run `python benchmarks/sglang/measure_sampling.py` to reproduce against port 6020.
