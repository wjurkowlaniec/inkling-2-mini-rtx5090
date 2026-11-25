# Agent context and compaction

These are starting points, not measured guarantees of agent quality. Check the client version, prompt accounting and actual workload. Server context includes system text, tool definitions, images, reasoning and output.

## Basalt (recommended setup)

The provided Basalt configuration sets **two slots of 131,072 tokens each**, including output. More requests wait. FP8 KV, dynamic Vision and expert residency affect memory use; two full windows can require substantially more GPU and host memory than two short windows. We checked two independent ~60k requests and one image request with ~130k input, not every combination or sustained full-window load.

- For focused subagents, start around **8k–32k** input tokens and include only relevant files and state.
- For ordinary coding, use **32k–60k** as an initial working range; longer cold prompts increase latency.
- A **96k client context setting** is a reasonable starting experiment. If its compaction threshold is a percentage of that window, try **60–70%** (~59k–69k tokens). This is a heuristic, not a measured optimal value or a verified Hermes setting.
- Reserve space for output, tool schemas, system messages and images. A practical starting reserve is **8k–16k or more**, depending on the task and client accounting. Compress before approaching the limit, and retain decisions, paths, errors and next steps in the summary.

Use task completion time, answer quality, compression frequency, queue time and cache hits to tune the client. A larger window can reduce compression frequency while increasing cold prefill cost; it does not automatically increase token generation speed.

## SGLang alternative

SGLang's configuration uses a **shared 139,264-token full-attention KV pool**, rather than Basalt's two full configured slots. Two independent 96k histories cannot both fit in that pool; requests may wait or cache may be evicted. For two agents, begin around 40k–60k input each, subject to output and memory overhead. A synthetic three-request ~60k test against two SGLang slots completed with one queued; the last first token arrived at 17.72 s and the batch completed at 19.72 s. This single test is not a stability guarantee.

See the [Basalt benchmark report](../benchmarks/basalt/README.md), including the [context sweep](../benchmarks/basalt/README.md#production-context-sweep--2026-11-25), and [SGLang results](../benchmarks/sglang/README.md) for test methods and limitations.
