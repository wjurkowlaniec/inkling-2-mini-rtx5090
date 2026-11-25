# Agent context and compression guidance

These are starting points for the RTX 5090 setup in this repository, not model-quality guarantees. They depend on your agent's actual system prompt, tools, output budget, compaction implementation and number of concurrent sessions.

## Context and memory limits

- The server accepts up to **131,072 tokens total per request**, including its chat template, tool schemas, images, prompt and output. `max_tokens` and reasoning also need room.
- The tested deployment has **139,264 full-attention KV slots shared** between all active and cached prefixes, not that amount per request. Independent prompts compete for the pool. Shared prefixes may reduce use, but do not assume a cache hit.
- The production recipe has two active slots. Extra requests wait in the scheduler queue; cache entries may be evicted, and under decode memory pressure SGLang can retract/recompute a request. A request exceeding its context limit may fail rather than wait. Measure actual use, as cache overhead, sliding-window state and reserved output affect admission.
- In one test, three concurrent requests of about 60,000 prompt tokens each plus 512 output tokens ran against two slots. All completed; the latest first-token arrival was 17.72 s and the batch completed in 19.72 s. The scheduler logged two active requests and one queued. One synthetic trial is not a stability or latency guarantee.

## Suggested starting budgets

| Workload | Suggested starting point | Reason |
| --- | --- | --- |
| Short tool calls or subagents | Aim for 8k–32k input tokens | Keeps prefills short and leaves memory for other requests. Supply only relevant files, task and state. |
| Normal coding agent | Aim for 32k–60k input tokens | Balances detail against the cost of rereading and sharing the GPU; use retrieval and summaries for older material. |
| Long single-agent task | Configure the client for around **98,304 tokens (96k)**, with a separate output budget | A reasonable trial while leaving headroom below the server's 131,072 limit. Measure the actual serialized prompt; the client setting is not necessarily its compression threshold. |
| Two independent long agents | Prefer around 40k–60k input tokens each, subject to output and cache overhead | Two 96k histories exceed the shared 139k KV pool. Even two near-64k requests with large output budgets may need to wait. |

**Compression:** If the agent uses a threshold equal to a fraction of its configured window, start around **60–70% of a 96k client window** (about 59k–69k tokens), then adjust based on actual compaction frequency, queueing and quality. For example, a 50% threshold on 96k triggers around 49k; this depends on the agent implementation and is not verified for every Hermes release. Avoid compressing at every turn: it costs time and may discard details. Keep a structured checkpoint with goals, decisions, paths, failures and outstanding work. Reserve at least 8k output/system/tool headroom; for complex reasoning use 16k or more as appropriate. Check how your client accounts for the output budget before assuming these numbers.

Do not confuse token throughput with task completion time. DFlash2 can be fast on repetitive text, while long cold prompts still need prefill and tool calls may dominate. Track task completion time, compaction time/count, queue time, cache hits, errors, and whether the agent retained necessary facts. Compare similar real tasks before choosing a threshold.

## What the measurements show

On 2026-11-25, a synthetic context sweep used temperature 0, fresh random prefixes, repeated filler and fixed 512-token completions. Each 1- and 2-request point has 3 batches. Two-slot configuration; each row below reports **batch wall time**, including prefill, for all requests in the batch. TTFT and output rates are in the raw results.

| Prompt per request | One request | Two concurrent requests |
| ---: | ---: | ---: |
| ~1k | 1.42 s | 1.91 s |
| ~16k | 2.59 s | 4.48 s |
| ~30k | 4.11 s | 7.13 s |
| ~60k | 6.71 s | 12.79 s |

A separate temporary three-slot configuration (same 128k setting and vision enabled) completed three ~1k requests in a median 2.19 s and three ~30k requests in 9.38 s, over two batches per size. A third request sent to the **two-slot** configuration waited. Three slots are an experiment, not the default: the shared pool is unchanged, and three long independent contexts will queue or compete for memory. The temporary three-slot run did not constitute a broad vision or agent-quality validation.

Synthetic benchmark responses were forced to 512 output tokens and often ended at that limit, sometimes entirely in reasoning; these figures do not establish answer quality. Full per-request data and method are in [`benchmarks/results/`](benchmarks/results/).
