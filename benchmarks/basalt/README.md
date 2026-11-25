# Basalt 0.5.7 experiment on RTX 5090

Measured 2026-11-25 on one RTX 5090, using the same OpenUncensored checkpoint as the SGLang deployment. This is an experimental comparison, not a replacement of the repository's SGLang setup or a comprehensive quality evaluation. Basalt uses the model's DSpark, while SGLang uses Inco AI's DFlash2. Basalt's converter rounds the checkpoint's attention and other dense projections to Q8_0 and uses FP8 KV; SGLang uses BF16 KV. Consequently these are different numerical implementations, and output differences are expected. Native NVFP4/MXFP8 expert codes are preserved by the converter. The Basalt pack contains vision and the built-in draft. Both tests used temperature 0 and two configured slots with Vision enabled.

Basalt binary: official `v0.5.7`, source tag `v0.5.7`, commit `400b411`. The installed SGLang image and DFlash2 revision are specified in the main README. The Basalt pack was converted from the local uncensored safetensors using the Basalt v0.5.7 converter with 4 workers. The ephemeral pack is not included in this repository. Basalt was served with `--parallel 2 --slot-full-context --max-context 131072 --kv fp8 --vision dynamic`; each Basalt slot is configured for 131,072 tokens, whereas SGLang's 139,264 KV tokens are shared across both requests. This difference matters under large simultaneous workloads.

## Reproducing the Basalt setup

This separately converts the chosen checkpoint; it requires extra storage roughly the size of the model and working memory. Use a free GPU and stop the existing server only when interrupting active requests is acceptable. This is the tested command shape, with local paths chosen by the operator:

```bash
git clone --branch v0.5.7 --depth 1 https://github.com/jesdga95/basalt.git
python -m pip install -r basalt/requirements.txt
python basalt/tools/convert/convert.py --checkpoint /path/to/Inkling-2-Mini-Uncensored \
  --out /path/to/inkling-uncensored.basalt --workers 4
# Use the matching v0.5.7 official release binary, with CUDA 13 installed.
/path/to/basalt-v0.5.7-linux-x64/bin/basalt serve --model /path/to/inkling-uncensored.basalt --alias inkling \
  --host 127.0.0.1 --port 6022 --parallel 2 --slot-full-context \
  --vision dynamic --max-context 131072 --kv fp8
```

Our exact test ran the official `basalt-v0.5.7-linux-x64` archive in the SGLang container image for its CUDA/Python environment. Release archive SHA-256 `a390c775c41d54454ca21c1e547855bf6c251732533ca387e8b890e538090525`; converted file SHA-256 `927c5779e5d0181c7468a612225cdc33a77183467a4d8914c5c6aee25a4c35d5`. Reproducing this file hash requires the exact same input checkpoint and converter. The conversion deliberately changes dense projection precision; it is not a bit-identical checkpoint copy.

## Identical chat prompts, natural EOS

`compare_chat.py` sends an identical short English job-queue prompt for one request; for two requests it simultaneously sends that prompt and a Python LRU implementation prompt. Three batches, 1024 maximum output tokens, streaming, reasoning effort none. All requests in the comparison reached their 1024-token cap, so these are not completed-answer or quality measurements. The prompts and synthetic outputs are in the JSON files. Generated text differs between engines and the client sees reasoning as well as answer tokens. The first Basalt request includes a cold startup wait; later requests reuse a short prefix. SGLang also had a warmed server.

| Server | One request: median estimated decode | One request: output / total time | Two requests: combined output / batch wall time |
| --- | ---: | ---: | ---: |
| SGLang + DFlash2 | 285.2 tok/s | 278.1 tok/s | 469.0 tok/s |
| Basalt + DSpark | 387.9 tok/s | 385.6 tok/s | 493.5 tok/s |

Decode estimate is completion tokens divided by first visible chunk to end of stream; it is not kernel-only throughput and includes parser/transport overhead. Aggregate is all output divided by whole-batch wall time including prefill. The single-prompt estimate is about 36% higher for Basalt; the paired aggregate is about 5% higher. These results do not establish an average across tasks or equal quality. See [SGLang one](chat-sglang-c1.json), [SGLang two](chat-sglang-c2.json), [Basalt one](chat-basalt-c1.json), [Basalt two](chat-basalt-c2.json).

Reproduce the chat test against a compatible local endpoint, when idle:

```bash
python benchmarks/basalt/compare_chat.py --url http://127.0.0.1:6022 --concurrency 1 --output chat-c1.json
python benchmarks/basalt/compare_chat.py --url http://127.0.0.1:6022 --concurrency 2 --output chat-c2.json
```

## Context and function checks

- With 130,214 input tokens including an image, Basalt returned the correct synthetic embedded code, red square and blue circle. Total response time was 16.38 s; its server reported prompt reading 13.1 s and 231 output tokens. This is one retrieval test, not a long-context benchmark suite.
- Two independent synthetic 60,038-token inputs submitted together returned their respective keys in 8.64 s. Both completed naturally, with 145 and 136 output tokens. These results cannot be compared by wall time alone to earlier tests that forced 512 tokens.
- A small image identification and a `get_weather` tool call passed. See [long validation](basalt-long-validation.json) and [short smoke](basalt-smoke.json).
- [Context sweep, one request](basalt-context-c1.json) and [two](basalt-context-c2.json): randomized repetitive filler, 1k/16k/30k/60k, two batches per point, up to 512 output tokens. Unlike the original SGLang sweep, EOS is permitted, so completion lengths vary and the tests are not direct wall-time comparisons. A high rate on repetitive text does not imply the same rate on real agent work.

The Basalt engine ran in a temporary container, and SGLang was restored afterwards. These tests do not establish broader coding quality, long-run stability, Hermes interoperability, or lossless equivalence with the original BF16 projections.

## Completed task sample

On 2026-11-25, six short, public prompts asked for a Python TTL cache, arithmetic, incident extraction, SQL, webhook design and JSON. Each task ran **once** per engine at temperature 0 with a 3,072-token cap and natural EOS; then the same six tasks ran again in three pairs. See [workload script](agent-workload.py) and raw results: [SGLang single](agent-sglang-c1.json), [Basalt single](agent-basalt-c1.json), [SGLang pairs](agent-sglang-c2.json), [Basalt pairs](agent-basalt-c2.json). Answers and reasoning are included for inspection.

All 24 requests ended naturally and passed their basic checks (keywords, a number, or JSON parsing); the checks do **not** establish code correctness or comprehensive answer quality. Outputs and their token counts differ. First-to-end estimated decode rates for three longer single requests:

| Task | SGLang + DFlash2 | Basalt + DSpark | Output tokens, SGLang / Basalt |
| --- | ---: | ---: | ---: |
| Python TTL cache | 418 tok/s | 459 tok/s | 1,105 / 922 |
| PostgreSQL query | 380 tok/s | 456 tok/s | 911 / 828 |
| Webhook design | 289 tok/s | 402 tok/s | 1,590 / 1,656 |

Short answers are sensitive to first-chunk and stream-end timing and showed mixed results. Summed wall time of the three sequentially submitted pairs was 11.93 s for SGLang and 8.59 s for Basalt, but they generated 4,254 and 3,734 tokens respectively, so this is not equal-work throughput. The Basalt single-run first request included approximately 2.4 s cold-start wait. No confidence intervals or general speedup claims are justified by one pass per task.

```bash
python benchmarks/basalt/agent-workload.py --url http://127.0.0.1:6020 --concurrency 1 --output tasks-c1.json
python benchmarks/basalt/agent-workload.py --url http://127.0.0.1:6020 --concurrency 2 --output tasks-c2.json
```

## DSpark speculation cap experiment

On the same converted uncensored pack with `--parallel 2 --slot-full-context --max-context 131072 --kv fp8 --vision dynamic`, we varied the Basalt `--spec` cap in order 8, 4, 6, 8 on 2026-11-25. Each server ran the same `compare_chat.py` workload: three single requests and then three pairs, each request capped at 1,024 output tokens and temperature 0. The first single request after each restart included roughly 2.3 seconds of model-load wait, so the first-to-end estimate excludes that wait. Two later single requests had a cached 48-token prompt prefix. The greedy single-request output text matched across these four runs.

| Spec cap | Single request: median estimated decode | Two requests: median combined output / batch wall |
| ---: | ---: | ---: |
| 8, first control | 388.68 tok/s | 491.22 tok/s |
| 4 | 384.04 tok/s | 500.23 tok/s |
| 6 | 387.80 tok/s | 492.88 tok/s |
| 8, final control | 388.06 tok/s | 490.93 tok/s |

The two-request figure includes prompt processing and differs from the single-request decode estimate. The roughly 1.9% advantage of cap 4 on the pair is from only three batches of one prompt pair and may be scheduling or run variation; it is not enough to recommend changing the default. This is not a broad sweep of tasks or long contexts, and the responses reached their output cap. Raw data: [8 first, one](spec-8a-c1.json), [8 first, two](spec-8a-c2.json), [4, one](spec-4-c1.json), [4, two](spec-4-c2.json), [6, one](spec-6-c1.json), [6, two](spec-6-c2.json), [8 final, one](spec-8b-c1.json), [8 final, two](spec-8b-c2.json).

## FP8 versus FP16 KV cache

A separate FP16 KV run kept Vision, two full 131,072-token slots and the same DSpark cap 8. On the identical 1,024-token capped chat fixture, the median estimated decode rate was 377.47 tok/s with FP16 versus 388.06 tok/s in the preceding FP8 control; the median combined output divided by two-request batch wall time was 471.72 versus 490.93 tok/s. Each number uses three requests or three pairs, respectively, on one prompt or prompt pair; tests were not interleaved, and output/numerical behavior may differ. A six-task natural-EOS sample also completed; one test per task cannot establish model quality or a general performance ranking.

With FP16, one image plus 130,214 input tokens returned the right image details and embedded code; two concurrent 60,038-token synthetic requests returned their own keys. These are narrow checks. Raw FP16 results: [one chat](basalt-fp16-chat-c1.json), [two chats](basalt-fp16-chat-c2.json), [six tasks](basalt-fp16-agent-c1.json), [long checks](basalt-fp16-long-validation.json). FP8 remains the speed-oriented recommendation; FP16 offers a different KV precision tradeoff.

## Existing proxy compatibility smoke check

On 2026-11-25, the repository's Basalt Compose file was temporarily run on the same localhost port as the existing local-models authenticated proxy backend. Through that proxy, a synthetic image was identified correctly; `get_weather` was called with Warsaw, followed by a tool response and a streamed final answer; and two independent streamed requests completed concurrently with their respective identifiers. [Result summary](proxy-smoke-results.json). The proxy and credentials are specific to the test host and are not included here. This validates a few OpenAI-compatible flows, not a complete Windows Hermes session, production resilience, or behavior on all tool schemas. The original SGLang server was restored after the check.

## Basalt standard engine benchmark

Ran the official `basalt-bench` v0.5.7 Inkling fixtures on the converted **OpenUncensored** pack, one RTX 5090, FP8 KV, max context 131,072, DSpark default, seed 42, two repetitions on 2026-11-25. GPU power limit was 575 W. These are single-sequence engine measurements, not two simultaneous API requests; the benchmark does not exercise Vision. The separate server tests above cover Vision and two slots. Generation figures use the engine's decode time; PP figures use its prefill time and the uncached tokens it actually processed.

| Fixture | Output tokens per repetition | Run 1 | Run 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Conversation | 1,200 | 363.9 tok/s | 375.8 tok/s | 369.8 tok/s |
| Structured JSON | 8,192 | 703.8 tok/s | 709.2 tok/s | 706.5 tok/s |
| Prose | 968 | 321.1 tok/s | 321.2 tok/s | 321.2 tok/s |
| Prompt, 16k setting | 15,810 newly read | 19,210 tok/s | 19,127 tok/s | 19,168 tok/s |
| Prompt, 65k setting | 63,189 newly read | 14,574 tok/s | 14,541 tok/s | 14,558 tok/s |

The JSON fixture is highly predictable and reaches its 8,192-token cap; its throughput should not be presented as ordinary chat speed. Conversation and prose use temperature 0.6; JSON uses temperature 0. Prompt fixture lengths are the actual newly read token counts, which differ from requested setting names. These two runs are not a broad model quality benchmark. The JSON `lossy` engine-flags field does not account for conversion of dense projections to Q8_0 or FP8 KV: those precision changes are present as described above. Full per-run measurements and hardware metadata: [basalt-standard-2026-11-25.json](basalt-standard-2026-11-25.json).

To reproduce with the official Linux archive and a converted model, use the correct Inkling fixture directory:

```bash
/path/to/basalt-v0.5.7-linux-x64/bin/basalt-bench \
  --model /path/to/inkling.basalt \
  --fixtures /path/to/basalt-v0.5.7-linux-x64/data/bench/inkling2 \
  -t conv,struct,prose -p 16384,65536 -r 2 --seed 42 -o json \
  -- --max-context 131072 --kv fp8 > basalt-benchmark.json
```

## Official checkpoint comparison — 2026-11-25

The [Basalt pack converted from the official Thinking Machines checkpoint](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), `Inkling-2-Mini-NVFP4-MX8.basalt`, SHA-256 `28ec70723feaa037942a02f55b9d5e898276f3e62a3b1fa083745f932a041b99`, was run with the same Basalt v0.5.7, GPU, FP8 KV, DSpark and benchmark command as above. It is a separately distributed, converted pack, not the original safetensors. The uncensored pack was locally converted using the v0.5.7 converter. Two repetitions, seed 42, one engine sequence; no Vision or parallel requests in this engine measurement.

| Test | Official pack mean | Uncensored pack mean |
| --- | ---: | ---: |
| Conversation | 360.1 tok/s | 369.8 tok/s |
| Structured JSON | 711.3 tok/s | 706.5 tok/s |
| Prose | 318.9 tok/s | 321.2 tok/s |
| PP 16k setting | 19,196 tok/s | 19,168 tok/s |
| PP 65k setting | 14,583 tok/s | 14,558 tok/s |

These speed figures come from generated content that can differ between checkpoints. The JSON fixture is unusually predictable. The six short agent tasks passed limited checks both singly and in pairs. Through the existing authenticated proxy, an image, a tool round-trip with streamed answer, and two simultaneous streams passed. A synthetic image-and-text request with 130,214 input tokens returned its correct marker and image details; two separate 60,038-token requests returned their keys concurrently. These are narrow checks, not broad accuracy, Hermes-session or stability guarantees. Raw results: [standard](basalt-official-standard.json), [tasks one](basalt-official-agent-c1.json), [tasks two](basalt-official-agent-c2.json), [proxy](basalt-official-proxy-results.json), [long](basalt-official-long-validation.json).

## Production context sweep — 2026-11-25

Repeated the original context-size benchmark, adapted for Basalt's tokenizer and streaming API, on the persistent uncensored server after deployment: `--parallel 2 --slot-full-context --vision dynamic --max-context 131072 --kv fp8`. Three batches per point, randomized prefixes with repetitive filler, temperature 0, up to 512 generated tokens, EOS allowed. Requests often ended before the cap, so durations are not directly comparable with SGLang's forced-512 sweep. This test did not send an image; image support was checked separately.

| Input per request | 1 request: median estimated decode | 1 request: median wall | 2 requests: median batch wall | 2 requests: median combined output / batch wall |
| ---: | ---: | ---: | ---: | ---: |
| ~1k | 405 tok/s | 1.35 s | 1.85 s | 474 tok/s |
| ~16k | 342 tok/s | 1.75 s | 3.47 s | 199 tok/s |
| ~30k | 324 tok/s | 2.76 s | 5.20 s | 142 tok/s |
| ~60k | 283 tok/s | 5.00 s | 10.44 s | 98 tok/s |

The first-visible-to-end rate is an estimate, not kernel-only speed; batch throughput includes prefill, network and scheduling. Different response lengths and generated content affect each point. Results: [one request](basalt-production-context-c1.json), [two requests](basalt-production-context-c2.json). Reproduce against the active Basalt API:

```bash
python benchmarks/basalt/benchmark_basalt.py --url http://127.0.0.1:6020 --concurrency 1 --iterations 3 --results context-c1.json
python benchmarks/basalt/benchmark_basalt.py --url http://127.0.0.1:6020 --concurrency 2 --iterations 3 --results context-c2.json
```
