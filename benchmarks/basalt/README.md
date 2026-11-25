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
