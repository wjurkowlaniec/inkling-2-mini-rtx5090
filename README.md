# Inkling 2 Mini on one RTX 5090

**370 tok/s conversation · 321 tok/s prose · 706 tok/s structured JSON** on one RTX 5090, using Basalt and the uncensored checkpoint. These are means of two single-sequence runs of Basalt's standard engine fixtures, measured **2026-11-25**. The 8,192-token JSON fixture is highly predictable; it is not a typical chat result. Vision and parallel requests were tested separately. [Raw results and method](benchmarks/basalt/README.md#basalt-standard-engine-benchmark).

The recommended configuration is **Basalt 0.5.7 + DSpark + FP8 KV**, with Vision, tool calling and **two slots configured for 128k tokens each**, including output. We tested both [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored). Basalt rounds dense projections to Q8_0; results can differ from the original checkpoint. These tests do not establish broad quality equivalence or a global speed record.

## Get started

1. Install Linux, the NVIDIA driver, NVIDIA Container Toolkit and Docker Compose; provide an RTX 5090 with 32 GB VRAM.
2. Follow the **[Basalt installation guide](basalt/README.md)** to download the verified release and official converted model, or convert a checkpoint you already have. Model files and runtime are not stored in Git.
3. From this repository's root, start the server and make a request:

```bash
docker compose -f basalt/docker-compose.yml up -d
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

The unauthenticated API listens on localhost. For LAN access, put an authenticated proxy in front of it. Only one inference server should use the GPU and port at a time. [Setup, checksums and configuration](basalt/README.md).

## Measured generation speed

All numbers below are from **2026-11-25**, one RTX 5090 and the uncensored checkpoint. They measure different workloads, so rows are not interchangeable. The model's reasoning tokens count toward output.

| Workload | Result | Measurement |
| --- | ---: | --- |
| Standard conversation | **370 tok/s** | Single sequence, engine decode; 1,200 tokens, 2 runs, temperature 0.6 |
| Standard prose | **321 tok/s** | Single sequence, engine decode; 968 tokens, 2 runs, temperature 0.6 |
| Repetitive structured JSON | **706 tok/s** | Single sequence, engine decode; 8,192-token cap, 2 runs, temperature 0 |
| Short agent chat | **388 vs 285 tok/s** | Basalt vs SGLang estimated decode, median of 3, same prompt but different generated outputs |
| Two short agent requests | **494 vs 469 tok/s** | Basalt vs SGLang combined output divided by whole batch wall time, median of 3 |

[Full Basalt experiments](benchmarks/basalt/README.md) · [SGLang experiments](benchmarks/sglang/README.md) · [Raw JSON and scripts](benchmarks/)

The standard JSON fixture is especially easy for speculative decoding to predict. In a separate 24-record extraction test at temperature 0, the same prompt took **0.79 s with `response_format: json_object` vs 1.67 s when JSON was requested only in text**. Both returned the expected records in 3/3 single-request trials. JSON mode generated 611 tokens with no visible reasoning, versus 1,220 tokens with reasoning for prompt-only mode: shorter output is a major reason for the time difference. In this sample, strict `json_schema` and `json_object` were similarly fast for one request. Encoding the same input as JSON did not improve speed and used more prompt tokens. Results are task-specific; JSON syntax validation does not prove factual correctness. [Structured-output experiment](benchmarks/basalt/STRUCTURED_OUTPUT.md).

## Context and concurrency

The deployed Basalt sweep used repetitive synthetic input, randomized prefixes, up to 512 output tokens and three batches per point. First-to-end decode is an estimate; pair wall time includes prompt processing and client overhead. Outputs often ended before the cap.

| Input per request | 1 request: estimated decode | 2 requests: batch wall |
| ---: | ---: | ---: |
| ~1k | 405 tok/s | 1.85 s |
| ~16k | 342 tok/s | 3.47 s |
| ~30k | 324 tok/s | 5.20 s |
| ~60k | 283 tok/s | 10.44 s |

A synthetic 130,214-token image-and-text retrieval and two concurrent ~60k retrievals passed, as did simple image, tool and streaming checks through a proxy. This is not a comprehensive quality or long-term stability test. More than two requests queue in the provided Basalt configuration. [Context data and tests](benchmarks/basalt/README.md#production-context-sweep--2026-11-25) · [Agent context and compaction guidance](docs/AGENT_CONTEXT.md).

## Other configurations tested

| Option | Finding |
| --- | --- |
| [Basalt FP16 KV](benchmarks/basalt/README.md#fp8-versus-fp16-kv-cache) | 377 vs 388 tok/s FP8 on one short chat fixture; both passed narrow long-context checks. |
| [Basalt DSpark caps 4, 6 and 8](benchmarks/basalt/README.md#dspark-speculation-cap-experiment) | No reliable gain over default 8 in the short single and paired tests. |
| [SGLang + DFlash2, BF16 KV](sglang/README.md) | Alternative with original dense weights; its 139,264-token KV pool is shared across two slots. The matched short chat measured 285 vs Basalt's 388 estimated decode tok/s. |
| [SGLang DFlash2 widths 6, 8 and 12](benchmarks/sglang/README.md#dflash2-verification-width-experiment-2026-11-25) | Width 8 led on tested short prose and code; 12 led on a repetitive synthetic case. |
| [SGLang three-slot experiment](benchmarks/sglang/README.md#dflash2-context-sweep-2026-11-25) | Short and medium synthetic requests ran; shared KV and Vision stability need further testing. |

The [Basalt report](benchmarks/basalt/README.md) includes official and uncensored checkpoint results, exact settings, test limitations and raw data. See [SGLang benchmark details](benchmarks/sglang/README.md) for its historical measurements. Speeds depend on workload, sampling, context, cache, batching, precision and hardware.

## Repository layout

```text
basalt/                 recommended Docker Compose and setup guide
sglang/                 alternative Dockerfile, Compose, patch and guide
benchmarks/basalt/      Basalt and cross-engine tests, results and reports
benchmarks/sglang/      SGLang tests, results and reports
docs/                   agent context and compaction guidance
```

Model weights, runtime files and local caches are ignored by Git. Basalt is from [jesdga95/basalt](https://github.com/jesdga95/basalt); SGLang and its model-specific DFlash2 patch are credited in the [alternative guide](sglang/README.md).
