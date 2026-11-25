# Inkling 2 Mini on RTX 5090

A single-GPU SGLang configuration for the official [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): 128k context, image input, tool calling, DFlash2 speculative decoding, and up to two concurrent requests. It also supports [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored). The checkpoint is downloaded separately and is not included in this repository.

## Hardware and requirements

- NVIDIA RTX 5090 with 32 GB VRAM, Linux, NVIDIA driver and NVIDIA Container Toolkit configured for Docker.
- Docker with Compose and GPU access; roughly 20 GB for the model plus space for the image and caches. The deployed container image is large (about 56 GB before layer sharing).
- A free GPU: the tested server reserves about 31 GB of VRAM. Stop other GPU inference servers before launching it.
- Python with a recent `huggingface_hub` CLI for downloading the checkpoint and the ~1 GB DFlash2 draft.

This is a setup for an RTX 5090, not a guarantee that the same flags work on other GPUs. The SGLang base image is pinned to the digest tested on 2026-11-21; the Dockerfile applies the [model-specific DFlash2 compatibility patch](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/blob/4f9088ea66c21f2b99ed657d1941c306b03e84d2/sglang_inkling_dflash2.py) from the pinned draft revision. The model uses NVFP4 and MXFP8 experts; the KV cache is BF16.

## Quick start

```bash
python -m pip install --user -U huggingface_hub
mkdir -p models/Inkling-2-Mini cache/sglang
hf download thinkingmachines/Inkling-2-Mini \
  --local-dir models/Inkling-2-Mini
hf download incoai/Inkling-2-Mini-DFlash2 \
  --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2 \
  --exclude "assets/*" --local-dir models/Inkling-2-Mini-DFlash2

docker compose up -d --build
docker compose logs -f
```

The first start may take several minutes while GPU kernels and CUDA graphs are compiled. To reuse an existing checkpoint, set `MODEL_DIR` to its absolute path before starting. `DRAFT_DIR` selects the DFlash2 draft directory and `CACHE_DIR` selects the persistent SGLang cache directory. Neither the checkpoint nor the cache is tracked by Git.

To use the uncensored checkpoint instead, download it and select its directory before starting:

```bash
hf download OpenUncensored/Inkling-2-Mini-Uncensored \
  --local-dir models/Inkling-2-Mini-Uncensored
export MODEL_DIR="$(pwd)/models/Inkling-2-Mini-Uncensored"
docker compose up -d
```

Check readiness and ask a question:

```bash
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512}'
```

The API binds to **127.0.0.1** and has no authentication. It is intentionally only reachable from the host; for LAN access put an authenticated proxy in front of it. The proxy and its configuration are not part of this repository. `docker compose stop` stops the service; `docker compose up -d` starts it again. The container uses `restart: unless-stopped`.

## Measured results

Measurements on one RTX 5090 on 2026-11-21, with the pinned image and the **uncensored checkpoint**. Results depend on prompt, context, sampling, batching and hardware. Output token counts include reasoning tokens when applicable.

| Measurement | Result | Method |
| --- | --- | --- |
| Earlier DSpark baseline | ~232 output tok/s | 1,024-token fixed-output test, three warm runs, 4.42 s each, concurrency 1 |
| No speculative decoding | ~162 output tok/s | Same 1,024-token test, warm runs 6.30–6.34 s, concurrency 1 |
| Two short requests | 2.93 s parallel vs 5.83 s sequential | Two identical 512-token outputs; synthetic, not an agent workload |
| Long-context retrieval | 130,759 input tokens and 208 output tokens | Synthetic repeated filler and one embedded fact, correct result |
| Long-context + vision | 130,225 input tokens including image tokens | Synthetic image and text fact recognized correctly |

In one later real-use log sample, prompt-processing batches of at least 512 tokens had a median of ~11,558 tok/s, and single-request generation samples a median of ~213 tok/s. These are sampled server throughput values, not complete-request benchmarks or guaranteed speeds. In another earlier observation, two active requests yielded ~351 output tok/s combined at the median; workloads differed. Full numerical test artifacts are in [`benchmarks/`](benchmarks/).

### DFlash2 context sweep (2026-11-25)

Measured on the uncensored checkpoint, one RTX 5090, vision enabled, 128k limit, two server slots. The client submits fresh random prefixes with repetitive synthetic filler, temperature 0 and 512 forced output tokens. The first token time and batch wall time include prefill, network and queueing; output includes reasoning. Three batches per point. The server remained live during the runs; results are not quality evaluations.

| Prompt per request | 1 request: batch time | 2 requests: batch time | 1 request: estimated TG | 2 requests: aggregate output / batch time |
| ---: | ---: | ---: | ---: | ---: |
| ~1k | 1.42 s | 1.91 s | 385 tok/s | 537 tok/s |
| ~16k | 2.59 s | 4.48 s | 356 tok/s | 229 tok/s |
| ~30k | 4.11 s | 7.13 s | 280 tok/s | 144 tok/s |
| ~60k | 6.71 s | 12.79 s | 336 tok/s | 80 tok/s |

Estimated TG is output tokens divided by first-visible-chunk-to-end-of-stream time. Aggregate throughput is output tokens divided by wall time of the whole batch **including prefill**; these are different measurements. High TG on repetitive synthetic prompts should not be generalized to agent tasks. Full per-request measurements: [one request](benchmarks/results/dflash2-rtx5090-c1-2026-11-25.json) and [two requests](benchmarks/results/dflash2-rtx5090-c2-2026-11-25.json).

In an isolated three-slot experiment, three ~1k requests completed in 2.19 s and three ~30k requests in 9.38 s (medians of two batches). Production remains at two slots. See [agent guidance](AGENT_CONTEXT.md) for limits and the [experimental results](benchmarks/results/dflash2-rtx5090-c3-experimental-2026-11-25.json).

Reproduce with one or two simultaneous requests (creates load on the running server):

```bash
python benchmarks/benchmark_generation.py --concurrency 1 --iterations 3 --output-tokens 512
python benchmarks/benchmark_generation.py --concurrency 2 --iterations 3 --output-tokens 512
```

Reproduce the fixed-output and synthetic retrieval checks against a running server:

```bash
python benchmarks/measure_speed.py dspark
python benchmarks/test_long_context.py 30000 256
```

The speed script forces 1,024 output tokens and writes a local, ignored results file. The context script uses repetitive synthetic filler and writes its result to a separate ignored file. These tests create inference load; run them when the server is idle.

See [Agent context and compression recommendations](AGENT_CONTEXT.md) for suggested budgets and their limitations.

## Limits and tradeoffs

- Context is 131,072 tokens **including output**. Model reasoning also consumes output tokens; set a sufficient `max_tokens` budget.
- The tested setup allocated approximately 139,264 full-attention KV slots **shared across requests** and 6,912 sliding-window slots. Two independent 96k contexts cannot both fit in that pool at once; requests may wait or cached entries may be evicted. Shared prefixes can reduce memory use.
- The long-context tests are narrow retrieval checks, not comprehensive long-context accuracy or sustained-load evaluations. Test on your own workloads.
- BF16 KV is intentional: the image's FA4 FP8 path rejected SM120 in our MXFP8 KV attempt. The previous DSpark configuration rejected a shorter draft block; the current DFlash2 draft uses its separate 2,048-token ring. These observations apply to the pinned build, not necessarily newer releases.
- Vision and tools are enabled in the server. Multimodal long-context and a simple read/repair/test tool-call sequence were checked with the uncensored checkpoint.

## Credits

Model: [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini); optional variant: [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored). Server and day-zero image: [SGLang](https://github.com/sgl-project/sglang). Model licensing is described in its Hugging Face repository.
