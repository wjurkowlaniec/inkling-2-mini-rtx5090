# Inkling 2 Mini Uncensored on RTX 5090

A tested, single-GPU SGLang configuration for [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored): 128k context, image input, tool calling, native DSpark speculative decoding, and up to two concurrent requests. The model is downloaded separately and is not included in this repository.

## Hardware and requirements

- NVIDIA RTX 5090 with 32 GB VRAM, Linux, NVIDIA driver and NVIDIA Container Toolkit configured for Docker.
- Docker with Compose and GPU access; roughly 20 GB for the model plus space for the image and caches. The deployed container image is large (about 56 GB before layer sharing).
- A free GPU: the tested server reserves about 31 GB of VRAM. Stop other GPU inference servers before launching it.
- Python with a recent `huggingface_hub` CLI for downloading the checkpoint.

This is a setup for an RTX 5090, not a guarantee that the same flags work on other GPUs. The image is pinned to the digest tested on 2026-11-21. The model uses NVFP4 and MXFP8 experts; the KV cache is BF16.

## Quick start

```bash
python -m pip install --user -U huggingface_hub
mkdir -p models/Inkling-2-Mini-Uncensored cache/sglang
hf download OpenUncensored/Inkling-2-Mini-Uncensored \
  --local-dir models/Inkling-2-Mini-Uncensored

docker compose up -d
docker compose logs -f
```

The first start may take several minutes while GPU kernels and CUDA graphs are compiled. To reuse an existing checkpoint, set `MODEL_DIR` to its absolute path before starting. `CACHE_DIR` optionally selects the persistent SGLang cache directory. Neither the checkpoint nor the cache is tracked by Git.

Check readiness and ask a question:

```bash
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512}'
```

The API binds to **127.0.0.1** and has no authentication. It is intentionally only reachable from the host; for LAN access put an authenticated proxy in front of it. The proxy and its configuration are not part of this repository. `docker compose stop` stops the service; `docker compose up -d` starts it again. The container uses `restart: unless-stopped`.

## Measured results

Measurements on one RTX 5090 on 2026-11-21, with the pinned image and this model. Results depend on prompt, context, sampling, batching and hardware. Output token counts include reasoning tokens when applicable.

| Measurement | Result | Method |
| --- | --- | --- |
| Native DSpark | ~232 output tok/s | 1,024-token fixed-output test, three warm runs, 4.42 s each, concurrency 1 |
| No speculative decoding | ~162 output tok/s | Same 1,024-token test, warm runs 6.30–6.34 s, concurrency 1 |
| Two short requests | 2.93 s parallel vs 5.83 s sequential | Two identical 512-token outputs; synthetic, not an agent workload |
| Long-context retrieval | 130,759 input tokens and 208 output tokens | Synthetic repeated filler and one embedded fact, correct result |
| Long-context + vision | 130,225 input tokens including image tokens | Synthetic image and text fact recognized correctly |

In one later real-use log sample, prompt-processing batches of at least 512 tokens had a median of ~11,558 tok/s, and single-request generation samples a median of ~213 tok/s. These are sampled server throughput values, not complete-request benchmarks or guaranteed speeds. In another earlier observation, two active requests yielded ~351 output tok/s combined at the median; workloads differed. Full numerical test artifacts are in [`benchmarks/`](benchmarks/).

Reproduce the fixed-output and synthetic retrieval checks against a running server:

```bash
python benchmarks/measure_speed.py dspark
python benchmarks/test_long_context.py 30000 256
```

The speed script forces 1,024 output tokens and writes a local, ignored results file. The context script uses repetitive synthetic filler and writes its result to a separate ignored file. These tests create inference load; run them when the server is idle.

## Limits and tradeoffs

- Context is 131,072 tokens **including output**. Model reasoning also consumes output tokens; set a sufficient `max_tokens` budget.
- The tested setup allocated approximately 137,984 full-attention KV slots **shared across requests** and 6,784 sliding-window slots. Two independent 96k contexts cannot both fit in that pool at once; requests may wait or cached entries may be evicted. Shared prefixes can reduce memory use.
- The long-context tests are narrow retrieval checks, not comprehensive long-context accuracy or sustained-load evaluations. Test on your own workloads.
- BF16 KV is intentional: the image's FA4 FP8 path rejected SM120 in our MXFP8 KV attempt. Reducing DSpark's draft block was also rejected by the fixed checkpoint. These observations apply to the pinned build, not necessarily newer releases.
- Vision and tools are enabled in the server. Multimodal long-context and a simple read/repair/test tool-call sequence were checked.

## Credits

Model: [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored), based on [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). Server and day-zero image: [SGLang](https://github.com/sgl-project/sglang). Model licensing is described in its Hugging Face repository.
