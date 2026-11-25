# SGLang + DFlash2 on RTX 5090

This alternative retains the original checkpoint's dense weights and uses BF16 KV, DFlash2, Vision and two simultaneous requests. The full-attention KV pool is **139,264 tokens shared across requests**; it does not guarantee two independent 128k prompts at once. Each request's 131,072-token limit includes its output. More requests can wait or contend for cache. The settings were tested with the uncensored checkpoint on 2026-11-25; the official checkpoint is compatible in principle but was not independently validated in this SGLang build. See [measured results](../benchmarks/sglang/README.md) and [the Basalt comparison](../benchmarks/basalt/README.md).

Requirements: Linux, one NVIDIA RTX 5090 (32 GB), NVIDIA driver and Container Toolkit, Docker Compose, and room for the model, draft, cache, and image. The base image is approximately 56 GB before layer sharing. GPU memory is reserved aggressively; don't start this at the same time as Basalt on one card.

The pinned [model-specific DFlash2 compatibility patch](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/blob/4f9088ea66c21f2b99ed657d1941c306b03e84d2/sglang_inkling_dflash2.py) is copied into the build context under `scripts/`. The included `Dockerfile` applies it to the pinned SGLang image. Model weights are downloaded separately.

## Start

Run from the repository root:

```bash
python -m pip install --user -U huggingface_hub
mkdir -p sglang/models/Inkling-2-Mini sglang/cache/sglang
hf download thinkingmachines/Inkling-2-Mini --local-dir sglang/models/Inkling-2-Mini
hf download incoai/Inkling-2-Mini-DFlash2 --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2 \
  --exclude 'assets/*' --local-dir sglang/models/Inkling-2-Mini-DFlash2

docker compose -f sglang/docker-compose.yml up -d --build
docker compose -f sglang/docker-compose.yml logs -f
```

For the tested [OpenUncensored model](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored), download it and set `MODEL_DIR` to its **absolute** path before starting; `DRAFT_DIR` and `CACHE_DIR` can similarly select existing directories. Relative Compose bind mounts resolve from `sglang/`.

```bash
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

The unauthenticated API listens on localhost. For LAN access use an authenticated proxy; no proxy is included here. Stop with `docker compose -f sglang/docker-compose.yml down`. The first launch may take several minutes to compile kernels and capture graphs.
