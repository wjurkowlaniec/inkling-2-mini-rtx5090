# Recommended: Basalt on RTX 5090

This configuration uses Basalt v0.5.7 with FP8 KV, Vision, the built-in DSpark drafter and two slots, each configured for 131,072 total input + output tokens. Measurements, conversion precision and validation limits: [benchmark report](../benchmarks/basalt/README.md). The server uses the same `inkling` model alias and port 6020 as the SGLang setup; run only one GPU server at a time.

The official and uncensored checkpoints are supported by the configuration. Our end-to-end tests used the uncensored checkpoint. Conversion rounds dense projections to Q8_0 and the engine uses FP8 KV, so numerical behavior is not identical to SGLang's BF16 path. Validate your agent and workloads before relying on it. The first start of a model can take time and interrupted requests are not resumed by the server.

## Prepare files

Run from this repository's root. These commands install the official Basalt release and converter; check available disk space first. A checkpoint and the converted pack each occupy roughly 20 GB, and Docker image layers need additional space. Python, Git, curl and an NVIDIA GPU-enabled Docker installation are required.

```bash
mkdir -p basalt/runtime basalt/models basalt/downloads
curl -fL https://github.com/jesdga95/basalt/releases/download/v0.5.7/basalt-v0.5.7-linux-x64.tar.gz -o basalt/downloads/basalt.tar.gz
echo 'a390c775c41d54454ca21c1e547855bf6c251732533ca387e8b890e538090525  basalt/downloads/basalt.tar.gz' | sha256sum -c -
tar -xzf basalt/downloads/basalt.tar.gz -C basalt/runtime --strip-components=1
git clone --branch v0.5.7 --depth 1 https://github.com/jesdga95/basalt.git basalt/downloads/source
python3 -m venv basalt/downloads/venv
basalt/downloads/venv/bin/pip install -r basalt/downloads/source/requirements.txt
```

Download the official [Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini), or use an existing local [OpenUncensored checkpoint](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored). For the official checkpoint:

```bash
basalt/downloads/venv/bin/pip install -U huggingface_hub
basalt/downloads/venv/bin/hf download thinkingmachines/Inkling-2-Mini --local-dir basalt/models/checkpoint
basalt/downloads/venv/bin/python basalt/downloads/source/tools/convert/convert.py \
  --checkpoint basalt/models/checkpoint --out basalt/models/inkling.basalt --workers 4
```

For an existing checkpoint, replace `basalt/models/checkpoint` with its path in the conversion command. To use existing files without copying them, set `BASALT_DIR` to the absolute release directory and `BASALT_MODEL` to the absolute converted pack path.

## Start and verify

After stopping any other GPU server using port 6020, from the repository root:

```bash
docker compose -f basalt/docker-compose.yml up -d
docker compose -f basalt/docker-compose.yml logs -f
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

The unauthenticated API binds only to localhost. Use an authenticated proxy for LAN access. `docker compose -f basalt/docker-compose.yml down` stops this server. The SGLang alternative remains in the root Compose file.
