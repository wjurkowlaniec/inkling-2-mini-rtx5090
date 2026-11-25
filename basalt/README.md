# Basalt on RTX 5090

The recommended speed configuration uses Basalt **v0.5.7**, DSpark, FP8 KV, Vision and two slots configured for **131,072 tokens each**, including input and output. The model alias is `inkling`; the unauthenticated API binds to `127.0.0.1:6020`. Measurements and validation: [Basalt benchmark report](../benchmarks/basalt/README.md). Both the official and uncensored variants passed the narrow checks recorded there.

Requirements: Linux, RTX 5090 (32 GB), NVIDIA driver with CUDA 13 support, NVIDIA Container Toolkit, Docker Compose, Git, curl and Python. Allow roughly 20 GB for the converted model plus space for the large base image and caches. Use one GPU server at a time. Basalt quantizes dense projections to Q8_0 and uses FP8 KV; output can differ from SGLang. Both checkpoint variants are obtained separately.

## Install Basalt and the official model

Commands run from the repository root. This path uses the preconverted Basalt pack derived from the **official Thinking Machines checkpoint**. The pack is a third-party conversion; its native expert codes are preserved, but some dense weights are quantized. The SHA-256 checks pin the tested artifacts.

```bash
mkdir -p basalt/runtime basalt/models basalt/downloads
curl -fL https://github.com/jesdga95/basalt/releases/download/v0.5.7/basalt-v0.5.7-linux-x64.tar.gz -o basalt/downloads/basalt.tar.gz
echo 'a390c775c41d54454ca21c1e547855bf6c251732533ca387e8b890e538090525  basalt/downloads/basalt.tar.gz' | sha256sum -c -
tar -xzf basalt/downloads/basalt.tar.gz -C basalt/runtime --strip-components=1
python3 -m venv basalt/downloads/venv
basalt/downloads/venv/bin/pip install -U huggingface_hub
basalt/downloads/venv/bin/hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-MX8.basalt --local-dir basalt/models
echo '28ec70723feaa037942a02f55b9d5e898276f3e62a3b1fa083745f932a041b99  basalt/models/Inkling-2-Mini-NVFP4-MX8.basalt' | sha256sum -c -
mv basalt/models/Inkling-2-Mini-NVFP4-MX8.basalt basalt/models/inkling.basalt
```

[Official Thinking Machines model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Basalt model pack](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) · [OpenUncensored checkpoint](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored)

## Use an existing checkpoint

To use the uncensored checkpoint or convert the original official weights yourself, first download the chosen checkpoint and install the matching converter. Conversion needs room for both the input checkpoint and the roughly 20 GB output pack.

```bash
git clone --branch v0.5.7 --depth 1 https://github.com/jesdga95/basalt.git basalt/downloads/source
basalt/downloads/venv/bin/pip install -r basalt/downloads/source/requirements.txt
basalt/downloads/venv/bin/python basalt/downloads/source/tools/convert/convert.py \
  --checkpoint /absolute/path/to/checkpoint --out basalt/models/inkling.basalt --workers 4
```

To reuse existing artifacts without copying them, set `BASALT_DIR` to the absolute release directory and `BASALT_MODEL` to the absolute `.basalt` file path before starting Compose.

## Start and verify

```bash
docker compose -f basalt/docker-compose.yml up -d
docker compose -f basalt/docker-compose.yml logs -f
curl -f http://127.0.0.1:6020/health
curl http://127.0.0.1:6020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

The first load takes time. Interrupted requests are not resumed by the server. For LAN access, place an authenticated proxy in front of the localhost API. Stop with `docker compose -f basalt/docker-compose.yml down`. For the alternative engine, see [SGLang + DFlash2](../sglang/README.md).
